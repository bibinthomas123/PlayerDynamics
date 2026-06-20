"""
MatchOrchestrator — PlayerDynamics

Drives the full team-tactical pipeline incrementally and publishes results
onto Redis Streams. This is the "live" complement to the batch engines
already validated independently (see TACTICAL_EVENT.../POSSESSION_INTELLIGENCE.../
TEAMSTATE_*/COACH_*_IMPLEMENTATION.md):

    TacticalEvent -> Possession -> TeamState -> TeamStateTrend -> CoachInsight -> CoachSituation

Until this module, every engine above was a pure batch function
(generate(full_list) -> List[...]) exercised only by tests -- nothing in
this codebase actually ran the pipeline against arriving events, and
nothing published a single object onto any of the Redis Streams implemented
in REDIS_STREAM_IMPLEMENTATION.md. This module closes that gap. Note this
is a SEPARATE system from analysis/orchestrator.py's PlayersDataAnalysisPipeline,
which drives the unrelated per-player physiological/anomaly pipeline.

Design: incremental wrapper around unmodified batch engines
----------------------------------------------------------------
Rather than rewrite five already-tested, deterministic batch engines into
true online/incremental algorithms (high risk, large surface area for a v1),
MatchOrchestrator buffers every TacticalEvent it receives and, on tick(),
RE-RUNS the existing, UNMODIFIED PossessionEngine / TeamStateBuilder /
TeamStateTrendBuilder / CoachInsightEngine / CoachSituationEngine over the
full accumulated buffer. Match-length event volumes (~7,500 events for a
60-minute handball match) make a full batch recompute on every tick cheap
(well under a second on this hardware -- see MATCH_ORCHESTRATOR_IMPLEMENTATION.md's
real-data timing). "Live" here means "ticks frequently enough to feel
live", not "O(1) work per incoming event".

Every published object's content is therefore IDENTICAL to what the
already-validated batch engines would produce from the same events -- this
module adds no new analytical logic, only WHEN to publish what.

The "tail is provisional" rule
----------------------------------
A batch recompute over a still-growing buffer necessarily treats "now" as
"end of stream":
  - the most recent Possession may have been force-closed by
    PossessionEngine's end-of-stream rule (see analysis/possession.py) --
    a real terminator/team-switch event closes every OTHER possession in
    the list, so only the very last entry can be an end-of-stream artifact;
  - the most recent TeamState/TeamStateTrend/CoachInsight/CoachSituation
    tick covers a window that has not finished accumulating events yet.
Both can change on the NEXT tick once more events arrive. So on every
tick():
  - Possession: every possession except the LAST one in the freshly
    recomputed list is published exactly once (tracked by possession_id);
    the last one is republished every tick until finalize() is called.
  - TeamState / TeamStateTrend / CoachInsight / CoachSituation: for each
    window_seconds independently, the entry/entries at the MAXIMUM
    timestamp in the freshly recomputed list are the "tail" (republished
    every tick); everything at an earlier timestamp is final (windows only
    look backward, so an earlier tick's content cannot change once a later
    tick exists) and is published exactly once.

finalize() must be called once, at match end, to flush whatever is still
"tail" (the final live possession + the final window of each layer) as
permanently final.

Redis Streams integration
----------------------------
tick()/finalize() return {stream_name: [dataclass, ...]}, ready for
RedisStreamProducer.publish_dataclass(). consume_tracking_events() pulls
new entries off the (PlayerDynamics-internal-only) tracking.events
RedisStreamConsumer, decodes them via ingestion.stream_codec, feeds them
into ingest_event(), and acks them. consume_match_events()/
consume_match_context() do the same for the two Backend-owned streams
(match.events, match.context -- see ingestion/match_event.py and
BACKEND_INTEGRATION_IMPLEMENTATION.md §1 for the ownership rule), storing
what they decode rather than feeding it into the existing engines.
publish() is the other direction ("PlayerDynamics -> Redis Stream ->
Backend").
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config.redis_client import RedisStreamConsumer, RedisStreamProducer, StreamTopics
from analysis.coach_insight import CoachInsightEngine
from analysis.coach_situation import CoachSituationEngine
from analysis.possession import PossessionEngine
from analysis.team_state import TeamStateBuilder
from analysis.team_state_trend import TeamStateTrendBuilder
from ingestion.match_event import MatchContext, MatchEvent
from ingestion.tactical_event import TacticalEvent

logger = logging.getLogger(__name__)


def _finalize_by_tail(
    items: List[Any],
    key_fn: Callable[[Any], Tuple],
    published_keys: Set[Tuple],
    tail_ts: Optional[Any] = None,
) -> List[Any]:
    """
    Splits `items` (all sharing one window_seconds) into "final" (timestamp
    strictly before `tail_ts`) and "tail" (timestamp equal to `tail_ts`).
    Final entries are returned only the first time their key is seen (and
    recorded in published_keys); tail entries are returned every call
    (never recorded, so they keep being treated as live/current).

    tail_ts MUST be the evaluation point's own schedule tail (i.e. the max
    timestamp among the TeamStateTrend ticks for this window_seconds), NOT
    derived from `items` itself when `items` is CoachInsight/CoachSituation
    output -- both can legitimately be EMPTY at the true tail evaluation
    point (no insight/situation fired there is a valid, expected outcome,
    not a sign the point doesn't exist yet). Deriving the tail from an
    insight/situation list's own max would mistake "nothing fired at the
    true tail" for "the true tail is actually one tick earlier", repeatedly
    re-previewing that earlier tick until something finally fires later --
    inflating the publish count relative to a one-shot batch run. If
    `tail_ts` is omitted, it falls back to max(item.timestamp for item in
    items), which is safe only for TeamState/TeamStateTrend (exactly one
    entry per team always exists at every evaluation point).
    """
    if not items:
        return []
    max_ts = tail_ts if tail_ts is not None else max(i.timestamp for i in items)
    result = []
    for i in items:
        if i.timestamp == max_ts:
            result.append(i)  # tail -- always (re)published
        else:
            k = key_fn(i)
            if k not in published_keys:
                published_keys.add(k)
                result.append(i)
    return result


class MatchOrchestrator:
    """
    Per-match incremental driver for the team-tactical pipeline.

    Usage
    -----
        orch = MatchOrchestrator(match_id="3387")
        for event in incoming_tactical_events:
            orch.ingest_event(event)
            if should_tick():                      # e.g. every N events
                new_objects = orch.tick()
                orch.publish(producer, new_objects)
        final_objects = orch.finalize()
        orch.publish(producer, final_objects)

    Or, driven directly from Redis Streams:
        orch.consume_tracking_events(consumer, count=200)   # ingest
        new_objects = orch.tick()
        orch.publish(producer, new_objects)
    """

    def __init__(self, match_id: str, player_meta: Optional[Dict[int, Any]] = None) -> None:
        self.match_id = match_id
        self.player_meta = player_meta
        self.events: List[TacticalEvent] = []

        # Backend-owned context (see ingestion/match_event.py) -- received
        # and held for downstream consumers, never recomputed or validated
        # here. Not yet fused into the analytics engines below (TeamState/
        # Possession/etc. still derive purely from Kinexon TacticalEvents);
        # that fusion is deliberately out of scope for this integration --
        # see BACKEND_INTEGRATION_IMPLEMENTATION.md's "not done here" section.
        self.match_events: List[MatchEvent] = []
        self.latest_match_context: Optional[MatchContext] = None

        self._team_state_builder = TeamStateBuilder()
        self._trend_builder = TeamStateTrendBuilder()
        self._insight_engine = CoachInsightEngine()
        self._situation_engine = CoachSituationEngine()
        self._possession_engine = PossessionEngine()

        self._published_possession_count = 0  # count of possessions already finalized
        self._published_keys: Dict[str, Set[Tuple]] = {
            "team_state": set(),
            "trend": set(),
            "insight": set(),
            "situation": set(),
        }
        self._finalized = False

    # ─────────────────────────────────────────────────────────────────────
    # Ingestion
    # ─────────────────────────────────────────────────────────────────────

    def ingest_event(self, event: TacticalEvent) -> None:
        """Buffer one TacticalEvent. Does not trigger recomputation -- call tick() for that."""
        if self._finalized:
            logger.warning("MatchOrchestrator(%s): ingest_event() called after finalize()", self.match_id)
        self.events.append(event)

    def consume_tracking_events(self, consumer: RedisStreamConsumer, count: int = 200) -> int:
        """
        Pull up to `count` new entries off a tracking.events RedisStreamConsumer,
        decode each into a TacticalEvent, ingest it, and ack it. Returns the
        number of events ingested (0 if nothing new was available).
        """
        from ingestion.stream_codec import decode

        entries = consumer.read_new(count=count)
        for entry_id, fields in entries:
            event = decode(fields)
            self.ingest_event(event)
            consumer.ack(entry_id)
        return len(entries)

    def consume_match_events(self, consumer: RedisStreamConsumer, count: int = 200) -> int:
        """
        Pull up to `count` new entries off a match.events RedisStreamConsumer
        (Backend-published coach/match actions), decode each into a
        MatchEvent, store it, and ack it. Returns the number consumed.

        PlayerDynamics does not validate or recompute these -- Backend is
        their source of truth (see ingestion/match_event.py). They are held
        on self.match_events for any future consumer that wants to read
        them alongside the Kinexon-derived TacticalEvent stream; the
        existing analytics engines (Possession/TeamState/etc.) do not read
        this list today.
        """
        from ingestion.stream_codec import decode

        entries = consumer.read_new(count=count)
        for entry_id, fields in entries:
            event = decode(fields)
            self.match_events.append(event)
            consumer.ack(entry_id)
        return len(entries)

    def consume_match_context(self, consumer: RedisStreamConsumer, count: int = 200) -> int:
        """
        Pull up to `count` new entries off a match.context RedisStreamConsumer
        (Backend-published running match state: score/clock/period), decode
        each into a MatchContext, keep only the most recent one (it is a
        running snapshot, not a log), and ack every entry read. Returns the
        number consumed.
        """
        from ingestion.stream_codec import decode

        entries = consumer.read_new(count=count)
        for entry_id, fields in entries:
            context = decode(fields)
            if self.latest_match_context is None or context.timestamp >= self.latest_match_context.timestamp:
                self.latest_match_context = context
            consumer.ack(entry_id)
        return len(entries)

    # ─────────────────────────────────────────────────────────────────────
    # Recomputation
    # ─────────────────────────────────────────────────────────────────────

    def _recompute(self, include_tail_possession: bool) -> Dict[str, Any]:
        """
        include_tail_possession=False (tick()): the last possession in the
        freshly recomputed list may be an end-of-stream artifact (see module
        docstring) -- it is excluded from CoachSituation's possession
        aggregates so that a situation published as "final" at a non-tail
        trend timestamp can never later turn out to have been computed from
        a possession that still had more events coming. The possession
        itself is still returned in full under "possessions" (tick() needs
        the untrimmed list to detect newly-finalized possessions).

        include_tail_possession=True (finalize()): the match is over, so the
        end-of-stream closure of the last possession IS its genuine final
        state -- identical to what a one-shot batch run over the same full
        event list would produce. It must be included so the final window's
        situation matches that ground truth.
        """
        possessions = self._possession_engine.generate(self.events)
        situation_possessions = possessions if include_tail_possession else possessions[:-1]
        team_states = self._team_state_builder.build_dual_window(self.events)
        window_lengths = tuple(team_states.keys())
        trends = {ws: self._trend_builder.build(team_states[ws]) for ws in window_lengths}
        insights = {ws: self._insight_engine.generate(trends[ws]) for ws in window_lengths}
        situations = {
            ws: self._situation_engine.generate(situation_possessions, team_states[ws], trends[ws], insights[ws])
            for ws in window_lengths
        }
        return {
            "window_lengths": window_lengths,
            "possessions": possessions, "team_states": team_states,
            "trends": trends, "insights": insights, "situations": situations,
        }

    def tick(self) -> Dict[str, List[Any]]:
        """
        Recompute the full pipeline over the current buffer and return
        {stream_name: [newly-publishable objects]} per the tail-is-provisional
        rule described in this module's docstring.
        """
        state = self._recompute(include_tail_possession=False)
        result: Dict[str, List[Any]] = {topic: [] for topic in StreamTopics.OUTBOUND}

        possessions = state["possessions"]
        finalized_possessions = possessions[:-1] if possessions else []
        new_possessions = finalized_possessions[self._published_possession_count:]
        self._published_possession_count = len(finalized_possessions)
        result[StreamTopics.ANALYTICS_POSSESSIONS].extend(new_possessions)

        for ws in state["window_lengths"]:
            # The trend ticks are the authoritative evaluation-point schedule
            # for this window_seconds (TeamState/TeamStateTrend always have
            # exactly one entry per team per tick); insights/situations can
            # legitimately be empty at that same tick, so their tail must be
            # pinned to the trend's tail, never derived from their own
            # (possibly silent) list -- see _finalize_by_tail's docstring.
            trend_tail_ts = max((t.timestamp for t in state["trends"][ws]), default=None)

            result[StreamTopics.ANALYTICS_TEAMSTATE].extend(_finalize_by_tail(
                state["team_states"][ws], lambda s: (s.team_id, s.window_seconds, s.timestamp),
                self._published_keys["team_state"],
            ))
            result[StreamTopics.ANALYTICS_TRENDS].extend(_finalize_by_tail(
                state["trends"][ws], lambda t: (t.team_id, t.window_seconds, t.timestamp),
                self._published_keys["trend"],
            ))
            result[StreamTopics.ANALYTICS_INSIGHTS].extend(_finalize_by_tail(
                state["insights"][ws],
                lambda i: (i.team_id, i.metadata.get("window_seconds"), i.timestamp, i.category),
                self._published_keys["insight"], tail_ts=trend_tail_ts,
            ))
            result[StreamTopics.ANALYTICS_SITUATIONS].extend(_finalize_by_tail(
                state["situations"][ws],
                lambda s: (s.team_id, s.source_metrics.get("window_seconds"), s.timestamp, s.situation_type),
                self._published_keys["situation"], tail_ts=trend_tail_ts,
            ))

        return result

    def finalize(self) -> Dict[str, List[Any]]:
        """
        Call once at match end. Flushes whatever is still "tail" (the final
        live possession, and the final window of every other layer) as
        permanently final, then marks this orchestrator closed -- further
        ingest_event() calls log a warning (the match is considered over).
        """
        state = self._recompute(include_tail_possession=True)
        result: Dict[str, List[Any]] = {topic: [] for topic in StreamTopics.OUTBOUND}

        possessions = state["possessions"]
        new_possessions = possessions[self._published_possession_count:]
        self._published_possession_count = len(possessions)
        result[StreamTopics.ANALYTICS_POSSESSIONS].extend(new_possessions)

        for ws in state["window_lengths"]:
            # At finalize, every remaining entry (including the tail) is final --
            # register tail keys too so a stray later tick() can't republish them.
            for items, stream, key_fn, bucket in (
                (state["team_states"][ws], StreamTopics.ANALYTICS_TEAMSTATE,
                 lambda s: (s.team_id, s.window_seconds, s.timestamp), "team_state"),
                (state["trends"][ws], StreamTopics.ANALYTICS_TRENDS,
                 lambda t: (t.team_id, t.window_seconds, t.timestamp), "trend"),
                (state["insights"][ws], StreamTopics.ANALYTICS_INSIGHTS,
                 lambda i: (i.team_id, i.metadata.get("window_seconds"), i.timestamp, i.category), "insight"),
                (state["situations"][ws], StreamTopics.ANALYTICS_SITUATIONS,
                 lambda s: (s.team_id, s.source_metrics.get("window_seconds"), s.timestamp, s.situation_type),
                 "situation"),
            ):
                for item in items:
                    k = key_fn(item)
                    if k not in self._published_keys[bucket]:
                        self._published_keys[bucket].add(k)
                        result[stream].append(item)

        self._finalized = True
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Publishing
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def publish(producer: RedisStreamProducer, new_objects: Dict[str, List[Any]]) -> int:
        """Publish every object in new_objects onto its stream. Returns total published count."""
        n = 0
        for stream, objs in new_objects.items():
            for obj in objs:
                producer.publish_dataclass(stream, obj)
                n += 1
        return n
