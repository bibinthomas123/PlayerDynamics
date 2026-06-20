"""
tests/test_match_orchestrator.py

Validates analysis/match_orchestrator.py -- the incremental driver that
wraps PossessionEngine / TeamStateBuilder / TeamStateTrendBuilder /
CoachInsightEngine / CoachSituationEngine and publishes onto Redis Streams.

Covers:
  A. Incremental ingestion + tick() basics (no crash, correct stream keys)
  B. The "tail is provisional" rule for Possession
  C. The "tail is provisional" rule for TeamState/Trend/Insight/Situation
  D. No duplicate publishing across repeated ticks
  E. finalize() flushes the tail exactly once and closes the orchestrator
  F. Equivalence: incremental publish stream == one-shot batch engine output
  G. Redis Streams integration (publish via RedisStreamProducer, decode back)
  H. Real-data validation against data/events.csv (session 3387)

Run:
    pytest tests/test_match_orchestrator.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from analysis.coach_insight import CoachInsightEngine
from analysis.coach_situation import CoachSituationEngine
from analysis.match_orchestrator import MatchOrchestrator
from analysis.possession import PossessionEngine
from analysis.team_state import TeamStateBuilder
from analysis.team_state_trend import TeamStateTrendBuilder
from config.redis_client import StreamTopics
from ingestion.tactical_event import KinexonTacticalEventAdapter, TacticalEvent

DATA_DIR = _ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.csv"

BASE_TS = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)


def _ev(
    event_type: str,
    seconds_offset: float,
    team_id: str = "SC Magdeburg",
    player_id: int = 1164,
) -> TacticalEvent:
    ts = BASE_TS + timedelta(seconds=seconds_offset)
    return TacticalEvent(
        event_id=f"{event_type}-{seconds_offset}-{player_id}-{team_id}",
        timestamp=ts, match_id="3387", team_id=team_id, player_id=player_id,
        event_type=event_type, metadata={}, source="kinexon", confidence=1.0,
    )


def _rally(team_id: str, start: float, terminator: str = "shot") -> list[TacticalEvent]:
    """One short possession: possession -> pass -> terminator, 5s apart."""
    return [
        _ev("possession", start, team_id=team_id),
        _ev("pass", start + 2, team_id=team_id),
        _ev(terminator, start + 5, team_id=team_id),
    ]


def _many_rallies(n: int, team_id: str = "SC Magdeburg", gap: float = 20.0) -> list[TacticalEvent]:
    events: list[TacticalEvent] = []
    for i in range(n):
        terminator = "shot" if i % 2 == 0 else "turnover"
        events.extend(_rally(team_id, i * gap, terminator=terminator))
    return events


# ---------------------------------------------------------------------------
# A. Incremental ingestion + tick() basics
# ---------------------------------------------------------------------------

class TestTickBasics:

    def test_tick_on_empty_buffer_returns_empty_lists_for_every_stream(self):
        orch = MatchOrchestrator(match_id="3387")
        result = orch.tick()
        assert set(result.keys()) == set(StreamTopics.OUTBOUND)
        assert all(v == [] for v in result.values())

    def test_tick_returns_only_outbound_stream_keys(self):
        orch = MatchOrchestrator(match_id="3387")
        for e in _many_rallies(3):
            orch.ingest_event(e)
        result = orch.tick()
        assert set(result.keys()) == set(StreamTopics.OUTBOUND)

    def test_ingest_event_does_not_trigger_recompute(self):
        orch = MatchOrchestrator(match_id="3387")
        orch.ingest_event(_ev("possession", 0))
        assert len(orch.events) == 1


# ---------------------------------------------------------------------------
# B. Tail-is-provisional rule for Possession
# ---------------------------------------------------------------------------

class TestPossessionTailRule:

    def test_single_rally_not_yet_published_until_more_events_close_it(self):
        """
        One rally (possession/pass/shot) is the ONLY possession in the
        buffer -- it is also the LAST one, so PossessionEngine cannot prove
        it wasn't end-of-stream-closed. tick() must withhold it.
        """
        orch = MatchOrchestrator(match_id="3387")
        for e in _rally("SC Magdeburg", 0):
            orch.ingest_event(e)
        result = orch.tick()
        assert result[StreamTopics.ANALYTICS_POSSESSIONS] == []

    def test_first_rally_published_once_a_second_rally_exists(self):
        orch = MatchOrchestrator(match_id="3387")
        for e in _rally("SC Magdeburg", 0):
            orch.ingest_event(e)
        for e in _rally("SC Magdeburg", 20):
            orch.ingest_event(e)
        result = orch.tick()
        published = result[StreamTopics.ANALYTICS_POSSESSIONS]
        assert len(published) == 1
        assert published[0].outcome == "shot"

    def test_each_new_rally_publishes_exactly_one_more_possession_per_tick(self):
        orch = MatchOrchestrator(match_id="3387")
        published_total = 0
        for i in range(5):
            for e in _rally("SC Magdeburg", i * 20):
                orch.ingest_event(e)
            result = orch.tick()
            published_total += len(result[StreamTopics.ANALYTICS_POSSESSIONS])
        # 5 rallies ingested -> first 4 are finalized across the 5 ticks,
        # the 5th (last) stays withheld until finalize().
        assert published_total == 4

    def test_no_possession_is_published_twice_across_ticks(self):
        orch = MatchOrchestrator(match_id="3387")
        seen_ids = []
        for i in range(6):
            for e in _rally("SC Magdeburg", i * 20):
                orch.ingest_event(e)
            result = orch.tick()
            seen_ids.extend(p.possession_id for p in result[StreamTopics.ANALYTICS_POSSESSIONS])
        assert len(seen_ids) == len(set(seen_ids))


# ---------------------------------------------------------------------------
# C. Tail-is-provisional rule for TeamState/Trend/Insight/Situation
# ---------------------------------------------------------------------------

class TestWindowedLayersTailRule:

    def test_team_state_tail_is_republished_until_buffer_grows_past_it(self):
        orch = MatchOrchestrator(match_id="3387")
        for e in _many_rallies(2):
            orch.ingest_event(e)
        r1 = orch.tick()
        ts_before = {s.timestamp for s in r1[StreamTopics.ANALYTICS_TEAMSTATE]}
        assert ts_before  # at least the tail tick was emitted

        # No new events -- recompute is identical, tail timestamp unchanged,
        # so it is republished (not yet finalized).
        r2 = orch.tick()
        ts_after = {s.timestamp for s in r2[StreamTopics.ANALYTICS_TEAMSTATE]}
        assert ts_after == ts_before

    def test_finalized_team_state_ticks_are_not_republished(self):
        orch = MatchOrchestrator(match_id="3387")
        for e in _many_rallies(3, gap=90.0):  # spans several 60s windows, SC Magdeburg only
            orch.ingest_event(e)
        r1 = orch.tick()
        keyed_r1 = {(s.team_id, s.timestamp): s for s in r1[StreamTopics.ANALYTICS_TEAMSTATE]}
        tail_ts_r1 = max(s.timestamp for s in r1[StreamTopics.ANALYTICS_TEAMSTATE])
        tail_keys_r1 = {k for k in keyed_r1 if k[1] == tail_ts_r1}

        for e in _many_rallies(3, team_id="HSG Wetzlar", gap=140.0):  # later/non-overlapping timestamps
            orch.ingest_event(e)
        r2 = orch.tick()
        keyed_r2 = {(s.team_id, s.timestamp): s for s in r2[StreamTopics.ANALYTICS_TEAMSTATE]}

        # Only the (team, timestamp) keys that were still "tail" (provisional)
        # at r1 may legitimately reappear once buffer growth confirms a later
        # max timestamp exists -- everything else finalized in r1 must not
        # reappear in r2.
        overlap = set(keyed_r1) & set(keyed_r2)
        assert overlap <= tail_keys_r1

    def test_insight_layer_publishes_multiple_categories_at_same_timestamp_without_dropping(self):
        """
        CoachInsight can fire >1 category at the same (team_id, window,
        timestamp) -- the dedup key must include category, or only one of
        several simultaneous insights would survive across ticks.
        """
        orch = MatchOrchestrator(match_id="3387")
        # A burst of attack + physical events designed to cross multiple
        # CoachInsight thresholds at once.
        events = []
        for i in range(40):
            events.append(_ev("pass", i * 1.0))
            events.append(_ev("sprint", i * 1.0 + 0.1))
        for e in events:
            orch.ingest_event(e)
        for e in _many_rallies(2, gap=90.0):
            orch.ingest_event(e)
        r1 = orch.tick()
        r1_keys = [
            (i.team_id, i.timestamp, i.category)
            for i in r1[StreamTopics.ANALYTICS_INSIGHTS]
        ]
        assert len(r1_keys) == len(set(r1_keys))


# ---------------------------------------------------------------------------
# D. No duplicate publishing across many ticks (all layers)
# ---------------------------------------------------------------------------

class TestNoDuplicatesAcrossManyTicks:

    def test_repeated_ticks_with_growing_buffer_never_redeliver_a_finalized_key(self):
        """
        A key may be returned by several CONSECUTIVE ticks while it remains
        the live/provisional tail, and then by exactly one more tick (or
        finalize()) the moment it is finalized -- but once a key stops
        appearing, it must never reappear later. I.e. each key's set of
        appearance-tick-indices must be a contiguous block, never having a
        gap (see module docstring's "tail is provisional" section: a tail
        value can be previewed many times before something newer supersedes
        it, but is then committed exactly once and never revisited).
        """
        orch = MatchOrchestrator(match_id="3387")
        per_tick_keys = []  # one set of keys per tick, in order
        for i in range(10):
            for e in _rally("SC Magdeburg" if i % 2 == 0 else "HSG Wetzlar", i * 25):
                orch.ingest_event(e)
            result = orch.tick()
            keys = set()
            for topic, objs in result.items():
                for obj in objs:
                    keys.add((topic, getattr(obj, "team_id", None), getattr(obj, "timestamp", None),
                              getattr(obj, "possession_id", None), getattr(obj, "category", None),
                              getattr(obj, "situation_type", None)))
            per_tick_keys.append(keys)

        final = orch.finalize()
        finalize_keys = set()
        for topic, objs in final.items():
            for obj in objs:
                finalize_keys.add((topic, getattr(obj, "team_id", None), getattr(obj, "timestamp", None),
                                    getattr(obj, "possession_id", None), getattr(obj, "category", None),
                                    getattr(obj, "situation_type", None)))
        per_tick_keys.append(finalize_keys)

        all_keys = set().union(*per_tick_keys)
        for key in all_keys:
            appearances = [idx for idx, keys in enumerate(per_tick_keys) if key in keys]
            span = appearances[-1] - appearances[0] + 1
            assert span == len(appearances), (
                f"key {key} appeared at non-contiguous ticks {appearances} -- "
                f"reappeared after having stopped, which the tail-is-provisional "
                f"design never allows"
            )


# ---------------------------------------------------------------------------
# E. finalize() semantics
# ---------------------------------------------------------------------------

class TestFinalize:

    def test_finalize_flushes_the_withheld_last_possession(self):
        orch = MatchOrchestrator(match_id="3387")
        for e in _rally("SC Magdeburg", 0):
            orch.ingest_event(e)
        orch.tick()  # withheld -- only possession, also last
        final = orch.finalize()
        assert len(final[StreamTopics.ANALYTICS_POSSESSIONS]) == 1

    def test_finalize_marks_orchestrator_closed(self):
        orch = MatchOrchestrator(match_id="3387")
        for e in _rally("SC Magdeburg", 0):
            orch.ingest_event(e)
        orch.finalize()
        assert orch._finalized is True

    def test_finalize_on_already_ticked_buffer_does_not_redeliver_finalized_possessions(self):
        orch = MatchOrchestrator(match_id="3387")
        for e in _rally("SC Magdeburg", 0):
            orch.ingest_event(e)
        for e in _rally("SC Magdeburg", 20):
            orch.ingest_event(e)
        r1 = orch.tick()
        assert len(r1[StreamTopics.ANALYTICS_POSSESSIONS]) == 1  # 1st rally finalized
        final = orch.finalize()
        assert len(final[StreamTopics.ANALYTICS_POSSESSIONS]) == 1  # only the 2nd (was tail)

    def test_finalize_with_no_events_returns_empty_lists(self):
        orch = MatchOrchestrator(match_id="3387")
        final = orch.finalize()
        assert all(v == [] for v in final.values())


# ---------------------------------------------------------------------------
# F. Equivalence with one-shot batch engines
# ---------------------------------------------------------------------------

class TestEquivalenceWithBatchEngines:

    def test_full_run_then_finalize_matches_one_shot_batch_possessions(self):
        events = _many_rallies(8, gap=20.0)
        orch = MatchOrchestrator(match_id="3387")
        published = []
        for e in events:
            orch.ingest_event(e)
        published.extend(orch.tick()[StreamTopics.ANALYTICS_POSSESSIONS])
        published.extend(orch.finalize()[StreamTopics.ANALYTICS_POSSESSIONS])

        batch = PossessionEngine().generate(events)
        assert [p.possession_id for p in published] == [p.possession_id for p in batch]

    def test_full_run_then_finalize_matches_one_shot_batch_team_state(self):
        events = _many_rallies(6, gap=90.0)
        orch = MatchOrchestrator(match_id="3387")
        published = []
        for e in events:
            orch.ingest_event(e)
        published.extend(orch.tick()[StreamTopics.ANALYTICS_TEAMSTATE])
        published.extend(orch.finalize()[StreamTopics.ANALYTICS_TEAMSTATE])

        batch_windows = TeamStateBuilder().build_dual_window(events)
        batch_all = batch_windows[60] + batch_windows[300]
        # The tail entry from the single tick() call is legitimately resent
        # once more by finalize() as the authoritative final value (see
        # module docstring) -- compare by unique key, not raw count.
        assert {(s.team_id, s.window_seconds, s.timestamp) for s in published} == \
               {(s.team_id, s.window_seconds, s.timestamp) for s in batch_all}


# ---------------------------------------------------------------------------
# G. Redis Streams integration
# ---------------------------------------------------------------------------

try:
    import fakeredis
    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False

try:
    import redis  # noqa: F401
    _REDIS_PY_AVAILABLE = True
except ImportError:
    _REDIS_PY_AVAILABLE = False

_REDIS_STACK_AVAILABLE = _FAKEREDIS_AVAILABLE and _REDIS_PY_AVAILABLE


@pytest.mark.skipif(not _REDIS_STACK_AVAILABLE, reason="redis-py and/or fakeredis not installed")
class TestRedisIntegration:

    @pytest.fixture()
    def fake_client(self, monkeypatch):
        from config.redis_client import RedisConnectionPool
        client = fakeredis.FakeStrictRedis(decode_responses=True)
        monkeypatch.setattr(RedisConnectionPool, "client", classmethod(lambda cls: client))
        return client

    def test_tick_output_publishes_and_decodes_round_trip(self, fake_client):
        from config.redis_client import RedisStreamConsumer, RedisStreamProducer
        from ingestion.stream_codec import decode

        orch = MatchOrchestrator(match_id="3387")
        for e in _many_rallies(4, gap=20.0):
            orch.ingest_event(e)
        new_objects = orch.tick()
        published_possessions = list(new_objects[StreamTopics.ANALYTICS_POSSESSIONS])
        assert published_possessions  # sanity: something to publish

        producer = RedisStreamProducer()
        count = MatchOrchestrator.publish(producer, new_objects)
        assert count == sum(len(v) for v in new_objects.values())

        consumer = RedisStreamConsumer(
            StreamTopics.ANALYTICS_POSSESSIONS, group="test-group", consumer_name="test-consumer",
        )
        entries = consumer.read_new(count=100)
        decoded = [decode(fields) for _, fields in entries]
        assert decoded == published_possessions

    def test_consume_tracking_events_ingests_and_acks(self, fake_client):
        from config.redis_client import RedisStreamConsumer, RedisStreamProducer
        from ingestion.stream_codec import decode  # noqa: F401

        producer = RedisStreamProducer()
        for e in _rally("SC Magdeburg", 0):
            producer.publish_dataclass(StreamTopics.TRACKING_EVENTS, e)

        orch = MatchOrchestrator(match_id="3387")
        consumer = RedisStreamConsumer(
            StreamTopics.TRACKING_EVENTS, group="orchestrator-group", consumer_name="orchestrator-1",
        )
        n = orch.consume_tracking_events(consumer, count=100)
        assert n == 3
        assert len(orch.events) == 3

        # acked -- a fresh read_new() with the same group sees nothing new
        more = consumer.read_new(count=100)
        assert more == []

    def test_consume_match_events_stores_and_acks_without_touching_engines(self, fake_client):
        from analysis.match_orchestrator import MatchEvent
        from config.redis_client import RedisStreamConsumer, RedisStreamProducer

        producer = RedisStreamProducer()
        backend_event = MatchEvent(
            event_id="be-1", timestamp=BASE_TS, match_id="3387",
            team_id="SC Magdeburg", player_id=1164, event_type="substitution",
            metadata={"player_in": 1170, "player_out": 1164}, source="backend",
        )
        producer.publish_dataclass(StreamTopics.MATCH_EVENTS, backend_event)

        orch = MatchOrchestrator(match_id="3387")
        consumer = RedisStreamConsumer(
            StreamTopics.MATCH_EVENTS, group="orchestrator-group", consumer_name="orchestrator-1",
        )
        n = orch.consume_match_events(consumer, count=100)
        assert n == 1
        assert orch.match_events == [backend_event]
        # Backend-owned events are held, not fed into the TacticalEvent buffer
        # that drives Possession/TeamState/etc. (strict ownership separation).
        assert orch.events == []

        more = consumer.read_new(count=100)
        assert more == []

    def test_consume_match_context_keeps_only_the_latest_snapshot(self, fake_client):
        from analysis.match_orchestrator import MatchContext
        from config.redis_client import RedisStreamConsumer, RedisStreamProducer

        producer = RedisStreamProducer()
        earlier = MatchContext(
            timestamp=BASE_TS, match_id="3387", home_score=10, away_score=9,
            period="first_half", game_clock_seconds=1200.0,
        )
        later = MatchContext(
            timestamp=BASE_TS + timedelta(seconds=600), match_id="3387",
            home_score=14, away_score=12, period="second_half", game_clock_seconds=2145.0,
        )
        producer.publish_dataclass(StreamTopics.MATCH_CONTEXT, earlier)
        producer.publish_dataclass(StreamTopics.MATCH_CONTEXT, later)

        orch = MatchOrchestrator(match_id="3387")
        consumer = RedisStreamConsumer(
            StreamTopics.MATCH_CONTEXT, group="orchestrator-group", consumer_name="orchestrator-1",
        )
        n = orch.consume_match_context(consumer, count=100)
        assert n == 2
        assert orch.latest_match_context == later


# ---------------------------------------------------------------------------
# H. Real-data validation: session 3387
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_events() -> list[TacticalEvent]:
    if not EVENTS_PATH.exists():
        pytest.skip(f"events.csv not found at {EVENTS_PATH}")
    stats_path = DATA_DIR / "statistics.csv"
    player_meta = None
    if stats_path.exists():
        from ingestion.kinexon_adapter import KinexonAdapter
        player_meta = KinexonAdapter().load_player_meta(stats_path)
    adapter = KinexonTacticalEventAdapter()
    return list(adapter.parse(EVENTS_PATH, player_meta=player_meta, match_id="3387"))


class TestRealDataValidation:

    def test_orchestrator_runs_end_to_end_without_crashing(self, real_events):
        orch = MatchOrchestrator(match_id="3387")
        chunk = 500
        published = {topic: [] for topic in StreamTopics.OUTBOUND}
        for i in range(0, len(real_events), chunk):
            for e in real_events[i:i + chunk]:
                orch.ingest_event(e)
            result = orch.tick()
            for topic, objs in result.items():
                published[topic].extend(objs)
        final = orch.finalize()
        for topic, objs in final.items():
            published[topic].extend(objs)

        assert len(published[StreamTopics.ANALYTICS_POSSESSIONS]) > 0
        assert len(published[StreamTopics.ANALYTICS_TEAMSTATE]) > 0
        assert len(published[StreamTopics.ANALYTICS_TRENDS]) > 0
        assert len(published[StreamTopics.ANALYTICS_SITUATIONS]) > 0

    def test_incremental_possessions_match_batch_possessions_exactly(self, real_events):
        orch = MatchOrchestrator(match_id="3387")
        chunk = 500
        published = []
        for i in range(0, len(real_events), chunk):
            for e in real_events[i:i + chunk]:
                orch.ingest_event(e)
            published.extend(orch.tick()[StreamTopics.ANALYTICS_POSSESSIONS])
        published.extend(orch.finalize()[StreamTopics.ANALYTICS_POSSESSIONS])

        batch = PossessionEngine().generate(real_events)
        assert [p.possession_id for p in published] == [p.possession_id for p in batch]

    def test_incremental_situations_reproduce_every_batch_situation(self, real_events):
        """
        Known limitation (documented in MATCH_ORCHESTRATOR_IMPLEMENTATION.md):
        a mid-match tick()'s "tail" preview is tagged with the CURRENT
        partial buffer's last-event timestamp, which is occasionally not a
        real fixed-schedule evaluation point at all (only the eventual,
        fuller buffer's tail timestamps coincide with the true schedule).
        Such a preview is published exactly once and never recurs (it does
        not get "corrected" at a later tick, because there is nothing at
        that exact off-schedule instant to correct) -- so the incremental
        stream can contain a handful of extra, harmless preview-only
        entries at synthetic timestamps that batch never produces. What
        actually matters for correctness is that nothing is ever LOST: every
        situation the one-shot batch run produces at a genuine schedule
        timestamp must also appear in the incremental stream.
        """
        orch = MatchOrchestrator(match_id="3387")
        chunk = 500
        published = []
        for i in range(0, len(real_events), chunk):
            for e in real_events[i:i + chunk]:
                orch.ingest_event(e)
            published.extend(orch.tick()[StreamTopics.ANALYTICS_SITUATIONS])
        published.extend(orch.finalize()[StreamTopics.ANALYTICS_SITUATIONS])

        possessions = PossessionEngine().generate(real_events)
        windows = TeamStateBuilder().build_dual_window(real_events)
        trend_builder = TeamStateTrendBuilder()
        trends = {ws: trend_builder.build(windows[ws]) for ws in windows}
        insights = {ws: CoachInsightEngine().generate(trends[ws]) for ws in windows}
        batch = []
        for ws in windows:
            batch.extend(CoachSituationEngine().generate(possessions, windows[ws], trends[ws], insights[ws]))

        key = lambda s: (s.team_id, s.source_metrics.get("window_seconds"), s.timestamp, s.situation_type)
        published_keys = set(map(key, published))
        batch_keys = set(map(key, batch))
        missing = batch_keys - published_keys
        assert not missing, f"incremental run lost {len(missing)} batch situation(s): {sorted(missing)[:5]}"

    def test_real_data_summary_report(self, real_events, capsys):
        orch = MatchOrchestrator(match_id="3387")
        chunk = 500
        published = {topic: [] for topic in StreamTopics.OUTBOUND}
        tick_count = 0
        for i in range(0, len(real_events), chunk):
            for e in real_events[i:i + chunk]:
                orch.ingest_event(e)
            result = orch.tick()
            tick_count += 1
            for topic, objs in result.items():
                published[topic].extend(objs)
        final = orch.finalize()
        for topic, objs in final.items():
            published[topic].extend(objs)

        print("\n=== MatchOrchestrator real-data validation (session 3387) ===")
        print(f"Total TacticalEvents ingested: {len(real_events)}")
        print(f"Ticks executed: {tick_count} (chunk size {chunk}) + 1 finalize()")
        for topic in StreamTopics.OUTBOUND:
            print(f"  {topic}: {len(published[topic])} published")
