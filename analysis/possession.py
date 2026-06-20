"""
Possession — PlayerDynamics

The tactical layer that bridges TacticalEvent and TeamState:

    TacticalEvent  ->  Possession  ->  TeamState

Groups a TacticalEvent stream into continuous team possession spans —
"which team has the ball, from when, until when, and what happened during
that span" — using only the team_id already resolved on each TacticalEvent
(KinexonTacticalEventAdapter) plus the event_type itself. No new identity
resolution is introduced here.

Explicitly out of scope for this module:
    CoachSituation, RecommendationEngine, Frontend changes. Wiring
    Possession as a direct input to a future TeamState v2 (replacing direct
    TacticalEvent consumption) is a natural next step but is NOT done here —
    TeamStateBuilder (analysis/team_state.py) is untouched and still
    consumes TacticalEvent directly, exactly as validated in
    TEAMSTATE_V1_IMPLEMENTATION.md.

Determinism
------------
Every field is a count, a timestamp difference, or a fixed-formula ratio
of counts already on the Possession. No ML, no learned models. The same
ordered TacticalEvent stream always produces the same Possession sequence.

Possession boundary algorithm
--------------------------------
"Ball Possession Lost" / "Ball Possession Recovery" CSV rows (mapped to
TacticalEvent's "turnover" / "possession" types in ingestion/tactical_event.py)
do NOT reliably pair 1:1 by timestamp in real Kinexon data (only ~42% match
exactly in session 3387 — see ingestion/tactical_event.py's docstring), so
this engine does not try to pair them. Instead it runs a single state
machine over the full, chronologically merged event stream (both teams
together, since only one team can hold the ball at a time):

  - A "possession" event for a NEW team (different from the team currently
    tracked) closes the previous span (outcome="neutral" -- no explicit
    shot/turnover evidence was seen) and opens a new one.
  - A "possession" event for the SAME team currently tracked is just
    another touch within the same span (e.g. consecutive passes).
  - A "shot" event closes the current span immediately with outcome="shot".
    If no span was open for that team yet, a minimal span starting at the
    shot's own timestamp is opened and closed in the same step (duration 0).
  - A "turnover" event closes the current span immediately with
    outcome="turnover" (same minimal-span fallback as above).
  - Every other Tier-1 event type (pass, sprint_event, acceleration_event,
    deceleration_event, change_of_direction, exertion_event, impact_event,
    jump_event) is attached to the currently open span ONLY if it belongs to
    the team that currently holds the ball -- a defending player's physical
    events are never counted into the opponent's possession.
  - Any span still open when the stream ends is closed with outcome="neutral".

Tie-breaking at identical timestamps
---------------------------------------
Kinexon frequently logs a "Ball Possession Lost" (turnover) row and its
matching "Ball Possession Recovery" row (possession, gained_via="recovery")
at the EXACT same millisecond timestamp. Sorting purely by timestamp leaves
the tie order undefined (falls back to original file row order), which can
process the gaining team's "possession" event before the losing team's
"turnover" event -- switching the tracked team away before the real
terminator arrives, and producing a spurious zero-duration turnover/shot
for whichever team it belonged to. To prevent this, events are sorted by
(timestamp, is_terminator) where shot/turnover sort BEFORE possession/pass/
physical events at an identical timestamp -- a genuine end-of-possession
signal for the team currently holding the ball is always applied first.

Metrics (Part 2 -- deterministic, explainable, no ML/no LLM)
----------------------------------------------------------------
attack_intensity      = (pass_count + shot_count) * (60 / duration_seconds)
physical_intensity    = physical_action_count      * (60 / duration_seconds)
transition_intensity  = (sprint_count + acceleration_count) * (60 / duration_seconds)
possession_quality    = quality_outcome_weight * outcome_score
                       + quality_attack_weight  * min(1, attack_intensity / reference_attack_rate_per_min)
  where outcome_score = 1.0 (shot) / 0.5 (neutral) / 0.0 (turnover).
All rates are 0.0 for a zero-duration possession (no time base to rate against).
See PossessionConfig (config/settings.py) for the two tunables.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from config.settings import CONFIG, PossessionConfig
from analysis.team_state import PHYSICAL_EVENT_TYPES
from ingestion.tactical_event import TacticalEvent

logger = logging.getLogger(__name__)

_OUTCOME_SCORE = {"shot": 1.0, "neutral": 0.5, "turnover": 0.0}
_TERMINATOR_TYPES = ("shot", "turnover")


@dataclass
class Possession:
    """One continuous team possession span derived from a TacticalEvent stream."""
    possession_id: str
    team_id: Optional[str]
    match_id: Optional[str]
    start_timestamp: datetime
    end_timestamp: datetime
    duration_seconds: float

    # Counts
    pass_count: int
    shot_count: int
    turnover_count: int
    sprint_count: int
    acceleration_count: int
    physical_action_count: int

    # "shot" | "turnover" | "neutral"
    outcome: str

    # Metrics (Part 2)
    attack_intensity: float
    physical_intensity: float
    transition_intensity: float
    possession_quality: float


def _count(events: List[TacticalEvent], event_type: str) -> int:
    return sum(1 for e in events if e.event_type == event_type)


def _make_possession_id(
    match_id: Optional[str], team_id: Optional[str], start: datetime, end: datetime, seq: int
) -> str:
    key = f"{match_id}|{team_id}|{int(start.timestamp()*1000)}|{int(end.timestamp()*1000)}|{seq}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _build_possession(
    team_id: Optional[str],
    start_ts: datetime,
    events: List[TacticalEvent],
    end_ts: datetime,
    outcome: str,
    seq: int,
    config: PossessionConfig,
) -> Possession:
    match_id = events[0].match_id if events else None

    pass_count = _count(events, "pass")
    shot_count = _count(events, "shot")
    turnover_count = _count(events, "turnover")
    sprint_count = _count(events, "sprint_event")
    acceleration_count = _count(events, "acceleration_event")
    physical_action_count = sum(_count(events, t) for t in PHYSICAL_EVENT_TYPES)

    duration_seconds = max((end_ts - start_ts).total_seconds(), 0.0)
    per_minute = (60.0 / duration_seconds) if duration_seconds > 0 else 0.0

    attack_intensity = (pass_count + shot_count) * per_minute
    physical_intensity = physical_action_count * per_minute
    transition_intensity = (sprint_count + acceleration_count) * per_minute

    outcome_score = _OUTCOME_SCORE[outcome]
    attack_norm = (
        min(1.0, attack_intensity / config.reference_attack_rate_per_min)
        if config.reference_attack_rate_per_min > 0 else 0.0
    )
    possession_quality = round(
        config.quality_outcome_weight * outcome_score + config.quality_attack_weight * attack_norm, 4
    )

    return Possession(
        possession_id=_make_possession_id(match_id, team_id, start_ts, end_ts, seq),
        team_id=team_id,
        match_id=match_id,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        duration_seconds=round(duration_seconds, 3),
        pass_count=pass_count,
        shot_count=shot_count,
        turnover_count=turnover_count,
        sprint_count=sprint_count,
        acceleration_count=acceleration_count,
        physical_action_count=physical_action_count,
        outcome=outcome,
        attack_intensity=round(attack_intensity, 4),
        physical_intensity=round(physical_intensity, 4),
        transition_intensity=round(transition_intensity, 4),
        possession_quality=possession_quality,
    )


class _OpenSpan:
    __slots__ = ("team_id", "start_ts", "events")

    def __init__(self, team_id: Optional[str], start_ts: datetime) -> None:
        self.team_id = team_id
        self.start_ts = start_ts
        self.events: List[TacticalEvent] = []


class PossessionEngine:
    """
    Usage
    -----
        engine = PossessionEngine()
        possessions = engine.generate(tactical_events)   # Iterable[TacticalEvent]
    """

    def __init__(self, config: Optional[PossessionConfig] = None) -> None:
        self.config = config or CONFIG.possession

    def generate(self, events: Iterable[TacticalEvent]) -> List[Possession]:
        event_list = sorted(
            events,
            key=lambda e: (e.timestamp, 0 if e.event_type in _TERMINATOR_TYPES else 1),
        )

        possessions: List[Possession] = []
        current: Optional[_OpenSpan] = None
        seq = 0

        def close(span: _OpenSpan, end_ts: datetime, outcome: str) -> None:
            nonlocal seq
            possessions.append(
                _build_possession(span.team_id, span.start_ts, span.events, end_ts, outcome, seq, self.config)
            )
            seq += 1

        for event in event_list:
            et = event.event_type
            team = event.team_id

            if et == "shot":
                if current is None or current.team_id != team:
                    span = _OpenSpan(team, event.timestamp)
                else:
                    span = current
                span.events.append(event)
                close(span, event.timestamp, "shot")
                current = None
                continue

            if et == "turnover":
                if current is not None and current.team_id != team:
                    # A different team's span was left open with no explicit
                    # terminator -- close it neutrally before starting fresh.
                    close(current, current.events[-1].timestamp, "neutral")
                    current = None
                span = current if current is not None else _OpenSpan(team, event.timestamp)
                span.events.append(event)
                close(span, event.timestamp, "turnover")
                current = None
                continue

            if et == "possession":
                if current is None:
                    current = _OpenSpan(team, event.timestamp)
                    current.events.append(event)
                elif team == current.team_id:
                    current.events.append(event)
                else:
                    close(current, current.events[-1].timestamp, "neutral")
                    current = _OpenSpan(team, event.timestamp)
                    current.events.append(event)
                continue

            # pass, sprint_event, acceleration_event, deceleration_event,
            # change_of_direction, exertion_event, impact_event, jump_event:
            # only attach to the team that currently holds the ball.
            if current is not None and team == current.team_id:
                current.events.append(event)
            # else: belongs to the non-possessing team (e.g. defensive
            # physical events) -- not part of any possession span.

        if current is not None and current.events:
            close(current, current.events[-1].timestamp, "neutral")

        return possessions
