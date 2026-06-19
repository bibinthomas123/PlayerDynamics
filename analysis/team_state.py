"""
TeamState — PlayerDynamics

Deterministic team-level aggregation built on top of the TacticalEvent
stream (ingestion/tactical_event.py). Produces TeamState snapshots over
rolling wall-clock windows so that future Coach Insight logic has a
reliable, auditable team-level state object to consume.

Explicitly out of scope for this module:
    Coach recommendations, Frontend, PostgreSQL integration.

Determinism
------------
Every metric here is a count, ratio, or rate derived directly from the
TacticalEvent stream -- no ML, no LLMs, no heuristic requiring data this
session doesn't have (e.g. no heart rate, no fatigue-decay curve fit). The
same TacticalEvent stream always produces the same TeamState snapshots.

Window semantics
------------------
Windows are wall-clock ticks anchored to the timestamp of the first event
in the stream, NOT triggered by individual event arrivals. This keeps the
snapshot cadence regular even though event density varies enormously by
type (exertion_event fires constantly; shot is rare). For a window of
length W and step S (S defaults to W -- tumbling/non-overlapping):

    tick_1 = t0 + W,  tick_2 = tick_1 + S,  ...,  final tick = t_last_event

Each tick's window covers events with (tick - W) < timestamp <= tick. The
final tick is always t_last_event itself (even if it falls short of a full
step), so the tail of the match is never silently dropped.

One TeamState is produced per (team_id, tick) pair. team_id=None is its own
bucket for events whose team could not be resolved (see
KinexonTacticalEventAdapter) -- it is never merged into a real team's
counts.

Metric formulas (see TeamStateConfig in config/settings.py for tunables)
---------------------------------------------------------------------------
possession_pressure = turnover_count / max(possession_count + turnover_count, 1)
    -- in [0, 1]; higher means a larger share of this team's
       possession-related events ended in a turnover within the window.
       0.0 when the team had neither possessions nor turnovers (no signal).

attack_activity = (pass_count + shot_count) * (60 / window_seconds)
    -- attacking-event rate, normalised to "events per minute" so 60s and
       300s windows are directly comparable.

physical_load = (sprint_count + acceleration_count + deceleration_count
                  + change_of_direction_count + exertion_count
                  + impact_count + jump_count) * (60 / window_seconds)
    -- combined physical-effort event rate per minute across ALL Tier-1
       physical event types, not just the three exposed as standalone
       counts (sprint_count, acceleration_count, exertion_count are kept as
       individual fields for transparency; the rest still feed physical_load).

fatigue_burden = physical_load / max(active_player_count, 1)
    -- physical_load normalised per player currently producing events, i.e.
       average physical-effort rate per active player. Deliberately not a
       fatigue-decay model: HR is absent from this export (see
       KinexonConfig.hr_sensor_present) and any "fatigue accumulation"
       curve would be unfounded without it (see SEMANTIC_VALIDATION_REPORT.md).

confidence = min(1.0, n_events_in_window / scaled_min_events)
    scaled_min_events = min_events_for_full_confidence_per_60s * (window_seconds / 60)
    -- a window with very few underlying events (e.g. right after kickoff,
       or a goalkeeper-only team bucket) is less reliable than a
       well-populated one; confidence reflects event-volume sufficiency,
       not a statistical/ML estimate. 0.0 when the window has zero events.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional

from config.settings import CONFIG, TeamStateConfig
from ingestion.tactical_event import TacticalEvent

logger = logging.getLogger(__name__)

# Tier-1 event types treated as physical-effort signals for physical_load.
PHYSICAL_EVENT_TYPES: tuple = (
    "sprint_event",
    "acceleration_event",
    "deceleration_event",
    "change_of_direction",
    "exertion_event",
    "impact_event",
    "jump_event",
)


@dataclass
class TeamState:
    """
    Deterministic team-level snapshot over a single rolling window.

    window_seconds is carried on every snapshot (not just implied by the
    caller) so that 60s and 300s snapshots can be freely mixed in one list
    or store without losing which window each one represents.
    """
    timestamp: datetime
    team_id: Optional[str]
    window_seconds: int

    # Possession metrics
    possession_count: int
    turnover_count: int
    possession_pressure: float

    # Attack metrics
    pass_count: int
    shot_count: int
    attack_activity: float

    # Physical metrics
    sprint_count: int
    acceleration_count: int
    exertion_count: int
    physical_load: float

    # Player metrics
    active_player_count: int
    fatigue_burden: float

    # Quality
    confidence: float


def _count(events: List[TacticalEvent], event_type: str) -> int:
    return sum(1 for e in events if e.event_type == event_type)


def _compute_snapshot(
    team_id: Optional[str],
    timestamp: datetime,
    window_seconds: int,
    window_events: List[TacticalEvent],
    config: TeamStateConfig,
) -> TeamState:
    possession_count = _count(window_events, "possession")
    turnover_count = _count(window_events, "turnover")
    possession_total = possession_count + turnover_count
    possession_pressure = (turnover_count / possession_total) if possession_total > 0 else 0.0

    pass_count = _count(window_events, "pass")
    shot_count = _count(window_events, "shot")
    per_minute = 60.0 / window_seconds
    attack_activity = (pass_count + shot_count) * per_minute

    sprint_count = _count(window_events, "sprint_event")
    acceleration_count = _count(window_events, "acceleration_event")
    exertion_count = _count(window_events, "exertion_event")
    physical_event_total = sum(_count(window_events, t) for t in PHYSICAL_EVENT_TYPES)
    physical_load = physical_event_total * per_minute

    active_player_count = len({e.player_id for e in window_events if e.player_id is not None})
    fatigue_burden = (physical_load / active_player_count) if active_player_count > 0 else 0.0

    scaled_min_events = config.min_events_for_full_confidence_per_60s * (window_seconds / 60.0)
    confidence = min(1.0, len(window_events) / scaled_min_events) if scaled_min_events > 0 else 0.0

    return TeamState(
        timestamp=timestamp,
        team_id=team_id,
        window_seconds=window_seconds,
        possession_count=possession_count,
        turnover_count=turnover_count,
        possession_pressure=round(possession_pressure, 4),
        pass_count=pass_count,
        shot_count=shot_count,
        attack_activity=round(attack_activity, 4),
        sprint_count=sprint_count,
        acceleration_count=acceleration_count,
        exertion_count=exertion_count,
        physical_load=round(physical_load, 4),
        active_player_count=active_player_count,
        fatigue_burden=round(fatigue_burden, 4),
        confidence=round(confidence, 4),
    )


class TeamStateBuilder:
    """
    Builds TeamState snapshot timelines from a TacticalEvent stream.

    Usage
    -----
        builder = TeamStateBuilder()
        short_snapshots = builder.build(events, window_seconds=60)
        long_snapshots  = builder.build(events, window_seconds=300)

        # or, to materialize the event stream only once for both windows:
        windows = builder.build_dual_window(events)   # {60: [...], 300: [...]}
    """

    def __init__(self, config: Optional[TeamStateConfig] = None) -> None:
        self.config = config or CONFIG.team_state

    def build(
        self,
        events: Iterable[TacticalEvent],
        window_seconds: Optional[int] = None,
        step_seconds: Optional[int] = None,
    ) -> List[TeamState]:
        """
        Returns one TeamState per (team_id, tick), sorted by team_id then
        timestamp. team_id=None (unresolved team) is its own bucket.

        window_seconds defaults to config.short_window_seconds (60s).
        step_seconds defaults to window_seconds (tumbling/non-overlapping).
        """
        window_seconds = window_seconds if window_seconds is not None else self.config.short_window_seconds
        step_seconds = step_seconds if step_seconds is not None else window_seconds

        event_list = sorted(events, key=lambda e: e.timestamp)
        if not event_list:
            return []

        t0 = event_list[0].timestamp
        t_end = event_list[-1].timestamp

        by_team: Dict[Optional[str], List[TacticalEvent]] = {}
        for e in event_list:
            by_team.setdefault(e.team_id, []).append(e)

        ticks: List[datetime] = []
        tick = t0 + timedelta(seconds=window_seconds)
        while tick < t_end:
            ticks.append(tick)
            tick += timedelta(seconds=step_seconds)
        if not ticks or ticks[-1] != t_end:
            ticks.append(t_end)  # tail window, possibly shorter than window_seconds

        snapshots: List[TeamState] = []
        for team_id in sorted(by_team.keys(), key=lambda t: (t is None, str(t))):
            team_events = by_team[team_id]
            for idx, tick_ts in enumerate(ticks):
                window_start = tick_ts - timedelta(seconds=window_seconds)
                # Windows are normally left-open, right-closed: (window_start, tick_ts].
                # This keeps consecutive tumbling windows non-overlapping. The very
                # first tick is the exception: its window_start lands exactly on t0
                # (the earliest event in the whole stream), so a strict "<" there
                # would silently drop that first event forever (no earlier window
                # exists to catch it). Only the first tick uses an inclusive
                # lower bound.
                if idx == 0:
                    window_events = [
                        e for e in team_events if window_start <= e.timestamp <= tick_ts
                    ]
                else:
                    window_events = [
                        e for e in team_events if window_start < e.timestamp <= tick_ts
                    ]
                snapshots.append(
                    _compute_snapshot(team_id, tick_ts, window_seconds, window_events, self.config)
                )

        return snapshots

    def build_dual_window(
        self, events: Iterable[TacticalEvent]
    ) -> Dict[int, List[TeamState]]:
        """
        Convenience wrapper: materializes the event stream once, then builds
        both configured rolling windows (config.short_window_seconds and
        config.long_window_seconds) from it.

        Returns {window_seconds: [TeamState, ...]}.
        """
        event_list = list(events)
        return {
            self.config.short_window_seconds: self.build(
                event_list,
                window_seconds=self.config.short_window_seconds,
                step_seconds=self.config.short_step_seconds,
            ),
            self.config.long_window_seconds: self.build(
                event_list,
                window_seconds=self.config.long_window_seconds,
                step_seconds=self.config.long_step_seconds,
            ),
        }
