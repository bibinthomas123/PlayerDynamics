"""
TeamStateTrend — PlayerDynamics

Deterministic temporal layer over TeamState (analysis/team_state.py)
snapshots. Converts a static per-window snapshot stream into interpretable
changes over time: signed deltas between consecutive snapshots, plus three
categorical trend labels derived from those deltas via fixed thresholds.

This layer is intended as the direct input to a future Coach Insight layer.

Explicitly out of scope for this module:
    Coach recommendations, LLM logic, Frontend.

Determinism
------------
Every output is a difference of two existing TeamState fields, or a fixed-
threshold classification of that difference. No ML, no learned models. The
same ordered TeamState sequence always produces the same TeamStateTrend
sequence.

Comparison scope
------------------
Trends are computed strictly WITHIN one team's own timeline at one window
length -- a trend is never computed between two different teams, or between
a 60s snapshot and a 300s snapshot. TeamStateTrendBuilder groups the input
by (team_id, window_seconds) before differencing, so passing in a combined
multi-team / multi-window list (e.g. the flattened output of
TeamStateBuilder.build_dual_window()) is safe.

The first snapshot in each (team_id, window_seconds) timeline has no
predecessor and therefore produces no TeamStateTrend -- a timeline with N
input snapshots produces N-1 output trends.

Threshold calibration
-----------------------
attack_activity / physical_load / fatigue_burden are already rate-
normalised (events per minute, or events per minute per active player --
see TeamState's docstring), so a single absolute threshold is comparable
across both the 60s and 300s windows. Default thresholds
(TeamStateTrendConfig in config/settings.py) were set to roughly half the
mean |consecutive delta| observed across both real teams in session 3387's
full match. Below the threshold, a swing is treated as within-window noise
("stable"); at or above it, as a genuine directional change.

possession_pressure_delta and confidence_delta are exposed as raw deltas
only
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from config.settings import CONFIG, TeamStateTrendConfig
from analysis.team_state import TeamState

logger = logging.getLogger(__name__)


@dataclass
class TeamStateTrend:
    """
    Deterministic delta between two consecutive TeamState snapshots for the
    same team_id and window_seconds.

    timestamp is the LATER (current) snapshot's timestamp -- the trend
    describes the change arriving AT this point in time, relative to the
    immediately preceding snapshot in the same (team_id, window_seconds)
    timeline.
    """
    timestamp: datetime
    team_id: Optional[str]
    window_seconds: int

    # Trend metrics (current - previous)
    possession_pressure_delta: float
    attack_activity_delta: float
    physical_load_delta: float
    fatigue_burden_delta: float
    confidence_delta: float

    # State labels: "increasing" | "stable" | "decreasing"
    attack_trend: str
    load_trend: str
    fatigue_trend: str


def _label(delta: float, threshold: float) -> str:
    if abs(delta) < threshold:
        return "stable"
    return "increasing" if delta > 0 else "decreasing"


def _compute_trend(
    previous: TeamState, current: TeamState, config: TeamStateTrendConfig
) -> TeamStateTrend:
    possession_pressure_delta = current.possession_pressure - previous.possession_pressure
    attack_activity_delta = current.attack_activity - previous.attack_activity
    physical_load_delta = current.physical_load - previous.physical_load
    fatigue_burden_delta = current.fatigue_burden - previous.fatigue_burden
    confidence_delta = current.confidence - previous.confidence

    return TeamStateTrend(
        timestamp=current.timestamp,
        team_id=current.team_id,
        window_seconds=current.window_seconds,
        possession_pressure_delta=round(possession_pressure_delta, 4),
        attack_activity_delta=round(attack_activity_delta, 4),
        physical_load_delta=round(physical_load_delta, 4),
        fatigue_burden_delta=round(fatigue_burden_delta, 4),
        confidence_delta=round(confidence_delta, 4),
        attack_trend=_label(attack_activity_delta, config.attack_activity_threshold),
        load_trend=_label(physical_load_delta, config.physical_load_threshold),
        fatigue_trend=_label(fatigue_burden_delta, config.fatigue_burden_threshold),
    )


class TeamStateTrendBuilder:
    """
    Builds TeamStateTrend sequences from ordered TeamState snapshots.

    Usage
    -----
        trend_builder = TeamStateTrendBuilder()
        trends = trend_builder.build(team_states)   # one (team, window) timeline

        # Safe to pass a combined multi-team / multi-window list too --
        # grouped internally by (team_id, window_seconds) before differencing:
        windows = team_state_builder.build_dual_window(events)
        trends = trend_builder.build(windows[60] + windows[300])
    """

    def __init__(self, config: Optional[TeamStateTrendConfig] = None) -> None:
        self.config = config or CONFIG.team_state_trend

    def build(self, snapshots: Iterable[TeamState]) -> List[TeamStateTrend]:
        groups: Dict[Tuple[Optional[str], int], List[TeamState]] = {}
        for s in snapshots:
            groups.setdefault((s.team_id, s.window_seconds), []).append(s)

        trends: List[TeamStateTrend] = []
        for key in sorted(groups.keys(), key=lambda k: (k[0] is None, str(k[0]), k[1])):
            group = sorted(groups[key], key=lambda s: s.timestamp)
            for previous, current in zip(group, group[1:]):
                trends.append(_compute_trend(previous, current, self.config))

        return trends
