"""
tests/test_team_state_trend.py

Validates TeamStateTrend v1 (analysis/team_state_trend.py):

    TeamStateTrend         -- canonical dataclass
    TeamStateTrendBuilder   -- converts ordered TeamState snapshots into deltas + labels

Covers:
  A. Increasing attack activity
  B. Decreasing attack activity
  C. Stable windows (below threshold -> "stable", not noise-triggered)
  D. Mixed trends (independent attack/load/fatigue directions in one step)
  E. Grouping safety (multi-team, multi-window, first-snapshot-has-no-trend)
  F. Real-data validation against data/events.csv (session 3387)

Run:
    pytest tests/test_team_state_trend.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from analysis.team_state import TeamState, TeamStateBuilder
from analysis.team_state_trend import TeamStateTrend, TeamStateTrendBuilder
from config.settings import TeamStateTrendConfig
from ingestion.tactical_event import KinexonTacticalEventAdapter

DATA_DIR = _ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.csv"

BASE_TS = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)


def _snap(
    minute_offset: int,
    team_id: str = "SC Magdeburg",
    window_seconds: int = 60,
    possession_count: int = 0,
    turnover_count: int = 0,
    possession_pressure: float = 0.0,
    pass_count: int = 0,
    shot_count: int = 0,
    attack_activity: float = 0.0,
    sprint_count: int = 0,
    acceleration_count: int = 0,
    exertion_count: int = 0,
    physical_load: float = 0.0,
    active_player_count: int = 1,
    fatigue_burden: float = 0.0,
    confidence: float = 1.0,
) -> TeamState:
    return TeamState(
        timestamp=BASE_TS + timedelta(minutes=minute_offset),
        team_id=team_id,
        window_seconds=window_seconds,
        possession_count=possession_count,
        turnover_count=turnover_count,
        possession_pressure=possession_pressure,
        pass_count=pass_count,
        shot_count=shot_count,
        attack_activity=attack_activity,
        sprint_count=sprint_count,
        acceleration_count=acceleration_count,
        exertion_count=exertion_count,
        physical_load=physical_load,
        active_player_count=active_player_count,
        fatigue_burden=fatigue_burden,
        confidence=confidence,
    )


@pytest.fixture()
def builder() -> TeamStateTrendBuilder:
    return TeamStateTrendBuilder(config=TeamStateTrendConfig())


# ---------------------------------------------------------------------------
# A. Increasing attack activity
# ---------------------------------------------------------------------------

class TestIncreasingAttackActivity:

    def test_attack_trend_increasing_above_threshold(self, builder):
        snaps = [_snap(0, attack_activity=2.0), _snap(1, attack_activity=10.0)]
        trends = builder.build(snaps)
        assert len(trends) == 1
        assert trends[0].attack_activity_delta == pytest.approx(8.0)
        assert trends[0].attack_trend == "increasing"

    def test_multi_step_increasing_sequence(self, builder):
        snaps = [_snap(i, attack_activity=float(i * 5)) for i in range(4)]
        trends = builder.build(snaps)
        assert all(t.attack_trend == "increasing" for t in trends)
        assert all(t.attack_activity_delta == pytest.approx(5.0) for t in trends)


# ---------------------------------------------------------------------------
# B. Decreasing attack activity
# ---------------------------------------------------------------------------

class TestDecreasingAttackActivity:

    def test_attack_trend_decreasing_below_negative_threshold(self, builder):
        snaps = [_snap(0, attack_activity=15.0), _snap(1, attack_activity=5.0)]
        trends = builder.build(snaps)
        assert trends[0].attack_activity_delta == pytest.approx(-10.0)
        assert trends[0].attack_trend == "decreasing"

    def test_multi_step_decreasing_sequence(self, builder):
        snaps = [_snap(i, attack_activity=float(20 - i * 5)) for i in range(4)]
        trends = builder.build(snaps)
        assert all(t.attack_trend == "decreasing" for t in trends)


# ---------------------------------------------------------------------------
# C. Stable windows
# ---------------------------------------------------------------------------

class TestStableWindows:

    def test_zero_delta_is_stable_for_all_three_labels(self, builder):
        snaps = [
            _snap(0, attack_activity=10.0, physical_load=20.0, fatigue_burden=2.0),
            _snap(1, attack_activity=10.0, physical_load=20.0, fatigue_burden=2.0),
        ]
        trends = builder.build(snaps)
        t = trends[0]
        assert t.attack_trend == "stable"
        assert t.load_trend == "stable"
        assert t.fatigue_trend == "stable"
        assert t.attack_activity_delta == 0.0
        assert t.physical_load_delta == 0.0
        assert t.fatigue_burden_delta == 0.0

    def test_small_swing_below_threshold_is_stable_not_noise(self, builder):
        """A swing smaller than the configured threshold must not flip the label."""
        cfg = TeamStateTrendConfig(attack_activity_threshold=3.0)
        b = TeamStateTrendBuilder(config=cfg)
        snaps = [_snap(0, attack_activity=10.0), _snap(1, attack_activity=12.5)]  # delta=2.5 < 3.0
        trends = b.build(snaps)
        assert trends[0].attack_trend == "stable"

    def test_delta_exactly_at_threshold_is_not_stable(self, builder):
        """Boundary: abs(delta) < threshold is stable; == threshold is directional."""
        cfg = TeamStateTrendConfig(attack_activity_threshold=3.0)
        b = TeamStateTrendBuilder(config=cfg)
        snaps = [_snap(0, attack_activity=10.0), _snap(1, attack_activity=13.0)]  # delta == 3.0
        trends = b.build(snaps)
        assert trends[0].attack_trend == "increasing"


# ---------------------------------------------------------------------------
# D. Mixed trends
# ---------------------------------------------------------------------------

class TestMixedTrends:

    def test_independent_directions_in_one_step(self, builder):
        """attack up, load down, fatigue stable -- in the same transition."""
        snaps = [
            _snap(0, attack_activity=5.0, physical_load=30.0, fatigue_burden=2.0),
            _snap(1, attack_activity=15.0, physical_load=10.0, fatigue_burden=2.1),
        ]
        trends = builder.build(snaps)
        t = trends[0]
        assert t.attack_trend == "increasing"
        assert t.load_trend == "decreasing"
        assert t.fatigue_trend == "stable"

    def test_possession_pressure_and_confidence_deltas_have_no_label(self, builder):
        snaps = [
            _snap(0, possession_pressure=0.1, confidence=0.5),
            _snap(1, possession_pressure=0.9, confidence=1.0),
        ]
        t = builder.build(snaps)[0]
        assert t.possession_pressure_delta == pytest.approx(0.8)
        assert t.confidence_delta == pytest.approx(0.5)
        assert not hasattr(t, "possession_pressure_trend")
        assert not hasattr(t, "confidence_trend")

    def test_alternating_sequence_produces_alternating_labels(self, builder):
        snaps = [
            _snap(0, attack_activity=0.0),
            _snap(1, attack_activity=10.0),   # up
            _snap(2, attack_activity=0.0),    # down
            _snap(3, attack_activity=10.0),   # up
        ]
        trends = builder.build(snaps)
        assert [t.attack_trend for t in trends] == ["increasing", "decreasing", "increasing"]


# ---------------------------------------------------------------------------
# E. Grouping safety
# ---------------------------------------------------------------------------

class TestGroupingSafety:

    def test_first_snapshot_produces_no_trend(self, builder):
        snaps = [_snap(0)]
        assert builder.build(snaps) == []

    def test_n_snapshots_produce_n_minus_1_trends(self, builder):
        snaps = [_snap(i, attack_activity=float(i)) for i in range(6)]
        trends = builder.build(snaps)
        assert len(trends) == 5

    def test_teams_never_cross_compared(self, builder):
        snaps = [
            _snap(0, team_id="SC Magdeburg", attack_activity=0.0),
            _snap(0, team_id="HSG Wetzlar", attack_activity=100.0),
            _snap(1, team_id="SC Magdeburg", attack_activity=1.0),
            _snap(1, team_id="HSG Wetzlar", attack_activity=101.0),
        ]
        trends = builder.build(snaps)
        # If cross-team comparison happened, deltas would be ~100; must not be.
        assert all(abs(t.attack_activity_delta) < 5 for t in trends)
        assert {t.team_id for t in trends} == {"SC Magdeburg", "HSG Wetzlar"}

    def test_window_lengths_never_cross_compared(self, builder):
        snaps = [
            _snap(0, window_seconds=60, attack_activity=0.0),
            _snap(0, window_seconds=300, attack_activity=100.0),
            _snap(1, window_seconds=60, attack_activity=1.0),
            _snap(1, window_seconds=300, attack_activity=101.0),
        ]
        trends = builder.build(snaps)
        assert all(abs(t.attack_activity_delta) < 5 for t in trends)
        assert {t.window_seconds for t in trends} == {60, 300}

    def test_unsorted_input_is_sorted_internally(self, builder):
        snaps = [_snap(2, attack_activity=2.0), _snap(0, attack_activity=0.0), _snap(1, attack_activity=1.0)]
        trends = builder.build(snaps)
        assert [t.timestamp for t in trends] == sorted(t.timestamp for t in trends)
        assert all(t.attack_activity_delta == pytest.approx(1.0) for t in trends)

    def test_empty_input_returns_empty_list(self, builder):
        assert builder.build([]) == []

    def test_determinism(self, builder):
        snaps = [_snap(i, attack_activity=float(i * 3 % 7)) for i in range(10)]
        a = builder.build(list(snaps))
        b = builder.build(list(snaps))
        assert a == b


# ---------------------------------------------------------------------------
# F. Real-data validation: session 3387
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_team_states() -> dict:
    if not EVENTS_PATH.exists():
        pytest.skip(f"events.csv not found at {EVENTS_PATH}")
    stats_path = DATA_DIR / "statistics.csv"
    player_meta = None
    if stats_path.exists():
        from ingestion.kinexon_adapter import KinexonAdapter
        player_meta = KinexonAdapter().load_player_meta(stats_path)
    adapter = KinexonTacticalEventAdapter()
    events = list(adapter.parse(EVENTS_PATH, player_meta=player_meta, match_id="3387"))
    return TeamStateBuilder().build_dual_window(events)


class TestRealDataValidation:

    def test_builds_without_crashing(self, real_team_states):
        trends = TeamStateTrendBuilder().build(real_team_states[60])
        assert len(trends) > 0

    def test_n_minus_2_per_team_for_60s(self, real_team_states):
        """2 teams x (117 snapshots - 1) = 232 trends at 60s."""
        trends = TeamStateTrendBuilder().build(real_team_states[60])
        assert len(trends) == 2 * (117 - 1)

    def test_labels_are_one_of_three_values(self, real_team_states):
        trends = TeamStateTrendBuilder().build(
            real_team_states[60] + real_team_states[300]
        )
        allowed = {"increasing", "stable", "decreasing"}
        for t in trends:
            assert t.attack_trend in allowed
            assert t.load_trend in allowed
            assert t.fatigue_trend in allowed

    def test_both_directions_observed_in_real_match(self, real_team_states):
        """A 60-minute match must show genuine ebb and flow, not a monotone trend."""
        trends = TeamStateTrendBuilder().build(real_team_states[60])
        attack_labels = {t.attack_trend for t in trends}
        load_labels = {t.load_trend for t in trends}
        assert "increasing" in attack_labels and "decreasing" in attack_labels
        assert "increasing" in load_labels and "decreasing" in load_labels

    def test_determinism_on_real_data(self, real_team_states):
        a = TeamStateTrendBuilder().build(list(real_team_states[60]))
        b = TeamStateTrendBuilder().build(list(real_team_states[60]))
        assert a == b

    def test_real_data_summary_report(self, real_team_states):
        """Not assertion-heavy -- prints the deliverable summary requested for
        the TeamStateTrend report (run with -s to see it)."""
        from collections import Counter

        print("\n--- TeamStateTrend v1: session 3387 summary ---")
        for window_seconds in (60, 300):
            trends = TeamStateTrendBuilder().build(real_team_states[window_seconds])
            print(f"\nWindow = {window_seconds}s -> {len(trends)} trends")

            by_team: dict = {}
            for t in trends:
                by_team.setdefault(t.team_id, []).append(t)

            for team_id, team_trends in by_team.items():
                attack_dist = Counter(t.attack_trend for t in team_trends)
                load_dist = Counter(t.load_trend for t in team_trends)
                fatigue_dist = Counter(t.fatigue_trend for t in team_trends)
                print(f"  team={team_id!r}")
                print(f"    attack_trend:  {dict(attack_dist)}")
                print(f"    load_trend:    {dict(load_dist)}")
                print(f"    fatigue_trend: {dict(fatigue_dist)}")

                strongest = max(team_trends, key=lambda t: abs(t.physical_load_delta))
                print(
                    f"    strongest physical_load swing: {strongest.physical_load_delta:+.1f} "
                    f"at {strongest.timestamp.isoformat()}"
                )
        assert True
