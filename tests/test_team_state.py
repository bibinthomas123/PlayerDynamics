"""
tests/test_team_state.py

Validates TeamState v1 (analysis/team_state.py):

    TeamState         -- canonical dataclass
    TeamStateBuilder   -- builds TeamState snapshots from a TacticalEvent stream

Covers:
  A. Possession aggregation (possession_count, turnover_count, possession_pressure)
  B. Attack metrics (pass_count, shot_count, attack_activity)
  C. Physical metrics (sprint_count, acceleration_count, exertion_count, physical_load)
  D. Player metrics (active_player_count, fatigue_burden)
  E. Mixed event streams (multi-team, multi-tick, confidence)
  F. Real-data validation against data/events.csv (session 3387)

Run:
    pytest tests/test_team_state.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from analysis.team_state import TeamState, TeamStateBuilder, PHYSICAL_EVENT_TYPES
from config.settings import TeamStateConfig
from ingestion.tactical_event import KinexonTacticalEventAdapter, TacticalEvent

DATA_DIR = _ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.csv"

BASE_TS = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)


def _ev(
    event_type: str,
    seconds_offset: float,
    team_id: str = "SC Magdeburg",
    player_id: int = 1164,
    event_id: str | None = None,
) -> TacticalEvent:
    ts = BASE_TS + timedelta(seconds=seconds_offset)
    return TacticalEvent(
        event_id=event_id or f"{event_type}-{seconds_offset}-{player_id}",
        timestamp=ts,
        match_id="3387",
        team_id=team_id,
        player_id=player_id,
        event_type=event_type,
        metadata={},
        source="kinexon",
        confidence=1.0,
    )


@pytest.fixture()
def builder() -> TeamStateBuilder:
    return TeamStateBuilder(config=TeamStateConfig())


# ---------------------------------------------------------------------------
# A. Possession aggregation
# ---------------------------------------------------------------------------

class TestPossessionAggregation:

    def test_possession_count(self, builder):
        events = [_ev("possession", t) for t in (1, 5, 10)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].possession_count == 3

    def test_turnover_count(self, builder):
        events = [_ev("turnover", t) for t in (1, 5)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].turnover_count == 2

    def test_possession_pressure_ratio(self, builder):
        # 3 possessions, 1 turnover -> pressure = 1 / 4 = 0.25
        events = [_ev("possession", t) for t in (1, 2, 3)] + [_ev("turnover", 4)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].possession_pressure == pytest.approx(0.25)

    def test_possession_pressure_zero_when_no_possession_events(self, builder):
        events = [_ev("pass", 1)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].possession_pressure == 0.0

    def test_possession_pressure_one_when_all_turnovers(self, builder):
        events = [_ev("turnover", t) for t in (1, 2, 3)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].possession_pressure == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# B. Attack metrics
# ---------------------------------------------------------------------------

class TestAttackMetrics:

    def test_pass_and_shot_counts(self, builder):
        events = [_ev("pass", t) for t in (1, 2, 3)] + [_ev("shot", 4)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].pass_count == 3
        assert snaps[-1].shot_count == 1

    def test_attack_activity_is_per_minute_rate(self, builder):
        # 4 pass + 1 shot = 5 attacking events in a 60s window -> 5/min
        events = [_ev("pass", t) for t in (1, 2, 3, 4)] + [_ev("shot", 5)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].attack_activity == pytest.approx(5.0)

    def test_attack_activity_normalised_across_window_lengths(self, builder):
        """Same raw counts in a 300s window should yield 1/5th the rate of a 60s window."""
        events = [_ev("pass", t) for t in (1, 2, 3, 4, 5)]
        snaps_60 = builder.build(events, window_seconds=60)
        snaps_300 = builder.build(events, window_seconds=300)
        assert snaps_60[-1].pass_count == snaps_300[-1].pass_count == 5
        assert snaps_60[-1].attack_activity == pytest.approx(snaps_300[-1].attack_activity * 5)


# ---------------------------------------------------------------------------
# C. Physical metrics
# ---------------------------------------------------------------------------

class TestPhysicalMetrics:

    def test_individual_counts(self, builder):
        events = (
            [_ev("sprint_event", t) for t in (1, 2)]
            + [_ev("acceleration_event", 3)]
            + [_ev("exertion_event", t) for t in (4, 5, 6)]
        )
        snaps = builder.build(events, window_seconds=60)
        s = snaps[-1]
        assert s.sprint_count == 2
        assert s.acceleration_count == 1
        assert s.exertion_count == 3

    def test_physical_load_includes_all_physical_types_not_just_the_named_three(self, builder):
        """deceleration/change_of_direction/impact/jump aren't standalone fields
        but must still feed physical_load."""
        events = [_ev(t, i) for i, t in enumerate(PHYSICAL_EVENT_TYPES, start=1)]
        snaps = builder.build(events, window_seconds=60)
        s = snaps[-1]
        assert s.physical_load == pytest.approx(len(PHYSICAL_EVENT_TYPES) * (60.0 / 60))
        # Only 3 of the 7 physical types have standalone counters
        assert s.sprint_count == 1
        assert s.acceleration_count == 1
        assert s.exertion_count == 1

    def test_non_physical_events_excluded_from_physical_load(self, builder):
        events = [_ev("pass", 1), _ev("shot", 2), _ev("possession", 3)]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].physical_load == 0.0


# ---------------------------------------------------------------------------
# D. Player metrics
# ---------------------------------------------------------------------------

class TestPlayerMetrics:

    def test_active_player_count_counts_distinct_players(self, builder):
        events = [
            _ev("pass", 1, player_id=100),
            _ev("pass", 2, player_id=200),
            _ev("shot", 3, player_id=100),  # same player again
        ]
        snaps = builder.build(events, window_seconds=60)
        assert snaps[-1].active_player_count == 2

    def test_fatigue_burden_is_physical_load_per_active_player(self, builder):
        events = (
            [_ev("sprint_event", 1, player_id=1), _ev("sprint_event", 2, player_id=2)]
        )
        snaps = builder.build(events, window_seconds=60)
        s = snaps[-1]
        assert s.active_player_count == 2
        assert s.fatigue_burden == pytest.approx(s.physical_load / 2)

    def test_fatigue_burden_zero_when_no_active_players(self, builder):
        snaps = builder.build([], window_seconds=60)
        assert snaps == []  # empty stream -> no snapshots at all


# ---------------------------------------------------------------------------
# E. Mixed event streams / multi-team / confidence
# ---------------------------------------------------------------------------

class TestMixedStreams:

    def test_two_teams_aggregated_independently(self, builder):
        events = [
            _ev("pass", 1, team_id="SC Magdeburg", player_id=1),
            _ev("pass", 2, team_id="SC Magdeburg", player_id=1),
            _ev("pass", 3, team_id="HSG Wetzlar", player_id=2),
        ]
        snaps = builder.build(events, window_seconds=60)
        by_team = {s.team_id: s for s in snaps}
        assert by_team["SC Magdeburg"].pass_count == 2
        assert by_team["HSG Wetzlar"].pass_count == 1

    def test_unresolved_team_is_its_own_bucket(self, builder):
        events = [
            _ev("pass", 1, team_id="SC Magdeburg"),
            _ev("pass", 2, team_id=None),
        ]
        snaps = builder.build(events, window_seconds=60)
        team_ids = {s.team_id for s in snaps}
        assert None in team_ids
        by_team = {s.team_id: s for s in snaps}
        assert by_team[None].pass_count == 1
        assert by_team["SC Magdeburg"].pass_count == 1

    def test_multiple_ticks_produced_for_long_stream(self, builder):
        # t0=0, t_end=250s -> 60s tumbling ticks at 60,120,180,240, plus the
        # tail tick at t_end=250 -> 5 ticks total.
        events = [_ev("pass", t, player_id=1) for t in range(0, 251, 10)]
        snaps = builder.build(events, window_seconds=60)
        assert len(snaps) == 5
        assert [s.timestamp for s in snaps] == sorted(s.timestamp for s in snaps)

    def test_events_outside_window_are_excluded(self, builder):
        events = [_ev("pass", 1, player_id=1), _ev("pass", 200, player_id=1)]
        # First snapshot tick is at t=61 (1 + window_seconds), covering [1,61].
        # The second event at t=200 must not appear there.
        snaps = builder.build(events, window_seconds=60)
        first_tick = snaps[0]
        assert first_tick.pass_count == 1

    def test_first_event_in_stream_is_not_dropped_on_window_boundary(self, builder):
        """
        Regression: t0 (the earliest event in the whole stream) sits exactly on
        the first window's lower boundary. A naive left-open interval there
        would silently drop it forever, since no earlier window exists to
        catch it on a later tick.
        """
        events = [_ev("pass", 0, player_id=1)]  # single event, ts == t0 exactly
        snaps = builder.build(events, window_seconds=60)
        assert len(snaps) == 1
        assert snaps[0].pass_count == 1

    def test_confidence_scales_with_event_volume(self, builder):
        few_events = [_ev("pass", 1)]
        many_events = [_ev("pass", t) for t in range(1, 30)]
        snaps_few = builder.build(few_events, window_seconds=60)
        snaps_many = builder.build(many_events, window_seconds=60)
        assert snaps_few[-1].confidence < snaps_many[-1].confidence
        assert snaps_many[-1].confidence <= 1.0

    def test_confidence_default_min_events_for_full_confidence(self, builder):
        cfg = TeamStateConfig(min_events_for_full_confidence_per_60s=5.0)
        b = TeamStateBuilder(config=cfg)
        events = [_ev("pass", t) for t in range(1, 6)]  # exactly 5 events
        snaps = b.build(events, window_seconds=60)
        assert snaps[-1].confidence == pytest.approx(1.0)

    def test_step_seconds_controls_overlap(self, builder):
        """A smaller step than window_seconds should produce overlapping windows
        (more snapshots covering the same data)."""
        events = [_ev("pass", t, player_id=1) for t in range(0, 180, 10)]
        tumbling = builder.build(events, window_seconds=60, step_seconds=60)
        sliding = builder.build(events, window_seconds=60, step_seconds=30)
        assert len(sliding) > len(tumbling)

    def test_window_seconds_recorded_on_snapshot(self, builder):
        events = [_ev("pass", 1)]
        snaps60 = builder.build(events, window_seconds=60)
        snaps300 = builder.build(events, window_seconds=300)
        assert snaps60[-1].window_seconds == 60
        assert snaps300[-1].window_seconds == 300

    def test_build_dual_window_uses_config_defaults(self):
        cfg = TeamStateConfig(short_window_seconds=60, long_window_seconds=300)
        b = TeamStateBuilder(config=cfg)
        events = [_ev("pass", t) for t in range(0, 400, 20)]
        result = b.build_dual_window(events)
        assert set(result.keys()) == {60, 300}
        assert all(s.window_seconds == 60 for s in result[60])
        assert all(s.window_seconds == 300 for s in result[300])

    def test_determinism_same_input_same_output(self, builder):
        events = [_ev("pass", t, player_id=(t % 3)) for t in range(0, 200, 7)]
        snaps_a = builder.build(list(events), window_seconds=60)
        snaps_b = builder.build(list(events), window_seconds=60)
        assert snaps_a == snaps_b


# ---------------------------------------------------------------------------
# F. Real-data validation: session 3387
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_tactical_events() -> list[TacticalEvent]:
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

    def test_builds_without_crashing(self, real_tactical_events):
        builder = TeamStateBuilder()
        snaps = builder.build(real_tactical_events, window_seconds=60)
        assert len(snaps) > 0

    def test_dual_window_real_session(self, real_tactical_events):
        builder = TeamStateBuilder()
        windows = builder.build_dual_window(real_tactical_events)
        assert len(windows[60]) > 0
        assert len(windows[300]) > 0
        assert len(windows[60]) > len(windows[300]), (
            "Shorter windows must produce more (or equal) snapshots than longer ones"
        )

    def test_real_teams_present(self, real_tactical_events):
        builder = TeamStateBuilder()
        snaps = builder.build(real_tactical_events, window_seconds=60)
        team_ids = {s.team_id for s in snaps}
        assert "SC Magdeburg" in team_ids
        assert "HSG Wetzlar" in team_ids

    def test_metrics_are_non_negative_and_bounded(self, real_tactical_events):
        builder = TeamStateBuilder()
        for window_seconds in (60, 300):
            snaps = builder.build(real_tactical_events, window_seconds=window_seconds)
            for s in snaps:
                assert s.possession_count >= 0
                assert s.turnover_count >= 0
                assert 0.0 <= s.possession_pressure <= 1.0
                assert s.pass_count >= 0
                assert s.shot_count >= 0
                assert s.attack_activity >= 0.0
                assert s.sprint_count >= 0
                assert s.acceleration_count >= 0
                assert s.exertion_count >= 0
                assert s.physical_load >= 0.0
                assert s.active_player_count >= 0
                assert s.fatigue_burden >= 0.0
                assert 0.0 <= s.confidence <= 1.0

    def test_determinism_on_real_data(self, real_tactical_events):
        builder = TeamStateBuilder()
        snaps_a = builder.build(list(real_tactical_events), window_seconds=60)
        snaps_b = builder.build(list(real_tactical_events), window_seconds=60)
        assert snaps_a == snaps_b

    def test_real_data_summary_report(self, real_tactical_events):
        """Not an assertion-heavy test -- prints the deliverable summary
        requested for the TeamState v1 report (run with -s to see it)."""
        builder = TeamStateBuilder()
        windows = builder.build_dual_window(real_tactical_events)

        print("\n--- TeamState v1: session 3387 summary ---")
        for window_seconds, snaps in windows.items():
            print(f"\nWindow = {window_seconds}s -> {len(snaps)} snapshots")
            by_team: dict = {}
            for s in snaps:
                by_team.setdefault(s.team_id, []).append(s)
            for team_id, team_snaps in by_team.items():
                confidences = [s.confidence for s in team_snaps]
                loads = [s.physical_load for s in team_snaps]
                attacks = [s.attack_activity for s in team_snaps]
                pressures = [s.possession_pressure for s in team_snaps]
                print(
                    f"  team={team_id!r:<16} n={len(team_snaps):<3} "
                    f"confidence=[{min(confidences):.2f},{max(confidences):.2f}] "
                    f"physical_load=[{min(loads):.1f},{max(loads):.1f}] "
                    f"attack_activity=[{min(attacks):.1f},{max(attacks):.1f}] "
                    f"possession_pressure=[{min(pressures):.2f},{max(pressures):.2f}]"
                )
        assert all(len(s) > 0 for s in windows.values())
