"""
tests/test_possession.py

Validates the Possession layer (analysis/possession.py):

    Possession        -- canonical dataclass
    PossessionEngine   -- groups a TacticalEvent stream into team possession spans

Covers:
  A. Basic possession boundaries (shot-ended, turnover-ended, team-switch-ended)
  B. Counts (pass/shot/turnover/sprint/acceleration/physical_action)
  C. Metrics (attack_intensity, physical_intensity, transition_intensity, possession_quality)
  D. Edge cases (empty stream, single event, end-of-stream neutral close, ignored opponent events)
  E. Real-data validation against data/events.csv (session 3387)

Run:
    pytest tests/test_possession.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from analysis.possession import Possession, PossessionEngine
from config.settings import PossessionConfig
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
def engine() -> PossessionEngine:
    return PossessionEngine(config=PossessionConfig())


# ---------------------------------------------------------------------------
# A. Basic possession boundaries
# ---------------------------------------------------------------------------

class TestPossessionBoundaries:

    def test_shot_ends_possession_with_shot_outcome(self, engine):
        events = [_ev("possession", 0), _ev("pass", 5), _ev("shot", 10)]
        possessions = engine.generate(events)
        assert len(possessions) == 1
        p = possessions[0]
        assert p.outcome == "shot"
        assert p.start_timestamp == BASE_TS
        assert p.end_timestamp == BASE_TS + timedelta(seconds=10)
        assert p.duration_seconds == pytest.approx(10.0)

    def test_turnover_ends_possession_with_turnover_outcome(self, engine):
        events = [_ev("possession", 0), _ev("pass", 3), _ev("turnover", 8)]
        possessions = engine.generate(events)
        assert len(possessions) == 1
        assert possessions[0].outcome == "turnover"

    def test_team_switch_closes_previous_possession_as_neutral(self, engine):
        events = [
            _ev("possession", 0, team_id="SC Magdeburg"),
            _ev("pass", 2, team_id="SC Magdeburg"),
            _ev("possession", 10, team_id="HSG Wetzlar"),
        ]
        possessions = engine.generate(events)
        # The SC span closes neutrally on the team switch; the HSG span is
        # then also closed neutrally at end-of-stream (no terminator ever
        # arrives for it) -- both are expected, separate possessions.
        assert len(possessions) == 2
        sc = possessions[0]
        assert sc.team_id == "SC Magdeburg"
        assert sc.outcome == "neutral"
        assert sc.end_timestamp == BASE_TS + timedelta(seconds=2)  # last SC event, not the switch event
        hsg = possessions[1]
        assert hsg.team_id == "HSG Wetzlar"
        assert hsg.outcome == "neutral"

    def test_consecutive_passes_by_same_team_form_one_possession(self, engine):
        events = [
            _ev("possession", 0),
            _ev("pass", 2), _ev("pass", 4), _ev("pass", 6),
            _ev("shot", 8),
        ]
        possessions = engine.generate(events)
        assert len(possessions) == 1
        assert possessions[0].pass_count == 3

    def test_two_full_possessions_in_sequence(self, engine):
        events = [
            _ev("possession", 0, team_id="SC Magdeburg"),
            _ev("shot", 5, team_id="SC Magdeburg"),
            _ev("possession", 6, team_id="HSG Wetzlar"),
            _ev("turnover", 15, team_id="HSG Wetzlar"),
        ]
        possessions = engine.generate(events)
        assert len(possessions) == 2
        assert possessions[0].team_id == "SC Magdeburg" and possessions[0].outcome == "shot"
        assert possessions[1].team_id == "HSG Wetzlar" and possessions[1].outcome == "turnover"


# ---------------------------------------------------------------------------
# B. Counts
# ---------------------------------------------------------------------------

class TestCounts:

    def test_pass_shot_turnover_counts(self, engine):
        events = [
            _ev("possession", 0),
            _ev("pass", 1), _ev("pass", 2), _ev("pass", 3),
            _ev("shot", 4),
        ]
        p = engine.generate(events)[0]
        assert p.pass_count == 3
        assert p.shot_count == 1
        assert p.turnover_count == 0

    def test_sprint_and_acceleration_counts(self, engine):
        events = [
            _ev("possession", 0),
            _ev("sprint_event", 1), _ev("sprint_event", 2),
            _ev("acceleration_event", 3),
            _ev("turnover", 4),
        ]
        p = engine.generate(events)[0]
        assert p.sprint_count == 2
        assert p.acceleration_count == 1

    def test_physical_action_count_includes_all_seven_physical_types(self, engine):
        from analysis.team_state import PHYSICAL_EVENT_TYPES
        events = [_ev("possession", 0)] + [
            _ev(t, i + 1) for i, t in enumerate(PHYSICAL_EVENT_TYPES)
        ] + [_ev("turnover", 20)]
        p = engine.generate(events)[0]
        assert p.physical_action_count == len(PHYSICAL_EVENT_TYPES)


# ---------------------------------------------------------------------------
# C. Metrics
# ---------------------------------------------------------------------------

class TestMetrics:

    def test_attack_intensity_is_per_minute_rate(self, engine):
        # 2 passes + 1 shot over a 30s possession -> 3 events / 30s = 6/min
        events = [_ev("possession", 0), _ev("pass", 10), _ev("pass", 20), _ev("shot", 30)]
        p = engine.generate(events)[0]
        assert p.attack_intensity == pytest.approx(6.0)

    def test_physical_intensity_is_per_minute_rate(self, engine):
        events = [
            _ev("possession", 0),
            _ev("sprint_event", 10), _ev("sprint_event", 20),
            _ev("turnover", 30),
        ]
        p = engine.generate(events)[0]
        # 2 physical events over 30s -> 4/min
        assert p.physical_intensity == pytest.approx(4.0)

    def test_transition_intensity_only_counts_sprint_and_acceleration(self, engine):
        events = [
            _ev("possession", 0),
            _ev("sprint_event", 10),
            _ev("acceleration_event", 15),
            _ev("jump_event", 20),  # physical, but NOT a transition type
            _ev("turnover", 30),
        ]
        p = engine.generate(events)[0]
        assert p.transition_intensity == pytest.approx(4.0)  # 2 events / 30s * 60
        assert p.physical_intensity == pytest.approx(6.0)    # 3 events / 30s * 60

    def test_possession_quality_shot_outcome_higher_than_turnover(self, engine):
        shot_events = [_ev("possession", 0), _ev("pass", 5), _ev("shot", 10)]
        turnover_events = [_ev("possession", 0), _ev("pass", 5), _ev("turnover", 10)]
        shot_p = engine.generate(shot_events)[0]
        turnover_p = engine.generate(turnover_events)[0]
        assert shot_p.possession_quality > turnover_p.possession_quality

    def test_possession_quality_bounded_zero_to_one(self, engine):
        events = [_ev("possession", 0)] + [_ev("pass", i) for i in range(1, 50)] + [_ev("shot", 50)]
        p = engine.generate(events)[0]
        assert 0.0 <= p.possession_quality <= 1.0

    def test_zero_duration_possession_has_zero_rates(self, engine):
        """A possession opened and closed on the very same tick (e.g. an
        immediate shot with no prior touch) has no time base to rate against."""
        events = [_ev("shot", 0)]
        p = engine.generate(events)[0]
        assert p.duration_seconds == 0.0
        assert p.attack_intensity == 0.0
        assert p.physical_intensity == 0.0
        assert p.transition_intensity == 0.0
        # possession_quality is still defined (outcome_score contributes even at zero duration).
        # A Kinexon-only shot with no coach annotation uses outcome_score_shot_unknown=0.85.
        assert p.possession_quality == pytest.approx(0.5 * 0.85)  # 0.5*0.85(shot_unknown) + 0.5*0.0(no rate)


# ---------------------------------------------------------------------------
# D. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_stream_returns_empty_list(self, engine):
        assert engine.generate([]) == []

    def test_open_possession_at_end_of_stream_closes_neutral(self, engine):
        events = [_ev("possession", 0), _ev("pass", 5)]  # no terminator
        possessions = engine.generate(events)
        assert len(possessions) == 1
        assert possessions[0].outcome == "neutral"
        assert possessions[0].end_timestamp == BASE_TS + timedelta(seconds=5)

    def test_opponent_physical_events_not_counted_into_possession(self, engine):
        events = [
            _ev("possession", 0, team_id="SC Magdeburg"),
            _ev("sprint_event", 1, team_id="HSG Wetzlar"),  # defender's sprint, not SC's
            _ev("shot", 2, team_id="SC Magdeburg"),
        ]
        p = engine.generate(events)[0]
        assert p.team_id == "SC Magdeburg"
        assert p.sprint_count == 0

    def test_shot_with_no_open_possession_creates_minimal_span(self, engine):
        events = [_ev("shot", 0)]
        possessions = engine.generate(events)
        assert len(possessions) == 1
        assert possessions[0].outcome == "shot"
        assert possessions[0].duration_seconds == 0.0

    def test_turnover_with_no_open_possession_creates_minimal_span(self, engine):
        events = [_ev("turnover", 0)]
        possessions = engine.generate(events)
        assert len(possessions) == 1
        assert possessions[0].outcome == "turnover"

    def test_possession_id_is_deterministic(self, engine):
        events = [_ev("possession", 0), _ev("shot", 5)]
        ids_a = [p.possession_id for p in engine.generate(list(events))]
        ids_b = [p.possession_id for p in PossessionEngine().generate(list(events))]
        assert ids_a == ids_b

    def test_tied_timestamp_terminator_processed_before_new_possession(self, engine):
        """
        Regression: Kinexon often logs a turnover and the opponent's matching
        "recovery" possession event at the EXACT same timestamp. If the
        recovery event were processed first, it would switch the tracked
        team away before the turnover applies, producing a spurious
        zero-duration turnover instead of properly closing the losing
        team's accumulated possession.
        """
        events = [
            _ev("possession", 0, team_id="SC Magdeburg"),
            _ev("pass", 2, team_id="SC Magdeburg"),
            # Both at the same instant: SC loses it, HSG recovers it.
            _ev("turnover", 5, team_id="SC Magdeburg"),
            _ev("possession", 5, team_id="HSG Wetzlar"),
        ]
        possessions = engine.generate(events)
        sc = next(p for p in possessions if p.team_id == "SC Magdeburg")
        assert sc.outcome == "turnover"
        assert sc.duration_seconds == pytest.approx(5.0)  # NOT 0 -- the pass must count
        assert sc.pass_count == 1

    def test_unsorted_input_is_sorted_internally(self, engine):
        events = [_ev("shot", 10), _ev("possession", 0), _ev("pass", 5)]
        possessions = engine.generate(events)
        assert len(possessions) == 1
        assert possessions[0].pass_count == 1

    def test_determinism_full_run(self, engine):
        events = [
            _ev("possession", 0, team_id="SC Magdeburg"),
            _ev("pass", 2, team_id="SC Magdeburg"),
            _ev("shot", 5, team_id="SC Magdeburg"),
            _ev("possession", 6, team_id="HSG Wetzlar"),
            _ev("turnover", 15, team_id="HSG Wetzlar"),
        ]
        a = engine.generate(list(events))
        b = engine.generate(list(events))
        assert a == b


# ---------------------------------------------------------------------------
# E. Real-data validation: session 3387
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

    def test_generates_without_crashing(self, real_tactical_events):
        possessions = PossessionEngine().generate(real_tactical_events)
        assert len(possessions) > 0

    def test_outcomes_are_one_of_three_values(self, real_tactical_events):
        possessions = PossessionEngine().generate(real_tactical_events)
        allowed = {"shot", "turnover", "neutral"}
        assert {p.outcome for p in possessions}.issubset(allowed)

    def test_all_possession_ids_unique(self, real_tactical_events):
        possessions = PossessionEngine().generate(real_tactical_events)
        ids = [p.possession_id for p in possessions]
        assert len(ids) == len(set(ids))

    def test_durations_non_negative(self, real_tactical_events):
        possessions = PossessionEngine().generate(real_tactical_events)
        assert all(p.duration_seconds >= 0.0 for p in possessions)

    def test_both_teams_present(self, real_tactical_events):
        possessions = PossessionEngine().generate(real_tactical_events)
        team_ids = {p.team_id for p in possessions}
        assert "SC Magdeburg" in team_ids
        assert "HSG Wetzlar" in team_ids

    def test_quality_bounded_for_all_real_possessions(self, real_tactical_events):
        possessions = PossessionEngine().generate(real_tactical_events)
        assert all(0.0 <= p.possession_quality <= 1.0 for p in possessions)

    def test_determinism_on_real_data(self, real_tactical_events):
        a = PossessionEngine().generate(list(real_tactical_events))
        b = PossessionEngine().generate(list(real_tactical_events))
        assert a == b

    def test_real_data_summary_report(self, real_tactical_events):
        """Not assertion-heavy -- prints the deliverable summary requested for
        the Possession Engine report (run with -s to see it)."""
        from collections import Counter

        possessions = PossessionEngine().generate(real_tactical_events)
        durations = [p.duration_seconds for p in possessions]
        outcomes = Counter(p.outcome for p in possessions)
        qualities = [p.possession_quality for p in possessions]

        print("\n--- PossessionEngine v1: session 3387 summary ---")
        print(f"Total possessions: {len(possessions)}")
        print(f"Duration (s): min={min(durations):.1f} max={max(durations):.1f} "
              f"mean={sum(durations)/len(durations):.1f}")
        print(f"Outcomes: {dict(outcomes)}")
        print(f"Quality: min={min(qualities):.2f} max={max(qualities):.2f} "
              f"mean={sum(qualities)/len(qualities):.2f}")

        by_team: dict = {}
        for p in possessions:
            by_team.setdefault(p.team_id, []).append(p)
        for team_id, team_possessions in by_team.items():
            t_outcomes = Counter(p.outcome for p in team_possessions)
            t_durations = [p.duration_seconds for p in team_possessions]
            print(f"  team={team_id!r}: n={len(team_possessions)} outcomes={dict(t_outcomes)} "
                  f"mean_duration={sum(t_durations)/len(t_durations):.1f}s")
        assert True
