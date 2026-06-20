"""
tests/test_coach_situation.py

Validates CoachSituation v1 (analysis/coach_situation.py):

    CoachSituation        -- canonical dataclass
    CoachSituationEngine   -- aggregates Possession + TeamState + TeamStateTrend
                              + CoachInsight into a single tactical state per
                              (team_id, window_seconds, timestamp)

Covers:
  A. Each situation type firing in isolation (10 types)
  B. Priority ordering when multiple rules could match
  C. No-situation cases
  D. Severity / confidence / source_insights contract
  E. Real-data validation against data/events.csv (session 3387)

Run:
    pytest tests/test_coach_situation.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from analysis.coach_insight import CoachInsight, CoachInsightEngine
from analysis.coach_situation import CoachSituation, CoachSituationEngine
from analysis.possession import Possession, PossessionEngine
from analysis.team_state import TeamState, TeamStateBuilder
from analysis.team_state_trend import TeamStateTrend, TeamStateTrendBuilder
from config.settings import CoachSituationConfig
from ingestion.tactical_event import KinexonTacticalEventAdapter

DATA_DIR = _ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.csv"

BASE_TS = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)
TEAM = "SC Magdeburg"
WIN = 60


def _state(attack_activity: float = 0.0, team_id: str = TEAM, window_seconds: int = WIN,
           minute_offset: int = 0) -> TeamState:
    return TeamState(
        timestamp=BASE_TS + timedelta(minutes=minute_offset),
        team_id=team_id, window_seconds=window_seconds,
        possession_count=0, turnover_count=0, possession_pressure=0.0,
        pass_count=0, shot_count=0, attack_activity=attack_activity,
        sprint_count=0, acceleration_count=0, exertion_count=0, physical_load=0.0,
        active_player_count=1, fatigue_burden=0.0, confidence=1.0,
    )


def _trend(
    minute_offset: int = 0, team_id: str = TEAM, window_seconds: int = WIN,
    possession_pressure_delta: float = 0.0,
    attack_trend: str = "stable", load_trend: str = "stable", fatigue_trend: str = "stable",
) -> TeamStateTrend:
    return TeamStateTrend(
        timestamp=BASE_TS + timedelta(minutes=minute_offset),
        team_id=team_id, window_seconds=window_seconds,
        possession_pressure_delta=possession_pressure_delta,
        attack_activity_delta=0.0, physical_load_delta=0.0,
        fatigue_burden_delta=0.0, confidence_delta=0.0,
        attack_trend=attack_trend, load_trend=load_trend, fatigue_trend=fatigue_trend,
    )


def _insight(category: str, minute_offset: int = 0, team_id: str = TEAM,
             window_seconds: int = WIN, confidence: float = 0.8) -> CoachInsight:
    return CoachInsight(
        timestamp=BASE_TS + timedelta(minutes=minute_offset),
        team_id=team_id, severity="medium", category=category,
        message=f"{category} for {team_id}", confidence=confidence,
        metadata={"source_metrics": [], "values": {}, "thresholds_crossed": {}, "window_seconds": window_seconds},
    )


def _possession(minute_offset: float, outcome: str, quality: float,
                 team_id: str = TEAM) -> Possession:
    end = BASE_TS + timedelta(minutes=minute_offset)
    return Possession(
        possession_id=f"p-{minute_offset}-{team_id}-{outcome}",
        team_id=team_id, match_id="3387",
        start_timestamp=end - timedelta(seconds=10), end_timestamp=end,
        duration_seconds=10.0,
        pass_count=1, shot_count=1 if outcome == "shot" else 0,
        turnover_count=1 if outcome == "turnover" else 0,
        sprint_count=0, acceleration_count=0, physical_action_count=0,
        outcome=outcome,
        attack_intensity=0.0, physical_intensity=0.0, transition_intensity=0.0,
        possession_quality=quality,
    )


@pytest.fixture()
def engine() -> CoachSituationEngine:
    return CoachSituationEngine(config=CoachSituationConfig())


# ---------------------------------------------------------------------------
# A. Each situation type
# ---------------------------------------------------------------------------

class TestEachSituationType:

    def test_attacking_surge_with_risk(self, engine):
        trend = _trend(attack_trend="increasing", load_trend="increasing")
        state = _state(attack_activity=10.0)
        insight = _insight("high_attack_high_load")
        possessions = [_possession(-0.5, "turnover", 0.1), _possession(-0.8, "turnover", 0.2)]
        situations = engine.generate(possessions, [state], [trend], [insight])
        assert len(situations) == 1
        assert situations[0].situation_type == "ATTACKING_SURGE_WITH_RISK"

    def test_high_tempo_attack(self, engine):
        trend = _trend(attack_trend="increasing", load_trend="increasing")
        state = _state(attack_activity=12.0)
        possessions = [_possession(-0.5, "shot", 0.9), _possession(-0.8, "shot", 0.85)]
        situations = engine.generate(possessions, [state], [trend], [])
        assert situations[0].situation_type == "HIGH_TEMPO_ATTACK"

    def test_inefficient_high_effort(self, engine):
        trend = _trend(attack_trend="decreasing", load_trend="increasing")
        insight = _insight("low_attack_high_load")
        situations = engine.generate([], [], [trend], [insight])
        assert situations[0].situation_type == "INEFFICIENT_HIGH_EFFORT"

    def test_possession_instability(self, engine):
        trend = _trend(possession_pressure_delta=0.3)
        possessions = [_possession(-0.5, "turnover", 0.0), _possession(-0.8, "turnover", 0.0)]
        situations = engine.generate(possessions, [], [trend], [])
        assert situations[0].situation_type == "POSSESSION_INSTABILITY"

    def test_sustained_pressure(self, engine):
        trend = _trend(attack_trend="stable", possession_pressure_delta=-0.3)
        state = _state(attack_activity=12.0)
        possessions = [_possession(-0.3, "shot", 0.9), _possession(-0.6, "shot", 0.8)]
        situations = engine.generate(possessions, [state], [trend], [])
        assert situations[0].situation_type == "SUSTAINED_PRESSURE"

    def test_fatigue_onset(self, engine):
        trend = _trend(fatigue_trend="increasing", load_trend="stable", attack_trend="stable")
        situations = engine.generate([], [], [trend], [])
        assert situations[0].situation_type == "FATIGUE_ONSET"

    def test_defensive_recovery_phase(self, engine):
        trend = _trend(attack_trend="decreasing", load_trend="decreasing", possession_pressure_delta=-0.3)
        state = _state(attack_activity=2.0)
        situations = engine.generate([], [state], [trend], [])
        assert situations[0].situation_type == "DEFENSIVE_RECOVERY_PHASE"

    def test_recovery_consolidation(self, engine):
        trend = _trend(fatigue_trend="decreasing", load_trend="decreasing", possession_pressure_delta=-0.1)
        situations = engine.generate([], [], [trend], [])
        assert situations[0].situation_type == "RECOVERY_CONSOLIDATION"

    def test_efficient_transition(self, engine):
        trend = _trend(attack_trend="increasing", load_trend="decreasing", fatigue_trend="stable")
        situations = engine.generate([], [], [trend], [])
        assert situations[0].situation_type == "EFFICIENT_TRANSITION"

    def test_controlled_tempo(self, engine):
        trend = _trend(attack_trend="stable", load_trend="stable", fatigue_trend="stable",
                        possession_pressure_delta=0.01)
        situations = engine.generate([], [], [trend], [])
        assert situations[0].situation_type == "CONTROLLED_TEMPO"


# ---------------------------------------------------------------------------
# B. Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:

    def test_attacking_surge_takes_priority_over_high_tempo_attack(self, engine):
        """Both conditions are satisfiable simultaneously -- ATTACKING_SURGE_WITH_RISK
        (priority 1) must win over HIGH_TEMPO_ATTACK (priority 2)."""
        trend = _trend(attack_trend="increasing", load_trend="increasing")
        state = _state(attack_activity=12.0)
        insight = _insight("high_attack_high_load")
        possessions = [_possession(-0.3, "shot", 0.9), _possession(-0.6, "turnover", 0.0),
                       _possession(-0.9, "turnover", 0.0)]
        situations = engine.generate(possessions, [state], [trend], [insight])
        assert len(situations) == 1
        assert situations[0].situation_type == "ATTACKING_SURGE_WITH_RISK"

    def test_only_one_situation_per_evaluation_point(self, engine):
        trend = _trend(attack_trend="stable", load_trend="stable", fatigue_trend="stable")
        situations = engine.generate([], [], [trend], [])
        assert len(situations) <= 1


# ---------------------------------------------------------------------------
# C. No-situation cases
# ---------------------------------------------------------------------------

class TestNoSituationCases:

    def test_empty_inputs_produce_no_situations(self, engine):
        assert engine.generate([], [], [], []) == []

    def test_ambiguous_trend_with_no_supporting_state_produces_nothing(self, engine):
        """attack_trend=='increasing' alone, with no TeamState to confirm an
        absolute level and no possessions, must not satisfy HIGH_TEMPO_ATTACK
        or any other rule that depends on missing data."""
        trend = _trend(attack_trend="increasing", load_trend="stable", fatigue_trend="stable",
                        possession_pressure_delta=0.0)
        situations = engine.generate([], [], [trend], [])
        assert situations == []

    def test_mild_mixed_signals_produce_no_situation(self, engine):
        trend = _trend(attack_trend="increasing", load_trend="stable", fatigue_trend="decreasing",
                        possession_pressure_delta=0.02)
        state = _state(attack_activity=3.0)  # below floor -- shouldn't enable HIGH_TEMPO_ATTACK
        situations = engine.generate([], [state], [trend], [])
        assert situations == []


# ---------------------------------------------------------------------------
# D. Severity / confidence / source_insights contract
# ---------------------------------------------------------------------------

class TestContract:

    def test_severity_high_with_two_or_more_relevant_insights(self, engine):
        trend = _trend(attack_trend="increasing", load_trend="increasing")
        state = _state(attack_activity=12.0)
        insights = [_insight("high_attack_high_load"), _insight("attack_activity_rising")]
        possessions = [_possession(-0.5, "turnover", 0.0), _possession(-0.8, "turnover", 0.0)]
        situations = engine.generate(possessions, [state], [trend], insights)
        assert situations[0].severity == "high"
        assert set(situations[0].source_insights) == {"high_attack_high_load", "attack_activity_rising"}

    def test_severity_low_with_no_relevant_insights(self, engine):
        trend = _trend(fatigue_trend="increasing", load_trend="stable", attack_trend="stable")
        situations = engine.generate([], [], [trend], [])
        assert situations[0].severity == "low"
        assert situations[0].source_insights == []
        assert situations[0].confidence == pytest.approx(CoachSituationConfig().trend_only_confidence)

    def test_confidence_uses_min_of_contributing_insights(self, engine):
        trend = _trend(attack_trend="decreasing", load_trend="increasing")
        insights = [_insight("low_attack_high_load", confidence=0.9),
                    _insight("attack_activity_falling", confidence=0.4)]
        situations = engine.generate([], [], [trend], insights)
        assert situations[0].confidence == pytest.approx(0.4)

    def test_source_metrics_contains_trend_and_possession_aggregates(self, engine):
        trend = _trend(possession_pressure_delta=0.3)
        possessions = [_possession(-0.5, "turnover", 0.0), _possession(-0.8, "turnover", 0.0)]
        situations = engine.generate(possessions, [], [trend], [])
        sm = situations[0].source_metrics
        assert sm["window_seconds"] == WIN
        assert sm["recent_possession_count"] == 2
        assert sm["recent_turnover_rate"] == pytest.approx(1.0)
        assert "attack_trend" in sm and "possession_pressure_delta" in sm

    def test_explanation_is_nonempty_and_mentions_team(self, engine):
        trend = _trend(fatigue_trend="increasing", load_trend="stable", attack_trend="stable")
        situations = engine.generate([], [], [trend], [])
        assert TEAM in situations[0].explanation
        assert len(situations[0].explanation) > 10

    def test_determinism(self, engine):
        trend = _trend(attack_trend="increasing", load_trend="increasing")
        state = _state(attack_activity=12.0)
        insight = _insight("high_attack_high_load")
        possessions = [_possession(-0.5, "turnover", 0.0)]
        a = engine.generate(list(possessions), [state], [trend], [insight])
        b = engine.generate(list(possessions), [state], [trend], [insight])
        assert a == b


# ---------------------------------------------------------------------------
# E. Real-data validation: session 3387
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pipeline() -> dict:
    if not EVENTS_PATH.exists():
        pytest.skip(f"events.csv not found at {EVENTS_PATH}")
    stats_path = DATA_DIR / "statistics.csv"
    player_meta = None
    if stats_path.exists():
        from ingestion.kinexon_adapter import KinexonAdapter
        player_meta = KinexonAdapter().load_player_meta(stats_path)
    adapter = KinexonTacticalEventAdapter()
    events = list(adapter.parse(EVENTS_PATH, player_meta=player_meta, match_id="3387"))

    possessions = PossessionEngine().generate(events)
    windows = TeamStateBuilder().build_dual_window(events)
    trend_builder = TeamStateTrendBuilder()
    trends60 = trend_builder.build(windows[60])
    trends300 = trend_builder.build(windows[300])
    insights60 = CoachInsightEngine().generate(trends60)
    insights300 = CoachInsightEngine().generate(trends300)
    return {
        "possessions": possessions,
        "states": {60: windows[60], 300: windows[300]},
        "trends": {60: trends60, 300: trends300},
        "insights": {60: insights60, 300: insights300},
    }


class TestRealDataValidation:

    def test_generates_without_crashing(self, real_pipeline):
        engine = CoachSituationEngine()
        situations = engine.generate(
            real_pipeline["possessions"], real_pipeline["states"][60],
            real_pipeline["trends"][60], real_pipeline["insights"][60],
        )
        assert isinstance(situations, list)

    def test_only_known_situation_types_appear(self, real_pipeline):
        allowed = set(_RELEVANT_INSIGHTS_KEYS := [
            "ATTACKING_SURGE_WITH_RISK", "HIGH_TEMPO_ATTACK", "INEFFICIENT_HIGH_EFFORT",
            "POSSESSION_INSTABILITY", "SUSTAINED_PRESSURE", "FATIGUE_ONSET",
            "DEFENSIVE_RECOVERY_PHASE", "RECOVERY_CONSOLIDATION", "EFFICIENT_TRANSITION",
            "CONTROLLED_TEMPO",
        ])
        engine = CoachSituationEngine()
        situations = engine.generate(
            real_pipeline["possessions"], real_pipeline["states"][60] + real_pipeline["states"][300],
            real_pipeline["trends"][60] + real_pipeline["trends"][300],
            real_pipeline["insights"][60] + real_pipeline["insights"][300],
        )
        assert {s.situation_type for s in situations}.issubset(allowed)

    def test_every_situation_has_required_fields(self, real_pipeline):
        engine = CoachSituationEngine()
        situations = engine.generate(
            real_pipeline["possessions"], real_pipeline["states"][60],
            real_pipeline["trends"][60], real_pipeline["insights"][60],
        )
        for s in situations:
            assert s.severity in ("low", "medium", "high")
            assert 0.0 < s.confidence <= 1.0
            assert isinstance(s.explanation, str) and s.explanation
            assert "window_seconds" in s.source_metrics

    def test_at_least_one_situation_fires_on_real_match(self, real_pipeline):
        engine = CoachSituationEngine()
        situations = engine.generate(
            real_pipeline["possessions"], real_pipeline["states"][60],
            real_pipeline["trends"][60], real_pipeline["insights"][60],
        )
        assert len(situations) > 0

    def test_determinism_on_real_data(self, real_pipeline):
        engine = CoachSituationEngine()
        a = engine.generate(
            list(real_pipeline["possessions"]), list(real_pipeline["states"][60]),
            list(real_pipeline["trends"][60]), list(real_pipeline["insights"][60]),
        )
        b = engine.generate(
            list(real_pipeline["possessions"]), list(real_pipeline["states"][60]),
            list(real_pipeline["trends"][60]), list(real_pipeline["insights"][60]),
        )
        assert a == b

    def test_real_data_summary_report(self, real_pipeline):
        """Not assertion-heavy -- prints the deliverable summary requested for
        the CoachSituation report (run with -s to see it)."""
        from collections import Counter

        print("\n--- CoachSituationEngine v1: session 3387 summary ---")
        for window_seconds in (60, 300):
            engine = CoachSituationEngine()
            situations = engine.generate(
                real_pipeline["possessions"], real_pipeline["states"][window_seconds],
                real_pipeline["trends"][window_seconds], real_pipeline["insights"][window_seconds],
            )
            print(f"\nWindow = {window_seconds}s -> {len(situations)} situations")
            print(f"  by type: {dict(Counter(s.situation_type for s in situations))}")

            by_team: dict = {}
            for s in situations:
                by_team.setdefault(s.team_id, []).append(s)
            for team_id, team_situations in by_team.items():
                print(f"  team={team_id!r}: {len(team_situations)} situations, "
                      f"by type: {dict(Counter(s.situation_type for s in team_situations))}")

            if situations:
                strongest = max(situations, key=lambda s: s.confidence)
                print(f"  strongest: {strongest.situation_type} for {strongest.team_id} "
                      f"at {strongest.timestamp.isoformat()} (confidence={strongest.confidence:.2f})")
        assert True


from analysis.coach_situation import _RELEVANT_INSIGHTS  # noqa: E402  (used only in the test above)
