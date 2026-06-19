"""
tests/test_coach_insight.py

Validates CoachInsightEngine v1 (analysis/coach_insight.py):

    CoachInsight        -- canonical dataclass
    CoachInsightEngine   -- converts TeamStateTrend rows into deterministic
                            coach-facing observations

Covers:
  A. Each insight type (8 categories) firing in isolation
  B. Multiple simultaneous insights (composites + singulars together)
  C. No-insight cases (sub-threshold deltas, stable trends)
  D. Source metrics / threshold / confidence present on every insight
  E. Real-data validation against data/events.csv (session 3387)

Run:
    pytest tests/test_coach_insight.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from analysis.coach_insight import CoachInsight, CoachInsightEngine
from analysis.team_state import TeamStateBuilder
from analysis.team_state_trend import TeamStateTrend, TeamStateTrendBuilder
from config.settings import CoachInsightConfig
from ingestion.tactical_event import KinexonTacticalEventAdapter

DATA_DIR = _ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.csv"

BASE_TS = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)


def _trend(
    minute_offset: int = 0,
    team_id: str = "SC Magdeburg",
    window_seconds: int = 60,
    possession_pressure_delta: float = 0.0,
    attack_activity_delta: float = 0.0,
    physical_load_delta: float = 0.0,
    fatigue_burden_delta: float = 0.0,
    confidence_delta: float = 0.0,
    attack_trend: str = "stable",
    load_trend: str = "stable",
    fatigue_trend: str = "stable",
) -> TeamStateTrend:
    return TeamStateTrend(
        timestamp=BASE_TS + timedelta(minutes=minute_offset),
        team_id=team_id,
        window_seconds=window_seconds,
        possession_pressure_delta=possession_pressure_delta,
        attack_activity_delta=attack_activity_delta,
        physical_load_delta=physical_load_delta,
        fatigue_burden_delta=fatigue_burden_delta,
        confidence_delta=confidence_delta,
        attack_trend=attack_trend,
        load_trend=load_trend,
        fatigue_trend=fatigue_trend,
    )


@pytest.fixture()
def engine() -> CoachInsightEngine:
    return CoachInsightEngine(config=CoachInsightConfig())


# ---------------------------------------------------------------------------
# A. Each insight type
# ---------------------------------------------------------------------------

class TestEachInsightType:

    def test_attack_activity_rising(self, engine):
        t = _trend(attack_trend="increasing", attack_activity_delta=12.0)
        insights = engine.generate([t])
        categories = {i.category for i in insights}
        assert "attack_activity_rising" in categories

    def test_attack_activity_falling(self, engine):
        t = _trend(attack_trend="decreasing", attack_activity_delta=-12.0)
        insights = engine.generate([t])
        assert {i.category for i in insights} == {"attack_activity_falling"}

    def test_workload_spike(self, engine):
        t = _trend(load_trend="increasing", physical_load_delta=25.0)
        insights = engine.generate([t])
        assert {i.category for i in insights} == {"workload_spike"}

    def test_workload_drop(self, engine):
        t = _trend(load_trend="decreasing", physical_load_delta=-25.0)
        insights = engine.generate([t])
        assert {i.category for i in insights} == {"workload_drop"}

    def test_possession_pressure_increasing(self, engine):
        t = _trend(possession_pressure_delta=0.4)
        insights = engine.generate([t])
        assert {i.category for i in insights} == {"possession_pressure_increasing"}

    def test_possession_pressure_decreasing(self, engine):
        t = _trend(possession_pressure_delta=-0.4)
        insights = engine.generate([t])
        assert {i.category for i in insights} == {"possession_pressure_decreasing"}

    def test_high_attack_high_load_composite(self, engine):
        t = _trend(
            attack_trend="increasing", attack_activity_delta=12.0,
            load_trend="increasing", physical_load_delta=25.0,
        )
        insights = engine.generate([t])
        categories = {i.category for i in insights}
        assert "high_attack_high_load" in categories
        # Composite is emitted IN ADDITION to the singulars, not instead of them.
        assert "attack_activity_rising" in categories
        assert "workload_spike" in categories
        assert len(insights) == 3

    def test_low_attack_high_load_composite(self, engine):
        t = _trend(
            attack_trend="decreasing", attack_activity_delta=-12.0,
            load_trend="increasing", physical_load_delta=25.0,
        )
        insights = engine.generate([t])
        categories = {i.category for i in insights}
        assert "low_attack_high_load" in categories
        assert "attack_activity_falling" in categories
        assert "workload_spike" in categories
        assert len(insights) == 3


# ---------------------------------------------------------------------------
# B. Multiple simultaneous insights
# ---------------------------------------------------------------------------

class TestMultipleSimultaneousInsights:

    def test_attack_rising_and_possession_pressure_increasing_together(self, engine):
        t = _trend(
            attack_trend="increasing", attack_activity_delta=15.0,
            possession_pressure_delta=0.5,
        )
        insights = engine.generate([t])
        categories = {i.category for i in insights}
        assert categories == {"attack_activity_rising", "possession_pressure_increasing"}

    def test_independent_teams_evaluated_independently(self, engine):
        t1 = _trend(team_id="SC Magdeburg", attack_trend="increasing", attack_activity_delta=12.0)
        t2 = _trend(team_id="HSG Wetzlar", load_trend="decreasing", physical_load_delta=-25.0)
        insights = engine.generate([t1, t2])
        by_team = {i.team_id: i.category for i in insights}
        assert by_team["SC Magdeburg"] == "attack_activity_rising"
        assert by_team["HSG Wetzlar"] == "workload_drop"

    def test_insights_across_multiple_trend_rows(self, engine):
        trends = [
            _trend(minute_offset=0, attack_trend="increasing", attack_activity_delta=10.0),
            _trend(minute_offset=1, load_trend="decreasing", physical_load_delta=-20.0),
        ]
        insights = engine.generate(trends)
        assert len(insights) == 2
        assert [i.timestamp for i in insights] == sorted(i.timestamp for i in insights)


# ---------------------------------------------------------------------------
# C. No-insight cases
# ---------------------------------------------------------------------------

class TestNoInsightCases:

    def test_stable_trend_produces_no_insights(self, engine):
        t = _trend(attack_trend="stable", load_trend="stable", fatigue_trend="stable")
        assert engine.generate([t]) == []

    def test_label_says_increasing_but_delta_below_insight_threshold(self, engine):
        """
        The trend label can already say "increasing" (it cleared the looser
        TeamStateTrendConfig threshold) while still falling short of the
        stricter CoachInsightConfig threshold -- no insight should fire.
        """
        t = _trend(attack_trend="increasing", attack_activity_delta=4.0)  # < 9.0 threshold
        assert engine.generate([t]) == []

    def test_delta_above_threshold_but_label_disagrees(self, engine):
        """Defensive: a label/delta mismatch (e.g. stale data) must not fire."""
        t = _trend(attack_trend="stable", attack_activity_delta=20.0)
        assert engine.generate([t]) == []

    def test_possession_pressure_below_threshold_is_silent(self, engine):
        t = _trend(possession_pressure_delta=0.1)  # < 0.25 threshold
        assert engine.generate([t]) == []

    def test_empty_input_returns_empty_list(self, engine):
        assert engine.generate([]) == []


# ---------------------------------------------------------------------------
# D. Source metrics / threshold / confidence on every insight
# ---------------------------------------------------------------------------

class TestInsightContract:

    def test_singular_insight_carries_source_metric_and_threshold(self, engine):
        t = _trend(attack_trend="increasing", attack_activity_delta=18.0)
        insight = engine.generate([t])[0]
        assert insight.metadata["source_metrics"] == ["attack_activity_delta"]
        assert insight.metadata["values"]["attack_activity_delta"] == 18.0
        assert insight.metadata["thresholds_crossed"]["attack_activity_delta"] == 9.0
        assert insight.metadata["window_seconds"] == 60
        assert 0.0 < insight.confidence <= 1.0

    def test_falling_category_records_negative_threshold(self, engine):
        t = _trend(attack_trend="decreasing", attack_activity_delta=-18.0)
        insight = engine.generate([t])[0]
        assert insight.metadata["thresholds_crossed"]["attack_activity_delta"] == -9.0

    def test_composite_insight_carries_both_source_metrics(self, engine):
        t = _trend(
            attack_trend="increasing", attack_activity_delta=12.0,
            load_trend="increasing", physical_load_delta=25.0,
        )
        insights = engine.generate([t])
        composite = next(i for i in insights if i.category == "high_attack_high_load")
        assert set(composite.metadata["source_metrics"]) == {"attack_activity_delta", "physical_load_delta"}
        assert composite.metadata["values"]["attack_activity_delta"] == 12.0
        assert composite.metadata["values"]["physical_load_delta"] == 25.0

    def test_severity_increases_with_ratio(self, engine):
        cfg = CoachInsightConfig()
        bare = _trend(attack_trend="increasing", attack_activity_delta=cfg.attack_activity_insight_threshold)
        strong = _trend(attack_trend="increasing", attack_activity_delta=cfg.attack_activity_insight_threshold * 3)
        bare_insight = engine.generate([bare])[0]
        strong_insight = engine.generate([strong])[0]
        assert bare_insight.severity == "low"
        assert strong_insight.severity == "high"
        assert strong_insight.confidence > bare_insight.confidence

    def test_composite_confidence_is_min_of_components(self, engine):
        cfg = CoachInsightConfig()
        # Strong attack signal, barely-qualifying load signal.
        t = _trend(
            attack_trend="increasing", attack_activity_delta=cfg.attack_activity_insight_threshold * 5,
            load_trend="increasing", physical_load_delta=cfg.physical_load_insight_threshold,
        )
        insights = engine.generate([t])
        attack_insight = next(i for i in insights if i.category == "attack_activity_rising")
        load_insight = next(i for i in insights if i.category == "workload_spike")
        composite = next(i for i in insights if i.category == "high_attack_high_load")
        assert composite.confidence == pytest.approx(min(attack_insight.confidence, load_insight.confidence))

    def test_message_is_nonempty_human_readable_string(self, engine):
        t = _trend(attack_trend="increasing", attack_activity_delta=12.0)
        insight = engine.generate([t])[0]
        assert isinstance(insight.message, str) and len(insight.message) > 10

    def test_determinism(self, engine):
        t = _trend(
            attack_trend="increasing", attack_activity_delta=12.0,
            load_trend="increasing", physical_load_delta=25.0,
            possession_pressure_delta=0.4,
        )
        a = engine.generate([t])
        b = engine.generate([t])
        assert a == b


# ---------------------------------------------------------------------------
# E. Real-data validation: session 3387
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_trends() -> dict:
    if not EVENTS_PATH.exists():
        pytest.skip(f"events.csv not found at {EVENTS_PATH}")
    stats_path = DATA_DIR / "statistics.csv"
    player_meta = None
    if stats_path.exists():
        from ingestion.kinexon_adapter import KinexonAdapter
        player_meta = KinexonAdapter().load_player_meta(stats_path)
    adapter = KinexonTacticalEventAdapter()
    events = list(adapter.parse(EVENTS_PATH, player_meta=player_meta, match_id="3387"))
    windows = TeamStateBuilder().build_dual_window(events)
    trend_builder = TeamStateTrendBuilder()
    return {
        60: trend_builder.build(windows[60]),
        300: trend_builder.build(windows[300]),
    }


class TestRealDataValidation:

    def test_generates_without_crashing(self, real_trends):
        insights = CoachInsightEngine().generate(real_trends[60])
        assert isinstance(insights, list)

    def test_only_known_categories_appear(self, real_trends):
        allowed = {
            "attack_activity_rising", "attack_activity_falling",
            "workload_spike", "workload_drop",
            "possession_pressure_increasing", "possession_pressure_decreasing",
            "high_attack_high_load", "low_attack_high_load",
        }
        insights = CoachInsightEngine().generate(real_trends[60] + real_trends[300])
        assert {i.category for i in insights}.issubset(allowed)

    def test_every_insight_has_required_fields(self, real_trends):
        insights = CoachInsightEngine().generate(real_trends[60])
        for i in insights:
            assert i.severity in ("low", "medium", "high")
            assert isinstance(i.message, str) and i.message
            assert 0.0 < i.confidence <= 1.0
            assert i.metadata["source_metrics"]
            assert i.metadata["values"]
            assert i.metadata["thresholds_crossed"]
            assert "window_seconds" in i.metadata

    def test_at_least_one_insight_fires_on_real_match(self, real_trends):
        insights = CoachInsightEngine().generate(real_trends[60])
        assert len(insights) > 0

    def test_determinism_on_real_data(self, real_trends):
        a = CoachInsightEngine().generate(list(real_trends[60]))
        b = CoachInsightEngine().generate(list(real_trends[60]))
        assert a == b

    def test_real_data_summary_report(self, real_trends):
        """Not assertion-heavy -- prints the deliverable summary requested for
        the CoachInsightEngine report (run with -s to see it)."""
        from collections import Counter

        print("\n--- CoachInsightEngine v1: session 3387 summary ---")
        for window_seconds in (60, 300):
            insights = CoachInsightEngine().generate(real_trends[window_seconds])
            print(f"\nWindow = {window_seconds}s -> {len(insights)} insights")
            print(f"  by category: {dict(Counter(i.category for i in insights))}")

            by_team: dict = {}
            for i in insights:
                by_team.setdefault(i.team_id, []).append(i)
            for team_id, team_insights in by_team.items():
                print(f"  team={team_id!r}: {len(team_insights)} insights, "
                      f"by category: {dict(Counter(i.category for i in team_insights))}")

            if insights:
                strongest = max(insights, key=lambda i: i.confidence)
                print(f"  strongest signal: {strongest.category} for {strongest.team_id} "
                      f"at {strongest.timestamp.isoformat()} (confidence={strongest.confidence:.2f}, "
                      f"severity={strongest.severity})")
                print(f"    message: {strongest.message}")
        assert True
