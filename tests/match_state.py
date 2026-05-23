"""
test_match_state.py
===================
Tests for MatchState, MatchStateManager, and the orchestrator's
XAI → match-state wiring.

Run with:
    pytest test_match_state.py -v

No external services required. Ollama and the LSTM model are mocked.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from analysis.match_state import MatchState, MatchStateManager


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(player_id: int = 1, match_id: str = "match_001") -> MatchState:
    return MatchState(
        player_id=player_id,
        player_name="Test Player",
        position="CM",
        match_id=match_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. MatchState — basic counter logic
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchStateCounters:

    def test_fatigue_alert_increments_correct_counter(self):
        s = _make_state()
        s.record_alert("fatigue_alert", confidence=0.9, anomaly_score=0.8, elapsed_seconds=600)
        assert s.fatigue_alert_count == 1
        assert s.workload_alert_count == 0
        assert s.anomaly_alert_count == 0

    def test_workload_alert_increments_correct_counter(self):
        s = _make_state()
        s.record_alert("workload_warning", confidence=0.7, anomaly_score=0.6, elapsed_seconds=300)
        assert s.workload_alert_count == 1
        assert s.fatigue_alert_count == 0

    def test_unknown_type_goes_to_anomaly_counter(self):
        s = _make_state()
        s.record_alert("positional_drift", confidence=0.5, anomaly_score=0.4, elapsed_seconds=100)
        assert s.anomaly_alert_count == 1

    def test_consecutive_same_type_increments(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.8, 300)
        s.record_alert("fatigue_alert", 0.9, 0.7, 360)
        assert s.consecutive_alerts == 2

    def test_different_type_resets_consecutive(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.8, 300)
        s.record_alert("fatigue_alert", 0.9, 0.8, 360)
        s.record_alert("workload_warning", 0.7, 0.5, 420)
        assert s.consecutive_alerts == 1

    def test_first_alert_elapsed_captured_once(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.8, 600)
        s.record_alert("fatigue_alert", 0.9, 0.7, 700)
        assert s.first_alert_elapsed_s == 600   # must not be overwritten

    def test_peak_anomaly_score_tracks_maximum(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.5, 100)
        s.record_alert("fatigue_alert", 0.9, 0.95, 200)
        s.record_alert("fatigue_alert", 0.9, 0.3, 300)
        assert s.peak_anomaly_score == pytest.approx(0.95)

    def test_mean_anomaly_score_correct(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.4, 100)
        s.record_alert("fatigue_alert", 0.9, 0.6, 200)
        assert s.mean_anomaly_score == pytest.approx(0.5)

    def test_mean_anomaly_score_zero_when_no_alerts(self):
        s = _make_state()
        assert s.mean_anomaly_score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. MatchState — telemetry
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchStateTelemetry:

    def test_peak_hr_tracks_maximum(self):
        s = _make_state()
        s.record_telemetry(speed_ms=5.0, hr_bpm=160.0)
        s.record_telemetry(speed_ms=5.0, hr_bpm=185.0)
        s.record_telemetry(speed_ms=5.0, hr_bpm=170.0)
        assert s.peak_hr_bpm == pytest.approx(185.0)

    def test_sprint_counted_at_threshold(self):
        s = _make_state()
        s.record_telemetry(speed_ms=7.0, hr_bpm=160.0)   # exactly at threshold
        assert s.sprint_count == 1

    def test_sprint_not_counted_below_threshold(self):
        s = _make_state()
        s.record_telemetry(speed_ms=6.99, hr_bpm=160.0)
        assert s.sprint_count == 0

    def test_multiple_sprints_accumulate(self):
        s = _make_state()
        for _ in range(5):
            s.record_telemetry(speed_ms=8.5, hr_bpm=170.0)
        assert s.sprint_count == 5


# ─────────────────────────────────────────────────────────────────────────────
# 3. MatchState — build_llm_context output
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildLLMContext:

    def test_no_alerts_returns_default_message(self):
        s = _make_state()
        ctx = s.build_llm_context()
        assert ctx == "No prior alerts this match."

    def test_context_contains_player_name(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.8, 300)
        assert "Test Player" in s.build_llm_context()

    def test_context_contains_alert_counts(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.8, 300)
        s.record_alert("workload_warning", 0.7, 0.5, 400)
        ctx = s.build_llm_context()
        assert "fatigue=1" in ctx
        assert "workload=1" in ctx

    def test_context_contains_first_alert_minute(self):
        s = _make_state()
        s.record_alert("fatigue_alert", 0.9, 0.8, 3660)  # 61 minutes
        ctx = s.build_llm_context()
        assert "61 min" in ctx

    def test_context_contains_peak_hr(self):
        s = _make_state()
        s.record_telemetry(speed_ms=5.0, hr_bpm=191.0)
        s.record_alert("fatigue_alert", 0.9, 0.8, 100)
        ctx = s.build_llm_context()
        assert "191" in ctx


# ─────────────────────────────────────────────────────────────────────────────
# 4. MatchStateManager — lifecycle and isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestMatchStateManager:

    def test_get_or_create_returns_same_instance(self):
        mgr = MatchStateManager()
        mgr.start_match("m1")
        s1 = mgr.get_or_create(1, "Alice", "FW", match_id="m1")
        s2 = mgr.get_or_create(1, "Alice", "FW", match_id="m1")
        assert s1 is s2

    def test_different_players_get_different_states(self):
        mgr = MatchStateManager()
        mgr.start_match("m1")
        s1 = mgr.get_or_create(1, "Alice", "FW", match_id="m1")
        s2 = mgr.get_or_create(2, "Bob",   "CB", match_id="m1")
        assert s1 is not s2

    def test_no_cross_match_leakage(self):
        """State from match A must never bleed into match B for the same player."""
        mgr = MatchStateManager()

        mgr.start_match("match_A")
        s_a = mgr.get_or_create(7, "Carlo", "CM", match_id="match_A")
        s_a.record_alert("fatigue_alert", 0.9, 0.8, 300)
        assert s_a.fatigue_alert_count == 1

        mgr.end_match("match_A")
        mgr.start_match("match_B")
        s_b = mgr.get_or_create(7, "Carlo", "CM", match_id="match_B")

        assert s_b is not s_a
        assert s_b.fatigue_alert_count == 0

    def test_end_match_frees_memory(self):
        mgr = MatchStateManager()
        mgr.start_match("m1")
        mgr.get_or_create(1, "Alice", "FW", match_id="m1")
        assert len(mgr._states) == 1
        mgr.end_match("m1")
        assert len(mgr._states) == 0

    def test_start_match_clears_stale_state(self):
        """start_match for match_B should wipe state from match_A."""
        mgr = MatchStateManager()
        mgr.start_match("match_A")
        mgr.get_or_create(1, "Alice", "FW", match_id="match_A")

        mgr.start_match("match_B")   # does NOT call end_match first
        assert all(k[1] == "match_B" for k in mgr._states)

    def test_active_match_id_used_when_not_explicit(self):
        mgr = MatchStateManager()
        mgr.start_match("implicit_match")
        s = mgr.get_or_create(1, "Alice", "FW")   # no match_id kwarg
        assert s.match_id == "implicit_match"

    def test_none_active_falls_back_to_default(self):
        mgr = MatchStateManager()   # no start_match called
        s = mgr.get_or_create(1, "Alice", "FW")
        assert s.match_id == "default"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Thread safety — concurrent writes must not corrupt state
# ─────────────────────────────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_record_alert_correct_count(self):
        """
        100 threads each call record_alert once.
        Final fatigue_alert_count must be exactly 100 — no lost writes.
        """
        s = _make_state()
        N = 100
        barrier = threading.Barrier(N)

        def worker():
            barrier.wait()   # all threads start simultaneously
            s.record_alert("fatigue_alert", 0.9, 0.8, 300)

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert s.fatigue_alert_count == N

    def test_concurrent_record_telemetry_no_corruption(self):
        """Peak HR must equal the maximum value written across all threads."""
        s = _make_state()
        N = 50
        hr_values = [float(100 + i) for i in range(N)]
        barrier = threading.Barrier(N)

        def worker(hr):
            barrier.wait()
            s.record_telemetry(speed_ms=5.0, hr_bpm=hr)

        threads = [threading.Thread(target=worker, args=(hr,)) for hr in hr_values]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert s.peak_hr_bpm == pytest.approx(max(hr_values))

    def test_mean_anomaly_score_consistent_under_concurrency(self):
        """mean = sum/count must be internally consistent; must not raise ZeroDivisionError."""
        s = _make_state()
        N = 80
        barrier = threading.Barrier(N)
        errors = []

        def worker():
            barrier.wait()
            s.record_alert("fatigue_alert", 0.9, 0.5, 100)
            try:
                _ = s.mean_anomaly_score
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exceptions under concurrency: {errors}"
        assert s.mean_anomaly_score == pytest.approx(0.5)

    def test_build_llm_context_never_raises_under_concurrency(self):
        """Context reads and alert writes must not deadlock or raise."""
        s = _make_state()
        errors = []

        def writer():
            for _ in range(20):
                s.record_alert("fatigue_alert", 0.9, 0.7, 100)
                time.sleep(0.001)

        def reader():
            for _ in range(20):
                try:
                    s.build_llm_context()
                except Exception as exc:
                    errors.append(exc)
                time.sleep(0.001)

        threads = (
            [threading.Thread(target=writer) for _ in range(4)]
            + [threading.Thread(target=reader) for _ in range(4)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestrator wiring — match state updated inline with XAI
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorMatchStateWiring:
    """
    Verifies that process_live_event populates match state when XAI fires,
    without requiring a real model, SHAP, or Ollama.
    All heavy dependencies are patched.
    """

    def _make_pipeline(self):
        """Build a PlayersDataAnalysisPipeline with all external deps mocked."""
        with (
            patch("analysis.orchestrator.PatternAnalysisEngine"),
            patch("analysis.orchestrator.XAILayer"),
            patch("analysis.orchestrator.BaselineBuilder"),
            patch("analysis.orchestrator.FeedbackStore"),
            patch("analysis.orchestrator.RecalibrationPipeline"),
            patch("analysis.orchestrator.FairnessMonitor"),
            patch("analysis.orchestrator.TelemetryValidityLayer"),
            patch("analysis.orchestrator.SystemInvariantGuard"),
            patch("utils.reliability.determinism.f"),
            patch("utils.reliability.determinism.TemporalCausalityGuard"),
        ):
            from analysis.orchestrator import PlayersDataAnalysisPipeline
            pipeline = PlayersDataAnalysisPipeline()
        return pipeline

    def _register_player(self, pipeline, player_id: int = 7):
        pipeline.registry.register(
            player_id=player_id,
            external_id=f"p{player_id:03d}",
            name="Carlo Rossi",
            position="CM",
            age=27,
        )
        player = pipeline.registry.get(player_id)
        player["model"] = MagicMock(is_trained=True, model_version="v1",
                                    player_id=player_id)
        player["baseline"] = MagicMock()
        return player

    def _make_anomaly_result(self, player_id: int = 7):
        from utils.alert_manager import AlertLevel
        result = MagicMock()
        result.player_id        = player_id
        result.external_id      = f"p{player_id:03d}"
        result.is_anomaly       = True
        result.anomaly_score    = 0.88
        result.recommendation_type = "fatigue_alert"
        result.confidence       = 0.91
        result.workload_status  = "high_risk"
        result.alert_level      = AlertLevel.CRITICAL
        result.persistence_windows = 5
        result.model_type       = "lstm"
        result.feature_vector   = {}
        result.triggered_at     = __import__("datetime").datetime.now(
                                      __import__("datetime").timezone.utc)
        result.raw_sequence     = None
        result.raw_mask         = None
        return result

    def test_match_state_created_on_xai_trigger(self):
        pipeline = self._make_pipeline()
        pipeline.start_match("match_test_001")
        self._register_player(pipeline, player_id=7)

        # Fake pattern engine returning an alert
        pipeline.pattern_engine.analyze.return_value = self._make_anomaly_result(7)

        # Fake telemetry validity — VALID
        from analysis.telemetry_validity import TelemetryStatus
        validity = MagicMock(status=TelemetryStatus.VALID, issues=[])
        pipeline.tvl.validate_event.return_value = validity
        pipeline.causality_guard.validate_sequence.return_value = True
        pipeline.guard.check.return_value = None

        # Safe mode allows SHAP
        with patch("utils.reliability.safe_mode.safe_mode") as sm:
            sm.is_feature_enabled.return_value = True
            pipeline._last_xai_ts[7] = 0.0   # cooldown satisfied

            # XAI layer returns a fake SHAPExplanation
            fake_explanation = MagicMock()
            fake_explanation.shap_values = {}
            fake_explanation.nlg_summary = "Test summary"
            pipeline.xai_layer.explain_from_dict.return_value = fake_explanation

            with patch.object(pipeline, "_build_xai_feature_vector",
                              return_value={"window_avg_speed_ms": 8.5,
                                            "heart_rate_bpm": 172.0,
                                            "elapsed_seconds": 2700}):
                pipeline.process_live_event({"player_external_id": "p007"})

        # Match state must now exist and have one fatigue alert recorded
        state = pipeline._match_state.get_or_create(7, "Carlo Rossi", "CM",
                                                     match_id="match_test_001")
        assert state.fatigue_alert_count == 1
        assert state.peak_hr_bpm == pytest.approx(172.0)

    def test_match_state_not_created_when_xai_gated_out(self):
        """If XAI is suppressed (e.g. not a sustained alert), state must stay clean."""
        pipeline = self._make_pipeline()
        pipeline.start_match("match_gated")
        self._register_player(pipeline, player_id=7)

        from utils.alert_manager import AlertLevel
        result = self._make_anomaly_result(7)
        result.alert_level = AlertLevel.NONE          # below WARNING threshold — sustained_alert=False
        result.persistence_windows = 1                # below minimum
        pipeline.pattern_engine.analyze.return_value = result

        from analysis.telemetry_validity import TelemetryStatus
        validity = MagicMock(status=TelemetryStatus.VALID, issues=[])
        pipeline.tvl.validate_event.return_value = validity
        pipeline.causality_guard.validate_sequence.return_value = True
        pipeline.guard.check.return_value = None

        with patch("utils.reliability.safe_mode.safe_mode") as sm:
            sm.is_feature_enabled.return_value = True
            pipeline.process_live_event({"player_external_id": "p007"})

        state = pipeline._match_state.get_or_create(7, "Carlo Rossi", "CM",
                                                     match_id="match_gated")
        assert state.fatigue_alert_count == 0

    def test_match_context_passed_to_explain_from_dict(self):
        """match_state.build_llm_context() result must reach explain_from_dict."""
        pipeline = self._make_pipeline()
        pipeline.start_match("match_ctx")
        self._register_player(pipeline, player_id=7)

        # Pre-populate state so context is non-trivial
        state = pipeline._match_state.get_or_create(7, "Carlo Rossi", "CM",
                                                     match_id="match_ctx")
        state.record_alert("fatigue_alert", 0.9, 0.8, 1800)

        pipeline.pattern_engine.analyze.return_value = self._make_anomaly_result(7)

        from analysis.telemetry_validity import TelemetryStatus
        validity = MagicMock(status=TelemetryStatus.VALID, issues=[])
        pipeline.tvl.validate_event.return_value = validity
        pipeline.causality_guard.validate_sequence.return_value = True
        pipeline.guard.check.return_value = None

        fake_explanation = MagicMock()
        fake_explanation.shap_values = {}
        fake_explanation.nlg_summary = "ok"
        pipeline.xai_layer.explain_from_dict.return_value = fake_explanation
        pipeline._last_xai_ts[7] = 0.0

        with patch("utils.reliability.safe_mode.safe_mode") as sm:
            sm.is_feature_enabled.return_value = True
            with patch.object(pipeline, "_build_xai_feature_vector",
                              return_value={"window_avg_speed_ms": 5.0,
                                            "heart_rate_bpm": 160.0,
                                            "elapsed_seconds": 2000}):
                pipeline.process_live_event({"player_external_id": "p007"})

        call_kwargs = pipeline.xai_layer.explain_from_dict.call_args.kwargs
        assert "match_context" in call_kwargs
        assert "fatigue" in call_kwargs["match_context"]   # state summary present

    def test_start_match_end_match_lifecycle_on_pipeline(self):
        pipeline = self._make_pipeline()
        pipeline.start_match("life_001")
        assert pipeline._active_match_id == "life_001"
        assert pipeline._match_state._active_match_id == "life_001"

        pipeline.end_match()
        assert pipeline._active_match_id is None
        assert pipeline._match_state._active_match_id is None

    def test_second_match_has_clean_state(self):
        pipeline = self._make_pipeline()

        pipeline.start_match("game_1")
        s1 = pipeline._match_state.get_or_create(7, "Carlo", "CM", match_id="game_1")
        s1.record_alert("fatigue_alert", 0.9, 0.8, 300)
        pipeline.end_match()

        pipeline.start_match("game_2")
        s2 = pipeline._match_state.get_or_create(7, "Carlo", "CM", match_id="game_2")
        assert s2.fatigue_alert_count == 0
        assert s2 is not s1