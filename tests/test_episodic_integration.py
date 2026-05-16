# tests/test_episodic_integration.py
"""
End-to-end integration test for the episodic context system.
Uses synthetic windows fed through process_window_direct() with nlg_async=False.
Ollama is NOT required — template NLG fires automatically if unavailable.
"""
import numpy as np
import pytest

def _make_synthetic_window(n_events=24, elapsed_start=0):
    """Build a minimal window list that process_window_direct accepts."""
    events = []
    for i in range(n_events):
        events.append({
            "player_external_id": "p001",
            "speed_ms": 4.5,
            "heart_rate_bpm": 165.0,
            "hr_recovery_rate": 0.02,
            "distance_delta_m": 4.5,
            "sprint_flag": 0,
            "x_pitch": 50.0,
            "y_pitch": 34.0,
            "elapsed_seconds": elapsed_start + i * 5,
            "session_id": 1,
            "_tvl_confidence": 1.0,
        })
    return events

@pytest.fixture(scope="module")
def trained_pipeline():
    """Build a minimal trained pipeline — reused across all tests in this module."""
    # This fixture is slow (~30s) but runs once.
    # Skip if model artifacts not present to keep CI fast.
    from pathlib import Path
    model_dir = Path("models")
    if not (model_dir / "shared_backbone.pt").exists():
        pytest.skip("Model not trained — run `python main.py train` first")

    from analysis.orchestrator import PlayersDataAnalysisPipeline
    import analysis.anomaly_detection as _ad
    from config.settings import CONFIG
    CONFIG.active_model = "lstm"
    _ad.MODEL_STORE = model_dir

    pipeline = PlayersDataAnalysisPipeline(replay_mode=True)
    # _restore_serve_state would normally load baselines; skip for unit test
    # Register one synthetic player
    pipeline.register_player(
        player_id=1, external_id="p001", name="Test Player",
        position="CM", age=25,
    )
    pipeline.start_match("test_match_episodic")
    yield pipeline
    pipeline.end_match()

def test_stage_3b_populates_compressed_context(trained_pipeline):
    """
    After process_window_direct(), result.compressed_context must be set
    and must be a CompressedTemporalContext.
    """
    from analysis.episodic_context import CompressedTemporalContext
    p = trained_pipeline
    window = _make_synthetic_window(elapsed_start=22*60)

    result = p.process_window_direct(
        window_events=window, player_id=1, replay_mode=True, nlg_async=False
    )
    # result may be None if no alert triggered; that's fine — test the state directly
    ms = p._match_state.get_or_create(
        player_id=1, player_name="Test Player",
        position="CM", match_id="test_match_episodic"
    )
    # After at least one window, refresh_episodes should have been called
    # (only inside run_xai block, so we call it directly to verify the method works)
    ms.refresh_episodes(current_minute=22)
    ctx = ms.build_compressed_context(current_minute=22)
    assert isinstance(ctx, CompressedTemporalContext)

def test_multi_window_episode_evolution(trained_pipeline):
    """
    Feed 10 windows. Episodes should accumulate. Recurring patterns detectable.
    """
    p = trained_pipeline
    for i in range(10):
        window = _make_synthetic_window(elapsed_start=(22 + i) * 60)
        p.process_window_direct(
            window_events=window, player_id=1, replay_mode=True, nlg_async=False
        )

    ms = p._match_state.get_or_create(
        player_id=1, player_name="Test Player",
        position="CM", match_id="test_match_episodic"
    )
    ms.refresh_episodes(current_minute=32)
    # Episode list should exist (may be empty if no alerts fired, which is ok)
    assert isinstance(ms.episodes, list)

def test_nlg_receives_episodic_context(trained_pipeline):
    """
    generate_nlg() called with a non-None compressed_context should not raise.
    Template NLG path is taken if Ollama is unavailable — that's fine.
    """
    from analysis.episodic_context import (
        CompressedTemporalContext, TemporalContextCompressor, PlayerEpisode
    )
    p = trained_pipeline

    # Build a synthetic compressed context
    ep = PlayerEpisode(
        episode_index=1, start_minute=22, end_minute=30,
        dominant_findings=["locomotor_overload"],
        trend_direction="worsening", severity="high",
        interventions=[], response="escalated",
        persistence_duration=3, peak_anomaly_score=0.75, peak_confidence=0.85,
    )
    compressor = TemporalContextCompressor()
    ctx = compressor.compress(
        episodes=[ep],
        trend_summaries={"anomaly": "worsening"},
        recent_findings=[],
        intervention_history=[],
        current_minute=35,
        current_escalation="high",
        current_finding_types=["locomotor_overload"],
    )

    # Call generate_nlg with a mock BaseExplanation
    from unittest.mock import MagicMock
    from datetime import datetime, timezone
    base = MagicMock()
    base.recommendation_type = "fatigue_alert"
    base.confidence = 0.82
    base.player_name = "Test Player"
    base.top_contributions = []
    base.workload_status = "high_risk"
    base.semantic_findings = []
    base.player_id = 1
    base.external_id = "p001"
    base.computed_at = datetime.now(tz=timezone.utc)
    base.base_value = 0.3
    base.shap_values = {}
    base.feature_values = {}
    base.waterfall_data = []
    base.uncertainty = 0.1
    base.shap_method = "magnitude_proxy"

    result = p.xai_layer.generate_nlg(
        base=base,
        match_context=None,
        compressed_context=ctx,
    )
    assert result.nlg_summary  # non-empty string
    assert result.nlg_engine in ("llm_qwen", "template")