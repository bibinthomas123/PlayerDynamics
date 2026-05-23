# tests/test_prompt_content.py
"""
Verify that the LLM prompt contains the episodic context block.
Intercepts client.generate() so no Ollama needed.
"""

from unittest.mock import patch, MagicMock
from analysis.episodic_context import (
    CompressedTemporalContext,
    TemporalContextCompressor,
    PlayerEpisode,
)
from explainability.xai_layer import (
    XAILayer,
    format_episodic_context,
    FeatureContribution,
    SemanticFinding,
)


def _make_ctx():
    ep = PlayerEpisode(
        episode_index=1,
        start_minute=22,
        end_minute=35,
        dominant_findings=["locomotor_overload"],
        trend_direction="worsening",
        severity="high",
        interventions=[],
        response="escalated",
        persistence_duration=4,
        peak_anomaly_score=0.8,
        peak_confidence=0.88,
    )
    c = TemporalContextCompressor()
    return c.compress(
        episodes=[ep],
        trend_summaries={"anomaly": "worsening", "recovery": "worsening"},
        recent_findings=[],
        intervention_history=["substitution requested min~72"],
        current_minute=35,
        current_escalation="high",
        current_finding_types=["locomotor_overload"],
    )


def test_episodic_block_in_prompt():
    """The prompt sent to Ollama must contain the episodic context section."""
    xai = XAILayer()
    ctx = _make_ctx()
    captured_prompts = []

    mock_response = MagicMock()
    mock_response.text = "Test NLG output."
    mock_response.eval_count = 50
    mock_response.prompt_eval_count = 80

    def _fake_generate(prompt, **kwargs):
        captured_prompts.append(prompt)
        return mock_response

    with patch.object(xai._llm_nlg, "_available", True), patch.object(
        xai._llm_nlg, "_circuit_open", False
    ):

        mock_client = MagicMock()
        mock_client.generate.side_effect = _fake_generate
        xai._llm_nlg._client = mock_client

        xai._llm_nlg.generate(
            recommendation_type="fatigue_alert",
            confidence=0.82,
            player_name="Test Player",
            semantic_findings=[
                SemanticFinding(
                    finding_type="locomotor_suppression",
                    severity="high",
                    confidence=0.88,
                    summary="Locomotor suppression detected.",
                    supporting_features=["avg_speed"],
                    evidence={
                        "avg_speed": 1.2,
                        "acceleration": -0.4,
                    },
                )
            ],
            top_contributions=[
                FeatureContribution(
                    feature_name="avg_speed",
                    feature_value=1.2,
                    shap_value=0.42,
                    direction="decreasing",
                    human_label="locomotor suppression",
                    formatted_value="1.2 m/s",
                )
            ],
            workload_status="high_risk",
            compressed_context=ctx,
        )

    assert captured_prompts, "No prompt was captured"
    prompt = captured_prompts[0]

    # The episodic section label must be present
    assert "Match history context" in prompt
    # The episode content must appear
    assert "locomotor overload" in prompt.lower() or "Ep1" in prompt
    # The intervention must appear
    assert "substitution" in prompt
    # No raw sensor values
    import re

    assert not re.search(r"\b1[4-9]\d bpm\b", prompt)


def test_no_episodic_block_when_context_none():
    """When compressed_context=None, no episodic section should appear in the prompt."""
    xai = XAILayer()
    captured_prompts = []

    mock_response = MagicMock()
    mock_response.text = "Test."
    mock_response.eval_count = 20
    mock_response.prompt_eval_count = 30

    with patch.object(xai._llm_nlg, "_available", True), patch.object(
        xai._llm_nlg, "_circuit_open", False
    ):
        mock_client = MagicMock()
        mock_client.generate.side_effect = lambda prompt, **kw: (
            captured_prompts.append(prompt) or mock_response
        )
        xai._llm_nlg._client = mock_client

        xai._llm_nlg.generate(
            recommendation_type="fatigue_alert",
            confidence=0.82,
            player_name="Test Player",
            semantic_findings=[
                SemanticFinding(
                    finding_type="locomotor_suppression",
                    severity="high",
                    confidence=0.88,
                    summary="Locomotor suppression detected.",
                    supporting_features=["avg_speed"],
                    evidence={
                        "avg_speed": 1.2,
                        "acceleration": -0.4,
                    },
                )
            ],
            top_contributions=[
                FeatureContribution(
                    feature_name="avg_speed",
                    feature_value=1.2,
                    shap_value=0.42,
                    direction="decreasing",
                    human_label="locomotor suppression",
                    formatted_value="1.2 m/s",
                )
            ],
            compressed_context=None,
        )

    assert "Match history context" not in captured_prompts[0]


def test_format_episodic_context_none_safe():
    assert format_episodic_context(None) == ""


def test_format_episodic_context_empty_safe():
    from analysis.episodic_context import CompressedTemporalContext

    empty = CompressedTemporalContext(
        trajectory_narrative="",
        trend_summaries={},
        recent_context="",
        intervention_history=[],
        recurring_patterns=[],
        state_transitions=[],
        relevant_prior_episodes=[],
        current_escalation="normal",
        episode_count=0,
    )
    assert format_episodic_context(empty) == ""
