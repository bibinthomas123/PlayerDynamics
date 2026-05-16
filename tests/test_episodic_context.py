# tests/test_episodic_context.py

import re
import pytest

from analysis.episodic_context import (
    PlayerEpisode,
    TemporalContextCompressor,
    CompressedTemporalContext,
    MAX_CONTEXT_TOKENS,
)


def _make_findings(types_and_sevs):
    """Helper: build a findings list from [(type, severity, minute), ...]"""
    return [
        {
            "type": t,
            "severity": s,
            "minute": m,
            "confidence": 0.75,
            "state": "active",
        }
        for t, s, m in types_and_sevs
    ]


def test_episode_segmentation_single_type():
    findings = _make_findings([
        ("locomotor_overload", "high", 22),
        ("locomotor_overload", "high", 23),
        ("locomotor_overload", "critical", 24),
    ])

    c = TemporalContextCompressor()

    episodes = c.build_episodes_from_findings(
        findings,
        anomaly_scores_snapshot=[0.6, 0.7, 0.8],
    )

    assert len(episodes) == 1
    assert episodes[0].dominant_findings == ["locomotor_overload"]
    assert episodes[0].severity == "critical"


def test_episode_segmentation_type_change_opens_new():
    findings = _make_findings([
        ("locomotor_overload", "high", 22),
        ("locomotor_overload", "high", 23),

        # gap > merge window → new episode
        ("cardiovascular_overload", "high", 30),
        ("cardiovascular_overload", "high", 31),
    ])

    c = TemporalContextCompressor()

    episodes = c.build_episodes_from_findings(
        findings,
        anomaly_scores_snapshot=[0.6] * 4,
    )

    assert len(episodes) == 2
    assert episodes[0].dominant_findings[0] == "locomotor_overload"
    assert episodes[1].dominant_findings[0] == "cardiovascular_overload"


def test_recurring_patterns_detected():
    eps = [
        PlayerEpisode(
            1, 10, 20,
            ["locomotor_overload"],
            "worsening",
            "high",
            [],
            "escalated",
            3,
            0.7,
            0.8,
        ),
        PlayerEpisode(
            2, 30, 40,
            ["recovery_degradation"],
            "stable",
            "high",
            [],
            "persisted",
            2,
            0.6,
            0.7,
        ),
        PlayerEpisode(
            3, 50, 60,
            ["locomotor_overload"],
            "worsening",
            "critical",
            [],
            "unknown",
            4,
            0.85,
            0.9,
        ),
    ]

    c = TemporalContextCompressor()

    patterns = c.detect_recurring_patterns(eps)

    assert any("locomotor overload" in p.lower() for p in patterns)


def test_state_transitions_detected():
    eps = [
        PlayerEpisode(
            1, 10, 20,
            ["locomotor_overload"],
            "worsening",
            "high",
            [],
            "escalated",
            3,
            0.7,
            0.8,
        ),
        PlayerEpisode(
            2, 22, 30,
            ["cardiovascular_overload"],
            "worsening",
            "high",
            [],
            "persisted",
            2,
            0.6,
            0.7,
        ),
    ]

    c = TemporalContextCompressor()

    transitions = c.detect_state_transitions(eps)

    assert any(
        "locomotor overload" in t.lower()
        and "cardiovascular overload" in t.lower()
        for t in transitions
    )


def test_compressed_context_no_raw_telemetry():
    """
    Critical architecture rule:
    compressed episodic context must not contain raw telemetry.
    """

    findings = _make_findings([
        ("locomotor_overload", "high", 22),
        ("recovery_degradation", "high", 35),
    ])

    c = TemporalContextCompressor()

    episodes = c.build_episodes_from_findings(
        findings,
        [0.7, 0.75],
    )

    ctx = c.compress(
        episodes=episodes,
        trend_summaries={
            "anomaly": "worsening",
            "recovery": "worsening",
        },
        recent_findings=findings,
        intervention_history=[],
        current_minute=35,
        current_escalation="high",
    )

    block = ctx.to_prompt_block()

    assert block

    forbidden_patterns = [
        r"\bbpm\b",
        r"\bm/s\b",
        r"\banomaly_score\b",
        r"\bheart rate\b",
        r"\bsprint_count\b",
    ]

    for pattern in forbidden_patterns:
        assert not re.search(
            pattern,
            block,
            flags=re.IGNORECASE,
        )


def test_token_budget_respected():
    findings = _make_findings([
        ("locomotor_overload", "high", i)
        for i in range(50)
    ])

    c = TemporalContextCompressor()

    episodes = c.build_episodes_from_findings(
        findings,
        [0.7] * 50,
    )

    ctx = c.compress(
        episodes=episodes,
        trend_summaries={},
        recent_findings=findings,
        intervention_history=[],
        current_minute=50,
        current_escalation="high",
    )

    block = ctx.to_prompt_block()

    # small tolerance for truncation suffix
    assert len(block) <= MAX_CONTEXT_TOKENS + 30


def test_empty_findings_returns_empty_context():
    c = TemporalContextCompressor()

    ctx = c.compress(
        episodes=[],
        trend_summaries={},
        recent_findings=[],
        intervention_history=[],
        current_minute=None,
        current_escalation="normal",
    )

    assert ctx.is_empty()

    block = ctx.to_prompt_block()

    assert block == "" or "no significant episodes" in block.lower()


def test_prompt_contains_prior_episode_context():
    """
    Ensure temporal reasoning artifacts actually reach the prompt.
    """

    eps = [
        PlayerEpisode(
            1, 10, 20,
            ["locomotor_overload"],
            "worsening",
            "high",
            ["reduced intensity"],
            "persisted",
            3,
            0.75,
            0.8,
        ),
        PlayerEpisode(
            2, 30, 40,
            ["locomotor_overload"],
            "worsening",
            "critical",
            ["substitution requested"],
            "escalated",
            5,
            0.9,
            0.95,
        ),
    ]

    c = TemporalContextCompressor()

    ctx = c.compress(
        episodes=eps,
        trend_summaries={
            "locomotor_overload": "worsening",
        },
        recent_findings=_make_findings([
            ("locomotor_overload", "critical", 40),
        ]),
        intervention_history=[
            "reduced intensity",
            "substitution requested",
        ],
        current_minute=40,
        current_escalation="critical",
    )

    block = ctx.to_prompt_block().lower()

    expected_keywords = [
        "recurring",
        "previous",
        "trajectory",
        "worsening",
    ]

    assert any(k in block for k in expected_keywords)


def test_episode_serialization_roundtrip():
    ep = PlayerEpisode(
        episode_index=1,
        start_minute=10,
        end_minute=25,
        dominant_findings=["locomotor_overload"],
        trend_direction="worsening",
        severity="high",
        interventions=["substitution requested"],
        response="escalated",
        persistence_duration=4,
        peak_anomaly_score=0.82,
        peak_confidence=0.91,
    )

    restored = PlayerEpisode.from_dict(ep.to_dict())

    assert restored == ep