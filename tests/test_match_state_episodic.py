# tests/test_match_state_episodic.py
from analysis.match_state import MatchState

def _make_state():
    return MatchState(player_id=1, player_name="Test", position="CM", match_id="m1")

def test_log_intervention_stored():
    s = _make_state()
    s.log_intervention("substitution requested min~72")
    assert "substitution requested min~72" in s.intervention_history

def test_refresh_episodes_populates():
    s = _make_state()
    # Feed some findings manually
    for i in range(5):
        s.record_finding(
            {"finding_type": "locomotor_overload", "severity": "high",
             "confidence": 0.8, "trend": "stable", "domain": "locomotor"},
            elapsed_seconds=22*60 + i*30,
        )
    s.refresh_episodes(current_minute=22)
    assert len(s.episodes) > 0
    assert s.trend_summaries is not None  # populated, even if all "stable"

def test_build_compressed_context_no_raw_telemetry():
    s = _make_state()
    for i in range(6):
        s.record_finding(
            {"finding_type": "recovery_degradation", "severity": "high",
             "confidence": 0.78, "trend": "worsening", "domain": "cardiovascular"},
            elapsed_seconds=35*60 + i*30,
        )
    s.refresh_episodes(current_minute=35, current_escalation="high")
    ctx = s.build_compressed_context(
        current_finding_types=["recovery_degradation"],
        current_minute=35,
    )
    block = ctx.to_prompt_block()
    # Should produce content
    assert block
    # Should not contain raw numbers
    import re
    assert not re.search(r'\b\d{3} bpm\b', block)

def test_to_dict_from_dict_roundtrip_with_episodes():
    s = _make_state()
    s.log_intervention("load reduction min~55")
    for i in range(3):
        s.record_finding(
            {"finding_type": "locomotor_overload", "severity": "high",
             "confidence": 0.8, "trend": "stable", "domain": "locomotor"},
            elapsed_seconds=22*60,
        )
    s.refresh_episodes(current_minute=22)
    d = s.to_dict()

    # Schema version should be 2
    assert d["_schema_version"] == 2
    # Episodes should be serialized
    assert isinstance(d["episodes"], list)
    assert d["intervention_history"] == ["load reduction min~55"]

    # Roundtrip
    s2 = MatchState.from_dict(d)
    assert len(s2.episodes) == len(s.episodes)
    assert s2.intervention_history == s.intervention_history

def test_schema_version_2_rejects_old_checkpoints():
    s = _make_state()
    d = s.to_dict()
    d["_schema_version"] = 1  # simulate old checkpoint
    import pytest
    with pytest.raises(ValueError, match="schema"):
        MatchState.from_dict(d)