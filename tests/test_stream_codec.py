"""
tests/test_stream_codec.py

Validates ingestion/stream_codec.py — dataclass <-> Redis Stream field-dict
serialization for TacticalEvent, Possession, TeamState, TeamStateTrend, and
CoachInsight.

Covers:
  A. Round-trip identity for each of the 5 registered dataclasses
  B. Wire format contract (flat str->str fields, datetime handling)
  C. Versioning / error handling (unknown type, unrecognised schema_version)

Run:
    pytest tests/test_stream_codec.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from analysis.coach_insight import CoachInsight
from analysis.coach_situation import CoachSituation
from analysis.possession import Possession
from analysis.team_state import TeamState
from analysis.team_state_trend import TeamStateTrend
from ingestion.match_event import MatchContext, MatchEvent
from ingestion.stream_codec import SCHEMA_VERSION, decode, encode
from ingestion.tactical_event import TacticalEvent

BASE_TS = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)


def _tactical_event() -> TacticalEvent:
    return TacticalEvent(
        event_id="abc123", timestamp=BASE_TS, match_id="3387",
        team_id="SC Magdeburg", player_id=1164, event_type="pass",
        metadata={"distance_m": 1.23, "ball_speed_kmh": 9.77, "pass_type": "successful"},
        source="kinexon", confidence=1.0,
    )


def _possession() -> Possession:
    return Possession(
        possession_id="poss-1", team_id="SC Magdeburg", match_id="3387",
        start_timestamp=BASE_TS, end_timestamp=BASE_TS + timedelta(seconds=10),
        duration_seconds=10.0,
        pass_count=2, shot_count=1, turnover_count=0,
        sprint_count=0, acceleration_count=1, physical_action_count=1,
        outcome="shot",
        attack_intensity=18.0, physical_intensity=6.0, transition_intensity=6.0,
        possession_quality=0.83,
    )


def _team_state() -> TeamState:
    return TeamState(
        timestamp=BASE_TS, team_id="SC Magdeburg", window_seconds=60,
        possession_count=3, turnover_count=1, possession_pressure=0.25,
        pass_count=10, shot_count=2, attack_activity=12.0,
        sprint_count=1, acceleration_count=2, exertion_count=5, physical_load=30.0,
        active_player_count=6, fatigue_burden=5.0, confidence=0.9,
    )


def _team_state_trend() -> TeamStateTrend:
    return TeamStateTrend(
        timestamp=BASE_TS, team_id="SC Magdeburg", window_seconds=60,
        possession_pressure_delta=-0.1, attack_activity_delta=5.0,
        physical_load_delta=10.0, fatigue_burden_delta=1.0, confidence_delta=0.05,
        attack_trend="increasing", load_trend="increasing", fatigue_trend="stable",
    )


def _coach_insight() -> CoachInsight:
    return CoachInsight(
        timestamp=BASE_TS, team_id="SC Magdeburg", severity="high",
        category="attack_activity_rising", message="Attack activity rising for SC Magdeburg.",
        confidence=0.85,
        metadata={
            "source_metrics": ["attack_activity_delta"],
            "values": {"attack_activity_delta": 12.0},
            "thresholds_crossed": {"attack_activity_delta": 9.0},
            "window_seconds": 60,
        },
    )


def _coach_situation() -> CoachSituation:
    return CoachSituation(
        timestamp=BASE_TS, team_id="SC Magdeburg", situation_type="HIGH_TEMPO_ATTACK",
        severity="medium", confidence=0.7,
        source_insights=["attack_activity_rising"],
        source_metrics={"window_seconds": 60, "attack_trend": "increasing", "attack_activity": 12.0},
        explanation="SC Magdeburg: attack tempo rising from an already-high base.",
    )


def _match_event() -> MatchEvent:
    return MatchEvent(
        event_id="be-1", timestamp=BASE_TS, match_id="3387",
        team_id="SC Magdeburg", player_id=1164, event_type="substitution",
        metadata={"player_in": 1170, "player_out": 1164}, source="backend",
    )


def _match_context() -> MatchContext:
    return MatchContext(
        timestamp=BASE_TS, match_id="3387", home_score=14, away_score=12,
        period="second_half", game_clock_seconds=2145.0,
        metadata={"timeouts_remaining_home": 1}, source="backend",
    )


ALL_FACTORIES = {
    "TacticalEvent": _tactical_event,
    "Possession": _possession,
    "TeamState": _team_state,
    "TeamStateTrend": _team_state_trend,
    "CoachInsight": _coach_insight,
    "CoachSituation": _coach_situation,
    "MatchEvent": _match_event,
    "MatchContext": _match_context,
}


# ---------------------------------------------------------------------------
# A. Round-trip identity
# ---------------------------------------------------------------------------

class TestRoundTrip:

    @pytest.mark.parametrize("name", list(ALL_FACTORIES.keys()))
    def test_round_trip_equals_original(self, name):
        original = ALL_FACTORIES[name]()
        fields = encode(original)
        restored = decode(fields)
        assert restored == original

    @pytest.mark.parametrize("name", list(ALL_FACTORIES.keys()))
    def test_round_trip_preserves_type(self, name):
        original = ALL_FACTORIES[name]()
        restored = decode(encode(original))
        assert type(restored) is type(original)

    def test_round_trip_preserves_datetime_precision(self):
        ts = datetime(2026, 6, 7, 13, 1, 21, 123456, tzinfo=timezone.utc)
        obj = TacticalEvent(
            event_id="x", timestamp=ts, match_id="3387", team_id="SC Magdeburg",
            player_id=1164, event_type="pass", metadata={}, source="kinexon", confidence=1.0,
        )
        restored = decode(encode(obj))
        assert restored.timestamp == ts

    def test_round_trip_preserves_nested_metadata(self):
        obj = _tactical_event()
        restored = decode(encode(obj))
        assert restored.metadata == obj.metadata

    def test_round_trip_with_none_team_id(self):
        obj = _tactical_event()
        obj.team_id = None
        restored = decode(encode(obj))
        assert restored.team_id is None


# ---------------------------------------------------------------------------
# B. Wire format contract
# ---------------------------------------------------------------------------

class TestWireFormat:

    def test_encoded_fields_are_flat_str_to_str(self):
        fields = encode(_possession())
        assert set(fields.keys()) == {"schema_version", "type", "match_id", "payload"}
        for k, v in fields.items():
            assert isinstance(k, str) and isinstance(v, str)

    def test_match_id_extracted_to_top_level_field(self):
        fields = encode(_possession())
        assert fields["match_id"] == "3387"

    def test_match_id_empty_string_when_dataclass_has_none(self):
        """TeamState/TeamStateTrend/CoachInsight have no match_id field at all."""
        fields = encode(_team_state())
        assert fields["match_id"] == ""

    def test_type_field_matches_class_name(self):
        fields = encode(_coach_insight())
        assert fields["type"] == "CoachInsight"

    def test_schema_version_field_matches_module_constant(self):
        fields = encode(_team_state_trend())
        assert fields["schema_version"] == str(SCHEMA_VERSION)

    def test_payload_is_valid_json(self):
        import json
        fields = encode(_team_state())
        parsed = json.loads(fields["payload"])
        assert parsed["team_id"] == "SC Magdeburg"
        assert parsed["attack_activity"] == 12.0


# ---------------------------------------------------------------------------
# C. Versioning / error handling
# ---------------------------------------------------------------------------

class TestVersioningAndErrors:

    def test_encode_unregistered_type_raises(self):
        class NotRegistered:
            pass
        with pytest.raises(ValueError, match="No stream codec registered"):
            encode(NotRegistered())

    def test_decode_unregistered_type_raises(self):
        fields = {"schema_version": "1", "type": "NotRegistered", "match_id": "", "payload": "{}"}
        with pytest.raises(ValueError, match="No stream codec registered"):
            decode(fields)

    def test_decode_unrecognised_schema_version_raises(self):
        fields = encode(_team_state())
        fields["schema_version"] = "999"
        with pytest.raises(ValueError, match="Unrecognised stream schema_version"):
            decode(fields)

    def test_decode_does_not_guess_on_version_mismatch(self):
        """Fail-closed: a version mismatch must raise, never silently parse."""
        fields = encode(_coach_insight())
        fields["schema_version"] = "2"
        with pytest.raises(ValueError):
            decode(fields)
