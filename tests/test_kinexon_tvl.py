"""
test_kinexon_tvl.py

Proves that Kinexon events (which carry heart_rate_bpm=None) pass TVL
validation and are not blocked before inference.

Run:
    pytest tests/test_kinexon_tvl.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from analysis.telemetry_validity import TelemetryValidityLayer, TelemetryStatus, ValidityMetrics
from ingestion.kinexon_adapter import KinexonAdapter, KinexonObservation
from config.settings import KinexonConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tvl() -> TelemetryValidityLayer:
    return TelemetryValidityLayer()


def _kinexon_event(
    speed_ms: float = 3.5,
    distance_delta_m: float = 0.175,
    is_sprint: int = 0,
    heart_rate_bpm=None,
    x_pitch: float = 52.3,
    y_pitch: float = 45.1,
) -> dict:
    """Minimal event dict as produced by KinexonAdapter.to_event_dict()."""
    return {
        "speed_ms":           speed_ms,
        "distance_delta_m":   distance_delta_m,
        "is_sprint":          is_sprint,
        "heart_rate_bpm":     heart_rate_bpm,
        "x_pitch":            x_pitch,
        "y_pitch":            y_pitch,
    }


def _make_obs(
    speed_ms: float = 3.5,
    sprint_flag: bool = False,
    heart_rate_bpm=None,
    distance_delta_m: float = 0.175,
) -> KinexonObservation:
    return KinexonObservation(
        player_id=1796,
        player_name="Test Player",
        jersey_number=7,
        group_name="Team A",
        timestamp_ms=1_700_000_000_000,
        ts=datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc),
        x_m=2.0,
        y_m=1.0,
        x_pitch=55.0,
        y_pitch=55.0,
        speed_ms=speed_ms,
        acceleration_ms2=0.5,
        distance_delta_m=distance_delta_m,
        heart_rate_bpm=heart_rate_bpm,
        sprint_flag=sprint_flag,
        session_id="3387",
        match_id="match_001",
        valid=True,
        issues=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TVL unit tests — HR=None semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestHRSensorAbsent:
    """None heart_rate_bpm = sensor not worn; must not produce INVALID."""

    def test_not_invalid_when_hr_is_none(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=None))
        assert result.status != TelemetryStatus.INVALID, (
            f"Expected non-INVALID when HR=None, got {result.status} | issues={result.issues}"
        )

    def test_confidence_positive_when_hr_is_none(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=None))
        assert result.confidence > 0.0, (
            f"Expected confidence > 0 when HR=None, got {result.confidence}"
        )

    def test_hr_sensor_absent_issue_tagged(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=None))
        assert "hr_sensor_absent" in result.issues, (
            f"Expected 'hr_sensor_absent' in issues, got {result.issues}"
        )

    def test_confidence_lower_than_full_hr(self):
        """Absent HR must reduce confidence relative to a valid HR reading."""
        result_hr = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=145))
        result_no_hr = _tvl().validate_event(1797, _kinexon_event(heart_rate_bpm=None))
        assert result_no_hr.confidence < result_hr.confidence, (
            f"Expected confidence to drop when HR absent: "
            f"with_hr={result_hr.confidence} no_hr={result_no_hr.confidence}"
        )

    def test_valid_status_when_only_hr_absent(self):
        """A clean Kinexon event (no other issues) with HR=None should be VALID."""
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=None))
        assert result.status == TelemetryStatus.VALID, (
            f"Expected VALID for clean Kinexon event, got {result.status} "
            f"confidence={result.confidence} issues={result.issues}"
        )


class TestHRZeroIsInvalid:
    """Zero heart_rate_bpm = sensor malfunction/bad read; must be INVALID."""

    def test_hr_zero_produces_invalid(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=0))
        assert result.status == TelemetryStatus.INVALID, (
            f"Expected INVALID for HR=0, got {result.status} | issues={result.issues}"
        )

    def test_hr_zero_tagged_as_implausible(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=0))
        assert any("implausible_hr" in i for i in result.issues), (
            f"Expected 'implausible_hr' in issues for HR=0, got {result.issues}"
        )

    def test_hr_none_and_hr_zero_differ(self):
        """None and 0 must yield different TVL outcomes — the core semantics."""
        tvl = _tvl()
        none_result = tvl.validate_event(1796, _kinexon_event(heart_rate_bpm=None))
        tvl2 = _tvl()
        zero_result = tvl2.validate_event(1797, _kinexon_event(heart_rate_bpm=0))
        assert none_result.status != zero_result.status, (
            "heart_rate_bpm=None and heart_rate_bpm=0 must produce different TVL statuses"
        )


class TestHRValidReading:
    """Plausible HR values must pass without penalty."""

    def test_valid_hr_gives_full_confidence(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=155))
        assert result.status == TelemetryStatus.VALID
        assert result.confidence == 1.0

    def test_implausible_high_hr_invalid(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=250))
        assert result.status == TelemetryStatus.INVALID

    def test_implausible_low_hr_invalid(self):
        result = _tvl().validate_event(1796, _kinexon_event(heart_rate_bpm=20))
        assert result.status == TelemetryStatus.INVALID


# ─────────────────────────────────────────────────────────────────────────────
# Completeness — is_sprint field
# ─────────────────────────────────────────────────────────────────────────────

class TestIsSprintCompleteness:
    """TVL requires is_sprint; KinexonAdapter must emit it."""

    def test_event_without_is_sprint_fails_completeness(self):
        """Old adapter emitted sprint_flag but not is_sprint; HR also None in Kinexon.
        Result: speed_ms + distance_delta_m = 2/4 = 0.50 < 0.75 → INVALID."""
        event = {
            "speed_ms": 3.5,
            "distance_delta_m": 0.175,
            "sprint_flag": 0,         # old key name — TVL doesn't recognise this
            "heart_rate_bpm": None,   # absent sensor; doesn't count toward completeness
        }
        result = _tvl().validate_event(1796, event)
        # speed_ms + distance_delta_m only = 2/4 = 0.50 < 0.75 → INVALID
        assert result.status == TelemetryStatus.INVALID
        assert any("low_completeness" in i for i in result.issues)

    def test_event_with_is_sprint_passes_completeness(self):
        event = {
            "speed_ms": 3.5,
            "distance_delta_m": 0.175,
            "is_sprint": 0,
            "heart_rate_bpm": 155,
        }
        result = _tvl().validate_event(1796, event)
        assert result.status == TelemetryStatus.VALID

    def test_is_sprint_zero_counts_as_present(self):
        """is_sprint=0 (not sprinting) must count toward completeness."""
        event = {
            "speed_ms": 3.5,
            "distance_delta_m": 0.175,
            "is_sprint": 0,       # 0 is falsy but is not None — must count
            "heart_rate_bpm": None,
        }
        result = _tvl().validate_event(1796, event)
        # 3/4 = 0.75 → passes completeness
        assert "low_completeness" not in " ".join(result.issues), (
            f"is_sprint=0 should not cause low_completeness. issues={result.issues}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Adapter integration — to_event_dict() emits is_sprint
# ─────────────────────────────────────────────────────────────────────────────

class TestAdapterEmitsIsSprintField:

    def test_to_event_dict_contains_is_sprint(self):
        adapter = KinexonAdapter()
        obs = _make_obs()
        evt = adapter.to_event_dict(obs, elapsed_s=30.0)
        assert "is_sprint" in evt, (
            f"to_event_dict() must emit 'is_sprint'. Keys present: {list(evt.keys())}"
        )

    def test_is_sprint_value_matches_sprint_flag(self):
        adapter = KinexonAdapter()
        for sprint_flag, expected in [(False, 0), (True, 1)]:
            obs = _make_obs(sprint_flag=sprint_flag)
            evt = adapter.to_event_dict(obs, elapsed_s=0.0)
            assert evt["is_sprint"] == expected, (
                f"sprint_flag={sprint_flag} should give is_sprint={expected}, "
                f"got {evt['is_sprint']}"
            )

    def test_sprint_flag_also_retained(self):
        """sprint_flag must still be present for legacy consumers."""
        adapter = KinexonAdapter()
        evt = adapter.to_event_dict(_make_obs(), elapsed_s=0.0)
        assert "sprint_flag" in evt


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: adapter event → TVL passes (the core requirement)
# ─────────────────────────────────────────────────────────────────────────────

class TestKinexonEventReachesInference:
    """
    Proves the primary requirement:
        A Kinexon event with speed_ms + distance_delta_m + is_sprint +
        heart_rate_bpm=None passes TVL and is not blocked before inference.
    """

    def test_kinexon_event_not_blocked_by_tvl(self):
        adapter = KinexonAdapter()
        obs = _make_obs(speed_ms=4.2, sprint_flag=False, heart_rate_bpm=None)
        evt = adapter.to_event_dict(obs, elapsed_s=15.0)

        tvl = _tvl()
        result = tvl.validate_event(obs.player_id, evt)

        assert result.status != TelemetryStatus.INVALID, (
            f"Kinexon event must pass TVL (not INVALID). "
            f"Got status={result.status} confidence={result.confidence} issues={result.issues}"
        )

    def test_sprinting_kinexon_event_not_blocked(self):
        adapter = KinexonAdapter()
        obs = _make_obs(speed_ms=6.1, sprint_flag=True, heart_rate_bpm=None)
        evt = adapter.to_event_dict(obs, elapsed_s=30.0)

        result = _tvl().validate_event(obs.player_id, evt)
        assert result.status != TelemetryStatus.INVALID

    def test_full_kinexon_event_has_expected_issues(self):
        """VALID status but hr_sensor_absent must appear in issues."""
        adapter = KinexonAdapter()
        obs = _make_obs(heart_rate_bpm=None)
        evt = adapter.to_event_dict(obs)

        result = _tvl().validate_event(obs.player_id, evt)
        assert result.status == TelemetryStatus.VALID
        assert "hr_sensor_absent" in result.issues
        assert result.confidence > 0.0

    def test_confidence_after_tvl_satisfies_telemetry_valid_gate(self):
        """
        orchestrator.py gates EMA updates on _tvl_confidence >= 0.8.
        A clean Kinexon event (no other issues) must clear that threshold.
        """
        adapter = KinexonAdapter()
        obs = _make_obs(heart_rate_bpm=None)
        evt = adapter.to_event_dict(obs)

        result = _tvl().validate_event(obs.player_id, evt)
        assert result.confidence >= 0.8, (
            f"Confidence must reach >= 0.8 for EMA updates to proceed. "
            f"Got {result.confidence}. issues={result.issues}"
        )
