"""
tests/test_semantic_hr_absent.py

Validates that the semantic quality gate distinguishes:
  A. HR absent (sensor not worn, heart_rate_bpm=None -> 0.0)  -> NOT degraded
  B. HR=0 malfunction (sensor worn, reports 0)                -> degraded
  C. HR valid (155 bpm during movement)                       -> NOT degraded

All three cases must pass for the C1 semantic-unblocking fix to be correct.

Design:
  - Tests call SemanticInterpreter._assess_window_quality() directly.
  - CONFIG.kinexon.hr_sensor_present is patched per-test.
    False = sensor not equipped (default Kinexon config).
    True  = sensor equipped but reporting 0 = malfunction.
  - No PyTorch required.
"""

import pytest
from unittest.mock import patch

from config.settings import CONFIG
from explainability.semantics_layer import SemanticInterpreter, SemanticFinding


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def interp():
    return SemanticInterpreter()


def _moving_fv(heart_rate_bpm: float = 0.0, speed: float = 3.2) -> dict:
    """Feature vector for a player moving at given speed with given HR."""
    return {
        "heart_rate_bpm":       heart_rate_bpm,
        "window_avg_speed_ms":  speed,
        "window_distance_m":    speed * 30,
        "window_sprint_count":  0.0,
        "z_distance":           0.0,
        "z_sprint_count":       0.0,
        "z_top_speed":          0.0,
        "z_high_speed_dist":    0.0,
        "acwr":                 1.0,
        "fatigue_decay_residual": 0.0,
        "speed_drop_pct":       0.0,
        "positional_drift_score": 0.5,
        "hr_recovery_time_s":   0.0,
    }


# ---------------------------------------------------------------------------
# Class A — HR sensor absent: quality gate must NOT fire
# ---------------------------------------------------------------------------

class TestHRAbsent:
    """
    CONFIG.kinexon.hr_sensor_present = False (Kinexon default).
    heart_rate_bpm = 0.0 means "no wearable", not "malfunction".
    The quality gate should not mark the window degraded.
    """

    def test_not_degraded_when_moving(self, interp):
        fv = _moving_fv(heart_rate_bpm=0.0, speed=3.2)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"], (
            f"Window marked degraded despite hr_sensor_present=False. "
            f"Reasons: {quality['reasons']}"
        )

    def test_no_hr_reason_in_reasons_list(self, interp):
        fv = _moving_fv(heart_rate_bpm=0.0, speed=5.0)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            quality = interp._assess_window_quality(fv)
        hr_reasons = [r for r in quality["reasons"] if "HR" in r or "hr" in r.lower()]
        assert hr_reasons == [], f"Unexpected HR reason: {hr_reasons}"

    def test_not_degraded_at_high_speed(self, interp):
        fv = _moving_fv(heart_rate_bpm=0.0, speed=5.5)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"]

    def test_not_degraded_near_threshold_speed(self, interp):
        """speed just above 0.5 m/s should still be fine with absent sensor."""
        fv = _moving_fv(heart_rate_bpm=0.0, speed=0.6)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"]

    def test_interpret_returns_findings_when_locomotor_conditions_met(self, interp):
        """
        End-to-end: with sensor absent, locomotor_suppression rule should be
        reachable (quality gate does not block). A shap-driven suppression
        finding fires when speed is low and SHAP points at speed/distance.
        """
        fv = _moving_fv(heart_rate_bpm=0.0, speed=1.2)
        fv["window_distance_m"] = 36.0
        shap = {
            "window_avg_speed_ms":  0.22,
            "window_distance_m":    0.10,
        }
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            findings = interp.interpret(
                shap_values=shap,
                feature_values=fv,
                persistence_windows=0,
            )
        assert findings, (
            "Expected at least one finding (locomotor_suppression) but got none. "
            "HR-absent quality gate may still be blocking."
        )
        types = [f.finding_type for f in findings]
        assert "locomotor_suppression" in types, f"Expected locomotor_suppression, got: {types}"

    def test_findings_are_semantic_finding_objects(self, interp):
        fv = _moving_fv(heart_rate_bpm=0.0, speed=1.2)
        fv["window_distance_m"] = 36.0
        shap = {"window_avg_speed_ms": 0.22, "window_distance_m": 0.10}
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            findings = interp.interpret(
                shap_values=shap, feature_values=fv, persistence_windows=0
            )
        for f in findings:
            assert isinstance(f, SemanticFinding), f"Got {type(f)} instead of SemanticFinding"

    def test_stationary_player_also_not_degraded(self, interp):
        """speed=0.0 means check 2 condition (speed>0.5) is False regardless."""
        fv = _moving_fv(heart_rate_bpm=0.0, speed=0.0)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"]


# ---------------------------------------------------------------------------
# Class B — HR=0 malfunction: quality gate MUST fire
# ---------------------------------------------------------------------------

class TestHRZeroMalfunction:
    """
    CONFIG.kinexon.hr_sensor_present = True (sensor equipped).
    heart_rate_bpm = 0.0 during movement = biologically impossible = degraded.
    This preserves the existing malfunction detection for HR-equipped sources.
    """

    def test_degraded_when_hr_zero_during_movement(self, interp):
        fv = _moving_fv(heart_rate_bpm=0.0, speed=3.2)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            quality = interp._assess_window_quality(fv)
        assert quality["degraded"], "HR=0 during movement should degrade quality when sensor is equipped"

    def test_hr_dropout_reason_present(self, interp):
        fv = _moving_fv(heart_rate_bpm=0.0, speed=3.2)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            quality = interp._assess_window_quality(fv)
        reasons_text = " ".join(quality["reasons"])
        assert "dropout" in reasons_text.lower() or "HR=0" in reasons_text, (
            f"Expected HR dropout reason, got: {quality['reasons']}"
        )

    def test_interpret_returns_empty_when_degraded(self, interp):
        """All findings suppressed when quality is degraded."""
        fv = _moving_fv(heart_rate_bpm=0.0, speed=1.2)
        fv["window_distance_m"] = 36.0
        shap = {"window_avg_speed_ms": 0.22, "window_distance_m": 0.10}
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            findings = interp.interpret(
                shap_values=shap, feature_values=fv, persistence_windows=0
            )
        assert findings == [], f"Expected [] when HR=0 malfunction, got: {findings}"

    def test_hr_zero_at_threshold_speed(self, interp):
        """speed just above 0.5 triggers the dropout check when sensor is present."""
        fv = _moving_fv(heart_rate_bpm=0.0, speed=0.6)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            quality = interp._assess_window_quality(fv)
        assert quality["degraded"]

    def test_hr_zero_at_rest_not_degraded(self, interp):
        """HR=0 when stationary (speed=0.0) does not fire check 2 regardless."""
        fv = _moving_fv(heart_rate_bpm=0.0, speed=0.0)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"]


# ---------------------------------------------------------------------------
# Class C — HR present and valid: existing behaviour unchanged
# ---------------------------------------------------------------------------

class TestHRPresentBehavior:
    """
    HR sensor equipped and reporting valid bpm.
    Quality gate must pass; rules must fire normally.
    hr_sensor_present=True is the future state once wearables are integrated.
    """

    def test_valid_hr_not_degraded(self, interp):
        fv = _moving_fv(heart_rate_bpm=155.0, speed=4.0)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"], f"Valid HR=155 should not degrade: {quality['reasons']}"

    def test_high_hr_with_speed_not_degraded(self, interp):
        fv = _moving_fv(heart_rate_bpm=180.0, speed=5.5)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"]

    def test_implausible_hr_still_degraded(self, interp):
        """Check 4 still fires: HR=300 bpm (above 220) is physiologically impossible."""
        fv = _moving_fv(heart_rate_bpm=300.0, speed=3.0)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            quality = interp._assess_window_quality(fv)
        assert quality["degraded"], "HR=300 during movement should degrade quality"

    def test_locomotor_suppression_fires_with_valid_hr(self, interp):
        """Locomotor suppression fires when speed is low, HR is fine, SHAP drives it."""
        fv = _moving_fv(heart_rate_bpm=80.0, speed=1.2)
        fv["window_distance_m"] = 36.0
        shap = {"window_avg_speed_ms": 0.22, "window_distance_m": 0.10}
        with patch.object(CONFIG.kinexon, "hr_sensor_present", True):
            findings = interp.interpret(
                shap_values=shap, feature_values=fv, persistence_windows=0
            )
        assert any(f.finding_type == "locomotor_suppression" for f in findings), (
            f"Expected locomotor_suppression with valid HR, got: {[f.finding_type for f in findings]}"
        )

    def test_hr_absent_flag_false_does_not_affect_valid_hr(self, interp):
        """
        When hr_sensor_present=False but HR is actually valid (edge case after
        config change), valid HR still passes all checks normally.
        """
        fv = _moving_fv(heart_rate_bpm=155.0, speed=4.0)
        with patch.object(CONFIG.kinexon, "hr_sensor_present", False):
            quality = interp._assess_window_quality(fv)
        assert not quality["degraded"]
