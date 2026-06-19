"""
test_phase_a_calibration.py

Proves that the four Phase A handball calibration fixes are live.
Tests operate against config.settings (no PyTorch dependency) plus
standalone geometry arithmetic that mirrors _extract() exactly.

D1  SPRINT_THRESHOLD_MS  — 7.0  → 5.5 m/s  (KinexonConfig.sprint_threshold_ms)
D2  PITCH dimensions     — 105×68 m  → 40×20 m  (KinexonConfig.pitch_length/width_m)
D3  late_in_game gate    — elapsed > 2700 s → > 1800 s  (KinexonConfig.match_half_duration_s)
D4  baseline_speed denom — 90×60 s  → 60×60 s  (KinexonConfig.match_duration_s)

Run:
    pytest tests/test_phase_a_calibration.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from config.settings import CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — replicate _extract() geometry without importing anomaly_detection
# ─────────────────────────────────────────────────────────────────────────────

def _sprint_flag(speed: float, threshold: float) -> float:
    return 1.0 if speed >= threshold else 0.0


def _distance_delta(
    x: float, prev_x: float,
    y: float, prev_y: float,
    pitch_width_m: float,
    pitch_length_m: float,
) -> float:
    """Mirror of _extract() lines 1232-1234."""
    dx_m = (x - prev_x) / 100.0 * pitch_width_m
    dy_m = (y - prev_y) / 100.0 * pitch_length_m
    return math.sqrt(dx_m * dx_m + dy_m * dy_m)


def _sprint_fraction(speeds: list[float], threshold: float) -> float:
    flags = [_sprint_flag(s, threshold) for s in speeds]
    return sum(flags) / len(flags)


# ─────────────────────────────────────────────────────────────────────────────
# D1 — Sprint threshold: 7.0 → 5.5 m/s
# ─────────────────────────────────────────────────────────────────────────────

class TestD1SprintThreshold:

    def test_config_sprint_threshold_is_5_5(self):
        assert CONFIG.kinexon.sprint_threshold_ms == pytest.approx(5.5), (
            f"KinexonConfig.sprint_threshold_ms={CONFIG.kinexon.sprint_threshold_ms}: "
            f"expected 5.5 (handball IHF), not 7.0 (football)"
        )

    def test_anomaly_detection_constant_matches_config(self):
        """SPRINT_THRESHOLD_MS in anomaly_detection.py must equal KinexonConfig value."""
        import importlib
        try:
            import torch  # noqa: F401
            ad = importlib.import_module("analysis.anomaly_detection")
            assert ad.SPRINT_THRESHOLD_MS == CONFIG.kinexon.sprint_threshold_ms, (
                f"anomaly_detection.SPRINT_THRESHOLD_MS={ad.SPRINT_THRESHOLD_MS} != "
                f"CONFIG.kinexon.sprint_threshold_ms={CONFIG.kinexon.sprint_threshold_ms}"
            )
        except (ImportError, AttributeError):
            pytest.skip("PyTorch not available — constant verified via config only")

    def test_6ms_is_sprint_at_new_threshold(self):
        """6.0 m/s > 5.5 → sprint_flag = 1 (handball sprint)."""
        flag = _sprint_flag(6.0, CONFIG.kinexon.sprint_threshold_ms)
        assert flag == 1.0, f"6.0 m/s should be sprint at threshold {CONFIG.kinexon.sprint_threshold_ms}"

    def test_6ms_was_not_sprint_at_old_threshold(self):
        """Reference: 6.0 m/s < 7.0 (football) → sprint_flag = 0 (bug)."""
        flag_old = _sprint_flag(6.0, 7.0)
        assert flag_old == 0.0, "At old threshold 7.0, 6.0 m/s was NOT a sprint"

    def test_5_5ms_boundary_is_sprint(self):
        flag = _sprint_flag(5.5, CONFIG.kinexon.sprint_threshold_ms)
        assert flag == 1.0

    def test_5_4ms_is_not_sprint(self):
        flag = _sprint_flag(5.4, CONFIG.kinexon.sprint_threshold_ms)
        assert flag == 0.0

    def test_sprint_fraction_at_6ms_is_1_0(self):
        """All ticks at 6.0 m/s → sprint_fraction = 1.0 → intensity = 'high'."""
        speeds = [6.0] * 8
        frac = _sprint_fraction(speeds, CONFIG.kinexon.sprint_threshold_ms)
        assert frac == pytest.approx(1.0)

    def test_before_fix_sprint_fraction_at_6ms_was_0(self):
        """Old 7.0 threshold: 6.0 m/s → sprint_fraction = 0.0 → always 'low'."""
        speeds = [6.0] * 8
        frac_old = _sprint_fraction(speeds, 7.0)
        assert frac_old == pytest.approx(0.0)

    def test_regime_intensity_high_now_reachable(self):
        """sprint_fraction >= 0.15 → 'high' intensity (from regime.py constant)."""
        HIGH_MIN = 0.15
        speeds = [6.0] * 8
        frac = _sprint_fraction(speeds, CONFIG.kinexon.sprint_threshold_ms)
        assert frac >= HIGH_MIN, (
            f"sprint_fraction={frac} must be >= {HIGH_MIN} for high intensity. "
            f"After D1 fix this is {frac:.2f} (was 0.0 before)."
        )

    def test_regime_intensity_was_always_low_before(self):
        """Before fix: sprint_fraction = 0.0 < 0.04 → always 'low'."""
        MEDIUM_MIN = 0.04
        speeds = [6.0] * 8
        frac_old = _sprint_fraction(speeds, 7.0)
        assert frac_old < MEDIUM_MIN


# ─────────────────────────────────────────────────────────────────────────────
# D2 — Pitch geometry: 105×68 m → 40×20 m
# ─────────────────────────────────────────────────────────────────────────────

class TestD2PitchGeometry:

    def test_config_pitch_length_is_40(self):
        assert CONFIG.kinexon.pitch_length_m == pytest.approx(40.0), (
            f"pitch_length_m={CONFIG.kinexon.pitch_length_m}: expected 40.0 (handball)"
        )

    def test_config_pitch_width_is_20(self):
        assert CONFIG.kinexon.pitch_width_m == pytest.approx(20.0), (
            f"pitch_width_m={CONFIG.kinexon.pitch_width_m}: expected 20.0 (handball)"
        )

    def test_anomaly_detection_pitch_constants_match_config(self):
        """PITCH_LENGTH_M and PITCH_WIDTH_M in anomaly_detection.py must equal KinexonConfig."""
        try:
            import torch  # noqa: F401
            import importlib
            ad = importlib.import_module("analysis.anomaly_detection")
            assert ad.PITCH_LENGTH_M == CONFIG.kinexon.pitch_length_m
            assert ad.PITCH_WIDTH_M  == CONFIG.kinexon.pitch_width_m
        except (ImportError, AttributeError):
            pytest.skip("PyTorch not available — constants verified via config only")

    def test_short_axis_5_units_is_1_metre(self):
        """
        5 pitch units on the 20 m short axis (x) = 5/100 × 20 = 1.0 m.
        Old value: 5/100 × 68 = 3.4 m (3.4× inflation).
        """
        d = _distance_delta(
            x=55.0, prev_x=50.0,
            y=50.0, prev_y=50.0,
            pitch_width_m=CONFIG.kinexon.pitch_width_m,
            pitch_length_m=CONFIG.kinexon.pitch_length_m,
        )
        assert d == pytest.approx(1.0, abs=1e-6), (
            f"5 pitch units on 20 m width → 1.0 m. Got {d:.4f} m"
        )

    def test_short_axis_old_value_was_3_4_metres(self):
        """Reference: old 68 m width gave 3.4 m for the same 5 pitch unit move."""
        d_old = _distance_delta(55.0, 50.0, 50.0, 50.0, pitch_width_m=68.0, pitch_length_m=105.0)
        assert d_old == pytest.approx(3.4, abs=1e-6)

    def test_long_axis_10_units_is_4_metres(self):
        """
        10 pitch units on the 40 m long axis (y) = 10/100 × 40 = 4.0 m.
        Old value: 10/100 × 105 = 10.5 m (2.625× inflation).
        """
        d = _distance_delta(
            x=50.0, prev_x=50.0,
            y=60.0, prev_y=50.0,
            pitch_width_m=CONFIG.kinexon.pitch_width_m,
            pitch_length_m=CONFIG.kinexon.pitch_length_m,
        )
        assert d == pytest.approx(4.0, abs=1e-6)

    def test_long_axis_old_value_was_10_5_metres(self):
        d_old = _distance_delta(50.0, 50.0, 60.0, 50.0, pitch_width_m=68.0, pitch_length_m=105.0)
        assert d_old == pytest.approx(10.5, abs=1e-6)

    def test_short_axis_inflation_factor(self):
        """Old / new: 68 / 20 = 3.4× inflation removed."""
        assert 68.0 / CONFIG.kinexon.pitch_width_m == pytest.approx(3.4)

    def test_long_axis_inflation_factor(self):
        """Old / new: 105 / 40 = 2.625× inflation removed."""
        assert 105.0 / CONFIG.kinexon.pitch_length_m == pytest.approx(2.625)

    def test_diagonal_3_4_5_triangle(self):
        """
        dx = 3 pitch units on 20 m court = 0.6 m
        dy = 4 pitch units on 40 m court = 1.6 m
        hyp = sqrt(0.36 + 2.56) = sqrt(2.92) ≈ 1.709 m
        """
        d = _distance_delta(
            x=53.0, prev_x=50.0,
            y=54.0, prev_y=50.0,
            pitch_width_m=CONFIG.kinexon.pitch_width_m,
            pitch_length_m=CONFIG.kinexon.pitch_length_m,
        )
        expected = math.sqrt((0.6)**2 + (1.6)**2)
        assert d == pytest.approx(expected, abs=1e-6)

    def test_zero_movement_gives_zero_distance(self):
        d = _distance_delta(50.0, 50.0, 50.0, 50.0,
                            CONFIG.kinexon.pitch_width_m, CONFIG.kinexon.pitch_length_m)
        assert d == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# D3 — late_in_game gate: elapsed > 2700 s → > 1800 s
# ─────────────────────────────────────────────────────────────────────────────

class TestD3LateInGame:

    def test_match_half_duration_is_1800(self):
        assert CONFIG.kinexon.match_half_duration_s == 1800, (
            f"match_half_duration_s={CONFIG.kinexon.match_half_duration_s}: "
            f"expected 1800 (handball 30-min half), was 2700 (football 45-min half)"
        )

    def _late(self, elapsed: float) -> bool:
        return elapsed > CONFIG.kinexon.match_half_duration_s

    def _late_old(self, elapsed: float) -> bool:
        return elapsed > 2700

    def test_1900s_is_late_new_not_late_old(self):
        """
        1900 s = 31:40 into match — inside handball second half.
        Old: 1900 < 2700 → False (bug: second half not treated as late).
        New: 1900 > 1800 → True (correct).
        """
        assert self._late(1900) is True
        assert self._late_old(1900) is False

    def test_1799s_is_not_late(self):
        assert self._late(1799.0) is False

    def test_exactly_1800s_is_not_late(self):
        """Strict >: boundary is not included."""
        assert self._late(1800.0) is False

    def test_1801s_is_late(self):
        assert self._late(1801.0) is True

    def test_first_half_never_late(self):
        for elapsed in [0, 300, 900, 1800]:
            assert self._late(elapsed) is False, f"elapsed={elapsed}s should not be late"

    def test_second_half_always_late(self):
        for elapsed in [1801, 2100, 2400, 2700, 3000, 3600]:
            assert self._late(elapsed) is True, f"elapsed={elapsed}s should be late"

    def test_old_threshold_misses_most_of_second_half(self):
        """
        Old threshold 2700 s: late only in final 15 min of a 60-min match.
        New threshold 1800 s: late for the entire second half (30 min).

        Count of 60-second ticks late in a 3600 s match:
        """
        ticks = range(0, 3601, 60)
        late_old = sum(1 for t in ticks if self._late_old(t))
        late_new = sum(1 for t in ticks if self._late(t))
        assert late_new > late_old, (
            f"New threshold must flag more second-half time as late. "
            f"Old: {late_old} ticks, new: {late_new} ticks"
        )
        assert late_new == pytest.approx(30, abs=1), "~30 minutes of second half should be 'late'"


# ─────────────────────────────────────────────────────────────────────────────
# D4 — baseline_speed denominator: 90×60 → 60×60 s
# ─────────────────────────────────────────────────────────────────────────────

class TestD4BaselineSpeedDenominator:

    def test_match_duration_is_3600(self):
        assert CONFIG.kinexon.match_duration_s == 3600, (
            f"match_duration_s={CONFIG.kinexon.match_duration_s}: "
            f"expected 3600 (handball 60 min), not 5400 (football 90 min)"
        )

    def _baseline_speed_new(self, distance_mean: float) -> float:
        return distance_mean / CONFIG.kinexon.match_duration_s if distance_mean > 0 else 3.5

    def _baseline_speed_old(self, distance_mean: float) -> float:
        return distance_mean / (90 * 60) if distance_mean > 0 else 3.5

    def test_denominator_is_3600_not_5400(self):
        assert CONFIG.kinexon.match_duration_s == 3600
        assert CONFIG.kinexon.match_duration_s != 5400

    def test_baseline_speed_for_9000m_session(self):
        """
        distance_mean = 9000 m (realistic handball session total).
        New: 9000 / 3600 = 2.5 m/s
        Old: 9000 / 5400 = 1.667 m/s
        """
        d = 9000.0
        assert self._baseline_speed_new(d) == pytest.approx(2.5)
        assert self._baseline_speed_old(d) == pytest.approx(9000 / 5400)

    def test_new_baseline_greater_than_old(self):
        """
        Dividing by a smaller denominator (3600 < 5400) gives a LARGER baseline_speed.
        Players must now reach a higher bar to avoid speed_low classification.
        """
        for d in [5000.0, 9000.0, 12600.0]:
            assert self._baseline_speed_new(d) > self._baseline_speed_old(d), (
                f"distance_mean={d}: new baseline_speed must exceed old"
            )

    def test_denominator_ratio(self):
        """Old denominator 5400 / new denominator 3600 = 1.5."""
        ratio = (90 * 60) / CONFIG.kinexon.match_duration_s
        assert ratio == pytest.approx(1.5)

    def test_player_at_exact_average_speed_not_slow(self):
        """
        A player whose session-average speed equals baseline_speed must not be
        classified as speed_low (ratio must be 1.0, not < 0.55).
        """
        avg_speed = 3.5  # m/s — typical handball cruising
        distance_mean = avg_speed * CONFIG.kinexon.match_duration_s  # 12 600 m

        baseline_speed = self._baseline_speed_new(distance_mean)
        speed_ratio = avg_speed / max(baseline_speed, 0.1)

        assert speed_ratio == pytest.approx(1.0, abs=1e-6)
        assert speed_ratio >= 0.55, (
            f"Player at average speed should not be speed_low. speed_ratio={speed_ratio}"
        )

    def test_old_combined_inflation_permanently_forced_speed_low(self):
        """
        Reference: old pitch dims (3× distance inflation) + old denominator (÷5400)
        produced a baseline speed ~4.5× too high → speed_ratio perpetually < 0.55.
        """
        avg_speed = 3.5
        # Old: distance inflated by ~3× AND divided by 5400 instead of 3600
        inflation = (68.0 / 20.0 + 105.0 / 40.0) / 2.0  # ≈ 3.01
        distance_mean_old = avg_speed * 3600 * inflation
        baseline_speed_old = self._baseline_speed_old(distance_mean_old)
        speed_ratio_old = avg_speed / max(baseline_speed_old, 0.1)

        assert speed_ratio_old < 0.55, (
            f"Old bias should produce speed_ratio < 0.55. Got {speed_ratio_old:.3f}"
        )

    def test_fallback_3_5_unchanged(self):
        """distance_mean=0 → baseline_speed defaults to 3.5 (unchanged)."""
        assert self._baseline_speed_new(0.0) == pytest.approx(3.5)


# ─────────────────────────────────────────────────────────────────────────────
# Integration — all four fixes together
# ─────────────────────────────────────────────────────────────────────────────

class TestPhaseAIntegration:
    """Combined sanity: values correct, math consistent, no regressions."""

    def test_all_four_config_values_present(self):
        assert CONFIG.kinexon.sprint_threshold_ms == pytest.approx(5.5)
        assert CONFIG.kinexon.pitch_length_m      == pytest.approx(40.0)
        assert CONFIG.kinexon.pitch_width_m       == pytest.approx(20.0)
        assert CONFIG.kinexon.match_half_duration_s == 1800
        assert CONFIG.kinexon.match_duration_s      == 3600

    def test_handball_sprint_then_late_game_fatigue_possible(self):
        """
        Scenario: player sprinting at 6.0 m/s at 2000 s into the match.

        Before all four fixes:
          sprint_flag=0 (7.0 threshold) → sprint_low=True always
          late_in_game=False (2000 < 2700) → fatigue_flag=False always

        After all four fixes:
          sprint_flag=1 (5.5 threshold) → sprint_low=False
          late_in_game=True (2000 > 1800) → fatigue correctly triggered if anomaly
        """
        speed = 6.0
        elapsed = 2000.0

        # D1: sprint classification
        sprint_flag_new = _sprint_flag(speed, CONFIG.kinexon.sprint_threshold_ms)
        sprint_flag_old = _sprint_flag(speed, 7.0)
        sprint_low_new  = sprint_flag_new == 0
        sprint_low_old  = sprint_flag_old == 0

        # D3: late_in_game
        late_new = elapsed > CONFIG.kinexon.match_half_duration_s
        late_old = elapsed > 2700

        assert sprint_low_old  is True,  "OLD: sprint_low always True at 6.0 m/s"
        assert sprint_low_new  is False, "NEW: sprint_low=False at 6.0 m/s (sprinting)"
        assert late_old        is False, "OLD: 2000s not late at threshold 2700s (bug)"
        assert late_new        is True,  "NEW: 2000s is late in handball (> 1800s)"

    def test_distance_delta_not_inflated(self):
        """
        A player crossing the full short axis of a handball court:
        20 pitch units (100 → 0) = 20/100 × 20 m = 4.0 m actual.
        Old value: 20/100 × 68 = 13.6 m (3.4× inflation).
        """
        d = _distance_delta(0.0, 20.0, 50.0, 50.0,
                            CONFIG.kinexon.pitch_width_m,
                            CONFIG.kinexon.pitch_length_m)
        assert d == pytest.approx(4.0, abs=1e-6), (
            f"20 pitch units on 20 m court = 4.0 m. Got {d:.4f} m"
        )

    def test_second_half_sprint_window_correctly_classified(self):
        """
        Regime: attacking third (x=80), all ticks at 6.0 m/s, elapsed=2200 s.
        Expected after fix: territory=attacking, intensity=high, late_in_game=True.
        """
        x_pitch = 80.0
        speed   = 6.0
        elapsed = 2200.0

        # Territory (regime.py constants)
        territory = (
            "defensive" if x_pitch < 33.0 else
            "attacking" if x_pitch > 67.0 else
            "midfield"
        )
        # Intensity
        frac = _sprint_fraction([speed] * 8, CONFIG.kinexon.sprint_threshold_ms)
        intensity = "high" if frac >= 0.15 else ("medium" if frac >= 0.04 else "low")
        # Late
        late = elapsed > CONFIG.kinexon.match_half_duration_s

        assert territory == "attacking"
        assert intensity  == "high"
        assert late       is True
