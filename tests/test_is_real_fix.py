"""
test_is_real_fix.py

Proves that SequenceWindowBuilder.build_live_window() (and add_event /
build_from_session) treats Kinexon events as REAL after the is_real fix.

Before fix: is_real required both speed_ms AND heart_rate_bpm.
After fix:  is_real requires only speed_ms.

Run:
    pytest tests/test_is_real_fix.py -v
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pytest

from analysis.anomaly_detection import (
    SequenceWindowBuilder,
    N_SEQUENCE_FEATURES,
    SEQUENCE_FEATURE_NAMES,
)
from config.settings import CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kinexon_event(
    player_id: str = "1796",
    speed_ms: float = 3.5,
    heart_rate_bpm=None,
    x_pitch: float = 52.0,
    y_pitch: float = 48.0,
    distance_delta_m: float = 0.175,
    is_sprint: int = 0,
    ts: str = "2026-06-07T10:00:00+00:00",
) -> dict:
    """Minimal event dict as produced by KinexonAdapter.to_event_dict()."""
    return {
        "player_external_id": player_id,
        "ts": ts,
        "speed_ms": speed_ms,
        "heart_rate_bpm": heart_rate_bpm,
        "x_pitch": x_pitch,
        "y_pitch": y_pitch,
        "distance_delta_m": distance_delta_m,
        "is_sprint": is_sprint,
        "sprint_flag": is_sprint,
        "source": "kinexon",
    }


def _gps_event(
    player_id: str = "42",
    speed_ms: float = 4.0,
    heart_rate_bpm: float = 145.0,
    x_pitch: float = 50.0,
    y_pitch: float = 50.0,
) -> dict:
    """Minimal GPS event dict with real HR."""
    return {
        "player_external_id": player_id,
        "ts": "2026-06-07T11:00:00+00:00",
        "speed_ms": speed_ms,
        "heart_rate_bpm": heart_rate_bpm,
        "x_pitch": x_pitch,
        "y_pitch": y_pitch,
        "distance_delta_m": 0.2,
        "is_sprint": 0,
        "source": "gps",
    }


def _make_window(n: int = None, **kwargs) -> list[dict]:
    """Build a list of n identical Kinexon events."""
    n = n or CONFIG.window.window_steps
    return [_kinexon_event(**kwargs) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Core is_real semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestIsRealSemantics:

    def test_kinexon_event_hr_none_is_real(self):
        """After fix: speed_ms present → is_real=True regardless of HR."""
        builder = SequenceWindowBuilder()
        events = _make_window()
        seq, mask = builder.build_live_window(events)
        assert mask.all(), (
            f"All Kinexon ticks (HR=None) should be real after fix. "
            f"Got mask={mask.tolist()}"
        )

    def test_event_missing_speed_is_not_real(self):
        """speed_ms=None still produces padding (unchanged from before)."""
        builder = SequenceWindowBuilder()
        events = []
        for i in range(CONFIG.window.window_steps):
            e = _kinexon_event(heart_rate_bpm=None)
            if i % 2 == 0:
                e["speed_ms"] = None   # dropped GPS packet
            events.append(e)
        seq, mask = builder.build_live_window(events)
        # Even-indexed ticks should be padded
        for t in range(0, CONFIG.window.window_steps, 2):
            assert not mask[t], f"t={t} has speed_ms=None, should be padded"
        for t in range(1, CONFIG.window.window_steps, 2):
            assert mask[t], f"t={t} has valid speed, should be real"

    def test_gps_event_with_hr_is_real(self):
        """GPS events with HR still work correctly (HR present → real)."""
        builder = SequenceWindowBuilder()
        events = [_gps_event() for _ in range(CONFIG.window.window_steps)]
        seq, mask = builder.build_live_window(events)
        assert mask.all(), "GPS events with HR=145 should all be real"

    def test_gps_event_without_hr_is_still_real(self):
        """GPS events with dropped HR packet are now real (was padding before fix)."""
        builder = SequenceWindowBuilder()
        events = [_gps_event(heart_rate_bpm=None) for _ in range(CONFIG.window.window_steps)]
        seq, mask = builder.build_live_window(events)
        assert mask.all(), (
            "After fix: GPS tick with dropped HR (speed present) should be real, "
            "not padding."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector content
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureVectorContent:

    def test_sequence_not_all_zeros_for_kinexon(self):
        """After fix: sequence must contain real movement values, not all zeros."""
        builder = SequenceWindowBuilder()
        events = _make_window(speed_ms=4.2, x_pitch=55.0, y_pitch=45.0)
        seq, mask = builder.build_live_window(events)
        assert seq.sum() != 0.0, (
            "Feature sequence must be non-zero for Kinexon events after fix."
        )

    def test_speed_feature_populated(self):
        """speed_ms feature (index 0) must reflect actual speed, not 0."""
        builder = SequenceWindowBuilder()
        events = _make_window(speed_ms=6.0)
        seq, mask = builder.build_live_window(events)
        speed_col = seq[:, 0]   # SEQUENCE_FEATURE_NAMES[0] = speed_ms
        assert (speed_col > 0).any(), (
            f"speed_ms column should be non-zero for speed=6.0. Got {speed_col}"
        )

    def test_x_y_pitch_features_populated(self):
        """Positional features (x_pitch idx=4, y_pitch idx=5) must be real."""
        builder = SequenceWindowBuilder()
        events = _make_window(x_pitch=72.5, y_pitch=30.0)
        seq, mask = builder.build_live_window(events)
        x_col = seq[:, 4]
        y_col = seq[:, 5]
        assert (x_col > 0).any(), "x_pitch column should be non-zero"
        assert (y_col > 0).any(), "y_pitch column should be non-zero"

    def test_hr_feature_is_zero_for_kinexon(self):
        """HR feature (index 2) is 0.0 when sensor absent — not fabricated."""
        builder = SequenceWindowBuilder()
        events = _make_window(heart_rate_bpm=None)
        seq, mask = builder.build_live_window(events)
        hr_col = seq[:, 2]
        assert (hr_col == 0.0).all(), (
            f"HR feature must be 0.0 for Kinexon (sensor absent). Got {hr_col}"
        )

    def test_hr_feature_real_when_present(self):
        """When HR IS present, it must appear in the feature vector correctly."""
        builder = SequenceWindowBuilder()
        events = [_gps_event(heart_rate_bpm=155.0) for _ in range(CONFIG.window.window_steps)]
        seq, mask = builder.build_live_window(events)
        hr_col = seq[:, 2]
        assert (hr_col > 0).any(), (
            f"HR feature should reflect real HR=155. Got {hr_col}"
        )

    def test_mask_completeness_is_one_for_kinexon(self):
        """mask_completeness must be 1.0 for full Kinexon windows after fix."""
        builder = SequenceWindowBuilder()
        events = _make_window()
        seq, mask = builder.build_live_window(events)
        completeness = float(mask.mean())
        assert completeness == 1.0, (
            f"mask_completeness should be 1.0 for Kinexon window. Got {completeness}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# _masked_mse behaviour with movement-only data
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskedMSEBehaviour:

    def test_masked_mse_is_zero_when_all_padded(self):
        """Before-fix scenario: all-False mask → loss = 0 (verify the bug)."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from analysis.anomaly_detection import _masked_mse

        T, F = CONFIG.window.window_steps, N_SEQUENCE_FEATURES
        x = torch.ones(1, T, F)
        recon = torch.zeros(1, T, F)
        mask = torch.zeros(1, T, dtype=torch.bool)   # all False = before fix

        loss = _masked_mse(x, recon, mask)
        assert float(loss.item()) == pytest.approx(0.0), (
            "All-padded window must produce loss=0.0 — this is the bug we fixed."
        )

    def test_masked_mse_nonzero_when_real(self):
        """After-fix scenario: all-True mask + non-zero diff → loss > 0."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        from analysis.anomaly_detection import _masked_mse

        T, F = CONFIG.window.window_steps, N_SEQUENCE_FEATURES
        x = torch.ones(1, T, F)
        recon = torch.zeros(1, T, F)
        mask = torch.ones(1, T, dtype=torch.bool)   # all True = after fix

        loss = _masked_mse(x, recon, mask)
        assert float(loss.item()) > 0.0, (
            "After fix: real ticks should produce non-zero reconstruction loss."
        )


# ─────────────────────────────────────────────────────────────────────────────
# add_event path
# ─────────────────────────────────────────────────────────────────────────────

class TestAddEventPath:

    def test_add_event_kinexon_returns_window_when_full(self):
        """add_event must accumulate Kinexon events and return a window."""
        builder = SequenceWindowBuilder()
        result = None
        for _ in range(CONFIG.window.window_steps):
            result = builder.add_event(_kinexon_event())
        assert result is not None, (
            "add_event must return (seq, mask) after window_steps Kinexon events."
        )
        seq, mask = result
        assert mask.any(), "At least some ticks should be real after fix."

    def test_add_event_kinexon_mask_all_true(self):
        """add_event: Kinexon window mask must be all True after fix."""
        builder = SequenceWindowBuilder()
        result = None
        for _ in range(CONFIG.window.window_steps):
            result = builder.add_event(_kinexon_event(speed_ms=3.5))
        assert result is not None
        _, mask = result
        assert mask.all(), f"Expected all-True mask from add_event. Got {mask.tolist()}"


# ─────────────────────────────────────────────────────────────────────────────
# Distance delta propagation
# ─────────────────────────────────────────────────────────────────────────────

class TestDistanceDelta:

    def test_distance_delta_computed_across_real_ticks(self):
        """With all ticks real, _extract computes distance_delta from x/y."""
        builder = SequenceWindowBuilder()
        n = CONFIG.window.window_steps
        events = []
        for i in range(n):
            events.append(_kinexon_event(
                x_pitch=50.0 + i * 0.5,   # player moving right
                y_pitch=50.0,
                speed_ms=3.0,
                heart_rate_bpm=None,
            ))
        seq, mask = builder.build_live_window(events)
        # distance_delta is at index 6
        dist_col = seq[1:, 6]   # skip t=0 (first tick, delta=0)
        assert (dist_col > 0).any(), (
            "distance_delta_m must be non-zero when player moves across real ticks."
        )
