"""
tests/test_window_gap_reset.py

Validates the window-contamination fix in SequenceWindowBuilder.add_event():
a substitution/bench gap must reset the per-player buffer instead of letting
a window mix pre-gap and post-gap ticks.

Covers:
  A. Short dropout   (gap < gap_threshold_s)  -> buffer NOT reset
  B. Long bench gap  (gap > gap_threshold_s)  -> buffer reset
  C. First event after return                  -> prev=None (no stale delta)
  D. Window emission behaviour around a reset   -> no contaminated window ever
     emitted; window re-fills cleanly from post-gap ticks only

PyTorch is unavailable in this Python 3.14 environment (analysis.anomaly_detection
defines class SharedLSTMEncoder(nn.Module) at module level, which fails to import
when torch is None). Following the established pattern in
tests/test_phase_a_calibration.py, the import is attempted lazily per-fixture and
all tests skip cleanly when torch cannot be loaded.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from config.settings import CONFIG


def _import_builder():
    """Lazily import SequenceWindowBuilder; skip if PyTorch is unavailable."""
    try:
        import torch  # noqa: F401
        from analysis.anomaly_detection import (
            SequenceWindowBuilder,
            N_SEQUENCE_FEATURES,
            SEQUENCE_FEATURE_NAMES,
        )
        return SequenceWindowBuilder, N_SEQUENCE_FEATURES, SEQUENCE_FEATURE_NAMES
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"PyTorch not available — cannot import SequenceWindowBuilder: {exc}")


@pytest.fixture()
def builder():
    Builder, _, _ = _import_builder()
    return Builder()


@pytest.fixture()
def feature_names():
    _, _, names = _import_builder()
    return names


BASE_TS = datetime(2026, 6, 7, 10, 0, 0, tzinfo=timezone.utc)


def _event(
    player_id: str = "2059",
    ts: datetime = BASE_TS,
    speed_ms: float = 3.5,
    x_pitch: float = 52.0,
    y_pitch: float = 48.0,
    distance_delta_m: float = 0.175,
    is_sprint: int = 0,
) -> dict:
    """Minimal Kinexon-shaped event, matching KinexonAdapter.to_event_dict() keys."""
    return {
        "player_external_id": player_id,
        "ts": ts.isoformat(),
        "speed_ms": speed_ms,
        "heart_rate_bpm": None,
        "x_pitch": x_pitch,
        "y_pitch": y_pitch,
        "distance_delta_m": distance_delta_m,
        "is_sprint": is_sprint,
        "sprint_flag": is_sprint,
        "source": "kinexon",
    }


def _feed(builder, n, start_ts, interval_s, pid="2059", x_start=10.0, x_step=1.0):
    """Feed n events at fixed cadence starting at start_ts; x_pitch increments
    by x_step per tick so post/pre-gap ticks are distinguishable in the
    emitted window. Returns (last_result, last_ts, last_x)."""
    result = None
    ts = start_ts
    x = x_start
    for i in range(n):
        result = builder.add_event(_event(player_id=pid, ts=ts, x_pitch=x))
        ts = ts + timedelta(seconds=interval_s)
        x += x_step
    return result, ts, x


# ---------------------------------------------------------------------------
# A. Short dropout — gap below threshold must NOT reset the buffer
# ---------------------------------------------------------------------------

class TestShortDropoutNoReset:

    def test_buffer_not_cleared_below_threshold(self, builder):
        pid = "2059"
        threshold = CONFIG.window.gap_threshold_s
        # 3 normal ticks, then a 4th tick after a gap just under threshold.
        _feed(builder, 3, BASE_TS, interval_s=15)
        gap_ts = BASE_TS + timedelta(seconds=45 + threshold - 1)  # < threshold
        builder.add_event(_event(player_id=pid, ts=gap_ts))
        assert len(builder._buffers[pid]) == 4, (
            "Short dropout (below gap_threshold_s) must not reset the buffer"
        )

    def test_prev_event_preserved_across_short_gap(self, builder):
        pid = "2059"
        threshold = CONFIG.window.gap_threshold_s
        _feed(builder, 2, BASE_TS, interval_s=15)
        gap_ts = BASE_TS + timedelta(seconds=15 + threshold - 5)
        builder.add_event(_event(player_id=pid, ts=gap_ts))
        assert builder._prev_events.get(pid) is not None, (
            "prev_events must be preserved across a sub-threshold gap"
        )

    def test_no_window_emitted_prematurely(self, builder):
        """4 ticks with window_steps=8 must never emit a window."""
        pid = "2059"
        result, _, _ = _feed(builder, 4, BASE_TS, interval_s=15)
        assert result is None


# ---------------------------------------------------------------------------
# B. Long bench gap — must reset the buffer
# ---------------------------------------------------------------------------

class TestLongGapResetsBuffer:

    def test_buffer_cleared_above_threshold(self, builder):
        pid = "2059"
        window_steps = CONFIG.window.window_steps
        # Fill to one short of a full window.
        _, last_ts, _ = _feed(builder, window_steps - 1, BASE_TS, interval_s=15)
        assert len(builder._buffers[pid]) == window_steps - 1

        # Real Zehnder-style bench gap (1640s, from session 3387 audit).
        gap_ts = last_ts + timedelta(seconds=1640)
        builder.add_event(_event(player_id=pid, ts=gap_ts))

        assert len(builder._buffers[pid]) == 1, (
            "Buffer must contain exactly the post-gap tick after a reset — "
            f"got {len(builder._buffers[pid])}"
        )

    def test_mask_buffer_also_cleared(self, builder):
        pid = "2059"
        window_steps = CONFIG.window.window_steps
        _, last_ts, _ = _feed(builder, window_steps - 1, BASE_TS, interval_s=15)
        gap_ts = last_ts + timedelta(seconds=901)  # Zehnder gap #2
        builder.add_event(_event(player_id=pid, ts=gap_ts))
        assert len(builder._mask_buffers[pid]) == 1

    def test_no_window_emitted_immediately_after_reset(self, builder):
        """
        Before the fix: buffer was already full (window_steps-1 + this append
        would have reached window_steps), so a window would have been emitted
        on this very call, contaminated with stale pre-gap ticks.
        After the fix: the reset means the buffer only has 1 tick, so no
        window is emitted yet.
        """
        pid = "2059"
        window_steps = CONFIG.window.window_steps
        _, last_ts, _ = _feed(builder, window_steps - 1, BASE_TS, interval_s=15)
        gap_ts = last_ts + timedelta(seconds=808)  # Zehnder gap #3
        result = builder.add_event(_event(player_id=pid, ts=gap_ts))
        assert result is None, (
            "A window must NOT be emitted on the first tick after a gap reset "
            "(this is the exact bug: buffer was already full pre-fix, so this "
            "call would have emitted a contaminated window immediately)"
        )

    def test_short_then_long_gap_both_detected(self, builder):
        """Multiple gaps in sequence (Zehnder had 7) must each be handled."""
        pid = "2059"
        _, ts, _ = _feed(builder, 3, BASE_TS, interval_s=15)
        ts += timedelta(seconds=300)  # Zehnder gap #4
        builder.add_event(_event(player_id=pid, ts=ts))
        assert len(builder._buffers[pid]) == 1
        _, ts, _ = _feed(builder, 2, ts + timedelta(seconds=15), interval_s=15)
        ts += timedelta(seconds=1640)  # another large gap
        builder.add_event(_event(player_id=pid, ts=ts))
        assert len(builder._buffers[pid]) == 1, "Second gap must also reset the buffer"


# ---------------------------------------------------------------------------
# C. First event after return — prev must be None, not the stale event
# ---------------------------------------------------------------------------

class TestFirstEventAfterReturn:

    def test_prev_events_cleared_on_reset(self, builder):
        pid = "2059"
        _feed(builder, 3, BASE_TS, interval_s=15)
        assert builder._prev_events.get(pid) is not None
        gap_ts = BASE_TS + timedelta(seconds=45 + 1640)
        builder.add_event(_event(player_id=pid, ts=gap_ts))
        # _extract() is called with prev=None for this tick (first-tick
        # semantics), then this same event becomes the new _prev_events entry.
        assert builder._prev_events[pid]["ts"] == gap_ts.isoformat()

    def test_no_bogus_acceleration_spike_after_gap(self, builder, feature_names):
        """
        Before the fix: accel = (speed - prev_speed) / event_interval_s using
        the STALE prev (from before the gap), producing a spurious spike.
        After the fix: prev=None for the first post-gap tick, so accel=0.0
        (matches the existing first-tick-in-session convention).
        """
        pid = "2059"
        accel_idx = feature_names.index("acceleration_ms2")
        window_steps = CONFIG.window.window_steps

        # Pre-gap: walking pace.
        _feed(builder, 3, BASE_TS, interval_s=15, x_start=10.0)
        # Post-gap: fill a full window; first post-gap tick jumps to sprint speed.
        gap_ts = BASE_TS + timedelta(seconds=45 + 1640)
        ts = gap_ts
        result = None
        for i in range(window_steps):
            spd = 6.0 if i == 0 else 6.0  # sustained post-gap speed
            result = builder.add_event(_event(player_id=pid, ts=ts, speed_ms=spd, x_pitch=30.0 + i))
            ts += timedelta(seconds=15)

        assert result is not None, "Window should be complete after window_steps post-gap ticks"
        sequence, mask = result
        first_post_gap_accel = sequence[0, accel_idx]
        assert first_post_gap_accel == pytest.approx(0.0), (
            f"First tick after a gap must have accel=0.0 (prev=None), got {first_post_gap_accel}. "
            "A nonzero value here means the stale pre-gap event leaked into the delta computation."
        )


# ---------------------------------------------------------------------------
# D. Window emission behaviour — no contaminated window is ever returned
# ---------------------------------------------------------------------------

class TestWindowEmissionAfterGap:

    def test_window_emitted_before_gap_is_clean(self, builder):
        """Sanity check: filling exactly window_steps ticks emits a window
        (mechanism unaffected by the fix when there is no gap)."""
        window_steps = CONFIG.window.window_steps
        result, _, _ = _feed(builder, window_steps, BASE_TS, interval_s=15)
        assert result is not None
        sequence, mask = result
        assert sequence.shape[0] == window_steps
        assert mask.all()

    def test_window_after_reset_contains_only_post_gap_ticks(self, builder, feature_names):
        """
        Core contamination-elimination proof: fill a full pre-gap window,
        trigger a long gap, then feed window_steps post-gap ticks with a
        distinctive x_pitch range. The emitted window must contain ONLY
        the post-gap x_pitch values — none of the pre-gap window must
        leak through.
        """
        pid = "2059"
        window_steps = CONFIG.window.window_steps
        x_idx = feature_names.index("x_pitch")

        # Pre-gap window: x_pitch in [10, 10+window_steps).
        pre_result, last_ts, _ = _feed(
            builder, window_steps, BASE_TS, interval_s=15, x_start=10.0
        )
        assert pre_result is not None
        pre_x_values = set(pre_result[0][:, x_idx].tolist())
        assert pre_x_values == set(float(10 + i) for i in range(window_steps))

        # Long bench gap (Zehnder's largest: 1640s).
        gap_ts = last_ts + timedelta(seconds=1640)

        # Post-gap window: x_pitch in [90, 90+window_steps) — disjoint range,
        # easy to detect contamination.
        post_result = None
        ts = gap_ts
        for i in range(window_steps):
            post_result = builder.add_event(
                _event(pid, ts=ts, x_pitch=90.0 + i)
            )
            ts += timedelta(seconds=15)

        assert post_result is not None, "Post-gap window must be emitted once refilled"
        post_sequence, post_mask = post_result
        post_x_values = set(post_sequence[:, x_idx].tolist())

        assert post_x_values == set(float(90 + i) for i in range(window_steps)), (
            f"Post-gap window must contain ONLY post-gap x_pitch values [90..{90+window_steps-1}], "
            f"got {sorted(post_x_values)}. Any value < 90 indicates stale pre-gap contamination."
        )
        assert post_x_values.isdisjoint(pre_x_values), (
            "Post-gap window must not share any tick with the pre-gap window"
        )
        assert post_mask.all(), "All post-gap ticks are real (speed_ms present)"

    def test_intermediate_ticks_after_reset_return_none(self, builder):
        """Between the reset and the (window_steps)-th post-gap tick, every
        call must return None — no partial/premature window emission."""
        pid = "2059"
        window_steps = CONFIG.window.window_steps
        _, last_ts, _ = _feed(builder, window_steps, BASE_TS, interval_s=15)
        gap_ts = last_ts + timedelta(seconds=1640)

        ts = gap_ts
        for i in range(window_steps - 1):
            result = builder.add_event(_event(pid, ts=ts, x_pitch=90.0 + i))
            assert result is None, f"Tick {i+1} after reset must not emit a window yet"
            ts += timedelta(seconds=15)
