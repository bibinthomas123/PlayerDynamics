"""
tests/test_calibration_stability.py

Proves three stability invariants against HardenedRollingThresholdStore:

  1. BOUND:      threshold never exceeds threshold_ceiling
  2. RATE:       threshold cannot rise >max_upward_step_fraction per commit
  3. BREAKER:    after max_consecutive_upward rising commits, the next
                 upward proposal is rejected (circuit-breaker fires)
  4. RECOVERY:   after from_state_dict(), circuit-breaker counter and
                 last_committed_threshold are identical to pre-crash state
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from utils.reliability.calibration_store import HardenedRollingThresholdStore, CalibrationWindow


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_store(floor=0.0, ceiling=float("inf"), max_up=0.20, max_consec=3):
    store = HardenedRollingThresholdStore(player_id=1)

    store.threshold_floor = floor
    store.threshold_ceiling = ceiling
    store.max_upward_step_fraction = max_up
    store.max_consecutive_upward = max_consec

    # disable unrelated gates for invariant tests
    store.adaptation_cooldown_s = 0
    store.quarantine_threshold = 0
    store.drift_threshold = 999.0

    return store


def _seed_windows(store, losses, base_time=None):
    """Populate store.windows directly (bypasses quarantine for seeding)."""
    t = base_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    store.windows = [
        CalibrationWindow(
            window_id=f"seed_{i}",
            loss=l,
            timestamp=t + timedelta(seconds=i),
            confidence=0.9,
            is_healthy=True,
        )
        for i, l in enumerate(losses)
    ]


def _fill_quarantine(store, losses, base_time=None):
    t = base_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    store.quarantine = [
        CalibrationWindow(
            window_id=f"q_{i}",
            loss=l,
            timestamp=t + timedelta(seconds=i),
            confidence=0.9,
            is_healthy=True,
        )
        for i, l in enumerate(losses)
    ]


# ── Invariant 1: hard ceiling ─────────────────────────────────────────────────

def test_threshold_never_exceeds_ceiling():
    store = _make_store(ceiling=1.0)
    _seed_windows(store, np.linspace(0.1, 0.9, 40).tolist())
    store._last_committed_threshold = 0.8

    # Quarantine that would push proposed above 1.0
    _fill_quarantine(store, [1.5] * 6)

    result = store.compute_proposed_threshold()
    assert result is None, (
        f"Expected rejection (ceiling=1.0) but got proposed={result}"
    )


# ── Invariant 2: rate-of-change cap ──────────────────────────────────────────

def test_threshold_cannot_rise_above_rate_limit():
    store = _make_store(max_up=0.20)
    prior = 1.0
    _seed_windows(store, np.linspace(0.5, prior, 40).tolist())
    store._last_committed_threshold = prior

    # Quarantine whose p99.5 would be ~1.5, well above 1.0 * 1.20 = 1.20
    _fill_quarantine(store, [1.5] * 6)

    result = store.compute_proposed_threshold()
    assert result is None, (
        f"Expected rate-cap rejection but got proposed={result}"
    )


def test_threshold_may_rise_within_rate_limit():
    store = _make_store(max_up=0.20)
    prior = 1.0
    _seed_windows(store, np.linspace(0.8, prior, 40).tolist())
    store._last_committed_threshold = prior

    # Quarantine whose p99.5 would be ~1.10, within 20% of 1.0
    _fill_quarantine(store, [1.05, 1.08, 1.10, 1.06, 1.07, 1.09])

    result = store.compute_proposed_threshold()
    assert result is not None, "Expected proposal within rate limit to be accepted"
    assert result <= prior * 1.20 + 1e-9


# ── Invariant 3: consecutive-upward circuit-breaker ──────────────────────────

def test_circuit_breaker_fires_after_max_consecutive_upward():
    store = _make_store(max_up=0.50, max_consec=3)

    # Simulate 3 prior upward commits by setting the counter directly
    store._consecutive_upward_adaptations = 3
    prior = 1.0
    _seed_windows(store, np.linspace(0.8, prior, 40).tolist())
    store._last_committed_threshold = prior

    # Another upward proposal — should be rejected
    _fill_quarantine(store, [1.1, 1.12, 1.08, 1.11, 1.09, 1.10])

    result = store.compute_proposed_threshold()
    assert result is None, (
        "Expected circuit-breaker to reject 4th consecutive upward adaptation"
    )


def test_circuit_breaker_resets_after_neutral_commit():
    store = _make_store(max_up=0.50, max_consec=3)
    store._consecutive_upward_adaptations = 3
    prior = 1.0
    store._last_committed_threshold = prior

    # Simulate a neutral/downward apply_adaptation resetting the counter
    store._consecutive_upward_adaptations = 0
    store._last_committed_threshold = 0.95  # slight downward

    _seed_windows(store, np.linspace(0.7, 0.95, 40).tolist())
    _fill_quarantine(store, [0.97, 0.96, 0.98, 0.95, 0.97, 0.96])

    result = store.compute_proposed_threshold()
    assert result is not None, (
        "Expected proposal to be accepted after circuit-breaker reset"
    )


# ── Invariant 4: recovery restores stability state ───────────────────────────

def test_recovery_restores_circuit_breaker_state():
    store = _make_store(max_up=0.20, max_consec=3)
    _seed_windows(store, np.linspace(0.5, 1.0, 40).tolist())
    store._last_committed_threshold = 1.0
    store._consecutive_upward_adaptations = 3

    # Serialize and recover
    snapshot = store.full_state_dict()
    recovered = HardenedRollingThresholdStore.from_state_dict(1, snapshot)

    assert recovered._consecutive_upward_adaptations == 3, (
        "Circuit-breaker counter must survive recovery"
    )
    assert recovered._last_committed_threshold == pytest.approx(1.0), (
        "Last committed threshold must survive recovery"
    )

    # Confirm circuit-breaker is still active post-recovery
    recovered.adaptation_cooldown_s = 0
    recovered.quarantine_threshold = 5
    _fill_quarantine(recovered, [1.1, 1.12, 1.08, 1.11, 1.09, 1.10])

    result = recovered.compute_proposed_threshold()
    assert result is None, (
        "Circuit-breaker must still fire after recovery with counter=3"
    )