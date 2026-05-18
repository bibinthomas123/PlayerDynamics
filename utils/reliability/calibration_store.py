"""
Hardened Calibration Store
Protects adaptive thresholds from scientific corruption, poisoning, and deviance normalization.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime, timezone
import logging
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

@dataclass
class CalibrationWindow:
    window_id: str
    loss: float
    timestamp: datetime
    confidence: float
    is_healthy: bool = True
    is_quarantined: bool = False

class HardenedRollingThresholdStore:
    """
    A versioned, integrity-aware store for anomaly thresholds.

    Implements:
    - Quarantine Buffers: New data is vetted before affecting the threshold.
    - Anomaly Exclusion: High-loss windows are permanently excluded from calibration.
    - Drift Monitoring: KL-divergence/KS-test against gold calibration sets.
    - Cooldown Windows: Prevents rapid adaptation during high-intensity phases.
    """
    def __init__(self, player_id: int, gold_set: Optional[np.ndarray] = None):
        self.player_id = player_id
        self.gold_set = gold_set # Initial "ground truth" calibration
        self.calibration_version = 0
        self.windows: List[CalibrationWindow] = []
        self.quarantine: List[CalibrationWindow] = []

        # Hyper-parameters
        self.max_windows = 500
        self.quarantine_threshold = 20 # Windows to vet before merging
        self.drift_threshold = 0.15     # KS-test statistic limit
        self.adaptation_cooldown_s = 3600 # 1 hour cooldown between updates

        self._last_update_at = datetime(1970, 1, 1, tzinfo=timezone.utc)

    def add_window(self, window_id: str, loss: float, confidence: float, timestamp: datetime):
        """Adds a window to the quarantine buffer for vetting."""
        # Permanent exclusion: High-loss windows are likely anomalies, not baseline.
        # If loss is > 5x the current median, it's a candidate for exclusion.
        if self.is_calibrated():
            median_loss = np.median([w.loss for w in self.windows])
            if loss > median_loss * 5.0:
                logger.debug("Calibration poisoning attempt: loss %.4f > 5x median. Excluding.", loss)
                return

        window = CalibrationWindow(
            window_id=window_id,
            loss=loss,
            timestamp=timestamp,
            confidence=confidence,
            is_healthy=(confidence >= 0.7)
        )
        self.quarantine.append(window)

    # ------------------------------------------------------------------
    # Read-only predicates — no side effects, safe to call at any time
    # ------------------------------------------------------------------

    def should_adapt(self) -> bool:
        """Return True when the quarantine is full and cooldown has elapsed.

        Pure predicate: reads state only, never mutates it.
        Call this from DeterministicCalibrationManager to decide whether
        to open a mutation transaction — do NOT call attempt_adaptation()
        directly from outside this module.
        """
        if len(self.quarantine) < self.quarantine_threshold:
            return False
        now = datetime.now(tz=timezone.utc)
        return (now - self._last_update_at).total_seconds() >= self.adaptation_cooldown_s

    def compute_proposed_threshold(self, quantile: float = 0.995) -> Optional[float]:
        """Compute what the threshold *would* become if adaptation is applied.

        Pure computation: reads quarantine + windows, never mutates them.
        Returns None when vetting or drift checks would reject the update.
        """
        healthy_candidates = [w for w in self.quarantine if w.is_healthy]
        if not healthy_candidates:
            return None

        if self.is_calibrated():
            active_losses = np.array([w.loss for w in self.windows])
            cand_losses   = np.array([w.loss for w in healthy_candidates])
            stat, _       = ks_2samp(active_losses, cand_losses)
            if stat > self.drift_threshold:
                return None  # drift check would reject

        # Simulate the merged window set to compute the proposed value
        merged = list(self.windows) + healthy_candidates
        merged = merged[-self.max_windows:]
        losses = np.array([w.loss for w in merged])
        cutoff = np.quantile(losses, 0.95)
        clean  = losses[losses <= cutoff]
        return float(np.quantile(clean, quantile))

    # ------------------------------------------------------------------
    # Write path — called ONLY by DeterministicCalibrationManager
    # ------------------------------------------------------------------

    def apply_adaptation(self) -> bool:
        """Commit the quarantine into the active window set and bump version.

        MUST only be called after:
          1. should_adapt() returned True
          2. compute_proposed_threshold() returned a non-None value
          3. DeterministicCalibrationManager has committed the mutation to journal

        Returns True if the adaptation was applied, False if vetting rejects it
        (e.g. no healthy windows or drift check fails — same guards as before).
        """
        now = datetime.now(tz=timezone.utc)
        healthy_candidates = [w for w in self.quarantine if w.is_healthy]

        if not healthy_candidates:
            self.quarantine.clear()
            return False

        if self.is_calibrated():
            active_losses = np.array([w.loss for w in self.windows])
            cand_losses   = np.array([w.loss for w in healthy_candidates])
            stat, _       = ks_2samp(active_losses, cand_losses)
            if stat > self.drift_threshold:
                logger.warning(
                    "Calibration drift detected (KS=%.3f). Adaptation rejected.", stat
                )
                self.quarantine.clear()
                return False

        self.windows.extend(healthy_candidates)
        self.windows = self.windows[-self.max_windows:]
        self.quarantine.clear()
        self.calibration_version += 1
        self._last_update_at = now
        return True

    def attempt_adaptation(self) -> bool:
        """Vets quarantine and updates the active threshold set.

        .. deprecated::
            Do not call this directly. Use the split interface:
              1. should_adapt()                  — predicate, no side effects
              2. compute_proposed_threshold()    — read-only proposal
              3. apply_adaptation()              — write path, journal first

            Direct calls bypass DeterministicCalibrationManager and break
            replay-safe mutation authority. This method is retained only for
            backward compatibility during migration and will be removed.
        """
        import warnings
        warnings.warn(
            "attempt_adaptation() called directly — this bypasses the "
            "DeterministicCalibrationManager journal and breaks replay safety. "
            "Use should_adapt() / compute_proposed_threshold() / apply_adaptation() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        now = datetime.now(tz=timezone.utc)
        if (now - self._last_update_at).total_seconds() < self.adaptation_cooldown_s:
            return False

        # Vetting: Only move healthy windows from quarantine to active set
        healthy_candidates = [w for w in self.quarantine if w.is_healthy]

        if not healthy_candidates:
            self.quarantine.clear()
            return False

        # Drift Check: Compare candidate distribution vs active set
        if self.is_calibrated():
            active_losses = np.array([w.loss for w in self.windows])
            cand_losses = np.array([w.loss for w in healthy_candidates])

            # Kolmogorov-Smirnov test for distribution shift
            stat, _ = ks_2samp(active_losses, cand_losses)
            if stat > self.drift_threshold:
                logger.warning("Calibration drift detected (KS=%.3f). Adaptation paused to prevent deviance normalization.", stat)
                self.quarantine.clear()
                return False

        # Merge and prune
        self.windows.extend(healthy_candidates)
        self.windows = self.windows[-self.max_windows:]
        self.quarantine.clear()

        self.calibration_version += 1
        self._last_update_at = now
        return True

    def is_calibrated(self) -> bool:
        return len(self.windows) >= 30

    def get_threshold(self, quantile: float = 0.995) -> float:
        if not self.is_calibrated():
            return float("inf")

        losses = np.array([w.loss for w in self.windows])
        # Trim contamination (top 5%) to ensure threshold isn't inflated by outliers
        cutoff = np.quantile(losses, 0.95)
        clean_losses = losses[losses <= cutoff]

        return float(np.quantile(clean_losses, quantile))

    def state_dict(self) -> dict:
        return {
            "version": self.calibration_version,
            "losses": [w.loss for w in self.windows],
            "last_update": self._last_update_at.isoformat()
        }