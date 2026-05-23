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
        self.quarantine_threshold = 20  # Windows to vet before merging
        self.drift_threshold = 0.15     # KS-test statistic limit
        self.adaptation_cooldown_s = 3600  # 1 hour cooldown between updates

        # Stability bounds — prevent oscillation and deviance normalization
        self.threshold_floor: float = 0.0    # set to gold-set p99.5 at init time
        self.threshold_ceiling: float = float("inf")  # set by operator; hard upper cap
        self.max_upward_step_fraction: float = 0.20   # threshold may not rise >20% per commit
        self.max_downward_step_fraction: float = 0.50 # threshold may not fall >50% per commit
        self._consecutive_upward_adaptations: int = 0
        self.max_consecutive_upward: int = 3  # circuit-breaker: freeze after N rising commits

        self._last_update_at = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self._last_committed_threshold: Optional[float] = None

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
        proposed = float(np.quantile(clean, quantile))

        # ── Stability gate ─────────────────────────────────────────────
        # 1. Hard absolute bounds
        if proposed < self.threshold_floor:
            logger.warning(
                "Proposed threshold %.4f is below floor %.4f — clamping.",
                proposed, self.threshold_floor,
            )
            proposed = self.threshold_floor

        if proposed > self.threshold_ceiling:
            logger.warning(
                "Proposed threshold %.4f exceeds ceiling %.4f — rejecting.",
                proposed, self.threshold_ceiling,
            )
            return None  # reject entirely; do not adapt

        # 2. Rate-of-change cap (only enforceable once we have a prior commit)
        prior = self._last_committed_threshold
        if prior is not None and prior > 0.0:
            rise_limit = prior * (1.0 + self.max_upward_step_fraction)
            fall_limit = prior * (1.0 - self.max_downward_step_fraction)

            if proposed > rise_limit:
                logger.warning(
                    "Proposed threshold %.4f rises >%.0f%% from prior %.4f — rejecting "
                    "to prevent oscillatory inflation.",
                    proposed, self.max_upward_step_fraction * 100, prior,
                )
                return None

            if proposed < fall_limit:
                logger.warning(
                    "Proposed threshold %.4f falls >%.0f%% from prior %.4f — rejecting "
                    "to prevent over-correction.",
                    proposed, self.max_downward_step_fraction * 100, prior,
                )
                return None

        # 3. Consecutive-upward-adaptation circuit-breaker
        if prior is not None and proposed > prior:
            if self._consecutive_upward_adaptations >= self.max_consecutive_upward:
                logger.warning(
                    "Circuit breaker: %d consecutive upward adaptations reached. "
                    "Freezing threshold at %.4f until a downward or neutral commit occurs.",
                    self._consecutive_upward_adaptations, prior,
                )
                return None

        return proposed

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

        # Update stability tracking counters
        new_threshold = self.get_threshold()
        if self._last_committed_threshold is not None and new_threshold > self._last_committed_threshold:
            self._consecutive_upward_adaptations += 1
        else:
            self._consecutive_upward_adaptations = 0  # reset on neutral or downward move
        self._last_committed_threshold = new_threshold

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
    
    def full_state_dict(self) -> dict:
        """Complete serialization of all distribution state needed for replay recovery.

        Unlike state_dict(), this captures every field required to reproduce
        future adaptation timing, KS-test baselines, and quarantine contents
       exactly — so recovery is truly deterministic.
        """
        return {
            "calibration_version": self.calibration_version,
            "last_update_at": self._last_update_at.isoformat(),
            "last_committed_threshold": self._last_committed_threshold,
            "consecutive_upward_adaptations": self._consecutive_upward_adaptations,
            "windows": [
                {
                    "window_id": w.window_id,
                    "loss":      w.loss,
                    "timestamp": w.timestamp.isoformat(),
                    "confidence": w.confidence,
                    "is_healthy": w.is_healthy,
                    "is_quarantined": w.is_quarantined,
                }
                for w in self.windows
            ],
            "quarantine": [
                {
                    "window_id": w.window_id,
                    "loss":      w.loss,
                    "timestamp": w.timestamp.isoformat(),
                    "confidence": w.confidence,
                    "is_healthy": w.is_healthy,
                    "is_quarantined": w.is_quarantined,
                }
                for w in self.quarantine
            ],
        }

    @classmethod
    def from_state_dict(cls, player_id: int, data: dict) -> "HardenedRollingThresholdStore":
        """Reconstruct a store from a full_state_dict() snapshot.

        This is the only correct recovery path. Reconstructing from a partial
        snapshot (e.g. version number only) leaves distributions blank and
        causes KS-test divergence on the first post-recovery adaptation.
        """
        store = cls(player_id)
        store.calibration_version = data["calibration_version"]
        store._last_update_at = datetime.fromisoformat(data["last_update_at"])
        store._last_committed_threshold = data.get("last_committed_threshold")       
        store._consecutive_upward_adaptations = data.get(                            
            "consecutive_upward_adaptations", 0
        )

        def _deserialize(raw: list) -> list[CalibrationWindow]:
            return [
                CalibrationWindow(
                    window_id=r["window_id"],
                    loss=r["loss"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    confidence=r["confidence"],
                    is_healthy=r["is_healthy"],
                    is_quarantined=r["is_quarantined"],
                )
                for r in raw
            ]

        store.windows     = _deserialize(data.get("windows", []))
        store.quarantine  = _deserialize(data.get("quarantine", []))
        return store