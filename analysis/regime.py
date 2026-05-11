"""
Players Data — IBM CIC Germany
Session Regime Classifier + Regime-Aware Threshold Store

Problem being solved
────────────────────
The anomaly detection model assumes all 120-second windows are drawn from
one stationary behavioral distribution.  In football they are not.

A 120-second high-press window has systematically higher speed, higher
acceleration variance, and different positional geometry than a 120-second
possession-retention window.  When a global DynamicThresholdTracker pools
calibration losses from both regimes, the resulting threshold sits above
the true "normal" for both — raising false-positive risk during intensity
spikes and suppressing detection during low-intensity anomalies.

Solution
────────
Classify every window into a discrete (territory × intensity) regime.
Maintain a per-regime DynamicThresholdTracker.
At inference, compare the smoothed loss against the regime-specific threshold.
Fall back to the global tracker when a regime is under-calibrated.

Regime taxonomy
───────────────
Territory (mean x_pitch over window):
  defensive  — mean_x < 33   (own half, deep defensive block)
  midfield   — 33 ≤ mean_x ≤ 67
  attacking  — mean_x > 67   (opponent half, pressing / box entries)

Intensity (sprint_flag fraction over window):
  high    — sprint fraction ≥ 0.15  (transition / press)
  medium  — 0.04 ≤ sprint fraction < 0.15
  low     — sprint fraction < 0.04  (set piece, possession, recovery)

This gives 9 possible regime keys.  In practice 4–6 are well-populated per
player.  The global fallback handles the rest safely.

Why no match_phase (first/second half)?
────────────────────────────────────────
Match phase would be useful but is not available inside _calibrate() because
calibration sequences are passed without timestamp context.  Territory ×
Intensity already captures the dominant within-session variance; match phase
adds marginal signal at the cost of breaking the calibration interface.
If elapsed_seconds is ever threaded through to calibration, adding a third
axis is a one-line change to WindowRegime and classify().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature index constants (must match SEQUENCE_FEATURE_NAMES in settings.py) ──
# ["speed_ms", "acceleration_ms2", "heart_rate_bpm", "sprint_flag",
#  "x_pitch", "y_pitch", "distance_delta_m", "hr_recovery_rate"]
# hr_recovery_rate = fractional HR change per tick (hr-prev)/prev ∈ [-1,1];
# not used by the regime classifier (only sprint_flag and x_pitch are).
_IDX_SPRINT_FLAG = 3
_IDX_X_PITCH     = 4

# ── Territory thresholds (normalised pitch units, 0–100) ──────────────────────
_TERRITORY_DEFENSIVE_MAX  = 33.0
_TERRITORY_ATTACKING_MIN  = 67.0

# ── Intensity thresholds (sprint_flag fraction) ───────────────────────────────
_INTENSITY_HIGH_MIN   = 0.15   # ≥ 15 % of window steps are sprints
_INTENSITY_MEDIUM_MIN = 0.04   # 4–15 %: jog-dominated / transition warmup


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WindowRegime:
    """
    Immutable regime label for a single 120-second window.
    The key property is the dict/set key used throughout the threshold store.
    """
    territory: str   # "defensive" | "midfield" | "attacking"
    intensity: str   # "high" | "medium" | "low"

    @property
    def key(self) -> str:
        return f"{self.territory}__{self.intensity}"

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        return f"WindowRegime({self.key!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────
class SessionRegimeClassifier:
    """
    Maps a (T × N_FEATURES) sequence array to a WindowRegime.

    Deliberately stateless and dependency-free: takes only numpy arrays,
    reads no config, holds no state.  Call classify() as many times as needed.

    Example
    -------
    >>> clf = SessionRegimeClassifier()
    >>> regime = clf.classify(sequence)   # shape (T, 8)
    >>> print(regime.key)                 # "midfield__high"
    """

    def classify(self, sequence: np.ndarray) -> WindowRegime:
        """
        Parameters
        ----------
        sequence : np.ndarray, shape (T, N_SEQUENCE_FEATURES)
            Normalised feature matrix for one window.
            Columns must match SEQUENCE_FEATURE_NAMES order in settings.py.

        Returns
        -------
        WindowRegime
        """
        if sequence.ndim != 2 or sequence.shape[1] <= max(_IDX_SPRINT_FLAG, _IDX_X_PITCH):
            logger.warning(
                "SessionRegimeClassifier: unexpected sequence shape %s — returning midfield/medium",
                sequence.shape,
            )
            return WindowRegime(territory="midfield", intensity="medium")

        mean_x       = float(np.nanmean(sequence[:, _IDX_X_PITCH]))
        sprint_frac  = float(np.nanmean(sequence[:, _IDX_SPRINT_FLAG]))

        # Territory
        if mean_x < _TERRITORY_DEFENSIVE_MAX:
            territory = "defensive"
        elif mean_x > _TERRITORY_ATTACKING_MIN:
            territory = "attacking"
        else:
            territory = "midfield"

        # Intensity
        if sprint_frac >= _INTENSITY_HIGH_MIN:
            intensity = "high"
        elif sprint_frac >= _INTENSITY_MEDIUM_MIN:
            intensity = "medium"
        else:
            intensity = "low"

        return WindowRegime(territory=territory, intensity=intensity)


# ─────────────────────────────────────────────────────────────────────────────
# Regime-aware threshold store
# ─────────────────────────────────────────────────────────────────────────────
class RegimeAwareThresholdStore:
    """
    Drop-in replacement for a single DynamicThresholdTracker.

    Maintains one DynamicThresholdTracker per regime key AND a global tracker
    that receives every update.  At query time:

      • If the regime-specific tracker is calibrated → use its threshold.
      • Otherwise fall back to the global tracker.

    This guarantees the store is always usable once the global tracker reaches
    min_calibration_windows, even before rare regimes accumulate enough data.

    Parameters
    ----------
    inner_tracker_cls : callable
        Class or factory that produces a 
        Injected for testability; defaults to the real class.
    cfg : AnomalyScoringConfig | None
        Forwarded to every inner tracker.
    """

    def __init__(self, inner_tracker_cls=None, cfg=None):
        # Deferred import to avoid circular dependency with anomaly_detection
        from config.settings import CONFIG, AnomalyScoringConfig

        if inner_tracker_cls is None:
            from analysis.anomaly_detection import DynamicThresholdTracker
            inner_tracker_cls = DynamicThresholdTracker

        self._cls   = inner_tracker_cls
        self._cfg   = cfg or CONFIG.scoring
        self._global: object  = self._cls(self._cfg)
        self._per_regime: Dict[str, object] = {}

    # ── Write ──────────────────────────────────────────────────────────────
    def update(self, loss: float, regime_key: str) -> None:
        """
        Record a (EMA-smoothed) calibration loss for both the global and
        the regime-specific tracker.

        Parameters
        ----------
        loss : float
            EMA-smoothed reconstruction loss (same transform as infer_live).
        regime_key : str
            WindowRegime.key for this calibration window.
        """
        self._global.update(loss)
        if regime_key not in self._per_regime:
            self._per_regime[regime_key] = self._cls(self._cfg)
        self._per_regime[regime_key].update(loss)

    # ── Read ───────────────────────────────────────────────────────────────
    @property
    def is_calibrated(self) -> bool:
        """True once the global tracker has enough windows."""
        return self._global.is_calibrated

    def threshold_for(self, regime_key: str) -> float:
        """
        Return the threshold appropriate for *regime_key*.
        Falls back to the global threshold if the regime is under-calibrated.
        """
        tracker = self._per_regime.get(regime_key)
        if tracker is not None and tracker.is_calibrated:
            return tracker.threshold
        return self._global.threshold

    def confidence_for(self, loss: float, regime_key: str) -> float:
        """
        Empirical CDF value P(calib_loss ≤ loss) within *regime_key*.
        Falls back to the global tracker if the regime is under-calibrated.
        """
        tracker = self._per_regime.get(regime_key)
        if tracker is not None and tracker.is_calibrated:
            return tracker.confidence(loss)
        return self._global.confidence(loss)

    # ── Diagnostics ────────────────────────────────────────────────────────
    def regime_coverage(self) -> Dict[str, int]:
        """
        Number of calibration windows per regime.
        Useful to identify under-represented regimes that will fall back to global.

        Example
        -------
        {"defensive__low": 42, "midfield__medium": 61, "attacking__high": 12, ...}
        """
        return {key: len(tracker._losses) for key, tracker in self._per_regime.items()}

    def uncalibrated_regimes(self) -> List[str]:
        """
        Regime keys that have windows but not yet enough for a stable threshold.
        These will fall back to global at inference time.
        """
        return [
            key for key, tracker in self._per_regime.items()
            if not tracker.is_calibrated
        ]

    def summary(self) -> str:
        """Human-readable calibration summary for logging."""
        lines = [f"RegimeAwareThresholdStore — global n={len(self._global._losses)}"]
        for key, count in sorted(self.regime_coverage().items(), key=lambda x: -x[1]):
            tracker  = self._per_regime[key]
            calib    = "✓" if tracker.is_calibrated else "fallback"
            thr      = f"{tracker.threshold:.4f}" if tracker.is_calibrated else "–"
            lines.append(f"  {key:<30} n={count:<5} threshold={thr:<10} {calib}")
        return "\n".join(lines)

    # ── Serialisation ──────────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "global":     self._global.state_dict(),
            "per_regime": {k: v.state_dict() for k, v in self._per_regime.items()},
        }

    @classmethod
    def from_state_dict(cls, d: dict, inner_tracker_cls=None, cfg=None) -> "RegimeAwareThresholdStore":
        from analysis.anomaly_detection import DynamicThresholdTracker
        inner_tracker_cls = inner_tracker_cls or DynamicThresholdTracker

        obj = cls(inner_tracker_cls=inner_tracker_cls, cfg=cfg)
        obj._global = inner_tracker_cls.from_state_dict(d["global"])
        obj._per_regime = {
            k: inner_tracker_cls.from_state_dict(v)
            for k, v in d.get("per_regime", {}).items()
        }
        return obj