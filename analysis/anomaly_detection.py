"""
analysis.pattern_analysis
==========================
Pattern Analysis Engine — Anomaly Detection for Players Data (IBM CIC Germany).

Overview
--------
This module is the core machine-learning layer of the Players Data real-time
performance-monitoring system. It detects physiological and tactical anomalies
in player telemetry (GPS, heart rate, acceleration) during training sessions
and live matches.

Architecture
------------
The detection pipeline has four independently useful subsystems:

1. **Shared LSTM Autoencoder** (``SharedBackboneAutoencoder``)
   A single encoder–decoder trained jointly on every registered player.
   Player identity is injected via a learned FiLM (Feature-wise Linear
   Modulation) embedding so the shared latent space can still capture
   individual physiological baselines. At inference, the per-player
   reconstruction loss is compared against a *per-player, per-regime*
   threshold calibrated on a held-out calibration split.

2. **Transformer Autoencoder** (``TransformerAutoencoder``)
   Experimental alternative. Validity-weighted pooling in the bottleneck
   prevents padded windows from contaminating the latent representation.
   Requires ≥ 30 sessions per player. Disabled in production.

3. **Fatigue Curve Comparator**
   Built into ``PatternAnalysisEngine.analyze()``. Compares live speed
   and sprint rate against the player's personal baseline and flags
   physiological decline when the anomaly detector also fires.

4. **Positional Drift Scorer** (``PositionalDriftAnalyzer``)
   Detects tactical zone violations by comparing recent GPS positions
   against the player's historical positional centroid and spread.

Technical Novelty
-----------------
All thresholds and baselines are fitted on **per-player** historical data,
not squad averages. This is the core innovation: a goalkeeper's "low speed"
is structurally different from a striker's, and squad-level statistics would
produce systematically biased alerts for positional outliers.

Coordinate System
-----------------
Pitch coordinates are normalised to [0, 100] on both axes. Internally,
``SequenceWindowBuilder`` rescales to metres using standard FIFA dimensions
(105 m × 68 m) so that ``distance_delta`` and drift thresholds share the
same unit. All model features and threshold quantities are in SI units unless
otherwise stated in their docstrings.

Sequence Features (``SEQUENCE_FEATURE_NAMES``)
----------------------------------------------
Index  Name                Description
-----  ------------------  ---------------------------------------------------
0      speed_ms            Instantaneous speed in m/s
1      accel               Acceleration in m/s², clamped to ±10 m/s²
2      heart_rate_bpm      Heart rate in beats per minute
3      sprint_flag         Binary; 1 if speed_ms ≥ SPRINT_THRESHOLD_MS
4      x_pitch             Normalised x coordinate [0, 100]
5      y_pitch             Normalised y coordinate [0, 100]
6      distance_delta      True Euclidean displacement since last tick (metres)
7      hr_recovery         Fractional HR change per tick, clipped to [-1, 1]

Dependencies
------------
- PyTorch ≥ 2.0 (optional; graceful stub mode if absent)
- scikit-learn ≥ 1.3 (optional; ROC-AUC / PR-AUC disabled if absent)
- NumPy, Pandas (required)
- tqdm (optional; progress bars during shared-backbone training)

Thread Safety
-------------
``PatternAnalysisEngine`` is **not** thread-safe. The ``_ema_scores`` and
``_position_buffers`` dictionaries are mutated during ``analyze()``; callers
must serialize access (e.g. one engine per asyncio event loop or per process).
``SequenceWindowBuilder`` is similarly not thread-safe across player IDs.

Usage Example
-------------
::

    engine = PatternAnalysisEngine()
    for pid, baseline in player_baselines.items():
        engine.register_player(pid, baseline)

    # One-time training pass
    all_windows = {pid: engine.build_training_sequences(events_df, sessions_df)
                   for pid in player_ids}
    engine.train_player_model(all_windows)

    # Live inference
    result = engine.analyze(player_id=42, live_event=event_dict, sessions_df=df)
    if result and result.is_anomaly:
        trigger_coach_alert(result)
"""
from __future__ import annotations

import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from functools import wraps
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    from tqdm import tqdm as _tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
from utils.ema import EMASmoother
from utils.alert_manager import AlertManager, AlertLevel
from utils.reliability.calibration_store import HardenedRollingThresholdStore
from config.settings import LSTMAutoencoderConfig
import numpy as np
import pandas as pd
from utils.evaluation.episodes import (
    extract_episodes,
    match_episodes,
)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from torch.serialization import add_safe_globals

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from analysis.baseline import PlayerBaselineProfile, WorkloadTrendTracker
from analysis.regime import SessionRegimeClassifier, RegimeAwareThresholdStore
from config.settings import (
    CONFIG,
    SEQUENCE_FEATURE_NAMES, N_SEQUENCE_FEATURES,
    LSTMAutoencoderConfig, TransformerAutoencoderConfig,
    AnomalyScoringConfig, PositionalDriftConfig,
)

# Module-level regime classifier — stateless, safe to share across all callers.
_REGIME_CLASSIFIER = SessionRegimeClassifier()

logger = logging.getLogger(__name__)

MODEL_STORE = Path("./models")
MODEL_STORE.mkdir(parents=True, exist_ok=True)

SPRINT_THRESHOLD_MS = 7.0

# Standard FIFA pitch dimensions used for coordinate → metre conversion.
# x_pitch and y_pitch are normalised [0, 100]; these scale each axis back to metres
# so that distance_delta is geometrically correct and unit-consistent with
# drift thresholds and workload metrics that are expressed in metres.
PITCH_LENGTH_M = 105.0   # y-axis (goal-to-goal)
PITCH_WIDTH_M = 68.0    # x-axis (touchline-to-touchline)


def safe_float(v, default: float) -> float:
    """Convert an arbitrary value to a finite ``float``, falling back to *default*.

    This is a defensive utility used throughout the feature-engineering
    pipeline to handle sensor drop-outs, JSON ``null`` values, and
    IEEE-754 special values (``NaN``, ``±Inf``) without raising exceptions.

    Parameters
    ----------
    v:
        The value to convert.  Any type accepted; conversion is attempted
        via ``float(v)``.
    default:
        Returned when *v* is ``None``, non-numeric, ``NaN``, or infinite.

    Returns
    -------
    float
        A finite floating-point number.

    Examples
    --------
    >>> safe_float(None, 0.0)
    0.0
    >>> safe_float(float("nan"), -1.0)
    -1.0
    >>> safe_float("3.14", 0.0)
    3.14
    """
    if v is None:
        return default

    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────
def set_deterministic(seed: int = 42) -> None:
    """Seed all random-number generators for reproducible training runs.

    Sets seeds on NumPy, PyTorch (CPU and all CUDA devices), and disables
    cuDNN auto-tuning so convolution algorithms are deterministic.

    Parameters
    ----------
    seed:
        Integer seed value.  Defaults to ``42``.

    Notes
    -----
    This function is a no-op when PyTorch is not installed
    (``TORCH_AVAILABLE`` is ``False``).

    cuDNN determinism (``torch.backends.cudnn.deterministic = True``) may
    reduce throughput on some GPUs.  Acceptable for training; do not call
    during latency-critical inference paths.
    """
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
def resolve_device() -> "torch.device":
    """Select the best available compute device for PyTorch operations.

    Priority order: CUDA GPU → Apple MPS (Apple Silicon) → CPU.

    Returns
    -------
    torch.device or None
        The selected device, or ``None`` when PyTorch is not available.

    Notes
    -----
    When CUDA is available the device properties (name and VRAM) are logged
    at ``INFO`` level so deployment logs show which GPU was selected.
    """
    if not TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info("Device: CUDA (%s, %.1f GB VRAM)",
                    props.name, props.total_memory / 1e9)
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device: MPS (Apple Silicon)")
        return torch.device("mps")
    logger.info("Device: CPU")
    return torch.device("cpu")


DEVICE = resolve_device()
set_deterministic()


def to_device(t: "torch.Tensor") -> "torch.Tensor":
    """Move a tensor to the globally resolved compute device.

    Parameters
    ----------
    t:
        The tensor to move.

    Returns
    -------
    torch.Tensor
        The same tensor on ``DEVICE``.  If ``DEVICE`` is ``None``
        (PyTorch absent) the tensor is returned unchanged.
    """
    return t.to(DEVICE) if DEVICE is not None else t


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AnomalyResult:
    """Structured output of a single anomaly-detection inference pass.

    Produced by ``PatternAnalysisEngine.analyze()`` for every player window
    that passes the minimum sequence-length으로 threshold.

    Attributes
    ----------
    player_id:
        Internal integer player identifier (database primary key).
    external_id:
        Human-readable or federation player identifier (e.g. FIFA ID).
    ts:
        UTC timestamp of the event that closed this window.
    anomaly_score:
        Raw per-player reconstruction loss from the autoencoder.
        Higher values indicate greater deviation from the player's personal
        baseline.  Compare against ``DynamicThresholdTracker.threshold``
        or the regime-specific threshold in ``RegimeAwareThresholdStore``.
    is_anomaly:
        ``True`` when the EMA-smoothed ``anomaly_score`` exceeds the
        calibrated threshold for the current tactical regime.
    confidence:
        Empirical percentile rank P(calibration_loss ≤ anomaly_score) in
        [0.0, 1.0].  Intended as a human-readable severity indicator;
        *not* a Bayesian posterior.
    feature_vector:
        Flat dict of all features visible to the model at this step,
        including enriched fatigue/workload features from upstream.
        Used by the XAI layer (SHAP).
    sequence_shape:
        ``(window_steps, n_features)`` — shape of the raw input window.
    deviations:
        Optional per-feature deviation breakdown (populated by XAI layer).
    raw_sequence:
        Un-normalised ``(T, F)`` float32 array — the exact window forwarded
        through the model.  Stored so the SHAP explainer can perturb real
        sequences without re-building them from scratch.
    raw_mask:
        Boolean ``(T,)`` validity mask aligned with ``raw_sequence``.
        ``True`` = real telemetry; ``False`` = zero-padded missing tick.
    fatigue_flag:
        ``True`` when the player shows a physiological decline pattern
        (speed/sprint below 55 % of personal average, late in match) *and*
        the model also flags an anomaly.
    positional_drift_flag:
        ``True`` when ``PositionalDriftAnalyzer`` reports that the player
        has spent a significant fraction of recent time outside their
        historical positional zone.
    workload_flag:
        ``True`` when the acute:chronic workload ratio (ACWR) is outside
        the safe band [0.8, 1.5].
    workload_status:
        Human-readable ACWR category: ``"optimal"``, ``"high"``, ``"low"``,
        ``"very_high"``, etc.  Set by ``WorkloadTrendTracker``.
    recommendation_type:
        Highest-priority coaching action from ``PatternAnalysisEngine._recommend()``.
        One of: ``"substitution"``, ``"fatigue_alert"``,
        ``"positional_drift"``, ``"workload_warning"``,
        ``"anomaly_flag"``, or ``None``.
    triggered_at:
        UTC wall-clock time the result object was created (for audit logs).
    model_type:
        Identifier of the model that produced the score, e.g.
        ``"shared_lstm"``, ``"transformer"``, or ``"none"``.

    Operational Determinism Fields
    ------------------------------
    model_version:
        The exact git hash or version ID of the weights used.
    calibration_version:
        The version of the thresholds applied to this result.
    episode_id:
        Monotonic counter for the current alert episode to prevent replay divergence.
    telemetry_confidence:
        Score from TVL at the time of inference.
    inference_timestamp:
        The exact moment processing occurred.
    """
    player_id: int
    external_id: str
    ts: datetime
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    
    feature_vector: Dict[str, float] = field(default_factory=dict)
    sequence_shape: Tuple[int, int] = (0, 0)
    deviations: Dict[str, dict] = field(default_factory=dict)

    raw_sequence: Optional[np.ndarray] = field(default=None, repr=False)
    raw_mask:     Optional[np.ndarray] = field(default=None, repr=False)

    fatigue_flag: bool = False
    positional_drift_flag: bool = False
    workload_flag: bool = False
    workload_status: str = "optimal"

    recommendation_type: Optional[str] = None
    triggered_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    model_type: str = "lstm"

    # Determinism binding
    # Determinism binding
    model_version: str = "unknown"
    calibration_version: int = 0
    episode_id: int = 0
    telemetry_confidence: float = 1.0
    inference_timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc))

    # Alert state — populated by PatternAnalysisEngine.analyze()
    alert_level: AlertLevel = AlertLevel.NONE
    persistence_windows: int = 0

    # XAI / NLG — populated by orchestrator after inference
    nlg_summary: str = ""
    shap_values: Dict[str, float] = field(default_factory=dict)
    counterfactual: Optional[str] = None
    top_contributions: List = field(default_factory=list)  # List[FeatureContribution]

    # Async NLG sidecar — set when nlg_async=True so main.py can dispatch
    # generate_explanation_from_base() to the NLG thread pool without
    # blocking the SLA-measured inference path.
    base_explanation: Optional[object] = field(default=None, repr=False)
    semantic_state: Optional[object] = field(default=None, repr=False)
    # Raw kwargs for async build_base_explanation (SHAP + semantics + LLM).
    # Set instead of base_explanation when nlg_async=True so that all
    # 10 MPS forward passes happen off the SLA clock.
    _xai_kwargs: Optional[dict] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Per-player normaliser  (plain-array serialisation)
# ─────────────────────────────────────────────────────────────────────────────
class PerPlayerNormaliser:
    """Z-score normaliser fitted and applied per player (or per dataset split).

    Stores per-feature means and standard deviations as plain NumPy arrays
    so the state can be serialised to JSON-compatible dicts without pickle
    dependency on sklearn, making checkpoint files more portable.

    Features with a standard deviation below 1e-6 (effectively constant
    across the fitting data) are assigned a scale of 1.0 to prevent
    division-by-zero and NaN propagation.

    After ``transform()``, any residual NaN/±Inf values — which can arise
    from hardware faults or encoding bugs upstream — are replaced with 0.0
    so they do not contaminate the model's gradient computation.

    Attributes
    ----------
    means:
        (F,) float32 array of per-feature means.  ``None`` before fitting.
    stds:
        (F,) float32 array of per-feature standard deviations (minimum
        clamped to 1.0).  ``None`` before fitting.
    """

    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds:  Optional[np.ndarray] = None

    def fit(self, sequences: np.ndarray) -> None:
        """Compute per-feature statistics from a set of training sequences.

        Parameters
        ----------
        sequences:
            Array of shape ``(N, T, F)`` where N is the number of windows,
            T is the sequence length, and F is the number of features.
            The array is flattened along the first two axes so statistics
            are computed across all timesteps and all windows.

        Notes
        -----
        Must be called on the **training split only**.  Applying it to the
        full dataset (including validation / calibration) would constitute
        data leakage.
        """
        flat = sequences.reshape(-1, sequences.shape[-1])
        self.means = flat.mean(axis=0).astype(np.float32)
        raw_std = flat.std(axis=0).astype(np.float32)
        self.stds = np.where(raw_std > 1e-6, raw_std, 1.0).astype(np.float32)

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        """Apply Z-score normalisation to a sequence array.

        Parameters
        ----------
        sequences:
            Array of shape ``(N, T, F)`` or ``(T, F)`` containing raw
            (un-normalised) feature values.

        Returns
        -------
        numpy.ndarray
            Float32 array of the same shape with zero mean and unit variance
            per feature (relative to the fitting distribution).  All NaN and
            ±Inf values are replaced by 0.0.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called yet.
        """
        if self.means is None:
            raise RuntimeError("Normaliser not fitted.")
        out = ((sequences - self.means) / self.stds).astype(np.float32)

        out = np.nan_to_num(
            out,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return out

    def state_dict(self) -> dict:
        """Serialise normaliser state to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Keys ``"means"`` and ``"stds"``, each a Python list of floats
            (or ``None`` if not yet fitted).
        """
        return {
            "means": self.means.tolist() if self.means is not None else None,
            "stds":  self.stds.tolist() if self.stds is not None else None,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "PerPlayerNormaliser":
        """Reconstruct a ``PerPlayerNormaliser`` from a serialised state dict.

        Parameters
        ----------
        d:
            Dictionary previously returned by ``state_dict()``.

        Returns
        -------
        PerPlayerNormaliser
            A fitted normaliser instance ready for ``transform()`` calls.
        """
        obj = cls()
        obj.means = np.array(
            d["means"], dtype=np.float32) if d["means"] else None
        obj.stds = np.array(
            d["stds"],  dtype=np.float32) if d["stds"] else None
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic threshold tracker
# ─────────────────────────────────────────────────────────────────────────────
class DynamicThresholdTracker:
    """Calibration-split loss store that computes anomaly thresholds robustly.

    The tracker is populated **once** during the calibration phase after
    training (``update()``), and is never updated during inference.  This
    strict separation ensures that live anomaly scores do not shift the
    threshold distribution — a common source of threshold drift in online
    systems.

    Threshold Computation
    ---------------------
    Two regimes are supported, selected automatically by sample count:

    *Large calibration set* (≥ ``large_calib_threshold`` samples, default 150):
        ``threshold = np.quantile(clean_losses, threshold_quantile)``
        where ``clean_losses`` has the top ``calib_contamination_pct``
        (default 5 %) trimmed to remove calibration-split anomalies that
        would inflate the threshold and reduce sensitivity.

    *Small calibration set* (< 150 samples):
        ``threshold = median + k × (MAD × 1.4826)``
        where k = ``mad_multiplier`` (default 4.0).  MAD-based thresholds
        are more robust than quantile estimates when sample counts are in
        the tens, because quantiles at the 99th percentile of a 30-sample
        distribution have very high variance.

    Operational Override
    --------------------
    ``select_operational_threshold()`` replaces the computed threshold with
    one calibrated to a *target false-positive rate* (FP per 90-minute
    match).  When set, this overrides all subsequent calls to ``.threshold``.
    This is the recommended approach for production deployment because it
    ties the threshold to an operationally meaningful budget, not an
    arbitrary statistical percentile.

    Confidence Score
    ----------------
    ``confidence(loss)`` returns the empirical CDF value
    P(calibration_loss ≤ loss).  This is a percentile rank, not a
    probability in the Bayesian sense.  It is useful as a human-readable
    severity indicator for coaching staff.

    Parameters
    ----------
    cfg:
        ``AnomalyScoringConfig`` instance.  Defaults to ``CONFIG.scoring``.
    """

    def __init__(self, cfg: AnomalyScoringConfig = None):
        self.cfg = cfg or CONFIG.scoring
        self._losses: List[float] = []

    def update(self, loss: float) -> None:
        """Append one calibration-split reconstruction loss to the store.

        Parameters
        ----------
        loss:
            Scalar reconstruction loss from the autoencoder on a
            calibration-split window.

        Notes
        -----
        This method must **only** be called during the calibration phase
        (``train()`` or ``recalibrate()``).  Calling it with live inference
        losses will corrupt the threshold distribution.
        """
        self._losses.append(float(loss))

    def _clean_losses(self) -> np.ndarray:
        """Return calibration losses with the upper contamination tail removed.

        The calibration split may contain anomalous windows (injected
        anomalies, hardware faults, bad sessions).  Trimming the top
        ``calib_contamination_pct`` prevents these outliers from inflating
        the threshold and reducing detection sensitivity.

        Returns
        -------
        numpy.ndarray
            1-D float64 array.  Falls back to the full loss array if
            trimming leaves fewer than ``min_calibration_windows // 2``
            samples (minimum 5).
        """
        arr = np.array(self._losses, dtype=np.float64)
        cutoff = np.quantile(
            arr, 1.0 - getattr(self.cfg, "calib_contamination_pct", 0.05))
        clean = arr[arr <= cutoff]
        min_n = max(self.cfg.min_calibration_windows // 2, 5)
        return clean if len(clean) >= min_n else arr

    @property
    def is_calibrated(self) -> bool:
        """``True`` when enough calibration losses have been collected.

        The minimum required count is ``cfg.min_calibration_windows``
        (default 30).  Before this threshold is reached the tracker returns
        ``float("inf")`` as its threshold, effectively disabling anomaly
        detection for newly registered players.
        """
        return len(self._losses) >= self.cfg.min_calibration_windows

    @property
    def threshold(self) -> float:
        """Anomaly detection threshold derived from the calibration distribution.

        When an operational threshold has been set via
        ``select_operational_threshold()``, that value is returned directly.

        Returns
        -------
        float
            The detection threshold.  Returns ``float("inf")`` when the
            tracker is not yet calibrated.

        See Also
        --------
        select_operational_threshold : Set a PR-curve-derived threshold.
        """
        op_thr = getattr(self, "_operational_threshold", None)
        if op_thr is not None:
            return op_thr
        if not self.is_calibrated:
            return float("inf")

        # Conformal Prediction Logic
        # Use a calibrated epsilon to guarantee a la specific False Positive Rate.
        # threshold = quantile(losses, 1 - alpha)
        clean = self._clean_losses()
        large_n = getattr(self.cfg, "large_calib_threshold", 150)

        if len(clean) >= large_n:
            # High-confidence quantile estimate
            q = getattr(self.cfg, "threshold_quantile", 0.995)
            return float(np.quantile(clean, q))
        else:
            # Robust fallback for small samples (MAD-based)
            # Threshold = Median + k * (MAD * 1.4826)
            # This is effectively a quantile-like estimate for smaller distributions.
            median = float(np.median(clean))
            mad = float(np.median(np.abs(clean - median)))
            k = getattr(self.cfg, "mad_multiplier", 4.0)
            return median + k * (mad * 1.4826)

    def confidence(self, loss: float) -> float:
        """Compute the empirical CDF value for a given reconstruction loss.

        Parameters
        ----------
        loss:
            Reconstruction loss to rank against the calibration distribution.

        Returns
        -------
        float
            P(calibration_loss ≤ loss) in [0.0, 1.0].  Returns 0.0 when
            the tracker is not yet calibrated.
        """
        if not self.is_calibrated:
            return 0.0
        return float(np.mean(np.array(self._losses) <= loss))

    def select_operational_threshold(
        self,
        eval_scores:          np.ndarray,
        eval_labels:          np.ndarray,
        target_fp_per_90_min: float = 2.0,
        window_interval_s:    float = 120.0,
    ) -> float:
        """Replace the quantile threshold with one calibrated to a FP budget.

        This is the preferred production threshold-selection strategy.
        Rather than choosing an arbitrary quantile of the calibration loss
        distribution, it finds the *tightest* threshold that keeps the
        false-positive volume at or below ``target_fp_per_90_min`` spurious
        alerts per 90-minute match.

        The selected threshold overrides all subsequent calls to ``.threshold``.

        Parameters
        ----------
        eval_scores:
            (N,) float array of reconstruction losses from a *labeled*
            held-out evaluation set (not the training or calibration splits).
        eval_labels:
            (N,) int array with 1 = anomalous window, 0 = normal.
        target_fp_per_90_min:
            Maximum tolerable false alerts per 90-minute period.
            Typical coaching requirement: 2–5 false alerts per match.
        window_interval_s:
            Seconds between successive windows (step size used to convert
            window-level FP counts to per-90-minute rates).

        Returns
        -------
        float
            The selected threshold value (also stored internally).

        Notes
        -----
        Delegates threshold selection to the module-level
        ``_pr_curve_threshold()`` helper which walks the PR curve from the
        tightest to the loosest threshold.

        Requires scikit-learn.  Falls back to ``median(scores)`` if sklearn
        is absent.
        """
        thr = _pr_curve_threshold(
            eval_scores, eval_labels,
            target_fp_per_90_min=target_fp_per_90_min,
            window_interval_s=window_interval_s,
        )
        self._operational_threshold = thr
        logger.info(
            "Operational threshold set from PR curve: %.6f "
            "(target FP/90min=%.1f, window_interval=%.0fs)",
            thr, target_fp_per_90_min, window_interval_s,
        )
        return thr

    def state_dict(self) -> dict:
        """Serialise tracker state to a JSON-compatible dictionary.

        Returns
        -------
        dict
            Always contains key ``"losses"`` (list of floats).  Contains
            ``"operational_threshold"`` only when it has been explicitly set
            via ``select_operational_threshold()``.
        """
        d = {"losses": list(self._losses)}
        op = getattr(self, "_operational_threshold", None)
        if op is not None:
            d["operational_threshold"] = op
        return d

    @classmethod
    def from_state_dict(cls, d: dict,
                        cfg: AnomalyScoringConfig = None) -> "DynamicThresholdTracker":
        """Reconstruct a ``DynamicThresholdTracker`` from a serialised state dict.

        Parameters
        ----------
        d:
            Dictionary previously returned by ``state_dict()``.
        cfg:
            Optional scoring configuration.  Defaults to ``CONFIG.scoring``.

        Returns
        -------
        DynamicThresholdTracker
            Fully restored tracker, including the operational threshold
            override if one was saved.
        """
        obj = cls(cfg)
        obj._losses = list(d.get("losses", []))
        op = d.get("operational_threshold")
        if op is not None:
            obj._operational_threshold = float(op)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Sequence window builder  (returns (window, mask) pairs)
# ─────────────────────────────────────────────────────────────────────────────
class SequenceWindowBuilder:
    """Sliding-window builder that converts raw telemetry events into model inputs.

    Each call to ``add_event()`` appends one telemetry tick to a per-player
    circular buffer.  When the buffer reaches ``window_steps`` ticks, a
    ``(sequence, mask)`` pair is returned; otherwise ``None`` is returned.

    Mask Convention
    ---------------
    ``mask[t] = True``  → real telemetry data from the sensor.
    ``mask[t] = False`` → zero-padded placeholder (dropped packet or missing
                          event).

    Padded timesteps contribute zeros to the feature vector so the input
    tensor has a fixed shape.  The mask is forwarded to the autoencoder and
    the loss function so that padded positions are excluded from gradient
    updates and anomaly scoring.

    Feature Engineering
    -------------------
    Eight features are extracted per tick (see module docstring for the
    full table).  Derived features (``accel``, ``hr_recovery``,
    ``distance_delta``) are computed as deltas relative to the previous
    real tick.  First-tick deltas are set to 0.0.

    - ``accel`` is clamped to ±10 m/s² to suppress sensor spikes.
    - ``hr_recovery`` is expressed as a fractional HR change per tick
      (clipped to [−1, 1]) rather than raw bpm/s, which has extremely
      high inter-tick variance.
    - ``distance_delta`` uses true Euclidean displacement in metres with
      axis-specific scaling (PITCH_WIDTH_M for x, PITCH_LENGTH_M for y)
      to correct the geometric distortion of a non-square pitch.

    Attributes
    ----------
    window_steps:
        Number of ticks per window (from ``CONFIG.window.window_steps``).
    event_interval_s:
        Expected time between ticks in seconds
        (from ``CONFIG.window.event_interval_s``).

    Notes
    -----
    ``add_event()`` and ``build_from_session()`` maintain *separate* internal
    state.  ``build_from_session()`` creates fresh local buffers on each
    call and does not affect the streaming buffers used by ``add_event()``.

    Not thread-safe.  A separate instance should be used per worker process.
    """

    def __init__(self):
        self._buffers:      Dict[str, deque] = {}
        self._mask_buffers: Dict[str, deque] = {}
        self._prev_events:  Dict[str, dict] = {}
        cfg = CONFIG.window
        self.window_steps = cfg.window_steps
        self.event_interval_s = cfg.event_interval_s
        self.stride = self.window_steps
        self._tick_counters: Dict[str, int] = {}

    def add_event(
        self, event: dict
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Process one telemetry event and return a window when the buffer is full.

        Parameters
        ----------
        event:
            Dictionary containing at minimum:
            ``player_external_id`` (str), ``speed_ms`` (float or None),
            ``heart_rate_bpm`` (float or None), ``x_pitch`` (float),
            ``y_pitch`` (float).  Missing/None sensor readings mark the
            tick as padded (``mask = False``).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray) or None
            When the buffer contains exactly ``window_steps`` ticks:
            a tuple of ``(sequence, mask)`` where sequence has shape
            ``(window_steps, N_SEQUENCE_FEATURES)`` (float32) and mask
            has shape ``(window_steps,)`` (bool).

            Returns ``None`` for every event before the buffer is full.
        """
        pid = event.get("player_external_id", "")
        buf = self._buffers.setdefault(
            pid,      deque(maxlen=self.window_steps))
        mbuf = self._mask_buffers.setdefault(
            pid, deque(maxlen=self.window_steps))
        prev = self._prev_events.get(pid)

        is_real = (event.get("speed_ms") is not None
                   and event.get("heart_rate_bpm") is not None)
        fv = (self._extract(event, prev)
              if is_real
              else np.zeros(N_SEQUENCE_FEATURES, dtype=np.float32))

        self._prev_events[pid] = event if is_real else prev
        buf.append(fv)
        mbuf.append(is_real)

        if len(buf) == self.window_steps:
            return (
                np.array(list(buf),  dtype=np.float32),
                np.array(list(mbuf), dtype=bool),
            )
        return None

    def _extract(self, event: dict, prev: Optional[dict]) -> np.ndarray:
        """Extract an 8-dimensional feature vector from one telemetry tick.

        Computes instantaneous and delta features from the current event
        and the previous real event.  All derived features default to 0.0
        for the first event in a session (``prev`` is ``None``).

        Parameters
        ----------
        event:
            Current telemetry event dictionary.
        prev:
            Previous real telemetry event dictionary, or ``None`` if this
            is the first tick for the player.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(N_SEQUENCE_FEATURES,)`` with all
            NaN/±Inf values replaced by 0.0.

        Notes
        -----
        ``distance_delta`` is a true Euclidean displacement in metres,
        *not* ``speed × dt``.  The distinction matters because the pitch
        coordinate system is non-square: the x-axis spans 68 m and the
        y-axis spans 105 m, each requiring a different scale factor.
        Using a single scale factor introduces a ~35 % systematic geometric
        error on the longer axis.
        """
        speed = safe_float(event.get("speed_ms"), 0.0)
        hr = safe_float(event.get("heart_rate_bpm"), 0.0)
        x = safe_float(event.get("x_pitch"), 50.0)
        y = safe_float(event.get("y_pitch"), 50.0)

        sprint = 1.0 if speed >= SPRINT_THRESHOLD_MS else 0.0
        dt = self.event_interval_s

        if prev is not None:
            prev_speed = safe_float(prev.get("speed_ms"), 0.0)
            prev_hr = safe_float(prev.get("heart_rate_bpm"), hr)
            prev_x = safe_float(prev.get("x_pitch"), x)
            prev_y = safe_float(prev.get("y_pitch"), y)

            accel = (speed - prev_speed) / dt
            accel = float(np.clip(accel, -10.0, 10.0))

            if prev_hr > 0:
                hr_recovery = float(
                    np.clip((prev_hr - hr) / max(prev_hr, 1.0), -1.0, 1.0))
            else:
                hr_recovery = 0.0

            dx_m = (x - prev_x) / 100.0 * PITCH_WIDTH_M
            dy_m = (y - prev_y) / 100.0 * PITCH_LENGTH_M
            distance_delta = math.sqrt(dx_m * dx_m + dy_m * dy_m)
        else:
            accel = 0.0
            hr_recovery = 0.0
            distance_delta = 0.0

        features = np.array(
            [
                speed,
                accel,
                hr,
                sprint,
                x,
                y,
                distance_delta,
                hr_recovery,
            ],
            dtype=np.float32,
        )

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def build_from_session(
        self,
        events_df: pd.DataFrame,
        stride: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Build sliding windows from a session DataFrame.

        Uses configurable stride to reduce overlap inflation and
        dramatically improve runtime + operational realism.

        Parameters
        ----------
        events_df:
            Single-session events DataFrame.

        stride:
            Window stride. If None, defaults to 50% overlap.

        Returns
        -------
        list of (sequence, mask)
        """
        if events_df.empty or "speed_ms" not in events_df.columns:
            return []

        events_df = (
            events_df
            .sort_values("ts")
            .reset_index(drop=True)
        )

        if stride is None:
            stride = self.window_steps

        n = len(events_df)

        if n < self.window_steps:
            return []

        # ── Pre-extract rows once ─────────────────────────────────────
        feature_rows = []
        mask_rows = []
        prev_real = None

        rows = events_df.to_dict("records")
        feature_rows = []
        mask_rows = []
        prev_real = None

        for row in rows:

            is_real = (
                row.get("speed_ms") is not None
                and row.get("heart_rate_bpm") is not None
            )

            if is_real:
                fv = self._extract(
                    row,
                    prev_real,
                )
                prev_real = row
            else:
                fv = np.zeros(
                    N_SEQUENCE_FEATURES,
                    dtype=np.float32,
                )

            feature_rows.append(fv)
            mask_rows.append(is_real)

        feature_arr = np.asarray(
            feature_rows,
            dtype=np.float32,
        )
        mask_arr = np.asarray(
            mask_rows,
            dtype=bool,
        )
        # ── Sliding windows with reduced overlap ─────────────────────
        results = []

        total_rows = len(feature_arr)

        for start in range(
            0,
            total_rows - self.window_steps + 1,
            stride,
        ):
            end = start + self.window_steps
            seq = feature_arr[start:end]
            msk = mask_arr[start:end]

            if seq.shape[0] != self.window_steps:
                continue

            results.append((
                seq,
                msk,
            ))
        return results
    

    def build_live_window(
        self,
        events: List[dict],
        prev_event: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a completed accumulator window into model inputs.
    
        Stateless: reads ``self.window_steps`` and calls ``self._extract()``,
        but never touches ``self._buffers``, ``self._mask_buffers``, or
        ``self._prev_events``.
    
        Parameters
        ----------
        events:
            Ordered list of raw telemetry dicts emitted by
            ``LiveWindowAccumulator.push()``.  Length should equal
            ``self.window_steps``; longer lists are right-truncated, shorter
            lists are left-padded with zero rows and False mask entries.
        prev_event:
            Optional: the last real event from the *previous* window, used
            to compute delta features for ``events[0]`` across the window
            boundary.  Pass ``None`` (default) to accept delta = 0.0 at the
            boundary — which is the safe, stateless default.
    
        Returns
        -------
        seq : np.ndarray, shape (window_steps, N_SEQUENCE_FEATURES), float32
            Feature matrix ready for the autoencoder forward pass.
        mask : np.ndarray, shape (window_steps,), bool
            True for real sensor ticks, False for zero-padded positions.
        """
        window_steps = self.window_steps  # number of timesteps (e.g. 24)
    
        # ── Normalise length ──────────────────────────────────────────────────────
        if len(events) > window_steps:
            events = events[-window_steps:]          # right-truncate
        pad_count = window_steps - len(events)       # left-pad deficit
    
        feature_rows: List[np.ndarray] = []
        mask_rows:    List[bool]        = []
    
        # ── Left-pad with zero rows (masked out) ──────────────────────────────────
        for _ in range(pad_count):
            feature_rows.append(np.zeros(N_SEQUENCE_FEATURES, dtype=np.float32))
            mask_rows.append(False)
    
        # ── Extract features tick by tick ─────────────────────────────────────────
        # prev_real tracks the last real event *within this call* so delta
        # features are consistent inside the window.  It starts from prev_event
        # (cross-boundary continuity) or None (delta = 0.0 at t=0).
        prev_real: Optional[dict] = prev_event
    
        for event in events:
            is_real = (
                event.get("speed_ms")        is not None
                and event.get("heart_rate_bpm") is not None
            )
            if is_real:
                fv = self._extract(event, prev_real)   # (N_SEQUENCE_FEATURES,)
                prev_real = event
            else:
                fv = np.zeros(N_SEQUENCE_FEATURES, dtype=np.float32)
    
            feature_rows.append(fv)
            mask_rows.append(is_real)
    
        seq  = np.asarray(feature_rows, dtype=np.float32)   # (window_steps, N_SEQUENCE_FEATURES)
        mask = np.asarray(mask_rows,    dtype=bool)          # (window_steps,)
    
        # Sanity check — catches future N_SEQUENCE_FEATURES drift early
        assert seq.shape == (window_steps, N_SEQUENCE_FEATURES), (
            f"build_live_window: expected shape ({window_steps}, {N_SEQUENCE_FEATURES}), "
            f"got {seq.shape}.  Check N_SEQUENCE_FEATURES constant."
        )
    
        return seq, mask
    


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch modules
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:

    class _LSTMEncoder(nn.Module):
        """LSTM encoder with optional pack_padded_sequence for masked inputs.

        Encodes a variable-length (masked) sequence into a fixed-size latent
        vector using an LSTM followed by a linear projection.

        When ``lengths`` is supplied, the encoder uses
        ``pack_padded_sequence`` so that padded (zero) timesteps do not
        update the LSTM hidden state.  Without packing, padded zeros shift
        the latent embedding and make incomplete windows appear anomalous
        regardless of the player's actual physiology.

        Parameters
        ----------
        n_features:
            Number of input features per timestep.
        hidden:
            LSTM hidden state dimensionality.
        n_layers:
            Number of stacked LSTM layers.  Dropout is applied between
            layers only when ``n_layers > 1``.
        latent:
            Dimensionality of the output latent vector.
        dropout:
            Dropout rate applied between LSTM layers.
        """

        def __init__(self, n_features: int, hidden: int, n_layers: int,
                     latent: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, hidden, n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden, latent)

        def forward(
            self,
            x: "torch.Tensor",
            lengths: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """Encode a batch of sequences into latent vectors.

            Parameters
            ----------
            x:
                Input tensor of shape ``(B, T, F)``.
            lengths:
                Optional 1-D integer tensor of shape ``(B,)`` containing
                the number of real (non-padded) timesteps per sample.
                When provided, ``pack_padded_sequence`` is used.

            Returns
            -------
            torch.Tensor
                Latent tensor of shape ``(B, latent)`` with values squashed
                through ``tanh`` to the range [−1, 1].
            """
            if lengths is not None:
                safe_len = lengths.cpu().clamp(min=1)
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, safe_len, batch_first=True, enforce_sorted=False
                )
                _, (h_n, _) = self.lstm(packed)
            else:
                _, (h_n, _) = self.lstm(x)

            h_last = h_n[-1]
            return torch.tanh(self.fc(h_last))

    class _LSTMDecoder(nn.Module):
        """LSTM decoder that reconstructs a sequence from a latent vector.

        Replicates the latent vector across all ``seq_len`` timesteps and
        passes the result through a multi-layer LSTM, using the projected
        latent vector as the initial hidden state.

        Parameters
        ----------
        latent:
            Dimensionality of the input latent vector (encoder output).
        hidden:
            LSTM hidden state dimensionality (must match encoder).
        n_layers:
            Number of stacked LSTM layers.
        n_features:
            Number of output features per timestep (must match encoder input).
        seq_len:
            Fixed output sequence length.
        dropout:
            Dropout rate applied between LSTM layers.
        """

        def __init__(self, latent: int, hidden: int, n_layers: int,
                     n_features: int, seq_len: int, dropout: float):
            super().__init__()
            self.seq_len = seq_len
            self.fc_in = nn.Linear(latent, hidden)
            self.n_layers = n_layers
            self.lstm = nn.LSTM(
                latent, hidden, n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc_out = nn.Linear(hidden, n_features)

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            """Decode a batch of latent vectors into reconstructed sequences.

            Parameters
            ----------
            z:
                Latent tensor of shape ``(B, latent)``.

            Returns
            -------
            torch.Tensor
                Reconstructed sequence of shape ``(B, seq_len, n_features)``.
            """
            h0 = torch.tanh(self.fc_in(z)).unsqueeze(0)
            h0 = h0.repeat(self.n_layers, 1, 1)
            z = z.float()
            z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.lstm(z_seq, (h0.contiguous(), torch.zeros_like(h0)))
            return self.fc_out(out)

    class _LSTMAEModule(nn.Module):
        """Full LSTM autoencoder (encoder + decoder) as a single ``nn.Module``.

        Wraps ``_LSTMEncoder`` and ``_LSTMDecoder`` and routes the validity
        mask to the encoder's ``pack_padded_sequence`` logic.

        Parameters
        ----------
        cfg:
            ``LSTMAutoencoderConfig`` providing hidden size, latent dim,
            number of layers, and dropout.
        seq_len:
            Fixed sequence length (number of timesteps per window).
        """

        def __init__(self, cfg: LSTMAutoencoderConfig, seq_len: int):
            super().__init__()
            self.encoder = _LSTMEncoder(
                N_SEQUENCE_FEATURES, cfg.hidden_size,
                cfg.num_layers, cfg.latent_dim, cfg.dropout,
            )
            self.decoder = _LSTMDecoder(
                cfg.latent_dim, cfg.hidden_size, cfg.num_layers,
                N_SEQUENCE_FEATURES, seq_len, cfg.dropout,
            )

        def forward(
            self,
            x: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Encode then decode a batch of windows.

            Parameters
            ----------
            x:
                Input tensor of shape ``(B, T, F)``.
            mask:
                Optional bool tensor of shape ``(B, T)``.
                ``True`` = real data; ``False`` = padded.

            Returns
            -------
            (torch.Tensor, torch.Tensor)
                Tuple of ``(reconstruction, latent_vector)`` with shapes
                ``(B, T, F)`` and ``(B, latent_dim)`` respectively.
            """
            lengths = mask.long().sum(dim=1) if mask is not None else None
            z = self.encoder(x, lengths)
            recon = self.decoder(z)
            return recon, z

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding added to transformer input embeddings.

        Uses the standard Vaswani et al. (2017) formulation with separate
        sine and cosine frequencies for even and odd embedding dimensions.

        Parameters
        ----------
        d_model:
            Embedding dimensionality (must match the transformer d_model).
        max_len:
            Maximum supported sequence length (default 512).
        dropout:
            Dropout rate applied after adding the positional encoding.
        """

        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Add positional encoding to an embedded input sequence.

            Parameters
            ----------
            x:
                Embedding tensor of shape ``(B, T, d_model)``.

            Returns
            -------
            torch.Tensor
                Same shape as input, with positional encoding added and
                dropout applied.
            """
            return self.dropout(x + self.pe[:, :x.size(1)])

    class _TransformerAEModule(nn.Module):
        """Transformer autoencoder with validity-weighted bottleneck pooling.

        The encoder pools only over real (non-padded) timesteps:

        ``pooled = Σ_t m_t × h_t / Σ_t m_t``

        where ``m_t = 1`` for real data and ``0`` for padded.  Standard
        ``h.mean(dim=1)`` lets padded zeros bias the latent embedding,
        making incomplete windows appear anomalous purely due to packet
        loss rather than player physiology.

        The padding mask is also forwarded to the decoder so that padded
        output positions are not driven by the encoder's latent
        representation.

        Attention Weights
        -----------------
        The last encoder layer's attention matrix is captured after each
        forward pass and stored in ``_last_attn``.  **Warning**: attention
        weight magnitude does not equal feature importance.  Do not present
        as an XAI explanation without validation via SHAP or integrated
        gradients.

        Parameters
        ----------
        cfg:
            ``TransformerAutoencoderConfig`` with model hyperparameters.
        seq_len:
            Fixed sequence length.
        """

        def __init__(self, cfg: TransformerAutoencoderConfig, seq_len: int):
            super().__init__()
            D, L, F = cfg.d_model, cfg.latent_dim, N_SEQUENCE_FEATURES

            self.input_proj = nn.Linear(F, D)
            self.pos_enc = _PositionalEncoding(
                D, max_len=max(seq_len + 4, 64), dropout=cfg.dropout
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                enc_layer, num_layers=cfg.n_encoder_layers)
            self.fc_latent = nn.Linear(D, L)
            self.fc_expand = nn.Linear(L, D)

            dec_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                batch_first=True, norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(
                dec_layer, num_layers=cfg.n_decoder_layers)
            self.output_proj = nn.Linear(D, F)
            self.seq_len = seq_len
            self._last_attn: Optional[np.ndarray] = None

        @staticmethod
        def _mask_to_padding_mask(
            mask: Optional["torch.Tensor"],
        ) -> Optional["torch.Tensor"]:
            """Convert an internal validity mask to PyTorch's padding mask convention.

            Parameters
            ----------
            mask:
                Bool tensor of shape ``(B, T)``.
                ``True`` = real data (internal convention).

            Returns
            -------
            torch.Tensor or None
                Bool tensor of shape ``(B, T)`` where ``True`` = padded
                (PyTorch's ``src_key_padding_mask`` convention, which is the
                inverse of the internal convention).  Returns ``None`` when
                the input is ``None``.
            """
            if mask is None:
                return None
            return ~mask.bool()

        def _masked_pool(
            self,
            h: "torch.Tensor",
            mask: Optional["torch.Tensor"],
        ) -> "torch.Tensor":
            """Apply validity-weighted mean pooling over the time axis.

            Parameters
            ----------
            h:
                Encoder output of shape ``(B, T, D)``.
            mask:
                Bool tensor of shape ``(B, T)``.  ``True`` = real data.
                When ``None``, falls back to regular ``mean(dim=1)``.

            Returns
            -------
            torch.Tensor
                Pooled representation of shape ``(B, D)``.
            """
            if mask is None:
                return h.mean(dim=1)
            m = mask.float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1)
            count = m.sum(dim=1).clamp(min=1e-8)
            return pooled / count

        def _capture_attn_weights(
            self, h: "torch.Tensor"
        ) -> Optional[np.ndarray]:
            """Extract the final encoder layer's self-attention weights.

            Runs a no-gradient forward pass through the last encoder layer's
            self-attention head to retrieve the ``(B, T, T)`` attention matrix.

            Parameters
            ----------
            h:
                Encoder hidden states of shape ``(B, T, D)``.

            Returns
            -------
            numpy.ndarray or None
                Attention weight matrix of shape ``(B, T, T)`` averaged
                across heads, or ``None`` on failure.
            """
            try:
                last = self.encoder.layers[-1]
                with torch.no_grad():
                    _, w = last.self_attn(
                        h, h, h,
                        need_weights=True,
                        average_attn_weights=True,
                    )
                return w.cpu().numpy() if w is not None else None
            except Exception as exc:
                logger.debug("Attention capture failed: %s", exc)
                return None

        def forward(
            self,
            x: "torch.Tensor",
            mask: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            """Encode then decode a masked sequence batch.

            Parameters
            ----------
            x:
                Input tensor of shape ``(B, T, F)``.
            mask:
                Optional bool tensor of shape ``(B, T)``.
                ``True`` = real data (internal convention).

            Returns
            -------
            (torch.Tensor, torch.Tensor)
                ``(reconstruction, latent)`` with shapes ``(B, T, F)``
                and ``(B, latent_dim)`` respectively.
            """
            padding_mask = self._mask_to_padding_mask(mask)

            h = self.pos_enc(self.input_proj(x))
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            self._last_attn = self._capture_attn_weights(h)

            pooled = self._masked_pool(h, mask)
            z = torch.tanh(self.fc_latent(pooled))

            B, T = x.shape[:2]
            h_dec = torch.tanh(self.fc_expand(z))
            h_dec = h_dec.unsqueeze(1).repeat(1, T, 1)
            h_dec = self.pos_enc(h_dec)
            h_dec = self.decoder(h_dec, src_key_padding_mask=padding_mask)
            recon = self.output_proj(h_dec)
            return recon, z


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────
def profile_inference(func: Callable):
    """Decorator to measure and log inference latency and throughput.

    Sample counting rules:
    - ``predict_batch`` returns an ``np.ndarray`` of per-sample losses → len(result).
    - ``score_window`` returns a tuple (raw_loss, is_anomaly, confidence, model_type)
      → each call scores exactly 1 window.

    Output is routed through ``tqdm.write`` when tqdm is available so profiling
    messages don't interleave with active progress bars; falls back to
    ``logger.info`` otherwise.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time

        if isinstance(result, np.ndarray):
            # predict_batch: result is an (N,) loss array
            n_samples = len(result)
        elif isinstance(result, tuple):
            # score_window: returns (raw_loss, is_anomaly, confidence, model_type)
            n_samples = 1
        else:
            n_samples = 0

        throughput = n_samples / duration if duration > 0 and n_samples > 0 else 0
        # msg = (
        #     f"Profiling {func.__name__}: "
        #     f"duration={duration:.4f} s, samples={n_samples}, "
        #     f"throughput={throughput:.2f} windows/s"
        # )
        # if TQDM_AVAILABLE:
        #     _tqdm.write(msg)
        # else:
        #     logger.info(msg)
        return result
    return wrapper


class InferenceEngine:
    """
    Encapsulates the ML model backbone and per-player threshold calibration.

    Decouples the mathematical scoring (prediction + thresholding) from the
    operational orchestration (windowing, alert management, fatigue logic).
    """

    def __init__(self):
        self._shared_model: Optional[SharedBackboneAutoencoder] = None
        self._threshold_trackers: Dict[int, RegimeAwareThresholdStore] = {}

    @property
    def is_ready(self) -> bool:
        return self._shared_model is not None and self._shared_model.is_trained

    def train(self, all_windows: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]) -> dict:
        """Train the shared backbone and calibrate per-player thresholds."""
        player_ids = list(all_windows.keys())
        self._shared_model = SharedBackboneAutoencoder(
            n_players=len(player_ids))
        self._shared_model.register_players(player_ids)
        result = self._shared_model.train(all_windows)

        alpha = CONFIG.scoring.score_ema_alpha
        for pid, windows in all_windows.items():
            calib = windows[int(len(windows) * 0.80):]
            store = RegimeAwareThresholdStore()
            smoother = EMASmoother(alpha)
            for seq, mask, _ in calib:
                raw_loss, _, _ = self._shared_model.predict(pid, seq, mask)
                ema_val = smoother.update(raw_loss)
                regime_key = _REGIME_CLASSIFIER.classify(seq).key
                store.update(ema_val, regime_key)
            self._threshold_trackers[pid] = store
            logger.debug("InferenceEngine p%d calibration:\n%s",
                         pid, store.summary())

        try:
            saved_path = self._shared_model.save()
            logger.info("Shared backbone saved → %s", saved_path)
        except Exception as exc:
            logger.warning("Model save failed (non-fatal): %s", exc)

        return result

    # @profile_inference
    # def score_window(
    #     self,
    #     player_id: int,
    #     sequence: np.ndarray,
    #     mask: np.ndarray,
    #     smoothed_loss: Optional[float] = None
    # ) -> Tuple[float, bool, float, str]:
    #     """
    #     Compute the raw loss and check against calibrated thresholds.

    #     Returns:
    #         (raw_loss, is_anomaly, confidence, model_type)
    #     """
    #     if not self.is_ready:
    #         return 0.0, False, 0.0, "none"

    #     raw_loss, _, _ = self._shared_model.predict(player_id, sequence, mask)

    #     # Use smoothed_loss if provided (live inference), otherwise use raw_loss (batch/eval)
    #     score_for_threshold = smoothed_loss if smoothed_loss is not None else raw_loss

    #     tracker = self._threshold_trackers.get(player_id)
    #     regime_key = _REGIME_CLASSIFIER.classify(sequence).key

    #     if tracker and tracker.is_calibrated:
    #         is_anomaly = score_for_threshold > tracker.threshold_for(
    #             regime_key)
    #         confidence = tracker.confidence_for(
    #             score_for_threshold, regime_key)
    #     else:
    #         is_anomaly, confidence = False, 0.0

    #     return raw_loss, is_anomaly, confidence, self._shared_model.MODEL_TYPE

    def score_window(
        self,
        player_id: int,
        sequence: np.ndarray,
        mask: np.ndarray,
        
    ) -> Tuple[float, str]:
        if not self.is_ready:
            return 0.0, "none"

        raw_loss, _, _ = self._shared_model.predict(
            player_id,
            sequence,
            mask
        )

        return float(raw_loss), self._shared_model.MODEL_TYPE

    def get_tracker(self, player_id: int) -> Optional[RegimeAwareThresholdStore]:
        return self._threshold_trackers.get(player_id)

    def get_model(self) -> Optional[SharedBackboneAutoencoder]:
        return self._shared_model


def _masked_mse(
    x: "torch.Tensor",
    recon: "torch.Tensor",
    mask: Optional["torch.Tensor"],
) -> "torch.Tensor":
    """Compute mean-squared reconstruction error over real (non-padded) timesteps.

    Only timesteps where ``mask = True`` contribute to the loss.  This
    prevents padded zeros from generating artificial gradients and ensures
    that incomplete windows (due to packet loss) are scored fairly.

    Parameters
    ----------
    x:
        Original input tensor of shape ``(B, T, F)``.
    recon:
        Reconstructed tensor of shape ``(B, T, F)``.
    mask:
        Bool tensor of shape ``(B, T)``.  ``True`` = real timestep.
        When ``None``, all timesteps are included (standard MSE).

    Returns
    -------
    torch.Tensor
        Scalar loss value.

    Notes
    -----
    The denominator is ``(Σ_bt m_bt) × F + ε`` where ε = 1e-8 prevents
    division-by-zero when an entire batch contains only padded rows.
    """
    sq = (x - recon) ** 2
    if mask is not None:
        m = mask.float().unsqueeze(-1)
        return (sq * m).sum() / (m.sum() * sq.size(-1) + 1e-8)
    return sq.mean()


def _train_loop(
    module:        "nn.Module",
    sequences:     List[np.ndarray],
    masks:         List[np.ndarray],
    normaliser:    PerPlayerNormaliser,
    cfg_batch:     int,
    cfg_lr:        float,
    cfg_epochs:    int,
    cfg_patience:  int,
    player_id:     int,
    model_label:   str,
    latent_reg:    float = 1e-4,
    train_frac:    float = 0.70,
    val_frac:      float = 0.15,
) -> Tuple[dict, List[np.ndarray], List[np.ndarray]]:
    """Train an autoencoder module with a 3-way train/val/calibration split.

    Implements the standard per-player training loop shared by both
    ``LSTMAutoencoder`` and ``TransformerAutoencoder``.

    Data Splitting
    --------------
    Windows are randomly permuted (seeded by ``42 + player_id`` for
    reproducibility) and split as follows:

    * **Train** (``train_frac`` = 70 %): used for gradient updates.
    * **Validation** (``val_frac`` = 15 %): drives early stopping and
      the learning-rate scheduler.
    * **Calibration** (remaining ≈ 15 %): returned to the caller so the
      threshold tracker can be populated *after* training without data
      leakage.

    If a split ends up empty (e.g. very few windows), it falls back to a
    fraction of the training split to avoid crashes.

    Normalisation
    -------------
    ``normaliser.fit()`` is called on the **training split only**, then
    applied to all splits.  Fitting on the full dataset would constitute
    data leakage through the normalisation statistics.

    Training Details
    ----------------
    * Optimiser: Adam with the supplied learning rate.
    * Scheduler: ``ReduceLROnPlateau`` (factor 0.5, patience 3 epochs).
    * Gradient clipping: max norm 1.0.
    * Early stopping: triggered after ``cfg_patience`` epochs without a
      validation-loss improvement of at least 1e-5.
    * Best weights (by validation loss) are restored before returning.
    * Latent L2 regularisation (weight ``latent_reg``) prevents the
      autoencoder from learning a trivial identity or smoothing mapping.
      Without it the latent space collapses and reconstruction error
      becomes uniformly low, degrading anomaly separation.

    Parameters
    ----------
    module:
        An un-trained ``nn.Module`` autoencoder instance (LSTM or Transformer).
    sequences:
        List of ``(T, F)`` float32 arrays — one per sliding window.
    masks:
        List of ``(T,)`` bool arrays aligned with ``sequences``.
    normaliser:
        ``PerPlayerNormaliser`` instance.  ``fit()`` is called inside
        this function; the caller should pass an unfitted instance.
    cfg_batch:
        Mini-batch size.
    cfg_lr:
        Initial Adam learning rate.
    cfg_epochs:
        Maximum number of training epochs.
    cfg_patience:
        Early-stopping patience in epochs.
    player_id:
        Used to seed the random permutation for reproducibility.
    model_label:
        Short string for log messages, e.g. ``"LSTM"`` or ``"Transformer"``.
    latent_reg:
        L2 regularisation weight on the latent vector.
    train_frac:
        Fraction of windows used for training.
    val_frac:
        Fraction of windows used for validation.

    Returns
    -------
    (dict, list, list)
        * ``history`` — training metadata dict with keys:
          ``best_val_loss``, ``history`` (per-epoch losses),
          ``epochs_run``, ``n_train``, ``n_val``, ``n_calib``.
        * ``calib_sequences`` — held-out raw ``(T, F)`` arrays.
        * ``calib_masks``     — held-out ``(T,)`` bool arrays.
    """
    arr = np.stack(sequences)
    marr = np.stack(masks).astype(bool)
    N = len(arr)

    rng = np.random.default_rng(seed=42 + player_id)
    indices = rng.permutation(N)
    n_train = max(int(N * train_frac), 1)
    n_val = max(int(N * val_frac),   1)

    idx_train = indices[:n_train]
    idx_val = indices[n_train: n_train + n_val]
    idx_calib = indices[n_train + n_val:]

    if len(idx_val) == 0:
        idx_val = idx_train[:max(1, n_train // 5)]
    if len(idx_calib) == 0:
        idx_calib = idx_train[:max(1, n_train // 5)]

    normaliser.fit(arr[idx_train])

    def to_tensor_pair(idx: np.ndarray):
        X = torch.tensor(normaliser.transform(arr[idx]))
        M = torch.tensor(marr[idx])
        return X, M

    X_train, M_train = to_tensor_pair(idx_train)
    X_val,   M_val = to_tensor_pair(idx_val)

    loader = DataLoader(
        TensorDataset(X_train, M_train),
        batch_size=cfg_batch,
        shuffle=True,
        pin_memory=(str(DEVICE) == "cuda" or str(DEVICE) == "mps"),
    )

    module = module.to(DEVICE)
    optimizer = optim.Adam(module.parameters(), lr=cfg_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state = None
    patience_ct = 0
    history: List[dict] = []

    logger.info(
        "%s p%d: train=%d val=%d calib=%d device=%s",
        model_label, player_id,
        len(idx_train), len(idx_val), len(idx_calib), DEVICE,
    )

    for epoch in range(cfg_epochs):
        module.train()
        epoch_loss = 0.0
        for batch_x, batch_m in loader:
            batch_x = batch_x.to(DEVICE)
            batch_m = batch_m.to(DEVICE)
            optimizer.zero_grad()
            recon, z = module(batch_x, mask=batch_m)

            recon_loss = _masked_mse(batch_x, recon, batch_m)
            reg_loss = latent_reg * z.pow(2).mean()
            loss = recon_loss + reg_loss

            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += recon_loss.item() * len(batch_x)
        epoch_loss /= len(idx_train)

        module.eval()
        with torch.no_grad():
            xv = X_val.to(DEVICE)
            mv = M_val.to(DEVICE)
            rv, _ = module(xv, mask=mv)
            val_loss = _masked_mse(xv, rv, mv).item()

        history.append({"epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "val_loss":   val_loss})
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_ct = 0
            best_state = {k: v.cpu().clone()
                          for k, v in module.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= cfg_patience:
                logger.info(
                    "%s p%d: early stop epoch %d (val=%.5f)",
                    model_label, player_id, epoch + 1, val_loss,
                )
                break

        if (epoch + 1) % 10 == 0:
            logger.info(
                "%s p%d: epoch %d/%d  train=%.5f  val=%.5f",
                model_label, player_id, epoch + 1, cfg_epochs,
                epoch_loss, val_loss,
            )

    if best_state is not None:
        module.load_state_dict({k: v.to(DEVICE)
                               for k, v in best_state.items()})
    module.eval()

    calib_seqs = [sequences[i] for i in idx_calib]
    calib_masks = [masks[i] for i in idx_calib]

    return (
        {"best_val_loss": best_val_loss, "history": history,
         "epochs_run": len(history),
         "n_train": len(idx_train), "n_val": len(idx_val), "n_calib": len(idx_calib)},
        calib_seqs,
        calib_masks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────
def _mse_loss_single(
    module:   "nn.Module",
    seq_norm: "torch.Tensor",
    mask:     Optional["torch.Tensor"] = None,
) -> float:
    """Compute the masked reconstruction loss for a single normalised window.

    Runs the module in evaluation mode with ``torch.no_grad()`` for
    memory efficiency.

    Parameters
    ----------
    module:
        A trained autoencoder ``nn.Module`` (LSTM or Transformer).
    seq_norm:
        Normalised window tensor of shape ``(T, F)``.
    mask:
        Optional bool tensor of shape ``(T,)``.  ``True`` = real timestep.

    Returns
    -------
    float
        Scalar masked MSE reconstruction loss.
    """
    module.eval()
    with torch.no_grad():
        x = seq_norm.unsqueeze(0).to(DEVICE)
        m = mask.unsqueeze(0).to(DEVICE) if mask is not None else None
        recon, _ = module(x, mask=m)
        loss = _masked_mse(x, recon, m)
    return float(loss.item())


def _mse_loss_batch(
    module:   "nn.Module",
    X_norm:   "torch.Tensor",
    masks:    Optional["torch.Tensor"] = None,
) -> np.ndarray:
    """Compute per-sample masked reconstruction losses for a batch.

    Parameters
    ----------
    module:
        A trained autoencoder ``nn.Module``.
    X_norm:
        Normalised batch tensor of shape ``(N, T, F)``.
    masks:
        Optional bool tensor of shape ``(N, T)``.  ``True`` = real timestep.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(N,)`` containing the per-sample
        reconstruction loss.
    """
    module.eval()
    with torch.no_grad():
        x = X_norm.to(DEVICE)
        m = masks.to(DEVICE) if masks is not None else None
        recon, _ = module(x, mask=m)
        sq = (x - recon) ** 2
        if m is not None:
            mf = m.float().unsqueeze(-1)
            loss = (sq * mf).sum(dim=(1, 2)) / \
                (mf.sum(dim=(1, 2)) * sq.size(-1) + 1e-8)
        else:
            loss = sq.mean(dim=(1, 2))
    return loss.cpu().numpy()


class SharedLSTMEncoder(nn.Module):
    """Player-conditioned LSTM encoder trained jointly on all players.

    Extends the base ``_LSTMEncoder`` with a FiLM (Feature-wise Linear
    Modulation) conditioning layer.  A per-player embedding vector is used
    to compute a multiplicative scale and additive shift applied to the
    latent representation after the LSTM projection:

    ``z_conditioned = z × (1 + scale(embedding)) + shift(embedding)``

    Both scale and shift are squashed through ``tanh`` and multiplied by
    0.1 so that conditioning perturbs the shared latent space with small,
    bounded corrections rather than overwriting it entirely.  This keeps
    the shared backbone as the dominant signal and the player embedding as
    a fine-grained personalisation layer.

    A final ``nan_to_num`` clamp on the output latent ``z`` prevents NaN
    propagation into the decoder when an embedding weight is initialised
    near an exploding gradient.

    Parameters
    ----------
    cfg:
        ``LSTMAutoencoderConfig`` with LSTM hyperparameters.
    embedding_dim:
        Dimensionality of the player embedding vectors (default 16).
    """

    def __init__(self, cfg: LSTMAutoencoderConfig, embedding_dim: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(
            N_SEQUENCE_FEATURES, cfg.hidden_size, cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.latent_dim)
        self.film_scale = nn.Linear(embedding_dim, cfg.latent_dim)
        self.film_shift = nn.Linear(embedding_dim, cfg.latent_dim)

    def forward(self, x, embedding, lengths=None):
        """Encode a batch with per-player FiLM conditioning.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, T, F)``.
        embedding:
            Player embedding tensor of shape ``(B, embedding_dim)`` — one
            row per sample in the batch.
        lengths:
            Optional 1-D int tensor of shape ``(B,)`` for
            ``pack_padded_sequence``.

        Returns
        -------
        torch.Tensor
            Conditioned latent tensor of shape ``(B, latent_dim)``.
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        z = torch.tanh(self.fc(h_n[-1]))
        scale = 0.1 * torch.tanh(self.film_scale(embedding))
        shift = 0.1 * torch.tanh(self.film_shift(embedding))

        z = z * (1.0 + scale) + shift
        z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

        return z


class SharedBackboneAutoencoder:
    """Single autoencoder trained jointly on all registered players.

    This is the primary production model.  It replaces the original
    per-player models and solves the scalability
    problem of maintaining O(N_players) separate model files.

    Architecture
    ------------
    * **Shared encoder**: Learned conditioning (e.g. FiLM) to adapt a shared latent space.
    * **Shared decoder**: Reconstructs sequence from conditioned latent.
    * **Player embeddings**: Learned jointly during training.
    """
    MODEL_TYPE = "shared_lstm"

    def __init__(self, n_players: int, cfg: LSTMAutoencoderConfig = None,
                 embedding_dim: int = 16):
        self.cfg = cfg or CONFIG.lstm
        self.embedding_dim = embedding_dim
        self.n_players = n_players
        self.is_trained = False
        self.model_version = "untrained"
        self.normaliser = PerPlayerNormaliser()
        self._encoder: Optional[SharedLSTMEncoder] = None
        self._decoder: Optional[_LSTMDecoder] = None
        self._player_index: Dict[int, int] = {}
        self._embedding: Optional[nn.Embedding] = None
        self._threshold_trackers: Dict[int, RegimeAwareThresholdStore] = {}

    def register_players(self, player_ids: List[int]) -> None:
        """Build the player-ID → embedding-row index mapping.

        Must be called before ``train()`` with the complete list of player
        IDs that will appear in the training data.

        Parameters
        ----------
        player_ids:
            Ordered list of integer player identifiers.  The order
            determines which row of the embedding table each player maps to.
        """
        self._player_index = {pid: i for i, pid in enumerate(player_ids)}
        self.n_players = len(player_ids)

    def train(
    self, all_windows: Dict[int, List[Tuple[np.ndarray, np.ndarray, int]]]
) -> dict:
        """
        Train shared backbone using session-level splitting to prevent leakage.
        """

        if not TORCH_AVAILABLE:
            return {"status": "no_torch"}

        import gc

        try:
            from tqdm import tqdm as _tqdm

            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False

        torch.set_float32_matmul_precision("high")

        epoch_iter = (
            _tqdm(
                range(self.cfg.max_epochs),
                desc="Shared LSTM",
                unit="epoch",
                dynamic_ncols=True,
            )
            if TQDM_AVAILABLE
            else range(self.cfg.max_epochs)
        )

        seq_len = next(iter(all_windows.values()))[0][0].shape[0]

        self._encoder = SharedLSTMEncoder(
            self.cfg,
            self.embedding_dim,
        )

        self._decoder = _LSTMDecoder(
            self.cfg.latent_dim,
            self.cfg.hidden_size,
            self.cfg.num_layers,
            N_SEQUENCE_FEATURES,
            seq_len,
            self.cfg.dropout,
        )

        self._embedding = nn.Embedding(
            self.n_players,
            self.embedding_dim,
        )

        # ─────────────────────────────────────────────
        # Reverse mapping (embedding idx -> real pid)
        # ─────────────────────────────────────────────
        self._index_to_player_id = {v: k for k, v in self._player_index.items()}

        # ─────────────────────────────────────────────
        # Flatten windows
        # ─────────────────────────────────────────────
        all_seqs = []
        all_masks = []
        all_pids = []
        all_session_ids = []

        for pid, windows in all_windows.items():

            idx = self._player_index[pid]

            for seq, mask, session_id in windows:
                all_seqs.append(seq)
                all_masks.append(mask)
                all_pids.append(idx)
                all_session_ids.append(session_id)

        arr = np.stack(all_seqs).astype(np.float32, copy=False)
        marr = np.stack(all_masks).astype(bool, copy=False)
        parr = np.asarray(all_pids, dtype=np.int64)
        sarr = np.asarray(all_session_ids)

        del all_seqs
        del all_masks
        gc.collect()

        # ─────────────────────────────────────────────
        # Session-level split
        # ─────────────────────────────────────────────
        rng = np.random.default_rng(42)

        unique_sessions = np.unique(sarr)
        rng.shuffle(unique_sessions)

        n_sessions = len(unique_sessions)

        n_train = max(int(n_sessions * 0.70), 1)
        n_val = max(int(n_sessions * 0.15), 1)

        train_sessions = set(unique_sessions[:n_train])

        val_sessions = set(unique_sessions[n_train : n_train + n_val])

        calib_sessions = set(unique_sessions[n_train + n_val :])

        idx_train = np.where(np.isin(sarr, list(train_sessions)))[0]

        idx_val = np.where(np.isin(sarr, list(val_sessions)))[0]

        idx_calib = np.where(np.isin(sarr, list(calib_sessions)))[0]

        if len(idx_val) == 0:
            idx_val = idx_train[: max(1, len(idx_train) // 5)]

        if len(idx_calib) == 0:
            idx_calib = idx_train[: max(1, len(idx_train) // 5)]

        # ─────────────────────────────────────────────
        # Fit normaliser ONLY on training
        # ─────────────────────────────────────────────
        self.normaliser.fit(arr[idx_train])

        X_train = torch.from_numpy(
            self.normaliser.transform(arr[idx_train]).astype(np.float32)
        )

        X_val = torch.from_numpy(
            self.normaliser.transform(arr[idx_val]).astype(np.float32)
        )

        X_calib = torch.from_numpy(
            self.normaliser.transform(arr[idx_calib]).astype(np.float32)
        )

        M_train = torch.from_numpy(marr[idx_train])
        M_val = torch.from_numpy(marr[idx_val])
        M_calib = torch.from_numpy(marr[idx_calib])

        P_train = torch.from_numpy(parr[idx_train]).long()
        P_val = torch.from_numpy(parr[idx_val]).long()
        P_calib = torch.from_numpy(parr[idx_calib]).long()

        del arr
        del marr
        del parr
        gc.collect()

        # ─────────────────────────────────────────────
        # Datasets
        # ─────────────────────────────────────────────
        train_dataset = TensorDataset(
            X_train,
            M_train,
            P_train,
        )

        val_dataset = TensorDataset(
            X_val,
            M_val,
            P_val,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.cfg.batch_size, 512),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.cfg.batch_size, 512),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        # ─────────────────────────────────────────────
        # Store calibration tensors
        # ─────────────────────────────────────────────
        self._calib_sequences = X_calib
        self._calib_masks = M_calib
        self._calib_player_ids = P_calib

        params = (
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self._embedding.parameters())
        )

        optimizer = optim.Adam(
            params,
            lr=self.cfg.learning_rate,
        )

        self._encoder.to(DEVICE)
        self._decoder.to(DEVICE)
        self._embedding.to(DEVICE)

        best_loss = float("inf")
        best_state = None
        patience_ct = 0

        PATIENCE = self.cfg.patience
        MIN_IMPROVEMENT = 1e-4

        # ─────────────────────────────────────────────
        # Training
        # ─────────────────────────────────────────────
        for epoch in epoch_iter:

            self._encoder.train()
            self._decoder.train()

            train_loss = 0.0
            n_batches = 0

            for bx, bm, bp in train_loader:

                bx = bx.to(DEVICE)
                bm = bm.to(DEVICE)
                bp = bp.to(DEVICE)

                emb = self._embedding(bp)

                lengths = bm.long().sum(dim=1)

                z = self._encoder(
                    bx,
                    emb,
                    lengths,
                )

                recon = self._decoder(z)

                recon_loss = _masked_mse(
                    bx,
                    recon,
                    bm,
                )

                latent_reg = 1e-4
                reg_loss = latent_reg * z.pow(2).mean()

                loss = recon_loss + reg_loss

                optimizer.zero_grad(set_to_none=True)

                loss.backward()

                nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)

            # ─────────────────────────────────────────
            # Validation
            # ─────────────────────────────────────────
            self._encoder.eval()
            self._decoder.eval()

            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():

                for bx, bm, bp in val_loader:

                    bx = bx.to(DEVICE)
                    bm = bm.to(DEVICE)
                    bp = bp.to(DEVICE)

                    emb = self._embedding(bp)

                    lengths = bm.long().sum(dim=1)

                    z = self._encoder(
                        bx,
                        emb,
                        lengths,
                    )

                    recon = self._decoder(z)

                    recon_loss = _masked_mse(
                        bx,
                        recon,
                        bm,
                    )

                    latent_reg = 1e-4
                    reg_loss = latent_reg * z.pow(2).mean()

                    loss = recon_loss + reg_loss

                    val_loss += loss.item()
                    val_batches += 1

            val_loss /= max(val_batches, 1)

            if TQDM_AVAILABLE:
                epoch_iter.set_postfix(
                    train=f"{train_loss:.5f}",
                    val=f"{val_loss:.5f}",
                    best=f"{best_loss:.5f}",
                    patience=patience_ct,
                )

            if val_loss < best_loss - MIN_IMPROVEMENT:

                best_loss = val_loss
                patience_ct = 0

                best_state = {
                    "encoder": {
                        k: v.cpu().clone()
                        for k, v in self._encoder.state_dict().items()
                    },
                    "decoder": {
                        k: v.cpu().clone()
                        for k, v in self._decoder.state_dict().items()
                    },
                    "embedding": {
                        k: v.cpu().clone()
                        for k, v in self._embedding.state_dict().items()
                    },
                }

            else:

                patience_ct += 1

                if patience_ct >= PATIENCE:

                    logger.info(
                        "Shared LSTM early stop epoch=%d val=%.5f",
                        epoch + 1,
                        val_loss,
                    )

                    break

        # ─────────────────────────────────────────────
        # Restore best weights
        # ─────────────────────────────────────────────
        if best_state is not None:

            self._encoder.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state["encoder"].items()}
            )

            self._decoder.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state["decoder"].items()}
            )

            self._embedding.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state["embedding"].items()}
            )

        # ─────────────────────────────────────────────
        # Calibration thresholds
        # ─────────────────────────────────────────────
        self.is_trained = True

        alpha = CONFIG.scoring.score_ema_alpha

        for pid_tensor in self._calib_player_ids.unique():

            player_idx = pid_tensor.item()

            # FIX: convert embedding index -> real player ID
            pid = self._index_to_player_id[player_idx]

            mask = self._calib_player_ids == pid_tensor

            calib_seqs = self._calib_sequences[mask]
            calib_masks = self._calib_masks[mask]

            if len(calib_seqs) == 0:
                continue

            calib_np = calib_seqs.cpu().numpy().astype(np.float32)
            mask_np = calib_masks.cpu().numpy().astype(bool)

            pid_np = np.full(
                len(calib_np),
                pid,
                dtype=np.int64,
            )

            losses = self.predict_batch(
                player_ids=pid_np,
                sequences=calib_np,
                masks=mask_np,
                normalised=True,
            )

            store = RegimeAwareThresholdStore()

            smoother = EMASmoother(alpha)

            for loss, seq in zip(losses, calib_np):

                ema_val = smoother.update(float(loss))

                regime_key = _REGIME_CLASSIFIER.classify(seq).key

                store.update(
                    ema_val,
                    regime_key,
                )

            self._threshold_trackers[pid] = store

        self.is_trained = True

        self.model_version = f"shared_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return {
            "status": "trained",
            "n_players": self.n_players,
            "n_windows": int(len(sarr)),
        }

    @profile_inference
    def predict_batch(
        self,
        player_ids: np.ndarray,
        sequences: np.ndarray,
        masks: Optional[np.ndarray] = None,
        normalised: bool = False,
    ) -> np.ndarray:
        """
        Compute reconstruction losses for a batch of windows.

        Parameters
        ----------
        player_ids:
            Array of player IDs of shape (N,).

        sequences:
            Windows of shape (N, T, F).

        masks:
            Optional bool mask of shape (N, T).

        normalised:
            If True, assumes sequences are already normalised.

        Returns
        -------
        np.ndarray
            Reconstruction losses of shape (N,).
        """

        if not self.is_trained:
            return np.zeros(len(player_ids), dtype=np.float32)

        if len(player_ids) == 0:
            return np.empty(0, dtype=np.float32)

        # ─────────────────────────────────────────────
        # Resolve embedding indices
        # ─────────────────────────────────────────────
        indices = [
            self._player_index.get(int(pid), -1)
            for pid in player_ids
        ]

        unknown = [
            int(pid)
            for pid, idx in zip(player_ids, indices)
            if idx == -1
        ]

        if unknown:
            raise ValueError(
                f"Unknown player IDs in predict_batch(): "
                f"{unknown[:10]}"
            )

        # ─────────────────────────────────────────────
        # Tensor conversion (ZERO-COPY where possible)
        # ─────────────────────────────────────────────
        if normalised:
            X_input = sequences
        else:
            X_input = self.normaliser.transform(sequences)

        X_norm = torch.tensor(
            X_input,
            dtype=torch.float32
        ).to(DEVICE)

        P_tensor = torch.tensor(
            indices,
            dtype=torch.long,
            device=DEVICE,
        )

        emb = self._embedding(P_tensor)

        M_tensor = None
        lengths = None

        if masks is not None:
            M_tensor = torch.tensor(
                masks,
                dtype=torch.bool,
                device=DEVICE,
            )

            lengths = M_tensor.long().sum(dim=1)

        # ─────────────────────────────────────────────
        # Mixed precision inference
        # ─────────────────────────────────────────────
        use_amp = DEVICE.type != "cpu"

        self._encoder.eval()
        self._decoder.eval()

        with torch.no_grad():
                z = self._encoder(
                    X_norm,
                    emb,
                    lengths,
                ).float()

                recon = self._decoder(z)

                sq = (X_norm - recon).pow(2)

                # ─────────────────────────────────────
                # Masked MSE
                # ─────────────────────────────────────
                if M_tensor is not None:

                    valid_mask = M_tensor.unsqueeze(-1)

                    sq = sq.masked_fill(~valid_mask, 0.0)

                    denom = (
                        valid_mask.sum(dim=(1, 2)) *
                        sq.size(-1)
                    ).clamp(min=1)

                    loss = sq.sum(dim=(1, 2)) / denom

                else:
                    loss = sq.mean(dim=(1, 2))

        return (
            loss.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def predict(
        self,
        player_id: int,
        sequence: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[float, bool, float]:

        if not self.is_trained:
            return 0.0, False, 0.0

        idx = self._player_index.get(player_id)
        if idx is None:
            return 0.0, False, 0.0

        # -----------------------------
        # Normalize
        # -----------------------------
        sequence_norm = self.normaliser.transform(
            sequence[np.newaxis]
        )[0].astype(np.float32)

        norm = torch.tensor(
            sequence_norm,
            dtype=torch.float32,
            device=DEVICE
        )

        emb = self._embedding(
            torch.tensor([idx], dtype=torch.long, device=DEVICE)
        )

        mask_t = (
            torch.tensor(mask, dtype=torch.bool, device=DEVICE).unsqueeze(0)
            if mask is not None else None
        )

        with torch.no_grad():

            self._encoder.eval()
            self._decoder.eval()

            lengths = (
                mask_t.long().sum(dim=1)
                if mask_t is not None
                else None
            )

            # -----------------------------
            # Encode
            # -----------------------------
            z = self._encoder(
                norm.unsqueeze(0),
                emb,
                lengths
            )

            # -----------------------------
            # Decode
            # -----------------------------
            try:
                recon = self._decoder(
                    z,
                    target_len=norm.shape[0]
                )
            except TypeError:
                try:
                    recon = self._decoder(
                        z,
                        lengths
                    )
                except TypeError:
                    recon = self._decoder(z)

            # -----------------------------
            # Shape validation
            # -----------------------------
            expected_shape = norm.unsqueeze(0).shape

            if recon.shape != expected_shape:
                raise RuntimeError(
                    f"Decoder output shape mismatch. "
                    f"Expected {expected_shape}, got {recon.shape}"
                )

            # # -----------------------------
            # # Debug
            # # -----------------------------
            # print("\nPREDICT DEBUG")
            # print("norm shape :", norm.unsqueeze(0).shape)
            # print("recon shape:", recon.shape)

            # print("POST NORM MEAN", norm.mean().item())
            # print("POST NORM STD ", norm.std().item())

            # print("MODEL OUTPUT MEAN", recon.mean().item())
            # print("MODEL OUTPUT STD ", recon.std().item())

            # diff = (norm.unsqueeze(0) - recon).abs()

            # print("DIFF MEAN", diff.mean().item())
            # print("DIFF MAX ", diff.max().item())

            # print(
            #     "MASK SUM",
            #     mask_t.sum().item()
            #     if mask_t is not None else None
            # )

            # -----------------------------
            # Loss
            # -----------------------------
            loss = _masked_mse(
                norm.unsqueeze(0),
                recon,
                mask_t
            )

        return float(loss.item()), False, 0.0

        
    def reconstruction_loss_for_shap(
        self,
        player_id: int,
        sequences_norm: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Batch predict function designed for use with SHAP KernelExplainer.

        SHAP requires a function ``f(X) → scores`` where X is a 2-D matrix
        of perturbed samples.  This method accepts a batch of pre-normalised
        windows (already transformed by the caller using ``self.normaliser``)
        and returns per-sample reconstruction losses — the quantity SHAP
        will attribute to each input feature.

        This is the **only** correct way to generate SHAP attributions for
        this model: each perturbed sample passes through the real encoder
        and decoder so the loss reflects true model behaviour, not a
        surrogate approximation.

        Parameters
        ----------
        player_id:
            Player whose embedding to use.  The same embedding is broadcast
            to every sample in the batch.
        sequences_norm:
            Pre-normalised array of shape ``(N, T, F)`` (float32).
            The caller must normalise via ``self.normaliser.transform()``
            before passing here; this method does **not** normalise.
        mask:
            Optional ``(T,)`` bool array broadcast across all N samples.
            ``None`` means all timesteps are real.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(N,)`` — per-sample reconstruction
            losses.  Returns zeros for unknown players or untrained model.
        """
        if not self.is_trained:
            return np.zeros(len(sequences_norm), dtype=np.float32)

        idx = self._player_index.get(player_id)
        if idx is None:
            return np.zeros(len(sequences_norm), dtype=np.float32)

        N = len(sequences_norm)
        X = torch.tensor(sequences_norm, dtype=torch.float32).to(DEVICE)

        emb = self._embedding(
            torch.tensor([idx] * N, dtype=torch.long).to(DEVICE)
        )

        if mask is not None:
            M = torch.tensor(
                np.stack([mask] * N).astype(bool), dtype=torch.bool
            ).to(DEVICE)
            lengths = M.long().sum(dim=1)
        else:
            M = None
            lengths = None

        with torch.no_grad():
            self._encoder.eval()
            self._decoder.eval()
            z = self._encoder(X, emb, lengths)
            recon = self._decoder(z)

            sq = (X - recon) ** 2
            if M is not None:
                mf = M.float().unsqueeze(-1)
                loss = (sq * mf).sum(dim=(1, 2)) / (
                    mf.sum(dim=(1, 2)) * sq.size(-1) + 1e-8
                )
            else:
                loss = sq.mean(dim=(1, 2))

        return loss.cpu().numpy().astype(np.float32)

    def save(self, path: Path = None) -> Path:
        """Persist the trained shared backbone to disk.

        Saves all model weights, normaliser state, player-index mapping,
        and configuration in a single ``.pt`` file using ``torch.save``.

        Parameters
        ----------
        path:
            Optional destination path.  Defaults to
            ``MODEL_STORE / "shared_backbone.pt"``.

        Returns
        -------
        Path
            The path where the checkpoint was written.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save — model not yet trained.")
        path = path or (MODEL_STORE / "shared_backbone.pt")
        torch.save({
            "encoder":       self._encoder.state_dict(),
            "decoder":       self._decoder.state_dict(),
            "embedding":     self._embedding.state_dict(),
            "normaliser":    self.normaliser.state_dict(),
            "player_index":  self._player_index,
            "n_players":     self.n_players,
            "embedding_dim": self.embedding_dim,
            "model_version": self.model_version,
            "cfg":           self.cfg,
            "seq_len":       CONFIG.window.window_steps,
        }, path)
        logger.info("Shared backbone saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path = None) -> Optional["SharedBackboneAutoencoder"]:
        """Load a previously saved shared backbone from disk.

        Reconstructs the full model, normaliser, and player-index mapping
        from a checkpoint written by ``save()``.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint.  Defaults to
            ``MODEL_STORE / "shared_backbone.pt"``.

        Returns
        -------
        SharedBackboneAutoencoder or None
            Fully restored model ready for inference, or ``None`` if the
            file does not exist or PyTorch is not available.
        """
        path = path or (MODEL_STORE / "shared_backbone.pt")

        if not path.exists() or not TORCH_AVAILABLE:
            return None

        add_safe_globals([LSTMAutoencoderConfig])
        ckpt = torch.load(path, map_location=DEVICE)
        obj = cls(
            n_players=ckpt["n_players"],
            cfg=ckpt["cfg"],
            embedding_dim=ckpt["embedding_dim"],
        )
        obj._player_index = ckpt["player_index"]
        seq_len = ckpt.get("seq_len", CONFIG.window.window_steps)

        obj._encoder = SharedLSTMEncoder(obj.cfg, obj.embedding_dim).to(DEVICE)
        obj._decoder = _LSTMDecoder(
            obj.cfg.latent_dim, obj.cfg.hidden_size, obj.cfg.num_layers,
            N_SEQUENCE_FEATURES, seq_len, obj.cfg.dropout,
        ).to(DEVICE)
        obj._embedding = nn.Embedding(
            obj.n_players, obj.embedding_dim).to(DEVICE)

        obj._encoder.load_state_dict(ckpt["encoder"])
        obj._decoder.load_state_dict(ckpt["decoder"])
        obj._embedding.load_state_dict(ckpt["embedding"])
        obj.normaliser = PerPlayerNormaliser.from_state_dict(
            ckpt["normaliser"])
        obj.is_trained = True
        obj.model_version = ckpt["model_version"]
        logger.info("Shared backbone loaded ← %s (version=%s)",
                    path, obj.model_version)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# TransformerAutoencoder  (experimental, disabled)
# ─────────────────────────────────────────────────────────────────────────────
class TransformerAutoencoder:
    """Experimental transformer-based autoencoder for per-player anomaly detection.

    **Status: EXPERIMENTAL — do not use in production.**
    The default production model is ``SharedBackboneAutoencoder`` (LSTM).
    This class is retained for research comparisons and future productionisation.

    This autoencoder requires ≥ 30 sessions per player (``min_sessions_to_train``
    in config) to produce a reliable calibration distribution.  Below that
    threshold, ``train()`` returns ``{"status": "skipped"}``.

    Architecture
    ------------
    Input projection → positional encoding → transformer encoder (Pre-LN) →
    validity-weighted pooling → linear bottleneck → linear expansion →
    positional encoding → transformer decoder (Pre-LN) → output projection.

    Validity-weighted pooling (see ``_TransformerAEModule._masked_pool``)
    excludes padded timesteps from the bottleneck representation, preventing
    packet-loss artefacts from being mistaken for anomalies.

    Per-Regime Thresholds
    ---------------------
    Thresholds are stored in a ``RegimeAwareThresholdStore`` (one entry per
    tactical regime: high-press, low-block, transition, etc.).  Calibration
    losses pass through the same EMA smoothing used during inference so that
    the threshold distribution matches the inference score distribution exactly.

    Attention Weights
    -----------------
    ``last_attention_weights`` returns the final encoder layer's self-attention
    matrix.  **This does not equal feature importance**.  Validate with SHAP or
    integrated gradients before presenting as an explanation to coaching staff.

    Parameters
    ----------
    player_id:
        Internal integer player identifier.
    cfg:
        ``TransformerAutoencoderConfig`` instance.  Defaults to
        ``CONFIG.transformer``.
    """
    MODEL_TYPE = "transformer"
    EXPERIMENTAL = True

    def __init__(self, player_id: int, cfg: TransformerAutoencoderConfig = None):
        self.player_id = player_id
        self.cfg = cfg or CONFIG.transformer
        self.is_trained = False
        self.model_version = "untrained"
        self.normaliser = PerPlayerNormaliser()
        self.threshold_tracker = RegimeAwareThresholdStore()
        self._module: Optional["_TransformerAEModule"] = None
        if not TORCH_AVAILABLE:
            logger.warning(
                "PyTorch unavailable — TransformerAutoencoder in stub mode")
        warnings.warn(
            "TransformerAutoencoder is EXPERIMENTAL. "
            "Use LSTMAutoencoder for production.",
            stacklevel=2,
        )

    def train(self, session_windows: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
        """Train the transformer autoencoder on this player's session windows.

        Uses the shared ``_train_loop`` helper with a 70/15/15 split.
        After training, the calibration split is passed to ``_calibrate()``
        to populate per-regime thresholds.

        Parameters
        ----------
        session_windows:
            List of ``(sequence, mask)`` tuples for this player.
            Each sequence has shape ``(T, F)``; each mask has shape ``(T,)``.

        Returns
        -------
        dict
            ``{"status": "skipped", "n_windows": N}`` if fewer than
            ``cfg.min_sessions_to_train`` windows are provided.
            Otherwise, training metadata from ``_train_loop`` plus
            ``"status": "trained"``.
        """
        if not TORCH_AVAILABLE:
            return {"status": "no_torch"}

        sequences = [w for w, _ in session_windows]
        masks_np = [m for _, m in session_windows]

        if len(sequences) < self.cfg.min_sessions_to_train:
            return {"status": "skipped", "n_windows": len(sequences)}

        seq_len = sequences[0].shape[0]
        torch.manual_seed(self.cfg.random_state + self.player_id)
        self._module = _TransformerAEModule(self.cfg, seq_len)

        history, calib_seqs, calib_masks = _train_loop(
            self._module, sequences, masks_np, self.normaliser,
            self.cfg.batch_size, self.cfg.learning_rate,
            self.cfg.max_epochs, self.cfg.patience,
            self.player_id, "Transformer",
            latent_reg=getattr(self.cfg, "latent_reg", 1e-4),
        )
        self._calibrate(calib_seqs, calib_masks)
        self.model_version = (
            f"transformer_{datetime.now().strftime('%Y%m%d%H%M%S')}_p{self.player_id}"
        )
        return {"status": "trained", "n_windows": len(sequences),
                "device": str(DEVICE), **history}

    def _calibrate(self, calib_seqs, calib_masks):
        """Populate per-regime threshold stores from calibration-split windows.

        Two invariants are maintained:

        1. **EMA parity** — calibration losses are passed through the same
           EMA transformation applied in ``predict()`` so that threshold
           distributions match inference score distributions.

        2. **Regime separation** — each EMA loss is routed to both the global
           tracker and the regime-specific tracker in
           ``RegimeAwareThresholdStore``.  At inference, the regime-specific
           threshold is used when calibrated; the global is the fallback.

        Parameters
        ----------
        calib_seqs:
            List of raw ``(T, F)`` arrays from the calibration split.
        calib_masks:
            Corresponding ``(T,)`` bool arrays.
        """
        self.threshold_tracker = RegimeAwareThresholdStore()
        alpha = CONFIG.scoring.score_ema_alpha
        ema_val: Optional[float] = None
        for seq, msk in zip(calib_seqs, calib_masks):
            raw_loss = self._recon_loss(seq, msk)
            ema_val = raw_loss if ema_val is None else (
                alpha * raw_loss + (1 - alpha) * ema_val
            )
            regime_key = _REGIME_CLASSIFIER.classify(seq).key
            self.threshold_tracker.update(ema_val, regime_key)
        logger.debug("TransformerAutoencoder p%d calibration:\n%s",
                     self.player_id, self.threshold_tracker.summary())

    def recalibrate(self, coach_confirmed_normal):
        """Recalibrate thresholds from coach-confirmed normal windows.

        Called after a coach labels a set of windows as definitely normal
        (e.g. routine training session with no incidents).  The threshold
        store is rebuilt from scratch using only these windows.

        Parameters
        ----------
        coach_confirmed_normal:
            List of ``(sequence, mask)`` tuples verified as anomaly-free.
        """
        if not self.is_trained:
            return
        self._calibrate([w for w, _ in coach_confirmed_normal],
                        [m for _, m in coach_confirmed_normal])

    def _recon_loss(self, seq, mask=None):
        """Compute the reconstruction loss for one raw (un-normalised) window.

        Parameters
        ----------
        seq:
            Raw ``(T, F)`` float32 array.
        mask:
            Optional ``(T,)`` bool array.

        Returns
        -------
        float
            Scalar masked MSE reconstruction loss.
        """
        norm = torch.tensor(self.normaliser.transform(seq[np.newaxis])[0])
        mask_t = torch.tensor(mask) if mask is not None else None
        return _mse_loss_single(self._module, norm, mask_t)

    def predict(self, sequence, mask=None):
        """Predict the anomaly score for one window.

        Parameters
        ----------
        sequence:
            Raw ``(T, F)`` float32 array.
        mask:
            Optional ``(T,)`` bool array.

        Returns
        -------
        (float, bool, float)
            ``(reconstruction_loss, is_anomaly, confidence)``.
            Returns ``(0.0, False, 0.0)`` when the model is not trained.
        """
        if not self.is_trained or self._module is None:
            return 0.0, False, 0.0
        loss = self._recon_loss(sequence, mask)
        regime_key = _REGIME_CLASSIFIER.classify(sequence).key
        is_anomaly = (self.threshold_tracker.is_calibrated
                      and loss > self.threshold_tracker.threshold_for(regime_key))
        confidence = self.threshold_tracker.confidence_for(loss, regime_key)
        return loss, is_anomaly, confidence

    def predict_batch(self, sequences, masks=None):
        """Predict anomaly scores for a list of windows.

        Parameters
        ----------
        sequences:
            List of raw ``(T, F)`` float32 arrays.
        masks:
            Optional list of ``(T,)`` bool arrays aligned with ``sequences``.

        Returns
        -------
        list of (float, bool, float)
            Per-window ``(reconstruction_loss, is_anomaly, confidence)`` tuples.
        """
        if not self.is_trained or self._module is None:
            return [(0.0, False, 0.0)] * len(sequences)
        arr_norm = self.normaliser.transform(np.stack(sequences))
        X = torch.tensor(arr_norm)
        M = torch.tensor(np.stack(masks).astype(
            bool)) if masks is not None else None
        losses = _mse_loss_batch(self._module, X, M)
        results = []
        for seq, l in zip(sequences, losses):
            rk = _REGIME_CLASSIFIER.classify(seq).key
            thr = self.threshold_tracker.threshold_for(rk)
            results.append((
                float(l),
                self.threshold_tracker.is_calibrated and float(l) > thr,
                self.threshold_tracker.confidence_for(float(l), rk),
            ))
        return results

    @property
    def last_attention_weights(self) -> Optional[np.ndarray]:
        """Last encoder self-attention matrix, or ``None`` if unavailable.

        Returns the ``(B, T, T)`` attention weight tensor (averaged across
        heads) from the most recent forward pass.

        Warning
        -------
        Attention weight magnitude does **not** equal feature importance.
        Validate with SHAP or integrated gradients before presenting this
        to coaching staff as an explanation.
        """
        return self._module._last_attn if self._module else None

    def save(self) -> Path:
        """Persist the model to disk under ``MODEL_STORE``.

        Returns
        -------
        Path
            The checkpoint path.
        """
        path = MODEL_STORE / f"player_{self.player_id}_transformer_ae.pt"
        torch.save({
            "module_state":     self._module.state_dict() if self._module else None,
            "cfg":              self.cfg,
            "normaliser_state": self.normaliser.state_dict(),
            "threshold_state":  self.threshold_tracker.state_dict(),
            "model_version":    self.model_version,
            "is_trained":       self.is_trained,
            "seq_len":          CONFIG.window.window_steps,
        }, path)
        return path

    @classmethod
    def load(cls, player_id: int) -> Optional["TransformerAutoencoder"]:
        """Load a previously saved ``TransformerAutoencoder`` from disk.

        Parameters
        ----------
        player_id:
            Internal player ID used to locate the checkpoint file.

        Returns
        -------
        TransformerAutoencoder or None
            Fully restored model, or ``None`` if the file does not exist
            or PyTorch is not available.
        """
        path = MODEL_STORE / f"player_{player_id}_transformer_ae.pt"
        if not path.exists() or not TORCH_AVAILABLE:
            return None
        ckpt = torch.load(path, map_location=DEVICE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = cls(player_id, cfg=ckpt["cfg"])
        obj.normaliser = PerPlayerNormaliser.from_state_dict(
            ckpt["normaliser_state"])
        _ts = ckpt["threshold_state"]
        obj.threshold_tracker = (
            RegimeAwareThresholdStore.from_state_dict(_ts)
            if "per_regime" in _ts
            else DynamicThresholdTracker.from_state_dict(_ts)
        )
        obj.model_version = ckpt["model_version"]
        obj.is_trained = ckpt["is_trained"]
        if ckpt["module_state"] and obj.is_trained:
            seq_len = ckpt.get("seq_len", CONFIG.window.window_steps)
            obj._module = _TransformerAEModule(obj.cfg, seq_len).to(DEVICE)
            obj._module.load_state_dict(ckpt["module_state"])
            obj._module.eval()
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model: "TransformerAutoencoder",
    labeled_windows: List[Tuple[np.ndarray, np.ndarray, bool]],
    match_duration_seconds: float = 5400.0,
    window_interval_seconds: float = None,
) -> dict:
    """Compute standard anomaly detection metrics for a ``TransformerAutoencoder``.

    Evaluates the model against a set of manually labeled windows and returns
    a comprehensive metrics dictionary suitable for model-selection reports.

    Metrics Computed
    ----------------
    * ``roc_auc``        — area under the ROC curve (0.5 = random, 1.0 = perfect).
    * ``pr_auc``         — area under the precision-recall curve (handles class
                           imbalance better than ROC-AUC for rare anomalies).
    * ``precision_at_k`` — precision in the top-k scored windows where
                           k = the true anomaly count.
    * ``fp_per_90_min``  — false positives per simulated 90-minute match.
    * ``precision``      — binary precision at the current threshold.
    * ``recall``         — binary recall at the current threshold.
    * ``tp``, ``fp``, ``fn``, ``tn`` — confusion matrix elements.

    Parameters
    ----------
    model:
        A trained ``TransformerAutoencoder`` instance.
    labeled_windows:
        List of ``(sequence, mask, is_anomaly_label)`` triples.
        Both classes must be present; returns an error dict otherwise.
    match_duration_seconds:
        Simulated match duration for FP rate normalisation (default 5400 s = 90 min).
    window_interval_seconds:
        Seconds between successive windows.  Defaults to
        ``CONFIG.window.event_interval_s × CONFIG.window.window_steps``.

    Returns
    -------
    dict
        Metrics dictionary.  When sklearn is unavailable, ``roc_auc`` and
        ``pr_auc`` are set to ``None``.  Returns ``{"error": ...}`` on
        invalid input.

    Notes
    -----
    Window-level TP counts are inflated when anomalies span multiple
    consecutive windows (each window in a 10-window burst counts separately).
    Use ``evaluate_model_results()`` for event-level metrics that reflect
    operational reality.
    """
    if not model.is_trained:
        return {"error": "model not trained"}

    seqs = [w for w, _, _ in labeled_windows]
    masks = [m for _, m, _ in labeled_windows]
    labels = np.array([int(l) for _, _, l in labeled_windows])

    results = model.predict_batch(seqs, masks)
    scores = np.array([r[0] for r in results])
    preds = np.array([int(r[1]) for r in results])

    n_windows = len(labels)
    n_anomalies = int(labels.sum())
    n_normal = n_windows - n_anomalies

    if n_anomalies == 0 or n_normal == 0:
        return {"error": "labeled set needs both anomaly and normal examples",
                "n_windows": n_windows}

    metrics: dict = {
        "threshold":   model.threshold_tracker.threshold,
        "n_windows":   n_windows,
        "n_anomalies": n_anomalies,
    }

    if SKLEARN_AVAILABLE:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        logger.warning("sklearn not available — ROC-AUC / PR-AUC skipped")
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    k = n_anomalies
    top_k_idx = np.argsort(scores)[::-1][:k]
    prec_at_k = float(labels[top_k_idx].sum() / k)
    metrics["precision_at_k"] = prec_at_k

    if window_interval_seconds is None:
        window_interval_seconds = CONFIG.window.event_interval_s * CONFIG.window.window_steps
    windows_per_90 = match_duration_seconds / max(window_interval_seconds, 1.0)
    fp_rate = float((preds & (1 - labels)).sum()) / max(n_normal, 1)
    metrics["fp_per_90_min"] = round(fp_rate * windows_per_90, 2)

    tp = int((preds & labels).sum())
    fp = int((preds & (1 - labels)).sum())
    fn = int(((1 - preds) & labels).sum())
    tn = int(((1 - preds) & (1 - labels)).sum())
    metrics.update({
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall":    round(tp / max(tp + fn, 1), 4),
        "detection_latency_warning": (
            "latency not measurable without temporal ordering of windows"
        ),
    })

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Positional drift analyser
# ─────────────────────────────────────────────────────────────────────────────
class PositionalDriftAnalyzer:
    """Detects tactical zone violations from a player's recent GPS positions.

    Computes a drift score and flags the player when a significant fraction
    of recent positions fall outside their historical positional zone.

    Zone Definition
    ---------------
    The zone is centred on ``(baseline.avg_x, baseline.avg_y)`` with a
    radius of ``max(2 × baseline.position_std_radius, cfg.zone_radius_meters)``.
    Using twice the personal standard deviation (rather than a fixed radius)
    prevents false positives for players with naturally wide positional ranges
    (e.g. box-to-box midfielders).

    Drift Score
    -----------
    ``drift_score = mean_distance_from_centroid / threshold_radius``

    Values > 1.0 indicate the player is on average outside their zone.
    The flag is raised when ``fraction_outside_zone ≥ cfg.drift_fraction_threshold``.

    Parameters
    ----------
    cfg:
        ``PositionalDriftConfig`` instance.  Defaults to ``CONFIG.positional``.
    """

    def __init__(self, cfg: PositionalDriftConfig = None):
        self.cfg = cfg or CONFIG.positional

    def analyze(
        self,
        recent_positions: List[Tuple[float, float]],
        baseline: PlayerBaselineProfile,
    ) -> dict:
        """Analyse recent positions against the player's historical zone.

        Parameters
        ----------
        recent_positions:
            List of ``(x, y)`` coordinate pairs in normalised pitch units
            [0, 100] from the most recent GPS window (up to 60 ticks).
        baseline:
            Player's historical positional profile.  If ``baseline.avg_x``
            is ``None`` (insufficient history), returns all-zero/False output.

        Returns
        -------
        dict
            Keys:

            * ``drift_score`` — float, ratio of mean distance to threshold radius.
            * ``is_flagged`` — bool, ``True`` when fraction outside zone ≥
              ``cfg.drift_fraction_threshold``.
            * ``fraction_outside_zone`` — float in [0.0, 1.0].
            * ``avg_distance_from_norm_m`` — mean distance from centroid in metres.
            * ``threshold_radius_m`` — effective zone radius in metres.
        """
        if not recent_positions or baseline.avg_x is None:
            return {"drift_score": 0.0, "is_flagged": False, "fraction_outside_zone": 0.0}
        std_r = baseline.position_std_radius or self.cfg.zone_radius_meters
        thr = max(std_r * 2.0, self.cfg.zone_radius_meters)
        dists = [math.sqrt((x - baseline.avg_x)**2 + (y - baseline.avg_y)**2)
                 for x, y in recent_positions]
        frac = sum(1 for d in dists if d > thr) / len(dists)
        avg_d = float(np.mean(dists))
        return {
            "drift_score":              round(avg_d / (thr or 1.0), 3),
            "is_flagged":               frac >= self.cfg.drift_fraction_threshold,
            "fraction_outside_zone":    round(frac, 3),
            "avg_distance_from_norm_m": round(avg_d, 2),
            "threshold_radius_m":       round(thr, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight evaluate helper used by PatternAnalysisEngine.evaluate_player()
# ─────────────────────────────────────────────────────────────────────────────
def _merge_contiguous_events(
    binary_sequence: np.ndarray,
    gap_tolerance: int = 2,
) -> List[Tuple[int, int]]:
    """Merge a window-level binary sequence into contiguous anomaly episodes.

    Adjacent positive windows within ``gap_tolerance`` of each other are
    bridged into one event.  This converts a window-level prediction array
    into an episode-level event list, which is the appropriate unit for
    operational evaluation: a coach sees one substitution alert, not one
    alert per 2-second window during a 5-minute episode.

    Parameters
    ----------
    binary_sequence:
        1-D integer array of shape ``(N,)``.  1 = positive (anomaly), 0 = normal.
    gap_tolerance:
        Maximum number of consecutive negative windows that may appear inside
        a positive episode without ending it.  Default 2 (≈ 4 seconds at
        2-second window steps).

    Returns
    -------
    list of (int, int)
        List of ``(start_idx, end_idx)`` pairs (inclusive) for each
        detected episode.  Empty list if no positive windows exist.

    Examples
    --------
    >>> _merge_contiguous_events(np.array([0,1,1,0,1,0,0,0,1]), gap_tolerance=2)
    [(1, 4), (8, 8)]
    """
    events: List[Tuple[int, int]] = []
    in_event = False
    start = 0
    gap_count = 0

    for i, val in enumerate(binary_sequence):
        if val:
            if not in_event:
                in_event = True
                start = i
                gap_count = 0
            else:
                gap_count = 0
        else:
            if in_event:
                gap_count += 1
                if gap_count > gap_tolerance:
                    events.append((start, i - gap_count))
                    in_event = False
                    gap_count = 0

    if in_event:
        events.append((start, len(binary_sequence) - 1))

    return events


def _event_level_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    gap_tolerance: int = 2,
) -> dict:
    """Compute event-level precision, recall, and F1 for anomaly detection.

    Unlike window-level metrics, event-level metrics count one TP per
    detected episode regardless of how many windows it spans.  This
    prevents TP inflation from sustained alerts and gives a more honest
    picture of operational performance.

    TP/FP Assignment
    ----------------
    * A predicted episode is a **TP** if it overlaps with any ground-truth episode.
    * Multiple predicted episodes overlapping the *same* ground-truth episode
      count as ONE TP + (n-1) FPs.
    * A ground-truth episode not overlapped by any predicted episode is a FN.

    Parameters
    ----------
    pred_labels:
        ``(N,)`` int array — predicted window-level labels.
    true_labels:
        ``(N,)`` int array — ground-truth window-level labels.
    gap_tolerance:
        Passed through to ``_merge_contiguous_events()``.

    Returns
    -------
    dict
        Keys: ``event_tp``, ``event_fp``, ``event_fn``,
        ``event_precision``, ``event_recall``, ``event_f1``,
        ``n_pred_events``, ``n_true_events``.
    """
    pred_events = _merge_contiguous_events(pred_labels, gap_tolerance)
    true_events = _merge_contiguous_events(true_labels, gap_tolerance)

    matched_gt: set = set()
    event_tp = 0
    event_fp = 0

    for ps, pe in pred_events:
        pred_set = set(range(ps, pe + 1))
        matched = False
        for gt_idx, (gs, ge) in enumerate(true_events):
            if pred_set & set(range(gs, ge + 1)):
                matched = True
                matched_gt.add(gt_idx)
                break
        if matched:
            event_tp += 1
        else:
            event_fp += 1

    event_fn = len(true_events) - len(matched_gt)

    ep = event_tp / max(event_tp + event_fp, 1)
    er = event_tp / max(event_tp + event_fn, 1)
    ef1 = 2 * ep * er / max(ep + er, 1e-8)

    return {
        "event_tp":         event_tp,
        "event_fp":         event_fp,
        "event_fn":         event_fn,
        "event_precision":  round(ep,  4),
        "event_recall":     round(er,  4),
        "event_f1":         round(ef1, 4),
        "n_pred_events":    len(pred_events),
        "n_true_events":    len(true_events),
    }


def _pr_curve_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fp_per_90_min: float = 2.0,
    window_interval_s: float = 120.0,
) -> float:
    """Select a detection threshold calibrated to a false-positive budget.

    Walks the precision-recall curve from the tightest to the loosest
    threshold and returns the first value that keeps the per-90-minute
    false-positive rate at or below ``target_fp_per_90_min``.

    This is the operationally correct alternative to choosing an arbitrary
    quantile of the calibration loss distribution.  A head coach or sports
    scientist can specify a meaningful budget (e.g. "no more than 3 false
    alerts per match") and the threshold is set to meet it exactly.

    Parameters
    ----------
    scores:
        ``(N,)`` float array of raw model anomaly scores (higher = more
        anomalous).
    labels:
        ``(N,)`` int array — 1 = anomalous window, 0 = normal.
    target_fp_per_90_min:
        Maximum tolerable false alerts per 90-minute period.
    window_interval_s:
        Seconds between successive windows (step size).

    Returns
    -------
    float
        The selected threshold.  Falls back to the tightest threshold when
        no value achieves the FP budget, and to ``median(scores)`` when
        sklearn is unavailable.
    """
    if not SKLEARN_AVAILABLE:
        return float(np.median(scores))

    from sklearn.metrics import precision_recall_curve

    n_normal = int((labels == 0).sum())
    windows_per_90 = (90 * 60) / max(window_interval_s, 1.0)
    max_fpr = target_fp_per_90_min / \
        max(windows_per_90 * n_normal / len(labels), 1e-6)

    prec, rec, thresholds = precision_recall_curve(labels, scores)
    for thr in sorted(thresholds):
        preds = (scores >= thr).astype(int)
        fp = int((preds & (1 - labels)).sum())
        fpr = fp / max(n_normal, 1)
        if fpr <= max_fpr:
            return float(thr)

    best_thr = float(thresholds[-1])
    return best_thr


def evaluate_model_results(
    results: List[Tuple[float, bool, float]],
    labels: List[bool],
    threshold: float,
    ema_smoothed: bool = False,
    gap_tolerance: int = 2,
    target_fp_per_90_min: float = 2.0,
    window_interval_s: Optional[float] = None,
) -> dict:
    """
    Operational evaluation for anomaly-alert systems.

    Evaluates:
    - window ranking quality,
    - event/episode detection quality,
    - alert fragmentation,
    - operational FP burden,
    - lead time.

    NOTE:
    Event-level metrics are PRIMARY.
    Window metrics are SECONDARY diagnostics.
    """

    scores = np.asarray(
        [r[0] for r in results],
        dtype=np.float64,
    )

    preds = np.asarray(
        [int(r[1]) for r in results],
        dtype=np.int32,
    )

    labs = np.asarray(
        [int(v) for v in labels],
        dtype=np.int32,
    )

    n_windows = len(labs)

    if n_windows == 0:
        return {
            "error": "empty evaluation set",
            "n_windows": 0,
        }

    n_anomalies = int(labs.sum())
    n_normal = n_windows - n_anomalies

    if n_anomalies == 0 or n_normal == 0:
        return {
            "error": "need both anomaly and normal examples",
            "n_windows": n_windows,
        }

    # ─────────────────────────────────────────────
    # Duration normalization
    # ─────────────────────────────────────────────
    if window_interval_s is None:
        window_interval_s = (
            CONFIG.window.event_interval_s *
            CONFIG.window.window_steps
        )

    window_interval_s = max(
        float(window_interval_s),
        1.0,
    )

    evaluated_duration_s = (
        n_windows * window_interval_s
    )

    # ─────────────────────────────────────────────
    # Ranking metrics (SECONDARY)
    # ─────────────────────────────────────────────
    if SKLEARN_AVAILABLE:
        roc_auc = float(
            roc_auc_score(labs, scores)
        )

        pr_auc = float(
            average_precision_score(labs, scores)
        )
    else:
        roc_auc = None
        pr_auc = None

    k = max(n_anomalies, 1)

    topk = np.argsort(scores)[::-1][:k]

    precision_at_k = float(
        labs[topk].sum() / k
    )

    # ─────────────────────────────────────────────
    # Window confusion matrix
    # ─────────────────────────────────────────────
    tp = int(((preds == 1) & (labs == 1)).sum())
    fp = int(((preds == 1) & (labs == 0)).sum())
    fn = int(((preds == 0) & (labs == 1)).sum())
    tn = int(((preds == 0) & (labs == 0)).sum())

    window_precision = (
        tp / max(tp + fp, 1)
    )

    window_recall = (
        tp / max(tp + fn, 1)
    )

    window_f1 = (
        2 * window_precision * window_recall /
        max(window_precision + window_recall, 1e-12)
    )

    fp_per_90_window = (
        fp * (5400.0 / evaluated_duration_s)
    )

    # ─────────────────────────────────────────────
    # Episode extraction
    # ─────────────────────────────────────────────
    gt_episodes = extract_episodes(labs)
    pred_episodes = extract_episodes(preds)

    # ─────────────────────────────────────────────
    # Event-level matching (PRIMARY)
    # ─────────────────────────────────────────────
    matched_gt = set()

    event_tp = 0
    event_fp = 0

    lead_times = []

    for ps, pe in pred_episodes:

        overlap_found = False

        for gi, (gs, ge) in enumerate(gt_episodes):

            overlap = not (
                pe < gs or ps > ge
            )

            if overlap:

                overlap_found = True

                matched_gt.add(gi)

                lead_times.append(
                    (ps - gs) * window_interval_s
                )

                break

        if overlap_found:
            event_tp += 1
        else:
            event_fp += 1

    event_fn = (
        len(gt_episodes) -
        len(matched_gt)
    )

    event_precision = (
        event_tp / max(event_tp + event_fp, 1)
    )

    event_recall = (
        event_tp / max(event_tp + event_fn, 1)
    )

    event_f1 = (
        2 * event_precision * event_recall /
        max(event_precision + event_recall, 1e-12)
    )

    fp_per_90_event = (
        event_fp * (5400.0 / evaluated_duration_s)
    )

    fragmentation_ratio = (
        len(pred_episodes) /
        max(len(gt_episodes), 1)
    )

    mean_lead_time_s = (
        float(np.mean(lead_times))
        if lead_times else None
    )

    alert_stability = (
        1.0 / max(fragmentation_ratio, 1e-6)
    )

    # ─────────────────────────────────────────────
    # Operational threshold search
    # ─────────────────────────────────────────────
    candidate_thresholds = np.linspace(
        np.percentile(scores, 80),
        np.percentile(scores, 99.5),
        50,
    )

    best_threshold = threshold
    best_fp90 = float("inf")
    best_precision = 0.0
    best_recall = 0.0

    for th in candidate_thresholds:

        p = (scores >= th).astype(np.int32)

        fp_tmp = int(
            ((p == 1) & (labs == 0)).sum()
        )

        tp_tmp = int(
            ((p == 1) & (labs == 1)).sum()
        )

        fn_tmp = int(
            ((p == 0) & (labs == 1)).sum()
        )

        fp90 = (
            fp_tmp *
            (5400.0 / evaluated_duration_s)
        )
        precision = (
            tp_tmp /
            max(tp_tmp + fp_tmp, 1)
        )
        recall = (
            tp_tmp /
            max(tp_tmp + fn_tmp, 1)
        )

        if fp90 <= target_fp_per_90_min:
            best_threshold = float(th)
            best_fp90 = float(fp90)
            best_precision = float(precision)
            best_recall = float(recall)
            break

    # ─────────────────────────────────────────────
    # FINAL METRICS
    # ─────────────────────────────────────────────
    return {

        # metadata
        "threshold": threshold,
        "ema_smoothed": ema_smoothed,

        # dataset
        "n_windows": n_windows,
        "n_anomalies": n_anomalies,
        "n_normal": n_normal,

        # PRIMARY operational metrics
        "event_precision": float(event_precision),
        "event_recall": float(event_recall),
        "event_f1": float(event_f1),

        "fragmentation_ratio": float(fragmentation_ratio),
        "alert_stability": float(alert_stability),

        "mean_lead_time_s": mean_lead_time_s,

        "fp_per_90_min_event": float(fp_per_90_event),

        "event_tp": int(event_tp),
        "event_fp": int(event_fp),
        "event_fn": int(event_fn),

        "n_gt_episodes": len(gt_episodes),
        "n_pred_episodes": len(pred_episodes),

        # SECONDARY research metrics
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_k": precision_at_k,

        # window metrics
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "window_precision": float(window_precision),
        "window_recall": float(window_recall),
        "window_f1": float(window_f1),

        "fp_per_90_min_window": float(fp_per_90_window),

        # backward compatibility
        "fp_per_90_min": float(fp_per_90_event),

        # deployment recommendation
        "pr_curve_threshold": {
            "recommended_threshold": best_threshold,
            "target_fp_per_90_min": target_fp_per_90_min,
            "achieved_fp_per_90_min": best_fp90,
            "precision": best_precision,
            "recall": best_recall,
            "deployment_note":
                (
                    "Threshold selected under operational FP budget."
                    if np.isfinite(best_fp90)
                    else "No threshold satisfied FP budget."
                ),
        },
    }
# ─────────────────────────────────────────────────────────────────────────────
# Pattern Analysis Engine
# ─────────────────────────────────────────────────────────────────────────────


class PatternAnalysisEngine:
    """Top-level orchestrator for real-time player anomaly detection.

    Owns the shared LSTM backbone, per-player calibration stores, positional
    drift analyser, and workload tracker.  Provides a single ``analyze()``
    method that ingests one live telemetry event per call and returns a
    fully populated ``AnomalyResult`` when the window buffer is ready.

    Architecture
    ------------
    ::

        live_event → SequenceWindowBuilder → (sequence, mask)
                            ↓
                    SharedBackboneAutoencoder.predict()
                            ↓ reconstruction_loss
                    RegimeAwareThresholdStore.threshold_for()
                            ↓ is_anomaly, confidence
                    EMA smoothing (score_ema_alpha)
                            ↓
                    PositionalDriftAnalyzer.analyze()
                    WorkloadTrendTracker.compute_load_ratios()
                            ↓
                    AnomalyResult

    Scalability
    -----------
    Per-player isolated models become operationally heavy beyond ~500 players.
    The shared backbone (``SharedBackboneAutoencoder``) addresses this by
    training one model on all players simultaneously.  The per-player
    ``RegimeAwareThresholdStore`` objects are lightweight (a few hundred
    float values each) and add negligible memory overhead.

    Thread Safety
    -------------
    Not thread-safe.  ``_ema_scores``, ``_position_buffers``, and
    ``_threshold_trackers`` are mutated during ``analyze()``.  Use one
    engine per asyncio event loop or per OS process.

    Attributes
    ----------
    window_builder:
        ``SequenceWindowBuilder`` managing per-player sliding buffers.
    drift_analyzer:
        ``PositionalDriftAnalyzer`` for tactical zone monitoring.
    workload_tracker:
        ``WorkloadTrendTracker`` for ACWR computation.
    """

    def __init__(self):
        self.window_builder = SequenceWindowBuilder()
        self.drift_analyzer = PositionalDriftAnalyzer()
        self.workload_tracker = WorkloadTrendTracker()
        self.inference_engine = InferenceEngine()
        self._baselines:         Dict[int, PlayerBaselineProfile] = {}
        self._position_buffers:  Dict[int, List[Tuple[float, float]]] = {}
        self._ema_smoothers:     Dict[int, EMASmoother] = {}
        self._ema_scores:        Dict[int, float] = {}
        self._threshold_trackers: Dict[int, RegimeAwareThresholdStore] = {}
        self.alert_manager = AlertManager()
        self._shared_model: Optional[SharedBackboneAutoencoder] = None

    def reset_ema_state(
    self,
    player_id: int,
) -> None:
        """
        Reset EMA smoothing state for one player.
        Used after live stream/session discontinuities.
        """

        self._ema_smoothers.pop(player_id, None)
        self._ema_scores.pop(player_id, None)

        # logger.info(
        #     "EMA RESET | player=%s",
        #     player_id,
        # )

    def reset_player_runtime_state(
    self,
    player_id: int,
) -> None:
        """
        Reset transient online inference state for a player.

        Called when:
        - session changes
        - large temporal discontinuities occur
        - live stream reconnects
        """

        self._ema_smoothers.pop(player_id, None)
        self._ema_scores.pop(player_id, None)

        self._position_buffers.pop(player_id, None)

        logger.info(
            "RUNTIME RESET | player=%s",
            player_id,
        )

    def register_player(
        self,
        player_id: int,
        baseline:  PlayerBaselineProfile,
        model:     Optional[object] = None,
    ) -> None:
        """Register a player so the engine can process their events.

        Must be called for every player before ``analyze()`` is invoked
        for that player.  The ``baseline`` argument provides the player's
        historical performance profile used for fatigue detection, drift
        scoring, and workload comparison.

        Parameters
        ----------
        player_id:
            Internal integer player identifier.
        baseline:
            ``PlayerBaselineProfile`` built from the player's historical
            sessions.  Used for personalised threshold logic and drift scoring.
        model:
            Deprecated / ignored.  Included only for backward compatibility
            with call-sites that were written for the old per-player model API.
        """
        self._baselines[player_id] = baseline
        self._position_buffers[player_id] = []

    def train_player_model(
        self,
        all_windows: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
    ) -> dict:
        """Train the shared backbone and calibrate per-player thresholds."""
        result = self.inference_engine.train(all_windows)
        self._shared_model = self.inference_engine._shared_model
        return result

    def get_model_version(self):
        if self._shared_model is None:
            raise RuntimeError("Shared model not initialized")
        return self._shared_model.model_version

    def analyze(
        self,
        player_id:   int,
        live_event:  dict,
        sessions_df: pd.DataFrame,
    ) -> Optional[AnomalyResult]:
        """Process one live telemetry event and return an anomaly result.
 
        Appends the event to the per-player sliding buffer.  Returns ``None``
        while the buffer is still filling, then delegates to
        ``analyze_window()`` once a complete window is available.
 
        Old callers that use ``analyze()`` continue to work unchanged;
        new live-serving callers should use ``process_window_direct()``
        (which calls ``analyze_window()`` directly via ``build_live_window``).
        """
        baseline = self._baselines.get(player_id)
        if baseline is None:
            return None
 
        result = self.window_builder.add_event(live_event)
        if result is None:
            return None
        sequence, mask = result
 
        return self.analyze_window(
            player_id=player_id,
            sequence=sequence,
            mask=mask,
            live_event=live_event,
            sessions_df=sessions_df,
        )
    
    def analyze_window(
        self,
        player_id:   int,
        sequence:    np.ndarray,
        mask:        np.ndarray,
        live_event:  dict,
        sessions_df: pd.DataFrame,
    ) -> Optional[AnomalyResult]:
        """Run the full ML + heuristic scoring pipeline on a completed window.
 
        Called by ``analyze()`` (streaming path) and ``process_window_direct()``
        (live-serving path via ``build_live_window``).  Stateless with respect
        to the window buffer — ``sequence`` and ``mask`` must already be built
        by the caller.
 
        Parameters
        ----------
        player_id:
            Internal player ID.
        sequence:
            ``(window_steps, N_SEQUENCE_FEATURES)`` float32 array.
        mask:
            ``(window_steps,)`` bool array — True for real sensor ticks.
        live_event:
            The most recent raw telemetry dict (used for positional buffer,
            elapsed_seconds, enriched feature keys).
        sessions_df:
            Historical session DataFrame for ACWR workload computation.
 
        Returns
        -------
        AnomalyResult or None
            Fully populated result, or ``None`` when the player's baseline
            is not registered.
        """
        baseline = self._baselines.get(player_id)
        if baseline is None:
            return None
 
        # Run the ML inference engine for raw scoring
        raw_loss, model_type = self.inference_engine.score_window(
            player_id, sequence, mask
        )

        # logger.info(
        #         "WINDOW INFERENCE | player=%s",
        #         player_id,
        #     )
 
        x = live_event.get("x_pitch")
        y = live_event.get("y_pitch")
        # Gate position-buffer updates on TVL confidence, not field presence.
        # A GPS dropout can zero speed while x/y are still present (stale/frozen).
        # Only update when telemetry is trustworthy (confidence >= 0.8 = VALID).
        _tvl_confidence = float(live_event.get("_tvl_confidence", 1.0))
        _telemetry_valid = _tvl_confidence >= 0.8

        if x is not None and y is not None and _telemetry_valid:
            buf = self._position_buffers.setdefault(player_id, [])
            buf.append((float(x), float(y)))
            if len(buf) > 60:
                self._position_buffers[player_id] = buf[-60:]
 
        drift = self.drift_analyzer.analyze(
            self._position_buffers.get(player_id, []), baseline
        )
 
        acwr, workload_status = 1.0, "optimal"
        if not sessions_df.empty and "total_distance_m" in sessions_df.columns:
            w = self.workload_tracker.compute_load_ratios(0, sessions_df)
            acwr = w.get("acwr", 1.0)
            workload_status = w.get("workload_status", "optimal")
 
        workload_flag = acwr > 1.5 or acwr < 0.8
        elapsed = float(live_event.get("elapsed_seconds", 0))
 
        last = sequence[-1]
        fv = {SEQUENCE_FEATURE_NAMES[i]: float(last[i])
              for i in range(N_SEQUENCE_FEATURES)}
        fv.update({
            "acwr":                     float(acwr),
            "reconstruction_loss":      float(raw_loss),
            "positional_drift_score":   float(drift["drift_score"]),
            "mask_completeness":        float(mask.mean()),
        })
 
        for _enriched_key in ("fatigue_decay_residual", "speed_drop_pct",
                              "coach_fatigue_severity",
                              "coach_pre_match_status_encoded"):
            if _enriched_key in live_event:
                fv[_enriched_key] = float(live_event[_enriched_key])
 
        # Apply EMA smoothing to raw loss.
        # Gate on TVL confidence: corrupted windows produce artificially low
        # reconstruction loss (zero-filled features → easy reconstruction).
        # Updating the EMA from such windows pulls thresholds down and suppresses
        # future anomaly detection. Use the last known value during degradation.
        alpha_ema = CONFIG.scoring.score_ema_alpha
        smoother = self._ema_smoothers.setdefault(
            player_id, EMASmoother(alpha_ema))
        if _telemetry_valid:
            smoothed_loss = smoother.update(raw_loss)
            self._ema_scores[player_id] = smoothed_loss
        else:
            # Hold: read last known EMA without updating.
            # _tvl_confidence < 0.8 means DEGRADED; we don't know the true loss.
            smoothed_loss = self._ema_scores.get(player_id, raw_loss)
            logger.debug(
                "EMA hold | player=%d tvl_confidence=%.2f smoothed=%.6f (not updated)",
                player_id, _tvl_confidence, smoothed_loss,
            )

        regime_key = _REGIME_CLASSIFIER.classify(sequence).key
        tracker = self.inference_engine.get_tracker(player_id)
        # logger.info(
        #         "DEBUG SCORE | player=%s raw=%.6f smooth=%.6f regime=%s threshold=%.6f",
        #         player_id,
        #         raw_loss,
        #         smoothed_loss,
        #         regime_key,
        #         tracker.threshold_for(regime_key) if tracker else -1.0,
        #     )
        
        # logger.info(
        #         "TRACKER STATUS | player=%s calibrated=%s threshold=%s regime=%s",
        #         player_id,
        #         tracker.is_calibrated if tracker else None,
        #         tracker.threshold_for(regime_key) if tracker and tracker.is_calibrated else None,
        #         regime_key,
        #     )
        

        if tracker and tracker.is_calibrated:
            is_anomaly = smoothed_loss > tracker.threshold_for(regime_key)
            confidence = tracker.confidence_for(smoothed_loss, regime_key)
        else:
            is_anomaly, confidence = False, 0.0
 
        baseline_speed = (baseline.distance_mean / (90 * 60)
                          if baseline.distance_mean > 0 else 3.5)
        speed_ratio = fv.get("speed_ms", 999) / max(baseline_speed, 0.1)
        speed_low = speed_ratio < 0.55
        sprint_low = fv.get("sprint_flag", 1) == 0
        late_in_game = elapsed > 2700
        fatigue_flag = is_anomaly and (speed_low or sprint_low) and late_in_game
 
        _ACTIVE_LEVELS = (AlertLevel.WARNING, AlertLevel.SUSTAINED, AlertLevel.CRITICAL)
 
        anomaly_state = self.alert_manager.process_signal(
            player_id,
            "anomaly",
            is_anomaly,
        )

        fatigue_state = self.alert_manager.process_signal(
            player_id,
            "fatigue",
            fatigue_flag,
        )

        drift_state = self.alert_manager.process_signal(
            player_id,
            "drift",
            drift["is_flagged"],
        )

        workload_state = self.alert_manager.process_signal(
            player_id,
            "workload",
            workload_flag,
        )

        is_anomaly_alert = anomaly_state in _ACTIVE_LEVELS
        fatigue_alert    = fatigue_state in _ACTIVE_LEVELS
        drift_alert      = drift_state in _ACTIVE_LEVELS
        workload_alert   = workload_state in _ACTIVE_LEVELS

        # anomaly_state = self.alert_manager.get_state(player_id, "anomaly")
 
        return AnomalyResult(
            player_id=player_id,
            external_id=baseline.external_id,
            ts=datetime.now(tz=timezone.utc),
            anomaly_score=raw_loss,
            is_anomaly=is_anomaly_alert,
            confidence=confidence,
            feature_vector=fv,
            sequence_shape=sequence.shape,
            raw_sequence=sequence,
            raw_mask=mask,
            fatigue_flag=fatigue_alert,
            positional_drift_flag=drift_alert,
            workload_flag=workload_alert,
            workload_status=workload_status,
            recommendation_type=self._recommend(
                is_anomaly_alert, fatigue_alert,
                drift_alert, workload_alert, confidence,
            ),
            model_type=model_type,
            alert_level=anomaly_state,
            persistence_windows=getattr(
                self.alert_manager.get_state(player_id, "anomaly"),
                "persistence_count",
                0,
            ),
        )
    
    def build_training_sequences(
        self,
        events_df: pd.DataFrame,
        sessions_df: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Build all training windows from raw events and session DataFrames.

        Returns
        -------
        List of:
            (
                sequence,
                mask,
                session_id,
            )
        """
        all_pairs: List[
            Tuple[np.ndarray, np.ndarray, int]
        ] = []

        # Faster than iterrows()
        for session in sessions_df.itertuples(index=False):
            session_id = int(session.session_id)
            sess_ev = events_df.loc[
                events_df["session_id"] == session_id
            ]
            if sess_ev.empty:
                continue
            windows = self.window_builder.build_from_session(
                sess_ev
            )

            # Attach session_id to every window
            for seq, mask in windows:
                all_pairs.append(
                    (
                        seq.astype(np.float32, copy=False),
                        mask.astype(bool, copy=False),
                        session_id,
                    )
                )

        return all_pairs

    def analyze_players_sequential(
        self,
        player_ids: List[int],
        sequences:  List[np.ndarray],
        masks:      Optional[List[np.ndarray]] = None,
    ) -> Dict[int, Tuple[float, bool, float]]:
        """Run batch inference for multiple players using the shared backbone.

        Processes each player sequentially (not vectorised across players).
        Per-player regime-aware thresholds are applied from the inference engine.

        This method is suitable for end-of-session batch scoring or
        evaluation pipelines where all windows are available in memory.
        For real-time live inference, use ``analyze()`` instead.

        Parameters
        ----------
        player_ids:
            List of internal player IDs.
        sequences:
            List of raw ``(T, F)`` arrays, one per player, aligned with
            ``player_ids``.
        masks:
            Optional list of ``(T,)`` bool arrays.  Defaults to all-``None``
            (all timesteps treated as real).

        Returns
        -------
        dict
            Mapping from player ID to ``(reconstruction_loss, is_anomaly,
            confidence)``.  Returns ``(0.0, False, 0.0)`` for any player
            when the shared model is absent or untrained.
        """
        if masks is None:
            masks = [None] * len(player_ids)
        results = {}
        for pid, seq, msk in zip(player_ids, sequences, masks):
            raw_loss, is_anomaly, confidence, _ = self.inference_engine.score_window(
                pid, seq, msk
            )
            results[pid] = (raw_loss, is_anomaly, confidence)
        return results

    def evaluate_player(
        self,
        player_id: int,
        labeled_windows: List[Tuple[np.ndarray, np.ndarray, bool]],
    ) -> dict:
        """
        Batched evaluation with EMA-consistent operational scoring.
        """

        if not self.inference_engine.is_ready:
            return {
                "error": "no shared model trained"
            }

        shared = self.inference_engine._shared_model

        if shared._player_index.get(player_id) is None:
            return {
                "error": f"player {player_id} not registered in shared model"
            }

        tracker = self._threshold_trackers.get(player_id)

        if tracker is None or not tracker.is_calibrated:
            return {
                "error": f"player {player_id} tracker not calibrated"
            }

        if not labeled_windows:
            return {
                "error": "empty evaluation set"
            }

        # ─────────────────────────────────────────────
        # Batch extraction
        # ─────────────────────────────────────────────
        seqs = np.stack(
            [w for w, _, _ in labeled_windows]
        ).astype(np.float32, copy=False)

        masks = np.stack(
            [m for _, m, _ in labeled_windows]
        ).astype(bool, copy=False)

        labels = [
            bool(l)
            for _, _, l in labeled_windows
        ]

        pids = np.full(
            len(seqs),
            player_id,
            dtype=np.int64,
        )

        # ─────────────────────────────────────────────
        # ONE batched forward pass
        # ─────────────────────────────────────────────
        raw_losses = shared.predict_batch(
            player_ids=pids,
            sequences=seqs,
            masks=masks,
            normalised=False,
        )

        # ─────────────────────────────────────────────
        # EMA smoothing
        # ─────────────────────────────────────────────
        alpha_ema = CONFIG.scoring.score_ema_alpha

        smoother = EMASmoother(alpha_ema)

        smoothed_losses = np.asarray(
            [
                smoother.update(float(loss))
                for loss in raw_losses
            ],
            dtype=np.float32,
        )

        # ─────────────────────────────────────────────
        # Regime caching
        # ─────────────────────────────────────────────
        regime_cache = {}

        regime_keys = []

        for seq in seqs:

            cache_key = hash(
                seq[-1].tobytes()
            )

            regime_key = regime_cache.get(cache_key)

            if regime_key is None:

                regime_key = (
                    _REGIME_CLASSIFIER
                    .classify(seq)
                    .key
                )

                regime_cache[cache_key] = regime_key

            regime_keys.append(regime_key)


        default_threshold = tracker.threshold_for(
            "default"
        )

        # ─────────────────────────────────────────────
        # Operational predictions
        # ─────────────────────────────────────────────
        results = []

        for loss, regime_key in zip(
            smoothed_losses,
            regime_keys,
        ):

            base_thr = tracker.threshold_for(regime_key)

            regime_losses = np.asarray(
                tracker.get_regime_losses(regime_key),
                dtype=np.float32,
            )

            if len(regime_losses) >= 20:
                operational_threshold = max(
                    base_thr,
                    np.percentile(regime_losses, 99.9),
                )
            else:
                operational_threshold = base_thr

            is_anom = (
                loss > operational_threshold
            )

            conf = tracker.confidence_for(
                loss,
                regime_key,
            )

            results.append(
                (
                    float(loss),
                    bool(is_anom),
                    float(conf),
                )
            )
        # ─────────────────────────────────────────────
        # Evaluation metrics
        # ─────────────────────────────────────────────
        window_interval_s = float(
            CONFIG.window.event_interval_s
            * CONFIG.window.window_steps
        )
        return evaluate_model_results(
            results=results,
            labels=labels,
            threshold=default_threshold,
            ema_smoothed=True,
            window_interval_s=window_interval_s,
        )

    def _recommend(
        self,
        is_anomaly: bool,
        fatigue:    bool,
        drift:      bool,
        workload:   bool,
        conf:       float,
    ) -> Optional[str]:
        """Select the highest-priority coaching recommendation.

        Implements a strict priority ladder so that the most urgent alert
        is always surfaced first.  At most one recommendation is returned
        per inference call.

        Priority Order
        --------------
        1. ``"substitution"``    — anomaly + fatigue + confidence > 85 %.
        2. ``"fatigue_alert"``   — fatigue flag only (without high confidence).
        3. ``"positional_drift"`` — tactical zone violation.
        4. ``"workload_warning"`` — ACWR outside safe band [0.8, 1.5].
        5. ``"anomaly_flag"``    — model anomaly + confidence > 75 % (no fatigue).
        6. ``None``              — no actionable signal.

        Parameters
        ----------
        is_anomaly:
            Whether the smoothed reconstruction loss exceeds the threshold.
        fatigue:
            Whether the fatigue flag was raised in ``analyze()``.
        drift:
            Whether the positional drift flag was raised.
        workload:
            Whether the ACWR workload flag was raised.
        conf:
            Confidence (empirical CDF percentile) of the anomaly score.

        Returns
        -------
        str or None
            The recommendation string, or ``None`` when no action is needed.
        """
        if fatigue and is_anomaly and conf > 0.85:
            return "substitution"
        if fatigue:
            return "fatigue_alert"
        if drift:
            return "positional_drift"
        if workload:
            return "workload_warning"
        if is_anomaly and conf > 0.75:
            return "anomaly_flag"
        return None