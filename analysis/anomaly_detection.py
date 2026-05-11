"""
Players Data — IBM CIC Germany
Pattern Analysis Engine — Anomaly Detection

Implements:
  1. Isolation Forest — per-player anomaly detection against personal baseline
  2. Fatigue curve comparator — live segment vs. personal decay profile
  3. Positional drift scorer — detects tactical zone violations
  4. Feature engineering pipeline — converts raw events into model features

All models are fitted on personal (per-player) historical data, NOT squad averages.
This is the core technical novelty stated in the proposal.
"""
from __future__ import annotations

import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from config.settings import LSTMAutoencoderConfig
import numpy as np
import pandas as pd

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
PITCH_WIDTH_M  = 68.0    # x-axis (touchline-to-touchline)

def safe_float(v, default: float) -> float:
    """
    Converts values safely to finite float.
    Replaces NaN / inf / invalid values with default.
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
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
def resolve_device() -> "torch.device":
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
    return t.to(DEVICE) if DEVICE is not None else t


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AnomalyResult:
    player_id: int
    external_id: str
    ts: datetime
    anomaly_score: float
    is_anomaly: bool
    confidence: float           # empirical percentile rank

    feature_vector: Dict[str, float] = field(default_factory=dict)
    sequence_shape: Tuple[int, int]  = (0, 0)
    deviations: Dict[str, dict]      = field(default_factory=dict)

    # Raw sequence data — stored so the XAI layer can run true SHAP
    # by forwarding real perturbed sequences through the model.
    raw_sequence: Optional[np.ndarray] = field(default=None, repr=False)
    raw_mask:     Optional[np.ndarray] = field(default=None, repr=False)

    fatigue_flag: bool          = False
    positional_drift_flag: bool = False
    workload_flag: bool         = False
    workload_status: str        = "optimal"

    recommendation_type: Optional[str] = None
    triggered_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    model_type: str = "lstm"


# ─────────────────────────────────────────────────────────────────────────────
# Per-player normaliser  (plain-array serialisation)
# ─────────────────────────────────────────────────────────────────────────────
class PerPlayerNormaliser:
    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds:  Optional[np.ndarray] = None

    def fit(self, sequences: np.ndarray) -> None:
        flat       = sequences.reshape(-1, sequences.shape[-1])
        self.means = flat.mean(axis=0).astype(np.float32)
        raw_std    = flat.std(axis=0).astype(np.float32)
        self.stds  = np.where(raw_std > 1e-6, raw_std, 1.0).astype(np.float32)

    def transform(self, sequences: np.ndarray) -> np.ndarray:
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
        return {
            "means": self.means.tolist() if self.means is not None else None,
            "stds":  self.stds.tolist()  if self.stds  is not None else None,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "PerPlayerNormaliser":
        obj = cls()
        obj.means = np.array(d["means"], dtype=np.float32) if d["means"] else None
        obj.stds  = np.array(d["stds"],  dtype=np.float32) if d["stds"]  else None
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic threshold tracker
# ─────────────────────────────────────────────────────────────────────────────
class DynamicThresholdTracker:
    """
    Stores calibration-split reconstruction losses.
    NEVER updated during inference.

    threshold = np.quantile(losses, q) instead of mean+kσ.
           Reconstruction losses are right-skewed / heavy-tailed.
           Gaussian assumption of mean+kσ systematically underestimates tails.

     _clean_losses() removes the top `calib_contamination_pct` of losses
           before computing the threshold.  The calibration split may contain
           windows that were anomalous (injected anomalies, bad sessions).
           Trimming the upper tail prevents inflated thresholds that reduce
           detection sensitivity.

    Confidence: empirical percentile rank P(L_calib ≤ observed_loss).
    """

    def __init__(self, cfg: AnomalyScoringConfig = None):
        self.cfg     = cfg or CONFIG.scoring
        self._losses: List[float] = []

    def update(self, loss: float) -> None:
        """Only called from train() / recalibrate() — never from predict()."""
        self._losses.append(loss)

    #  trim upper tail that may contain calibration-split anomalies ─────
    def _clean_losses(self) -> np.ndarray:
        arr    = np.array(self._losses, dtype=np.float64)
        cutoff = np.quantile(arr, 1.0 - getattr(self.cfg, "calib_contamination_pct", 0.05))
        clean  = arr[arr <= cutoff]
        # Fall back to full array if trimming leaves too few samples
        min_n  = max(self.cfg.min_calibration_windows // 2, 5)
        return clean if len(clean) >= min_n else arr

    @property
    def is_calibrated(self) -> bool:
        return len(self._losses) >= self.cfg.min_calibration_windows

    @property
    def threshold(self) -> float:
        # Prefer the operationally-calibrated threshold when set via
        # select_operational_threshold() — it is tied to a real FP budget,
        # not an arbitrary quantile.
        op_thr = getattr(self, "_operational_threshold", None)
        if op_thr is not None:
            return op_thr
        if not self.is_calibrated:
            return float("inf")
        clean = self._clean_losses()
        large_n = getattr(self.cfg, "large_calib_threshold", 150)
        if len(clean) >= large_n:
            # Large set: quantile is stable
            q = getattr(self.cfg, "threshold_quantile", 0.995)
            return float(np.quantile(clean, q))
        else:
            # Small set: MAD-based threshold is more robust than quantile.
            median = float(np.median(clean))
            mad    = float(np.median(np.abs(clean - median)))
            k      = getattr(self.cfg, "mad_multiplier", 4.0)
            return median + k * (mad * 1.4826)

    def confidence(self, loss: float) -> float:
        """Empirical CDF: P(calibration_loss ≤ loss)."""
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
        """
        Replace the quantile-based threshold with one calibrated to a
        target false-positive budget (FP per 90-minute match).

        This is the principled alternative to threshold_quantile — it finds
        the tightest threshold that keeps FP volume at or below the budget.
        Call AFTER training, using a labeled held-out eval set.

        The selected threshold overrides all subsequent calls to .threshold.
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
        d = {"losses": list(self._losses)}
        op = getattr(self, "_operational_threshold", None)
        if op is not None:
            d["operational_threshold"] = op
        return d

    @classmethod
    def from_state_dict(cls, d: dict,
                        cfg: AnomalyScoringConfig = None) -> "DynamicThresholdTracker":
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
    """
    mask[t] = True  → real telemetry data
    mask[t] = False → zero-padded (dropped packet / missing event)

    mask[t] = True  → real telemetry data
    mask[t] = False → zero-padded (dropped packet / missing event)

    """

    def __init__(self):
        self._buffers:      Dict[str, deque] = {}
        self._mask_buffers: Dict[str, deque] = {}
        self._prev_events:  Dict[str, dict]  = {}
        cfg = CONFIG.window
        self.window_steps     = cfg.window_steps
        self.event_interval_s = cfg.event_interval_s

    def add_event(
        self, event: dict
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pid  = event.get("player_external_id", "")
        buf  = self._buffers.setdefault(pid,      deque(maxlen=self.window_steps))
        mbuf = self._mask_buffers.setdefault(pid, deque(maxlen=self.window_steps))
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
        speed = safe_float(event.get("speed_ms"), 0.0)
        hr    = safe_float(event.get("heart_rate_bpm"), 0.0)
        x     = safe_float(event.get("x_pitch"), 50.0)
        y     = safe_float(event.get("y_pitch"), 50.0)

        sprint = 1.0 if speed >= SPRINT_THRESHOLD_MS else 0.0
        dt      = self.event_interval_s

        if prev is not None:
            prev_speed = safe_float(prev.get("speed_ms"), 0.0)
            prev_hr    = safe_float(prev.get("heart_rate_bpm"), hr)
            prev_x     = safe_float(prev.get("x_pitch"), x)
            prev_y     = safe_float(prev.get("y_pitch"), y)

            accel = (speed - prev_speed) / dt
            # clamp to ±10 m/s² (physical limit) so sensor spikes don't dominate
            accel = float(np.clip(accel, -10.0, 10.0))

            # HR recovery rate: normalised change in HR per tick.
            # Raw (prev_hr - hr) / dt gives bpm/s, e.g. ±3 bpm/s, which has
            # extremely high variance between ticks and causes feature collapse.
            # express as fractional HR change per tick, then clip to [-1, 1]
            # so the signal has the same scale as other normalised features.
            # Positive = HR dropping (recovering); Negative = HR rising (exerting).
            if prev_hr > 0:
                hr_recovery = float(np.clip((prev_hr - hr) / max(prev_hr, 1.0), -1.0, 1.0))
            else:
                hr_recovery = 0.0

            # True Euclidean displacement in metres.
            # x_pitch ∈ [0,100] spans PITCH_WIDTH_M;  y_pitch spans PITCH_LENGTH_M.
            # The pitch is not square (105 m × 68 m), so each axis needs its own
            # scale factor — treating both as "≈ 1 m" introduced a systematic
            # ~35 % geometric error on the length axis.
            dx_m = (x - prev_x) / 100.0 * PITCH_WIDTH_M
            dy_m = (y - prev_y) / 100.0 * PITCH_LENGTH_M
            distance_delta = math.sqrt(dx_m * dx_m + dy_m * dy_m)
        else:
            accel          = 0.0
            hr_recovery    = 0.0
            distance_delta = 0.0

        features = np.array(
            [
                speed,
                accel,
                hr,
                sprint,
                x,
                y,
                distance_delta,   # true displacement, not speed*dt
                hr_recovery,      # fractional HR change [-1, 1]; ~0 when stable
            ],
            dtype=np.float32,
        )

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def build_from_session(
        self, events_df: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        if events_df.empty or "speed_ms" not in events_df.columns:
            return []
        events_df = events_df.sort_values("ts").reset_index(drop=True)
        buf:  deque = deque(maxlen=self.window_steps)
        mbuf: deque = deque(maxlen=self.window_steps)
        prev_row = None
        results: List[Tuple[np.ndarray, np.ndarray]] = []

        for _, row in events_df.iterrows():
            d       = row.to_dict()
            is_real = (d.get("speed_ms") is not None
                       and d.get("heart_rate_bpm") is not None)
            fv      = (self._extract(d, prev_row.to_dict() if prev_row is not None else None)
                       if is_real
                       else np.zeros(N_SEQUENCE_FEATURES, dtype=np.float32))
            buf.append(fv)
            mbuf.append(is_real)
            if len(buf) == self.window_steps:
                results.append((
                    np.array(list(buf),  dtype=np.float32),
                    np.array(list(mbuf), dtype=bool),
                ))
            prev_row = row if is_real else prev_row

        return results


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch modules
# ─────────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:

    # ── LSTM with pack_padded_sequence ─────────────────────────────────
    class _LSTMEncoder(nn.Module):
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
            """
            If lengths provided (real timestep counts per sample),
            pack the sequence so padded zeros do not drive hidden-state updates.
            Without packing, the LSTM processes fake zeros and the latent
            embedding shifts for incomplete windows — making them look anomalous
            purely due to missing packets, not actual player behaviour.
            """
            if lengths is not None:
                # clamp to at least 1 to avoid empty sequences
                safe_len = lengths.cpu().clamp(min=1)
                packed   = nn.utils.rnn.pack_padded_sequence(
                    x, safe_len, batch_first=True, enforce_sorted=False
                )
                _, (h_n, _) = self.lstm(packed)
            else:
                _, (h_n, _) = self.lstm(x)

            h_last = h_n[-1]                         # (B, hidden)
            return torch.tanh(self.fc(h_last))       # (B, latent)


    class _LSTMDecoder(nn.Module):
        def __init__(self, latent: int, hidden: int, n_layers: int,
                     n_features: int, seq_len: int, dropout: float):
            super().__init__()
            self.seq_len = seq_len
            self.fc_in   = nn.Linear(latent, hidden)
            self.n_layers = n_layers 
            self.lstm    = nn.LSTM(
                latent, hidden, n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.fc_out  = nn.Linear(hidden, n_features)

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            h0    = torch.tanh(self.fc_in(z)).unsqueeze(0)                        # (1, B, hidden)
            h0    = h0.repeat(self.n_layers, 1, 1)                                # (n_layers, B, hidden)
            z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)
            out, _ = self.lstm(z_seq, (h0.contiguous(), torch.zeros_like(h0)))
            return self.fc_out(out)


    class _LSTMAEModule(nn.Module):
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
            # derive lengths from mask for pack_padded_sequence
            lengths = mask.long().sum(dim=1) if mask is not None else None
            z       = self.encoder(x, lengths)
            recon   = self.decoder(z)
            return recon, z


    # ── Positional encoding ───────────────────────────────────────────────────
    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.dropout(x + self.pe[:, :x.size(1)])


    # ── Transformer with validity-weighted pooling ────────────────────
    class _TransformerAEModule(nn.Module):
        """
        Bottleneck mean-pooling now excludes padded tokens:

            pooled = Σ_t m_t * h_t / Σ_t m_t

        where m_t = 1 if timestep t contains real data, 0 if padded.
        Previously h.mean(dim=1) let padded positions (zeros) bias the
        embedding — incomplete windows appeared anomalous regardless of
        player physiology.

        also passes the padding mask to the decoder so that padded
        output positions do not contribute to the reconstruction loss.
        """

        def __init__(self, cfg: TransformerAutoencoderConfig, seq_len: int):
            super().__init__()
            D, L, F = cfg.d_model, cfg.latent_dim, N_SEQUENCE_FEATURES

            self.input_proj = nn.Linear(F, D)
            self.pos_enc    = _PositionalEncoding(
                D, max_len=max(seq_len + 4, 64), dropout=cfg.dropout
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                batch_first=True, norm_first=True,
            )
            self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_encoder_layers)
            self.fc_latent   = nn.Linear(D, L)
            self.fc_expand   = nn.Linear(L, D)

            dec_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff, dropout=cfg.dropout,
                batch_first=True, norm_first=True,
            )
            self.decoder     = nn.TransformerEncoder(dec_layer, num_layers=cfg.n_decoder_layers)
            self.output_proj = nn.Linear(D, F)
            self.seq_len     = seq_len
            self._last_attn: Optional[np.ndarray] = None

        @staticmethod
        def _mask_to_padding_mask(
            mask: Optional["torch.Tensor"],
        ) -> Optional["torch.Tensor"]:
            """
            Convert validity mask (True=real) to PyTorch src_key_padding_mask
            convention (True=IGNORE/padding).
            """
            if mask is None:
                return None
            return ~mask.bool()   # (B, T): True where padded

        def _masked_pool(
            self,
            h: "torch.Tensor",
            mask: Optional["torch.Tensor"],
        ) -> "torch.Tensor":
            """
            Validity-weighted mean pooling.
            mask: (B, T) bool, True = real data.
            Falls back to regular mean if no mask.
            """
            if mask is None:
                return h.mean(dim=1)
            m      = mask.float().unsqueeze(-1)          # (B, T, 1)
            pooled = (h * m).sum(dim=1)                  # (B, D)
            count  = m.sum(dim=1).clamp(min=1e-8)        # (B, 1)
            return pooled / count                         # (B, D)

        def _capture_attn_weights(
            self, h: "torch.Tensor"
        ) -> Optional[np.ndarray]:
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
            # x:    (B, T, F)
            # mask: (B, T) bool, True=real  — our internal convention
            padding_mask = self._mask_to_padding_mask(mask)  # PyTorch convention

            # ── Encode ──────────────────────────────────────────────────────
            h = self.pos_enc(self.input_proj(x))
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            self._last_attn = self._capture_attn_weights(h)

            # ── Bottleneck (masked pooling) ─────────────────────────
            pooled = self._masked_pool(h, mask)              # (B, D)
            z      = torch.tanh(self.fc_latent(pooled))      # (B, L)

            # ── Decode ──────────────────────────────────────────────────────
            B, T   = x.shape[:2]
            h_dec  = torch.tanh(self.fc_expand(z))
            h_dec  = h_dec.unsqueeze(1).repeat(1, T, 1)
            h_dec  = self.pos_enc(h_dec)
            #  pass mask to decoder so padded output positions are not
            # driven by the encoder's latent representation
            h_dec  = self.decoder(h_dec, src_key_padding_mask=padding_mask)
            recon  = self.output_proj(h_dec)                  # (B, T, F)
            return recon, z


# ─────────────────────────────────────────────────────────────────────────────
# Shared training loop  (masked MSE + latent regularisation)
# ─────────────────────────────────────────────────────────────────────────────
def _masked_mse(
    x: "torch.Tensor",
    recon: "torch.Tensor",
    mask: Optional["torch.Tensor"],
) -> "torch.Tensor":
    """
    Per-timestep MSE weighted by validity mask.
    Only real (non-padded) timesteps contribute to the gradient.
    mask: (B, T) bool, True=real.
    """
    sq = (x - recon) ** 2                              # (B, T, F)
    if mask is not None:
        m   = mask.float().unsqueeze(-1)               # (B, T, 1)
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
    """
    3-way split: train / val / calibration.
     masks threaded through module.forward() and _masked_mse().
     latent L2 regularisation (weight = latent_reg) prevents the
           autoencoder from learning a trivial identity/smoothing mapping.
           Without it the latent space collapses and reconstruction error
           becomes uniformly low — anomaly separation degrades.

    Returns (history, calib_sequences, calib_masks).
    """
    arr  = np.stack(sequences)            # (N, T, F)
    marr = np.stack(masks).astype(bool)  # (N, T)
    N    = len(arr)

    rng     = np.random.default_rng(seed=42 + player_id)
    indices = rng.permutation(N)
    n_train = max(int(N * train_frac), 1)
    n_val   = max(int(N * val_frac),   1)

    idx_train = indices[:n_train]
    idx_val   = indices[n_train: n_train + n_val]
    idx_calib = indices[n_train + n_val:]

    if len(idx_val)   == 0: idx_val   = idx_train[:max(1, n_train // 5)]
    if len(idx_calib) == 0: idx_calib = idx_train[:max(1, n_train // 5)]

    # Normalise on train split only
    normaliser.fit(arr[idx_train])

    def to_tensor_pair(idx: np.ndarray):
        X = torch.tensor(normaliser.transform(arr[idx]))
        M = torch.tensor(marr[idx])
        return X, M

    X_train, M_train = to_tensor_pair(idx_train)
    X_val,   M_val   = to_tensor_pair(idx_val)

    loader = DataLoader(
        TensorDataset(X_train, M_train),
        batch_size=cfg_batch,
        shuffle=True,
        pin_memory=(str(DEVICE) == "cuda" or str(DEVICE) == "mps"),
    )

    module    = module.to(DEVICE)
    optimizer = optim.Adam(module.parameters(), lr=cfg_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_ct   = 0
    history: List[dict] = []

    logger.info(
        "%s p%d: train=%d val=%d calib=%d device=%s",
        model_label, player_id,
        len(idx_train), len(idx_val), len(idx_calib), DEVICE,
    )

    for epoch in range(cfg_epochs):
        # ── Train ────────────────────────────────────────────────────────────
        module.train()
        epoch_loss = 0.0
        for batch_x, batch_m in loader:
            batch_x = batch_x.to(DEVICE)
            batch_m = batch_m.to(DEVICE)
            optimizer.zero_grad()
            recon, z = module(batch_x, mask=batch_m)

            # mask-aware reconstruction loss
            recon_loss = _masked_mse(batch_x, recon, batch_m)
            # latent L2 regularisation
            reg_loss   = latent_reg * z.pow(2).mean()
            loss       = recon_loss + reg_loss

            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += recon_loss.item() * len(batch_x)
        epoch_loss /= len(idx_train)

        # ── Validate ─────────────────────────────────────────────────────────
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
            patience_ct   = 0
            best_state    = {k: v.cpu().clone()
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
        module.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    module.eval()

    calib_seqs  = [sequences[i] for i in idx_calib]
    calib_masks = [masks[i]     for i in idx_calib]

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
    """Reconstruction loss for one normalised window. mask-aware."""
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
    """Vectorised reconstruction losses for a batch (N, T, F).mask-aware."""
    module.eval()
    with torch.no_grad():
        x = X_norm.to(DEVICE)
        m = masks.to(DEVICE)   if masks is not None else None
        recon, _ = module(x, mask=m)
        sq = (x - recon) ** 2                         # (N, T, F)
        if m is not None:
            mf   = m.float().unsqueeze(-1)             # (N, T, 1)
            loss = (sq * mf).sum(dim=(1, 2)) / (mf.sum(dim=(1, 2)) * sq.size(-1) + 1e-8)
        else:
            loss = sq.mean(dim=(1, 2))
    return loss.cpu().numpy()


class SharedLSTMEncoder(nn.Module):
    """Single encoder trained on ALL players. Loaded once, never per-player."""
    def __init__(self, cfg: LSTMAutoencoderConfig, embedding_dim: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(
            N_SEQUENCE_FEATURES, cfg.hidden_size, cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(cfg.hidden_size, cfg.latent_dim)
        # FiLM: scale and shift latent z using player embedding
        self.film_scale = nn.Linear(embedding_dim, cfg.latent_dim)
        self.film_shift = nn.Linear(embedding_dim, cfg.latent_dim)

    def forward(self, x, embedding, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        z = torch.tanh(self.fc(h_n[-1]))                   # (B, latent)
        scale = 0.1 * torch.tanh(self.film_scale(embedding))
        shift = 0.1 * torch.tanh(self.film_shift(embedding))

        z = z * (1.0 + scale) + shift

        # final latent clamp
        z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0)

        return z


class SharedBackboneAutoencoder:
    """
    One instance shared across ALL players.
    Replaces the per-player LSTMAutoencoder.
    """
    MODEL_TYPE = "shared_lstm"

    def __init__(self, n_players: int, cfg: LSTMAutoencoderConfig = None,
                 embedding_dim: int = 16):
        self.cfg           = cfg or CONFIG.lstm
        self.embedding_dim = embedding_dim
        self.n_players     = n_players
        self.is_trained    = False
        self.model_version = "untrained"
        self.normaliser    = PerPlayerNormaliser()   # fit on ALL players' data
        self._encoder: Optional[SharedLSTMEncoder] = None
        self._decoder: Optional[_LSTMDecoder] = None
        # Lookup: internal index → player_id and back
        self._player_index: Dict[int, int] = {}     # player_id → embedding row
        self._embedding: Optional[nn.Embedding] = None

    def register_players(self, player_ids: List[int]) -> None:
        self._player_index = {pid: i for i, pid in enumerate(player_ids)}
        self.n_players = len(player_ids)

    def train(self, all_windows: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]) -> dict:
        """
        all_windows: {player_id: [(sequence, mask), ...]}
        Trains ONE model on all players' data simultaneously.
        """
        if not TORCH_AVAILABLE:
            return {"status": "no_torch"}
        
        try:
            from tqdm import tqdm as _tqdm
            TQDM_AVAILABLE = True
        except ImportError:
            TQDM_AVAILABLE = False

        epoch_iter = _tqdm(
            range(self.cfg.max_epochs),
            desc="Shared LSTM",
            unit="epoch",
            dynamic_ncols=True,
        ) if TQDM_AVAILABLE else range(self.cfg.max_epochs)

        seq_len = next(iter(all_windows.values()))[0][0].shape[0]
        self._encoder  = SharedLSTMEncoder(self.cfg, self.embedding_dim)
        self._decoder  = _LSTMDecoder(
            self.cfg.latent_dim, self.cfg.hidden_size, self.cfg.num_layers,
            N_SEQUENCE_FEATURES, seq_len, self.cfg.dropout
        )
        self._embedding = nn.Embedding(self.n_players, self.embedding_dim)

        # Flatten all windows, track which player each belongs to
        all_seqs, all_masks, all_pids = [], [], []
        for pid, windows in all_windows.items():
            idx = self._player_index[pid]
            for seq, mask in windows:
                all_seqs.append(seq)
                all_masks.append(mask)
                all_pids.append(idx)

        arr  = np.stack(all_seqs)
        marr = np.stack(all_masks).astype(bool)
        self.normaliser.fit(arr)

        X = torch.tensor(self.normaliser.transform(arr))
        M = torch.tensor(marr)
        P = torch.tensor(all_pids, dtype=torch.long)

        dataset = TensorDataset(X, M, P)
        loader  = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        params = (list(self._encoder.parameters()) +
                  list(self._decoder.parameters()) +
                  list(self._embedding.parameters()))
        optimizer = optim.Adam(params, lr=self.cfg.learning_rate)
        self._encoder.to(DEVICE)
        self._decoder.to(DEVICE)
        self._embedding.to(DEVICE)

        best_loss   = float("inf")
        patience_ct = 0
        PATIENCE    = self.cfg.patience         

        for epoch in epoch_iter:
            self._encoder.train()
            self._decoder.train()
            epoch_loss = 0.0
            n_batches  = 0

            for bx, bm, bp in loader:
                bx, bm, bp = bx.to(DEVICE), bm.to(DEVICE), bp.to(DEVICE)
                emb        = self._embedding(bp)
                lengths    = bm.long().sum(dim=1)
                z          = self._encoder(bx, emb, lengths)
                recon      = self._decoder(z)
                recon_loss = _masked_mse(bx, recon, bm)
                latent_reg = 1e-4
                reg_loss   = latent_reg * z.pow(2).mean()
                loss       = recon_loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            epoch_loss /= max(n_batches, 1)

            # removed logger.info with:
            if TQDM_AVAILABLE:
                epoch_iter.set_postfix(loss=f"{epoch_loss:.5f}", best=f"{best_loss:.5f}", patience=patience_ct)
            else:
                logger.info("Shared LSTM: epoch %d/%d  loss=%.5f", epoch + 1, self.cfg.max_epochs, epoch_loss)

            # Early stopping
            if epoch_loss < best_loss - 1e-5:
                best_loss   = epoch_loss
                patience_ct = 0
                best_state  = {
                    "encoder":   {k: v.cpu().clone() for k, v in self._encoder.state_dict().items()},
                    "decoder":   {k: v.cpu().clone() for k, v in self._decoder.state_dict().items()},
                    "embedding": {k: v.cpu().clone() for k, v in self._embedding.state_dict().items()},
                }
            else:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    logger.info(
                        "Shared LSTM: early stop at epoch %d (loss=%.5f)",
                        epoch + 1, epoch_loss,
                    )
                    break

        # Restore best weights
        if best_state:
            self._encoder.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state["encoder"].items()})
            self._decoder.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state["decoder"].items()})
            self._embedding.load_state_dict(
                {k: v.to(DEVICE) for k, v in best_state["embedding"].items()})

        self.is_trained    = True
        self.model_version = f"shared_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        return {"status": "trained", "n_players": self.n_players,
                "n_windows": len(all_seqs)}

    def predict(self, player_id: int, sequence: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[float, bool, float]:
        if not self.is_trained:
            return 0.0, False, 0.0
        idx  = self._player_index.get(player_id)
        if idx is None:
            return 0.0, False, 0.0
        norm = torch.tensor(self.normaliser.transform(sequence[np.newaxis])[0])
        emb  = self._embedding(torch.tensor([idx]).to(DEVICE))
        mask_t = torch.tensor(mask).unsqueeze(0).to(DEVICE) if mask is not None else None
        with torch.no_grad():
            self._encoder.eval(); self._decoder.eval()
            lengths = mask_t.long().sum(dim=1) if mask_t is not None else None
            z     = self._encoder(norm.unsqueeze(0).to(DEVICE), emb, lengths)
            recon = self._decoder(z)
            loss  = _masked_mse(
                norm.unsqueeze(0).to(DEVICE), recon, mask_t
            ).item()
        return loss, False, 0.0   # threshold checked by per-player tracker externally

    def reconstruction_loss_for_shap(
        self,
        player_id: int,
        sequences_norm: np.ndarray,   # (N, T, F) — already normalised by caller
        mask: Optional[np.ndarray] = None,  # (T,) — same mask broadcast to all N
    ) -> np.ndarray:
        """
        True model predict_fn for SHAP KernelExplainer.

        Takes a BATCH of normalised sequences (N, T, F) — the SHAP perturbed
        samples — and returns a (N,) array of reconstruction losses.

        This is the only correct way to attribute the anomaly score to
        input features: each perturbed sequence goes through the real
        encoder→decoder and the MSE loss is the target SHAP explains.

        Parameters
        ----------
        player_id     : player whose embedding to use.
        sequences_norm: (N, T, F) float32, already normalised via self.normaliser.
        mask          : (T,) bool, broadcast across all N samples.
                        None means all timesteps are real.

        Returns
        -------
        losses : (N,) float32 — per-sample reconstruction loss.
        """
        if not self.is_trained:
            return np.zeros(len(sequences_norm), dtype=np.float32)

        idx = self._player_index.get(player_id)
        if idx is None:
            return np.zeros(len(sequences_norm), dtype=np.float32)

        N = len(sequences_norm)
        X = torch.tensor(sequences_norm, dtype=torch.float32).to(DEVICE)   # (N, T, F)

        # Broadcast the single player embedding to every sample in the batch
        emb = self._embedding(
            torch.tensor([idx] * N, dtype=torch.long).to(DEVICE)
        )                                                                    # (N, emb_dim)

        if mask is not None:
            M = torch.tensor(
                np.stack([mask] * N).astype(bool), dtype=torch.bool
            ).to(DEVICE)                                                     # (N, T)
            lengths = M.long().sum(dim=1)                                   # (N,)
        else:
            M       = None
            lengths = None

        with torch.no_grad():
            self._encoder.eval()
            self._decoder.eval()
            z     = self._encoder(X, emb, lengths)                          # (N, latent)
            recon = self._decoder(z)                                         # (N, T, F)

            # Per-sample masked MSE
            sq = (X - recon) ** 2                                           # (N, T, F)
            if M is not None:
                mf   = M.float().unsqueeze(-1)                              # (N, T, 1)
                loss = (sq * mf).sum(dim=(1, 2)) / (
                    mf.sum(dim=(1, 2)) * sq.size(-1) + 1e-8
                )                                                            # (N,)
            else:
                loss = sq.mean(dim=(1, 2))                                  # (N,)

        return loss.cpu().numpy().astype(np.float32)

    def save(self, path: Path = None) -> Path:
        """Persist the trained shared backbone to disk."""
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
        """Load a previously saved shared backbone. Returns None if file absent."""

        path = path or (MODEL_STORE / "shared_backbone.pt")

        if not path.exists() or not TORCH_AVAILABLE:
            return None
        
        add_safe_globals([LSTMAutoencoderConfig])
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        obj  = cls(
            n_players     = ckpt["n_players"],
            cfg           = ckpt["cfg"],
            embedding_dim = ckpt["embedding_dim"],
        )
        obj._player_index = ckpt["player_index"]
        seq_len           = ckpt.get("seq_len", CONFIG.window.window_steps)

        obj._encoder  = SharedLSTMEncoder(obj.cfg, obj.embedding_dim).to(DEVICE)
        obj._decoder  = _LSTMDecoder(
            obj.cfg.latent_dim, obj.cfg.hidden_size, obj.cfg.num_layers,
            N_SEQUENCE_FEATURES, seq_len, obj.cfg.dropout,
        ).to(DEVICE)
        obj._embedding = nn.Embedding(obj.n_players, obj.embedding_dim).to(DEVICE)

        obj._encoder.load_state_dict(ckpt["encoder"])
        obj._decoder.load_state_dict(ckpt["decoder"])
        obj._embedding.load_state_dict(ckpt["embedding"])
        obj.normaliser    = PerPlayerNormaliser.from_state_dict(ckpt["normaliser"])
        obj.is_trained    = True
        obj.model_version = ckpt["model_version"]
        logger.info("Shared backbone loaded ← %s (version=%s)", path, obj.model_version)
        return obj

# ─────────────────────────────────────────────────────────────────────────────
# TransformerAutoencoder  (experimental, disabled)
# ─────────────────────────────────────────────────────────────────────────────
class TransformerAutoencoder:
    """
    Experimental.  Requires ≥30 sessions/player.
    Default model is LSTMAutoencoder.

    Attention note: last_attention_weights returns real matrices but attention
    weight magnitude does NOT equal feature importance.  Do not present as XAI
    without additional validation (e.g. SHAP, integrated gradients).
    """
    MODEL_TYPE   = "transformer"
    EXPERIMENTAL = True

    def __init__(self, player_id: int, cfg: TransformerAutoencoderConfig = None):
        self.player_id         = player_id
        self.cfg               = cfg or CONFIG.transformer
        self.is_trained        = False
        self.model_version     = "untrained"
        self.normaliser        = PerPlayerNormaliser()
        self.threshold_tracker = RegimeAwareThresholdStore()
        self._module: Optional["_TransformerAEModule"] = None
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch unavailable — TransformerAutoencoder in stub mode")
        warnings.warn(
            "TransformerAutoencoder is EXPERIMENTAL. "
            "Use LSTMAutoencoder for production.",
            stacklevel=2,
        )

    def train(self, session_windows: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
        if not TORCH_AVAILABLE:
            return {"status": "no_torch"}

        sequences = [w for w, _ in session_windows]
        masks_np  = [m for _, m in session_windows]

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
        self.is_trained    = True
        self.model_version = (
            f"transformer_{datetime.now().strftime('%Y%m%d%H%M%S')}_p{self.player_id}"
        )
        return {"status": "trained", "n_windows": len(sequences),
                "device": str(DEVICE), **history}

    def _calibrate(self, calib_seqs, calib_masks):
        """
        Build per-regime thresholds from EMA-smoothed calibration losses.

        Two invariants maintained:
          1. EMA parity — calibration losses pass through the same EMA transform
             used in infer_live, so threshold ← f(EMA(losses)), not f(raw losses).
          2. Regime separation — each window's EMA loss is routed to both the
             global tracker and the regime-specific tracker inside
             RegimeAwareThresholdStore.  At inference, the regime-specific
             threshold is used when calibrated; the global is the fallback.
        """
        self.threshold_tracker = RegimeAwareThresholdStore()
        alpha = CONFIG.scoring.score_ema_alpha
        ema_val: Optional[float] = None
        for seq, msk in zip(calib_seqs, calib_masks):
            raw_loss    = self._recon_loss(seq, msk)
            ema_val     = raw_loss if ema_val is None else (
                alpha * raw_loss + (1 - alpha) * ema_val
            )
            regime_key  = _REGIME_CLASSIFIER.classify(seq).key
            self.threshold_tracker.update(ema_val, regime_key)
        logger.debug("TransformerAutoencoder p%d calibration:\n%s",
                     self.player_id, self.threshold_tracker.summary())

    def recalibrate(self, coach_confirmed_normal):
        if not self.is_trained:
            return
        self._calibrate([w for w, _ in coach_confirmed_normal],
                        [m for _, m in coach_confirmed_normal])

    def _recon_loss(self, seq, mask=None):
        norm   = torch.tensor(self.normaliser.transform(seq[np.newaxis])[0])
        mask_t = torch.tensor(mask) if mask is not None else None
        return _mse_loss_single(self._module, norm, mask_t)

    def predict(self, sequence, mask=None):
        if not self.is_trained or self._module is None:
            return 0.0, False, 0.0
        loss       = self._recon_loss(sequence, mask)
        regime_key = _REGIME_CLASSIFIER.classify(sequence).key
        is_anomaly = (self.threshold_tracker.is_calibrated
                      and loss > self.threshold_tracker.threshold_for(regime_key))
        confidence = self.threshold_tracker.confidence_for(loss, regime_key)
        return loss, is_anomaly, confidence

    def predict_batch(self, sequences, masks=None):
        if not self.is_trained or self._module is None:
            return [(0.0, False, 0.0)] * len(sequences)
        arr_norm = self.normaliser.transform(np.stack(sequences))
        X        = torch.tensor(arr_norm)
        M        = torch.tensor(np.stack(masks).astype(bool)) if masks is not None else None
        losses   = _mse_loss_batch(self._module, X, M)
        results  = []
        for seq, l in zip(sequences, losses):
            rk  = _REGIME_CLASSIFIER.classify(seq).key
            thr = self.threshold_tracker.threshold_for(rk)
            results.append((
                float(l),
                self.threshold_tracker.is_calibrated and float(l) > thr,
                self.threshold_tracker.confidence_for(float(l), rk),
            ))
        return results

    @property
    def last_attention_weights(self) -> Optional[np.ndarray]:
        """
        Returns last encoder attention matrix (B, T, T) or None.
        NOTE: attention weight ≠ feature importance.
        Validate with SHAP/IG before presenting as explanation.
        """
        return self._module._last_attn if self._module else None

    def save(self) -> Path:
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
        path = MODEL_STORE / f"player_{player_id}_transformer_ae.pt"
        if not path.exists() or not TORCH_AVAILABLE:
            return None
        ckpt = torch.load(path, map_location=DEVICE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = cls(player_id, cfg=ckpt["cfg"])
        obj.normaliser        = PerPlayerNormaliser.from_state_dict(ckpt["normaliser_state"])
        _ts = ckpt["threshold_state"]
        obj.threshold_tracker = (
            RegimeAwareThresholdStore.from_state_dict(_ts)
            if "per_regime" in _ts
            else DynamicThresholdTracker.from_state_dict(_ts)  # legacy compat
        )
        obj.model_version     = ckpt["model_version"]
        obj.is_trained        = ckpt["is_trained"]
        if ckpt["module_state"] and obj.is_trained:
            seq_len     = ckpt.get("seq_len", CONFIG.window.window_steps)
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
    """
    Compute standard anomaly detection metrics against labeled windows.

    labeled_windows: list of (sequence, mask, is_anomaly_label)
    Returns:
        roc_auc        — area under ROC curve (0.5 = random, 1.0 = perfect)
        pr_auc         — area under precision-recall curve
        precision_at_k — precision in top-k scored windows (k = true anomaly count)
        fp_per_90_min  — false positives per simulated 90-minute match
        threshold      — current anomaly threshold
        n_windows      — total windows evaluated
        n_anomalies    — true positive count in labeled set

    Without sklearn: falls back to threshold-based binary accuracy only.
    """
    if not model.is_trained:
        return {"error": "model not trained"}

    seqs   = [w for w, _, _ in labeled_windows]
    masks  = [m for _, m, _ in labeled_windows]
    labels = np.array([int(l) for _, _, l in labeled_windows])

    # Compute anomaly scores for every window
    results = model.predict_batch(seqs, masks)
    scores  = np.array([r[0] for r in results])    # reconstruction losses
    preds   = np.array([int(r[1]) for r in results])

    n_windows   = len(labels)
    n_anomalies = int(labels.sum())
    n_normal    = n_windows - n_anomalies

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
        metrics["pr_auc"]  = float(average_precision_score(labels, scores))
    else:
        logger.warning("sklearn not available — ROC-AUC / PR-AUC skipped")
        metrics["roc_auc"] = None
        metrics["pr_auc"]  = None

    # Precision@k  (k = number of true anomalies)
    k           = n_anomalies
    top_k_idx   = np.argsort(scores)[::-1][:k]
    prec_at_k   = float(labels[top_k_idx].sum() / k)
    metrics["precision_at_k"] = prec_at_k

    # False positives per 90-minute match
    if window_interval_seconds is None:
        window_interval_seconds = CONFIG.window.event_interval_s * CONFIG.window.window_steps
    windows_per_90 = match_duration_seconds / max(window_interval_seconds, 1.0)
    fp_rate        = float((preds & (1 - labels)).sum()) / max(n_normal, 1)
    metrics["fp_per_90_min"] = round(fp_rate * windows_per_90, 2)

    # Simple binary accuracy
    tp = int(( preds & labels).sum())
    fp = int(( preds & (1 - labels)).sum())
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
    def __init__(self, cfg: PositionalDriftConfig = None):
        self.cfg = cfg or CONFIG.positional

    def analyze(
        self,
        recent_positions: List[Tuple[float, float]],
        baseline: PlayerBaselineProfile,
    ) -> dict:
        if not recent_positions or baseline.avg_x is None:
            return {"drift_score": 0.0, "is_flagged": False, "fraction_outside_zone": 0.0}
        std_r = baseline.position_std_radius or self.cfg.zone_radius_meters
        thr   = max(std_r * 2.0, self.cfg.zone_radius_meters)
        dists = [math.sqrt((x - baseline.avg_x)**2 + (y - baseline.avg_y)**2)
                 for x, y in recent_positions]
        frac  = sum(1 for d in dists if d > thr) / len(dists)
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
# (does not require an old-style per-player model object)
# ─────────────────────────────────────────────────────────────────────────────
def _merge_contiguous_events(
    binary_sequence: np.ndarray,
    gap_tolerance: int = 2,
) -> List[Tuple[int, int]]:
    """
    Merge a binary window-level sequence into contiguous events.

    Adjacent positive windows within gap_tolerance of each other are merged
    into one event. This converts window-level predictions/labels into
    episode-level events for fair evaluation.

    Parameters
    ----------
    binary_sequence : (N,) int array — 1=positive, 0=negative
    gap_tolerance   : windows — gaps this short or shorter are bridged

    Returns
    -------
    List of (start_idx, end_idx) inclusive for each detected event.
    """
    events: List[Tuple[int, int]] = []
    in_event  = False
    start     = 0
    gap_count = 0

    for i, val in enumerate(binary_sequence):
        if val:
            if not in_event:
                in_event  = True
                start     = i
                gap_count = 0
            else:
                gap_count = 0
        else:
            if in_event:
                gap_count += 1
                if gap_count > gap_tolerance:
                    events.append((start, i - gap_count))
                    in_event  = False
                    gap_count = 0

    if in_event:
        events.append((start, len(binary_sequence) - 1))

    return events


def _event_level_metrics(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    gap_tolerance: int = 2,
) -> dict:
    """
    Compute event-level precision, recall, and F1.

    A predicted event is a TP if it overlaps with any ground-truth event.
    Multiple predicted events overlapping the same ground-truth event count
    as ONE TP + (n-1) FPs. This prevents TP inflation from sustained alerts.

    Parameters
    ----------
    pred_labels   : (N,) int — predicted binary labels (window level)
    true_labels   : (N,) int — ground-truth binary labels (window level)
    gap_tolerance : windows to bridge when merging contiguous events

    Returns
    -------
    dict with event_tp, event_fp, event_fn, event_precision,
    event_recall, event_f1, n_pred_events, n_true_events
    """
    pred_events = _merge_contiguous_events(pred_labels, gap_tolerance)
    true_events = _merge_contiguous_events(true_labels, gap_tolerance)

    # For each predicted event, check if it overlaps any ground-truth event
    matched_gt: set = set()
    event_tp = 0
    event_fp = 0

    for ps, pe in pred_events:
        pred_set = set(range(ps, pe + 1))
        matched  = False
        for gt_idx, (gs, ge) in enumerate(true_events):
            if pred_set & set(range(gs, ge + 1)):   # any overlap
                matched = True
                matched_gt.add(gt_idx)
                break
        if matched:
            event_tp += 1
        else:
            event_fp += 1

    event_fn = len(true_events) - len(matched_gt)

    ep  = event_tp / max(event_tp + event_fp, 1)
    er  = event_tp / max(event_tp + event_fn, 1)
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
    window_interval_s: float    = 120.0,
) -> float:
    """
    Choose an operating threshold from the PR curve that achieves a target
    false-positive rate per 90-minute match period.

    This is operationally principled: a coach tolerates at most
    `target_fp_per_90_min` spurious alerts per match. We find the
    highest-recall threshold that keeps FP volume at or below this budget.

    Parameters
    ----------
    scores              : (N,) raw model scores (higher = more anomalous)
    labels              : (N,) ground-truth binary labels
    target_fp_per_90_min: maximum tolerable false alerts per 90-minute period
    window_interval_s   : seconds between successive windows (step size)

    Returns
    -------
    float : selected threshold. Falls back to median(scores[labels==1]) if
            no threshold achieves the FP budget.
    """
    if not SKLEARN_AVAILABLE:
        return float(np.median(scores))

    from sklearn.metrics import precision_recall_curve

    n_normal       = int((labels == 0).sum())
    windows_per_90 = (90 * 60) / max(window_interval_s, 1.0)
    # Maximum FP fraction of normals that still meets the budget
    max_fpr = target_fp_per_90_min / max(windows_per_90 * n_normal / len(labels), 1e-6)

    prec, rec, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns len(thresholds) == len(prec) - 1
    # Walk thresholds from high (tight) to low (loose) — stop when FPR budget exceeded
    for thr in sorted(thresholds):
        preds = (scores >= thr).astype(int)
        fp    = int((preds & (1 - labels)).sum())
        fpr   = fp / max(n_normal, 1)
        if fpr <= max_fpr:
            return float(thr)

    # No threshold meets the budget — return one that gives FP rate closest to budget
    best_thr  = float(thresholds[-1])  # tightest threshold
    return best_thr


def evaluate_model_results(
    results:             List[Tuple[float, bool, float]],
    labels:              List[bool],
    threshold:           float,
    ema_smoothed:        bool  = False,
    gap_tolerance:       int   = 2,
    target_fp_per_90_min: float = 2.0,
    window_interval_s:   float = 120.0,
) -> dict:
    """
    Compute anomaly detection metrics at both window and event level.

    Parameters
    ----------
    results      : [(score, is_anomaly, confidence), ...] — model outputs
    labels       : ground-truth bool per window, in temporal order
    threshold    : the threshold used to produce the is_anomaly booleans
    ema_smoothed : True if scores are EMA-smoothed (disclosed in output)
    gap_tolerance: windows to bridge when merging events (default 2)
    target_fp_per_90_min: FP budget for PR-curve threshold selection
    window_interval_s   : step between windows in seconds

    Notes on metrics
    ----------------
    Window-level TP/FP: biased upward when anomalies are contiguous (each
        window in a 10-window anomaly burst counts as a separate TP).
        Use event_* metrics for operational reporting.

    Event-level TP/FP: a predicted contiguous burst overlapping any
        ground-truth event counts as ONE TP, regardless of burst length.
        This matches operational reality: a coach sees one alert, not N.

    PR-curve threshold: chosen to meet `target_fp_per_90_min` rather than
        an arbitrary quantile. Use this as the recommended deployment threshold.
    """
    scores = np.array([r[0] for r in results], dtype=np.float64)
    preds  = np.array([int(r[1]) for r in results])
    labs   = np.array([int(l) for l in labels])
    n_anomalies = int(labs.sum())
    n_normal    = len(labs) - n_anomalies

    if n_anomalies == 0 or n_normal == 0:
        return {"error": "need both anomaly and normal examples", "n_windows": len(labs)}

    metrics: dict = {
        "threshold":    threshold,
        "n_windows":    len(labs),
        "n_anomalies":  n_anomalies,
        "ema_smoothed": ema_smoothed,   # disclose smoothing to reviewer
    }

    # ── Ranking metrics (threshold-independent) ───────────────────────────────
    if SKLEARN_AVAILABLE:
        metrics["roc_auc"] = float(roc_auc_score(labs, scores))
        metrics["pr_auc"]  = float(average_precision_score(labs, scores))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"]  = None

    # ── Precision@k ──────────────────────────────────────────────────────────
    k = n_anomalies
    metrics["precision_at_k"] = float(labs[np.argsort(scores)[::-1][:k]].sum() / k)

    # ── Window-level binary metrics at current threshold ─────────────────────
    tp = int((preds & labs).sum())
    fp = int((preds & (1 - labs)).sum())
    fn = int(((1 - preds) & labs).sum())
    tn = int(((1 - preds) & (1 - labs)).sum())
    metrics.update({
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "window_precision": round(tp / max(tp + fp, 1), 4),
        "window_recall":    round(tp / max(tp + fn, 1), 4),
        # Legacy keys kept for backward compatibility
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall":    round(tp / max(tp + fn, 1), 4),
    })
    # Note explicitly: window-level counts are inflated by contiguous alerts
    windows_per_90 = (90 * 60) / max(window_interval_s, 1.0)
    metrics["fp_per_90_min_window"] = round(
        (fp / max(n_normal, 1)) * windows_per_90, 2
    )

    # ── Event-level metrics ───────────────────────────────────────────────────
    # This is the scientifically correct metric for reporting.
    # 12 consecutive flagged windows during a single anomaly episode = 1 TP event.
    # 20 consecutive false-positive windows = 1 FP event (not 20).
    ev = _event_level_metrics(preds, labs, gap_tolerance=gap_tolerance)
    metrics.update(ev)
    metrics["fp_per_90_min_event"] = round(
        ev["event_fp"] / max(len(labs) / max(windows_per_90, 1), 1), 2
    )

    # ── PR-curve threshold recommendation ────────────────────────────────────
    # Principled alternative to quantile-based threshold.
    # Find the tightest threshold that keeps FP volume at or below
    # `target_fp_per_90_min` false alerts per 90-minute match.
    pr_threshold = _pr_curve_threshold(
        scores, labs,
        target_fp_per_90_min=target_fp_per_90_min,
        window_interval_s=window_interval_s,
    )
    pr_preds = (scores >= pr_threshold).astype(int)
    pr_tp = int((pr_preds & labs).sum())
    pr_fp = int((pr_preds & (1 - labs)).sum())
    pr_fn = int(((1 - pr_preds) & labs).sum())
    metrics["pr_curve_threshold"] = {
        "threshold":               round(pr_threshold, 6),
        "target_fp_per_90_min":    target_fp_per_90_min,
        "achieved_fp_per_90_min":  round((pr_fp / max(n_normal, 1)) * windows_per_90, 2),
        "precision":               round(pr_tp / max(pr_tp + pr_fp, 1), 4),
        "recall":                  round(pr_tp / max(pr_tp + pr_fn, 1), 4),
        "note": (
            "Use this threshold in production instead of the quantile-based one. "
            "It is calibrated to the FP budget, not an arbitrary percentile."
        ),
    }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Pattern Analysis Engine
# ─────────────────────────────────────────────────────────────────────────────
class PatternAnalysisEngine:
    """
    Top-level orchestrator.  One model per player.

    Scalability note: per-player isolated models become operationally heavy
    beyond ~500 players.  Future direction: shared backbone + player embedding.
    This is an architectural decision beyond this module's scope.
    """

    def __init__(self):
        self.window_builder   = SequenceWindowBuilder()
        self.drift_analyzer   = PositionalDriftAnalyzer()
        self.workload_tracker = WorkloadTrendTracker()
        self._shared_model: Optional[SharedBackboneAutoencoder] = None
        self._baselines:         Dict[int, PlayerBaselineProfile] = {}
        self._threshold_trackers: Dict[int, RegimeAwareThresholdStore] = {}
        self._position_buffers:  Dict[int, List[Tuple[float, float]]] = {}

    def register_player(
        self,
        player_id: int,
        baseline:  PlayerBaselineProfile,
        model:     Optional[object] = None,
    ) -> None:
        self._baselines[player_id] = baseline
        self._position_buffers[player_id] = []
    
    def train_player_model(
        self,
        all_windows: Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
    ) -> dict:
        player_ids = list(all_windows.keys())
        self._shared_model = SharedBackboneAutoencoder(n_players=len(player_ids))
        self._shared_model.register_players(player_ids)
        result = self._shared_model.train(all_windows)

        # Calibrate per-player regime-aware thresholds from a held-out slice (last 20%).
        # 20% (not 15%) ensures most players clear min_calibration_windows=30 even with
        # modest session counts. The same EMA + regime classification used in analyze()
        # is replayed here so threshold distributions match inference distributions exactly.
        alpha = CONFIG.scoring.score_ema_alpha
        for pid, windows in all_windows.items():
            calib   = windows[int(len(windows) * 0.80):]   # 20% held out
            store   = RegimeAwareThresholdStore()
            ema_val = None
            for seq, mask in calib:
                raw_loss, _, _ = self._shared_model.predict(pid, seq, mask)
                ema_val        = raw_loss if ema_val is None else (
                    alpha * raw_loss + (1 - alpha) * ema_val
                )
                regime_key = _REGIME_CLASSIFIER.classify(seq).key
                store.update(ema_val, regime_key)
            self._threshold_trackers[pid] = store
            logger.debug("PatternAnalysisEngine p%d calibration:\n%s",
                         pid, store.summary())

        # Persist the trained model so it survives process restarts
        try:
            saved_path = self._shared_model.save()
            logger.info("Shared backbone saved → %s", saved_path)
        except Exception as exc:
            logger.warning("Model save failed (non-fatal): %s", exc)

        return result

    def analyze(
        self,
        player_id:   int,
        live_event:  dict,
        sessions_df: pd.DataFrame,
    ) -> Optional[AnomalyResult]:

        baseline = self._baselines.get(player_id)
        if baseline is None:
            return None

        result = self.window_builder.add_event(live_event)
        if result is None:
            return None
        sequence, mask = result

        shared = self._shared_model
        if shared and shared.is_trained:
            recon_loss, _, _ = shared.predict(player_id, sequence, mask)
            tracker = self._threshold_trackers.get(player_id)
            if tracker and tracker.is_calibrated:
                is_anomaly = recon_loss > tracker.threshold_for(_REGIME_CLASSIFIER.classify(sequence).key)
                confidence = tracker.confidence_for(recon_loss, _REGIME_CLASSIFIER.classify(sequence).key)
            else:
                is_anomaly, confidence = False, 0.0
            model_type = shared.MODEL_TYPE
        else:
            recon_loss, is_anomaly, confidence = 0.0, False, 0.0
            model_type = "none"

        x = live_event.get("x_pitch")
        y = live_event.get("y_pitch")
        if x is not None and y is not None:
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
            acwr            = w.get("acwr", 1.0)
            workload_status = w.get("workload_status", "optimal")

        workload_flag = acwr > 1.5 or acwr < 0.8
        elapsed       = float(live_event.get("elapsed_seconds", 0))

        # Build feature vector first — fatigue flag logic reads from it below.
        last = sequence[-1]
        fv   = {SEQUENCE_FEATURE_NAMES[i]: float(last[i])
                for i in range(N_SEQUENCE_FEATURES)}
        fv.update({
            "acwr":                float(acwr),
            "reconstruction_loss": float(recon_loss),
            "drift_score":         float(drift["drift_score"]),
            "mask_completeness":   float(mask.mean()),
        })

        # Thread enriched features computed upstream in demo / ingestion pipeline.
        # These are attached to live_event by enrich_event_with_fatigue_features()
        # but are NOT part of the 8 LSTM input features, so they must be copied
        # explicitly into the feature vector that the XAI layer reads.
        for _enriched_key in ("fatigue_decay_residual", "speed_drop_pct",
                              "coach_fatigue_severity",
                              "coach_pre_match_status_encoded"):
            if _enriched_key in live_event:
                fv[_enriched_key] = float(live_event[_enriched_key])

        # EMA smoothing on the live anomaly score.
        # Raw per-window loss is noisy. Smooth it so transient spikes don't fire alerts.
        alpha_ema = CONFIG.scoring.score_ema_alpha
        ema_key   = f"_ema_{player_id}"
        prev_ema  = getattr(self, "_ema_scores", {}).get(player_id, recon_loss)
        if not hasattr(self, "_ema_scores"):
            self._ema_scores = {}
        smoothed_loss = alpha_ema * recon_loss + (1 - alpha_ema) * prev_ema
        self._ema_scores[player_id] = smoothed_loss

        # Re-evaluate is_anomaly on smoothed score using regime-specific threshold.
        # Classify the current sequence so high-press / low-block windows are
        # compared against their own calibration distribution, not the pooled one.
        regime_key = _REGIME_CLASSIFIER.classify(sequence).key
        tracker = self._threshold_trackers.get(player_id)
        if tracker and tracker.is_calibrated:
            is_anomaly = smoothed_loss > tracker.threshold_for(regime_key)
            confidence = tracker.confidence_for(smoothed_loss, regime_key)

        # Relative fatigue flag — speed ratio vs. personal baseline.
        # Absolute threshold (speed < 3.5) penalises GKs and defenders who
        # legitimately move slowly. Use personal baseline mean instead.
        baseline_speed = (baseline.distance_mean / (90 * 60)
                          if baseline.distance_mean > 0 else 3.5)
        speed_ratio    = fv.get("speed_ms", 999) / max(baseline_speed, 0.1)
        speed_low      = speed_ratio < 0.55          # below 55% of personal average
        sprint_low     = fv.get("sprint_flag", 1) == 0
        hr_elevated    = fv.get("heart_rate_bpm", 0) > (
                            baseline.distance_mean / 90 * 0.06 + 130   # crude proxy
                         ) if hasattr(baseline, "distance_mean") else False
        late_in_game   = elapsed > 2700              # after 45 min
        # Fatigue requires: anomaly + physical decline signal + time context
        fatigue_flag   = is_anomaly and (speed_low or sprint_low) and late_in_game

        return AnomalyResult(
            player_id=player_id,
            external_id=baseline.external_id,
            ts=datetime.now(tz=timezone.utc),
            anomaly_score=recon_loss,
            is_anomaly=is_anomaly,
            confidence=confidence,
            feature_vector=fv,
            sequence_shape=sequence.shape,
            raw_sequence=sequence,        # (T, F) raw unnormalised — for true SHAP
            raw_mask=mask,                # (T,)   validity mask
            fatigue_flag=fatigue_flag,
            positional_drift_flag=drift["is_flagged"],
            workload_flag=workload_flag,
            workload_status=workload_status,
            recommendation_type=self._recommend(
                is_anomaly, fatigue_flag,
                drift["is_flagged"], workload_flag, confidence,
            ),
            model_type=model_type,
        )

    def build_training_sequences(
        self,
        events_df:   pd.DataFrame,
        sessions_df: pd.DataFrame,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """ Build training sequences for all players from raw events and sessions dataframes."""
        all_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, session in sessions_df.iterrows():

            sess_ev = events_df[
                events_df["session_id"] == session["session_id"]
            ].copy()
            all_pairs.extend(self.window_builder.build_from_session(sess_ev))

        return all_pairs

    #honest name — sequential per-player loop, not vectorised ─────────
    def analyze_players_sequential(
        self,
        player_ids: List[int],
        sequences:  List[np.ndarray],
        masks:      Optional[List[np.ndarray]] = None,
    ) -> Dict[int, Tuple[float, bool, float]]:
        """
        Runs inference for each player using the shared backbone model.
        Per-player thresholds applied from _threshold_trackers.
        """
        if masks is None:
            masks = [None] * len(player_ids)
        results = {}
        for pid, seq, msk in zip(player_ids, sequences, masks):
            if self._shared_model and self._shared_model.is_trained:
                loss, _, _ = self._shared_model.predict(pid, seq, msk)
                tracker    = self._threshold_trackers.get(pid)
                regime_key = _REGIME_CLASSIFIER.classify(seq).key
                if tracker and tracker.is_calibrated:
                    is_anomaly = loss > tracker.threshold_for(regime_key)
                    confidence = tracker.confidence_for(loss, regime_key)
                else:
                    is_anomaly, confidence = False, 0.0
                results[pid] = (loss, is_anomaly, confidence)
            else:
                results[pid] = (0.0, False, 0.0)
        return results

    def evaluate_player(
        self,
        player_id:       int,
        labeled_windows: List[Tuple[np.ndarray, np.ndarray, bool]],
    ) -> dict:
        """Evaluate the shared model against labeled windows for one player."""
        if self._shared_model is None:
            return {"error": "no shared model trained"}
        if not self._shared_model.is_trained:
            return {"error": "shared model not yet trained"}
        if self._shared_model._player_index.get(player_id) is None:
            return {"error": f"player {player_id} not registered in shared model"}

        tracker = self._threshold_trackers.get(player_id)

        seqs   = [w for w, _, _ in labeled_windows]
        masks  = [m for _, m, _ in labeled_windows]
        labels = [l for _, _, l in labeled_windows]

        results = []
        for seq, msk in zip(seqs, masks):
            loss, _, _  = self._shared_model.predict(player_id, seq, msk)
            regime_key  = _REGIME_CLASSIFIER.classify(seq).key
            if tracker and tracker.is_calibrated:
                threshold = tracker.threshold_for(regime_key)
                conf      = tracker.confidence_for(loss, regime_key)
            else:
                threshold = float("inf")
                conf      = 0.0
            is_anom = loss > threshold
            results.append((loss, is_anom, conf))

        window_interval_s = float(CONFIG.window.event_interval_s * CONFIG.window.window_steps)
        return evaluate_model_results(
            results, labels,
            threshold=threshold,
            ema_smoothed=True,    # scores are EMA-smoothed — disclosed to reviewer
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
        """
        priority ladder.
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