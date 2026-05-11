"""
Players Data — IBM CIC Germany, Group 11 / 2B
Configuration & Constants
Production-level configuration with environment variable support.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    host: str     = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int     = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str     = field(default_factory=lambda: os.getenv("DB_NAME", "players_data"))
    user: str     = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        return (f"postgresql+psycopg2://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.name}")

    @property
    def async_url(self) -> str:
        return (f"postgresql+asyncpg://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.name}")


# ─────────────────────────────────────────────
# Ingestion sources
# ─────────────────────────────────────────────
@dataclass
class GPSConfig:
    serial_port: str      = field(default_factory=lambda: os.getenv("GPS_SERIAL_PORT", "/dev/ttyUSB0"))
    baud_rate: int        = 9600
    tcp_host: Optional[str] = field(default_factory=lambda: os.getenv("GPS_TCP_HOST"))
    tcp_port: int         = field(default_factory=lambda: int(os.getenv("GPS_TCP_PORT", "2947")))
    sample_rate_hz: float = 10.0


@dataclass
class SportRadarConfig:
    api_key: str    = field(default_factory=lambda: os.getenv("SPORTRADAR_API_KEY", ""))
    base_url: str   = "https://api.sportradar.com/soccer/trial/v4/en"
    timeout_s: int  = 10
    retry_attempts: int = 3


@dataclass
class LiveEventWSConfig:
    url: str                    = field(default_factory=lambda: os.getenv("LIVE_WS_URL", "ws://localhost:8765"))
    reconnect_delay_s: float    = 2.0
    max_reconnect_attempts: int = 10
    heartbeat_interval_s: float = 30.0


@dataclass
class WearableSensorConfig:
    mqtt_broker: str  = field(default_factory=lambda: os.getenv("MQTT_BROKER", "localhost"))
    mqtt_port: int    = 1883
    topic_prefix: str = "players_data/sensors"
    qos: int          = 1


# ─────────────────────────────────────────────
# Sequence Feature Set
# 8 raw signals — exactly as recommended
# ─────────────────────────────────────────────
SEQUENCE_FEATURE_NAMES = [
    "speed_ms",           # Raw speed (m/s)
    "acceleration_ms2",   # delta_speed / delta_t
    "heart_rate_bpm",     # HR normalised per player
    "sprint_flag",        # Binary: 1 if speed >= 7.0 m/s
    "x_pitch",            # Pitch X [0, 100]
    "y_pitch",            # Pitch Y [0, 100]
    "distance_delta_m",   # Distance covered since last tick
    "hr_recovery_rate",   # delta_HR / delta_t (positive = recovering)
]
N_SEQUENCE_FEATURES = len(SEQUENCE_FEATURE_NAMES)   # 8


# ─────────────────────────────────────────────
# Sequence window
# ─────────────────────────────────────────────
@dataclass
class SequenceWindowConfig:
    window_seconds: int   = 120   # 120-second rolling window (as recommended)
    step_seconds: int     = 15   # Must match DT_OUT in data_generator.py (15 s/tick)
    min_events: int       = 5     # At 15 s/tick, 120s window = 8 ticks; require 5 valid
    event_interval_s: int = 15   # Must match DT_OUT=15 in data_generator.py
    # NOTE: for real GPS hardware at 1 Hz or 5 Hz, change both to 1 or 5.

    @property
    def window_steps(self) -> int:
        return self.window_seconds // self.event_interval_s   # = 8 steps at 15 s/tick


# ─────────────────────────────────────────────
# LSTM Autoencoder  (Phase 1)
# ─────────────────────────────────────────────
@dataclass
class LSTMAutoencoderConfig:
    hidden_size: int            = 64
    num_layers: int             = 2
    dropout: float              = 0.2
    latent_dim: int             = 16
    batch_size: int             = 32
    learning_rate: float        = 1e-3
    max_epochs: int             = 50
    patience: int               = 8
    min_sessions_to_train: int  = 15
    random_state: int           = 42


# ─────────────────────────────────────────────
# Transformer Autoencoder  (Phase 2)
# ─────────────────────────────────────────────
@dataclass
class TransformerAutoencoderConfig:
    d_model: int                = 64      # Must be divisible by n_heads
    n_heads: int                = 4       # Attention heads
    n_encoder_layers: int       = 3
    n_decoder_layers: int       = 3
    d_ff: int                   = 256     # Feed-forward inner dimension
    dropout: float              = 0.1
    latent_dim: int             = 32
    max_seq_len: int            = 16      # Max sequence length
    batch_size: int             = 64
    learning_rate: float        = 5e-4
    max_epochs: int             = 80
    patience: int               = 10
    min_sessions_to_train: int  = 30
    random_state: int           = 42


# ─────────────────────────────────────────────
# Anomaly scoring
# ─────────────────────────────────────────────
@dataclass
class AnomalyScoringConfig:
    """
    Dynamic per-player threshold.

    For large calibration sets (≥150 windows):
        threshold = quantile(clean_losses, threshold_quantile)

    For small sets (<150 windows):
        threshold = median + mad_multiplier * MAD * 1.4826
        MAD (median absolute deviation) is more robust than quantile on
        small samples — quantile(0.995) on N=50 is basically max(losses),
        which is trivially exceeded by any OOD live event.

    mad_multiplier = 5.0 (raised from 4.0).
        At 4.0 the threshold was too permissive — almost all live windows
        exceeded it, producing 100% confidence on every event.
        5.0 targets ~99.4% of a normal distribution (≈ 3σ in MAD units).

    score_ema_alpha = 0.25 (raised from 0.15).
        Higher alpha reacts faster to genuine sustained anomalies while
        still smoothing single-tick noise spikes.
    """
    k_sigma: float                  = 2.5
    min_calibration_windows: int    = 30
    large_calib_threshold: int      = 150
    score_ema_alpha: float          = 0.25
    threshold_quantile: float       = 0.995  # 0.95 was too loose (5% FPR on normals alone)
    mad_multiplier: float           = 5.0    # for small calib sets (< 150 windows)
    calib_contamination_pct: float  = 0.05


# ─────────────────────────────────────────────
# Personal baseline
# ─────────────────────────────────────────────
@dataclass
class BaselineConfig:
    rolling_window_days: int           = 28
    short_window_days: int             = 7
    min_sessions_for_baseline: int     = 5
    zscore_anomaly_threshold: float    = 2.5


@dataclass
class FatigueCurveConfig:
    window_minutes: int = 15
    min_segments: int   = 4
    decay_model: str    = "exponential"


@dataclass
class PositionalDriftConfig:
    zone_radius_meters: float       = 5.0
    drift_fraction_threshold: float = 0.3


# ─────────────────────────────────────────────
# Inference SLA
# ─────────────────────────────────────────────
@dataclass
class InferenceConfig:
    max_latency_ms: int = 200


# ─────────────────────────────────────────────
# XAI / SHAP
# ─────────────────────────────────────────────
@dataclass
class SHAPConfig:
    # n_background_samples: used for background mean in channel-ablation attribution.
    # 30 samples is sufficient; KernelExplainer is no longer used in sequence space
    # (it ran 2000+ model calls per event, causing 2-15s latency vs 200ms SLA).
    # Channel ablation runs exactly F+2 = 10 model calls per event (~30-50ms on CPU).
    n_background_samples: int       = 30
    max_display_features: int       = 8
    counterfactual_tolerance: float = 0.1


# ─────────────────────────────────────────────
# Feedback / Recalibration
# ─────────────────────────────────────────────
@dataclass
class FeedbackConfig:
    recalibration_cadence_days: int      = 7
    min_overrides_for_recalibration: int = 10
    threshold_adjustment_step: float     = 0.05
    per_player_sensitivity_decay: float  = 0.1


# ─────────────────────────────────────────────
# Fairness
# ─────────────────────────────────────────────
@dataclass
class FairnessConfig:
    flag_rate_disparity_threshold: float = 0.15
    protected_attributes: list = field(
        default_factory=lambda: ["position", "age_group", "nationality"]
    )
    audit_cadence_days: int = 7


# ─────────────────────────────────────────────
# Root config
# ─────────────────────────────────────────────
@dataclass
class PlayersDataConfig:
    db:          DatabaseConfig               = field(default_factory=DatabaseConfig)
    gps:         GPSConfig                    = field(default_factory=GPSConfig)
    sportradar:  SportRadarConfig             = field(default_factory=SportRadarConfig)
    live_ws:     LiveEventWSConfig            = field(default_factory=LiveEventWSConfig)
    wearable:    WearableSensorConfig         = field(default_factory=WearableSensorConfig)
    window:      SequenceWindowConfig         = field(default_factory=SequenceWindowConfig)
    lstm:        LSTMAutoencoderConfig        = field(default_factory=LSTMAutoencoderConfig)
    transformer: TransformerAutoencoderConfig = field(default_factory=TransformerAutoencoderConfig)
    scoring:     AnomalyScoringConfig         = field(default_factory=AnomalyScoringConfig)
    baseline:    BaselineConfig               = field(default_factory=BaselineConfig)
    fatigue:     FatigueCurveConfig           = field(default_factory=FatigueCurveConfig)
    positional:  PositionalDriftConfig        = field(default_factory=PositionalDriftConfig)
    inference:   InferenceConfig              = field(default_factory=InferenceConfig)
    shap:        SHAPConfig                   = field(default_factory=SHAPConfig)
    feedback:    FeedbackConfig               = field(default_factory=FeedbackConfig)
    fairness:    FairnessConfig               = field(default_factory=FairnessConfig)

    # Active model selection: "lstm" | "transformer"
    active_model: str = "lstm"


CONFIG = PlayersDataConfig()