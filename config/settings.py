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
    "sprint_flag",        # Binary: 1 if speed >= KinexonConfig.sprint_threshold_ms
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

    # Minimum real-world gap (seconds) between consecutive events for the same
    # player before SequenceWindowBuilder.add_event() treats it as a
    # substitution/bench gap and resets that player's buffer, instead of
    # silently mixing pre-gap and post-gap ticks into the same window.
    # Placeholder default (4x event_interval_s) — not yet calibrated against
    # real substitution patterns; see HANDBALL_CALIBRATION.md.
    gap_threshold_s: float = 60.0

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
    batch_size: int             = 256
    learning_rate: float        = 1e-3
    max_epochs: int             = 250
    patience: int               = 20
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

    # Pilot mode: minimum number of valid within-session windows required to
    # build a PROVISIONAL baseline (BaselineBuilder.compute_provisional()) when
    # fewer than min_sessions_for_baseline historical sessions exist yet.
    # Does not affect the historical (>= min_sessions_for_baseline) path.
    min_windows_for_provisional: int  = 5


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
# TeamState aggregation (built on TacticalEvent stream)
# ─────────────────────────────────────────────
@dataclass
class TeamStateConfig:
    """
    Rolling-window lengths for TeamStateBuilder (analysis/team_state.py).

    Windows are wall-clock ticks anchored to the first event timestamp in the
    stream, NOT tied to individual event arrival times -- this keeps the
    snapshot cadence regular even though event density varies hugely by type
    (e.g. exertion_event fires far more often than shot).

    step_seconds_* defaults to the matching window length (tumbling/
    non-overlapping windows) when left as None; set explicitly for an
    overlapping/sliding cadence instead.
    """
    short_window_seconds: int            = 60
    short_step_seconds: Optional[int]    = None
    long_window_seconds: int             = 300
    long_step_seconds: Optional[int]     = None

    # confidence = min(1.0, n_events_in_window / scaled_min_events), where
    # scaled_min_events = min_events_for_full_confidence_per_60s * (window_seconds / 60)
    min_events_for_full_confidence_per_60s: float = 10.0


# ─────────────────────────────────────────────
# TeamStateTrend (temporal layer over TeamState snapshots)
# ─────────────────────────────────────────────
@dataclass
class TeamStateTrendConfig:
    """
    Fixed thresholds for TeamStateTrendBuilder (analysis/team_state_trend.py).

    attack_activity / physical_load / fatigue_burden are already rate-
    normalised (events/min or events/min/active-player -- see TeamState's
    docstring), so a single absolute threshold is comparable across both the
    60s and 300s windows.

    Calibration: set to roughly half the mean |consecutive-snapshot delta|
    observed across both real teams over the full session 3387 match
    (attack_activity ~3.0-6.5, physical_load ~8.5-14, fatigue_burden
    ~0.9-1.75 depending on window length) -- below the threshold, a swing is
    treated as within-window noise ("stable"); at or above it, as a genuine
    directional change.
    """
    attack_activity_threshold: float  = 3.0
    physical_load_threshold: float    = 8.0
    fatigue_burden_threshold: float   = 1.0


# ─────────────────────────────────────────────
# CoachInsightEngine (deterministic observation layer over TeamStateTrend)
# ─────────────────────────────────────────────
@dataclass
class CoachInsightConfig:
    """
    Fixed insight-firing thresholds for CoachInsightEngine
    (analysis/coach_insight.py).

    These are deliberately set HIGHER than the underlying
    TeamStateTrendConfig thresholds -- not every "increasing"/"decreasing"
    trend label is significant enough to surface as a coach-facing
    observation. Calibrated to roughly the 75th percentile of |delta|
    observed across both real teams in session 3387's 60s-window trends, so
    only the most notable ~quarter of swings qualify as an insight:

        attack_activity_delta   75th pct (60s) ~= 9.0
        physical_load_delta     75th pct (60s) ~= 19.0
        possession_pressure_delta  ~between 75th (0.17) and 90th (0.30) pct

    severity = "high" / "medium" / "low" from ratio = |delta| / threshold
    (always >= 1.0 once an insight has fired):
        ratio >= severity_high_ratio    -> "high"
        ratio >= severity_medium_ratio  -> "medium"
        otherwise                       -> "low"

    confidence = min(1.0, confidence_base + confidence_slope * (ratio - 1.0))
    -- a bare threshold crossing (ratio=1.0) gets confidence_base; confidence
    rises linearly with how far past the threshold the delta is, capped at 1.0.
    """
    attack_activity_insight_threshold: float    = 9.0
    physical_load_insight_threshold: float      = 19.0
    possession_pressure_insight_threshold: float = 0.25

    severity_high_ratio: float   = 2.0
    severity_medium_ratio: float = 1.5

    confidence_base: float  = 0.4
    confidence_slope: float = 0.2


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
# Kinexon UWB Tracking Adapter
# ─────────────────────────────────────────────
@dataclass
class KinexonConfig:
    """
    Configuration for the Kinexon UWB tracking adapter.

    Kinexon uses Ultra-Wideband positioning, not GPS. All positions arrive
    in metres in a pitch-centred coordinate system where (0, 0) = court centre.

    Player identity: Kinexon 'mapped id' (int) is used directly as player_id
    in PlayerDynamics. It is NOT the same as the backend DB Player.id.
    A cross-system mapping table is required for backend integration.
    """
    sport: str                    = "handball"
    pitch_length_m: float         = 40.0    # long axis (m)
    pitch_width_m: float          = 20.0    # short axis (m)
    source: str                   = "kinexon"

    # IHF high-intensity sprint threshold for handball (lower than football's 7.0)
    sprint_threshold_ms: float    = 5.5     # m/s  ≈ 19.8 km/h
    high_intensity_threshold_ms: float = 4.17  # m/s  ≈ 15.0 km/h

    # Match timing — IHF handball: 2 × 30 min halves, 60 min total
    match_half_duration_s: int    = 1800   # 30 min; used for late_in_game gate
    match_duration_s: int         = 3600   # 60 min; used for baseline speed normalisation

    # Sample rates
    positions_sample_rate_hz: float  = 20.0   # positions.csv — continuous 20 Hz
    inertial_sample_rate_hz: float   = 91.0   # Inertial.csv  — variable ~91 Hz

    # Plausibility caps; readings above these are sensor artefacts
    max_speed_ms: float           = 12.0    # m/s  ≈ 43 km/h; top handball < 11 m/s
    max_accel_ms2: float          = 25.0    # m/s²

    # Source file names relative to the data directory
    positions_file: str           = "positions.csv"
    inertial_file: str            = "Inertial.csv"
    events_file: str              = "events.csv"
    statistics_file: str          = "statistics.csv"

    # HR wearable availability — set True when HR sensors are integrated.
    # Currently False: session 3387 and all known SC Magdeburg Kinexon exports
    # contain heart_rate_bpm=None throughout (wearable not worn / not synced).
    # When False, heart_rate_bpm=0.0 means "sensor absent", NOT "malfunction".
    hr_sensor_present: bool       = False


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
    kinexon:     KinexonConfig                = field(default_factory=KinexonConfig)
    window:      SequenceWindowConfig         = field(default_factory=SequenceWindowConfig)
    lstm:        LSTMAutoencoderConfig        = field(default_factory=LSTMAutoencoderConfig)
    transformer: TransformerAutoencoderConfig = field(default_factory=TransformerAutoencoderConfig)
    scoring:     AnomalyScoringConfig         = field(default_factory=AnomalyScoringConfig)
    baseline:    BaselineConfig               = field(default_factory=BaselineConfig)
    fatigue:     FatigueCurveConfig           = field(default_factory=FatigueCurveConfig)
    positional:  PositionalDriftConfig        = field(default_factory=PositionalDriftConfig)
    team_state:  TeamStateConfig              = field(default_factory=TeamStateConfig)
    team_state_trend: TeamStateTrendConfig    = field(default_factory=TeamStateTrendConfig)
    coach_insight: CoachInsightConfig         = field(default_factory=CoachInsightConfig)
    inference:   InferenceConfig              = field(default_factory=InferenceConfig)
    shap:        SHAPConfig                   = field(default_factory=SHAPConfig)
    feedback:    FeedbackConfig               = field(default_factory=FeedbackConfig)
    fairness:    FairnessConfig               = field(default_factory=FairnessConfig)

    # Active model selection: "lstm" | "transformer"
    active_model: str = "lstm"


CONFIG = PlayersDataConfig()