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
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "players_data"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


# ─────────────────────────────────────────────
# Data Ingestion Sources
# ─────────────────────────────────────────────
@dataclass
class GPSConfig:
    """Serial GPS device or NMEA over TCP."""
    serial_port: str = field(default_factory=lambda: os.getenv("GPS_SERIAL_PORT", "/dev/ttyUSB0"))
    baud_rate: int = 9600
    tcp_host: Optional[str] = field(default_factory=lambda: os.getenv("GPS_TCP_HOST"))
    tcp_port: int = field(default_factory=lambda: int(os.getenv("GPS_TCP_PORT", "2947")))
    sample_rate_hz: float = 10.0          # 10 Hz GPS


@dataclass
class SportRadarConfig:
    """SportRadar / Opta / Stats Perform REST API."""
    api_key: str = field(default_factory=lambda: os.getenv("SPORTRADAR_API_KEY", ""))
    base_url: str = "https://api.sportradar.com/soccer/trial/v4/en"
    timeout_s: int = 10
    retry_attempts: int = 3


@dataclass
class LiveEventWSConfig:
    """WebSocket endpoint for live match event stream."""
    url: str = field(default_factory=lambda: os.getenv("LIVE_WS_URL", "ws://localhost:8765"))
    reconnect_delay_s: float = 2.0
    max_reconnect_attempts: int = 10
    heartbeat_interval_s: float = 30.0


@dataclass
class WearableSensorConfig:
    """BLE / ANT+ heart-rate & accelerometer streams via MQTT bridge."""
    mqtt_broker: str = field(default_factory=lambda: os.getenv("MQTT_BROKER", "localhost"))
    mqtt_port: int = 1883
    topic_prefix: str = "players_data/sensors"
    qos: int = 1


# ─────────────────────────────────────────────
# Analysis Engine
# ─────────────────────────────────────────────
@dataclass
class IsolationForestConfig:
    contamination: float = 0.15         # Expected anomaly rate
    n_estimators: int = 200
    max_samples: str = "auto"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class FatigueCurveConfig:
    window_minutes: int = 15             # Segment width for decay analysis
    min_segments: int = 4                # Minimum segments needed for fitting
    decay_model: str = "exponential"     # "exponential" | "linear"


@dataclass
class BaselineConfig:
    rolling_window_days: int = 28
    short_window_days: int = 7
    min_sessions_for_baseline: int = 5   # Need at least 5 sessions to establish baseline
    zscore_anomaly_threshold: float = 2.5


@dataclass
class PositionalDriftConfig:
    zone_radius_meters: float = 5.0      # Acceptable drift radius from tactical zone
    drift_fraction_threshold: float = 0.3  # Flag if >30% of touches outside zone


@dataclass
class InferenceConfig:
    sliding_window_seconds: int = 30
    max_latency_ms: int = 200            # SLA from the proposal
    batch_size: int = 64


# ─────────────────────────────────────────────
# XAI / SHAP
# ─────────────────────────────────────────────
@dataclass
class SHAPConfig:
    n_background_samples: int = 100      # KernelExplainer background size
    max_display_features: int = 10       # Features shown in waterfall
    counterfactual_tolerance: float = 0.1


# ─────────────────────────────────────────────
# Feedback & Recalibration
# ─────────────────────────────────────────────
@dataclass
class FeedbackConfig:
    recalibration_cadence_days: int = 7
    min_overrides_for_recalibration: int = 10
    threshold_adjustment_step: float = 0.05
    per_player_sensitivity_decay: float = 0.1


# ─────────────────────────────────────────────
# Fairness
# ─────────────────────────────────────────────
@dataclass
class FairnessConfig:
    flag_rate_disparity_threshold: float = 0.15   # Alert if group flag rate differs > 15%
    protected_attributes: list = field(default_factory=lambda: ["position", "age_group", "nationality"])
    audit_cadence_days: int = 7


# ─────────────────────────────────────────────
# Root Config
# ─────────────────────────────────────────────
@dataclass
class PlayersDataConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    sportradar: SportRadarConfig = field(default_factory=SportRadarConfig)
    live_ws: LiveEventWSConfig = field(default_factory=LiveEventWSConfig)
    wearable: WearableSensorConfig = field(default_factory=WearableSensorConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    fatigue: FatigueCurveConfig = field(default_factory=FatigueCurveConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    positional: PositionalDriftConfig = field(default_factory=PositionalDriftConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    shap: SHAPConfig = field(default_factory=SHAPConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    fairness: FairnessConfig = field(default_factory=FairnessConfig)


# Singleton
CONFIG = PlayersDataConfig()
