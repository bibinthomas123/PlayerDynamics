"""
PlayerWorkloadEvent — PlayerDynamics

Coach-facing, model-free player workload telemetry. Every field is computed
directly from real Kinexon positions.csv / events.csv aggregates (see
analysis/player_workload.py) -- this dataclass has no reconstruction_loss,
confidence, SHAP, anomaly, or calibration field, and analysis/player_workload.py
does not import analysis.anomaly_detection or analysis.orchestrator.

Published onto Redis stream analytics.player_workload (config.redis_client.
StreamTopics.ANALYTICS_PLAYER_WORKLOAD) -- a separate topic from
analytics.players, which carries the PlayerDynamics pilot model's output for
the "PlayerDynamics Pilot" page and is left untouched by this dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class PlayerWorkloadEvent:
    player_id: int
    external_id: str
    player_name: str
    position: str
    match_id: Optional[str]
    timestamp: datetime
    elapsed_s: float

    # Real, observable per-window workload metrics -- see
    # analysis/player_workload.py module docstring for the exact formula
    # and data provenance behind each one.
    current_load: float
    load_trend: str             # "increasing" | "decreasing" | "stable"
    acceleration_load: float
    deceleration_load: float
    sprint_load: float
    high_intensity_load: float
    distance_covered: float
    performance_trend: str      # "increasing" | "decreasing" | "stable"
    workload_status: str        # "low" | "normal" | "high"
