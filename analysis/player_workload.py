"""
Player Workload Metrics — PlayerDynamics

Pure observable-data computation for coach-facing dashboards. Every metric
here is derived ONLY from KinexonResampler's per-window aggregates
(speed_ms, speed_ms_max, distance_traveled_m) plus
ingestion/kinexon_events_features.py's event-derived per-window aggregates
(accel_event_count, decel_event_count, sprint_distance_m, ...).

This module does NOT import analysis.anomaly_detection, analysis.orchestrator,
or any LSTM/autoencoder/calibration code -- there is no model inference
anywhere in this file. It computes no soreness, fatigue, recovery, or
injury-risk metric of any kind.

Field-by-field provenance (see PlayerWorkloadEvent for the wire schema)
------------------------------------------------------------------------
current_load        -- accel_event_count * accel_mean_ms2 (in this window)
                        + decel_event_count * |decel_mean_ms2| (in this
                        window). A simple count x intensity composite over
                        already-computed real per-window values -- not a
                        proprietary "load" formula, just count*magnitude for
                        each already-real event category, summed.
load_trend           -- increasing/decreasing/stable: mean of this player's
                        own current_load over the first half of windows seen
                        so far this match vs the second half.
acceleration_load    -- accel_event_count * accel_mean_ms2 (this window)
deceleration_load    -- decel_event_count * |decel_mean_ms2| (this window)
sprint_load          -- sprint_distance_m (this window, from events.csv)
high_intensity_load  -- distance_traveled_m (this window) if this window's
                        speed_ms_max >= KinexonConfig.high_intensity_threshold_ms,
                        else 0.0. Same convention KinexonResampler's own
                        _session_summary_row() already uses for
                        high_speed_distance_m at the session level --
                        applied per-window here instead of per-session.
distance_covered     -- distance_traveled_m (this window, from KinexonResampler)
performance_trend    -- increasing/decreasing/stable: mean of this player's
                        own speed_ms over the first half of windows seen so
                        far this match vs the second half.
workload_status      -- low/normal/high: this window's current_load against
                        the 33rd/66th percentile of ALL players' current_load
                        values across the whole match (computed once after
                        all players' windows are built).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

REQUIRED_EVENT_FEATURE_COLS = (
    "accel_event_count", "accel_mean_ms2",
    "decel_event_count", "decel_mean_ms2",
    "sprint_distance_m",
)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].fillna(0.0)
    return pd.Series(0.0, index=df.index)


def classify_trend(values: np.ndarray, stable_band_frac: float = 0.05) -> str:
    """increasing/decreasing/stable from first-half vs second-half mean.

    stable_band_frac: relative-change threshold below which a series is
    called "stable" rather than increasing/decreasing -- guards against
    labelling normal window-to-window noise as a trend.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return "stable"
    half = len(values) // 2
    first = float(np.mean(values[:half])) if half else float(values[0])
    second = float(np.mean(values[half:]))
    if first == 0:
        return "stable" if second == 0 else "increasing"
    rel_change = (second - first) / abs(first)
    if rel_change > stable_band_frac:
        return "increasing"
    if rel_change < -stable_band_frac:
        return "decreasing"
    return "stable"


def compute_player_workload_windows(
    player_id: int,
    df: pd.DataFrame,
    high_intensity_threshold_ms: float,
) -> List[dict]:
    """
    df: events_by_player[player_id] from KinexonResampler, merged with
    kinexon_events_features (use_event_features=True), sorted by ts.

    Returns one dict per window (one row of df), without playerName/position
    (caller fills those in from KinexonPlayerMeta) and without workload_status
    (caller fills that in after seeing all players' current_load values).
    """
    df = df.sort_values("ts").reset_index(drop=True)
    if df.empty:
        return []

    accel_count = _col(df, "accel_event_count")
    accel_mean = _col(df, "accel_mean_ms2")
    decel_count = _col(df, "decel_event_count")
    decel_mean = _col(df, "decel_mean_ms2")
    sprint_dist = _col(df, "sprint_distance_m")

    acceleration_load = accel_count * accel_mean
    deceleration_load = decel_count * decel_mean.abs()
    current_load = acceleration_load + deceleration_load
    distance = df["distance_traveled_m"].fillna(0.0)
    speed = df["speed_ms"].fillna(0.0)
    speed_max = df.get("speed_ms_max", df["speed_ms"]).fillna(0.0)
    high_intensity_load = np.where(speed_max >= high_intensity_threshold_ms, distance, 0.0)

    rows = []
    for i in range(len(df)):
        rows.append({
            "player_id": player_id,
            "elapsed_s": float(df["elapsed_s"].iloc[i]),
            "ts": df["ts"].iloc[i],
            "current_load": float(current_load.iloc[i]),
            "load_trend": classify_trend(current_load.iloc[: i + 1].to_numpy()),
            "acceleration_load": float(acceleration_load.iloc[i]),
            "deceleration_load": float(deceleration_load.iloc[i]),
            "sprint_load": float(sprint_dist.iloc[i]),
            "high_intensity_load": float(high_intensity_load[i]),
            "distance_covered": float(distance.iloc[i]),
            "performance_trend": classify_trend(speed.iloc[: i + 1].to_numpy()),
        })
    return rows


def assign_workload_status(rows_by_player: Dict[int, List[dict]]) -> None:
    """
    Mutates each row dict in-place, adding "workload_status" (low/normal/high)
    based on this window's current_load against the 33rd/66th percentile of
    ALL players' ALL windows' current_load values this match.
    """
    all_loads = [
        row["current_load"]
        for rows in rows_by_player.values()
        for row in rows
    ]
    if not all_loads:
        return
    p33, p66 = np.percentile(all_loads, [33, 66])

    for rows in rows_by_player.values():
        for row in rows:
            cl = row["current_load"]
            if cl <= p33:
                row["workload_status"] = "low"
            elif cl >= p66:
                row["workload_status"] = "high"
            else:
                row["workload_status"] = "normal"
