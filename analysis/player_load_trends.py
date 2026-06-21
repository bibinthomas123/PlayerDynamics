"""
Player Load Trend Engine — PlayerDynamics

Cross-match Acute Load / Chronic Load / ACWR + Sprint/HighIntensity/
Acceleration/Deceleration Load trends, built on top of
analysis/player_match_loads.py's per-(match_id, player_id) load rows.

Reuses analysis.player_trends.rolling_metric() for the
last_match/last_5/last_7/last_10/season_average/trend shape, so these new
metrics sit in player_trends.json next to distance_m/sprints/etc. with the
exact same structure the existing dashboard JSON consumers already expect.

No model code involved -- this module does not import
analysis.anomaly_detection or analysis.orchestrator.

Acute Load / Chronic Load / ACWR -- adaptation note
-----------------------------------------------------
The standard sports-science ACWR (Gabbett, 2016, "The training-injury
prevention paradox") compares a 7-day acute load against a 28-day rolling
chronic load -- a 1:4 timeframe ratio. This dataset has no daily
training-load tracking, only match-level Kinexon exports, so the same 1:4
ratio is adapted to match cadence instead of days:

    acute_load   = current_load (== acceleration_load + deceleration_load,
                    the SAME composite analysis/player_workload.py already
                    defines and the live analytics.player_workload stream
                    already shows) of the most recent match.
    chronic_load = mean current_load over the last CHRONIC_WINDOW_MATCHES
                    matches (default 4 -- mirrors the 28:7 day ratio;
                    falls back to however many matches exist if fewer).
    acwr         = acute_load / chronic_load, or None if chronic_load == 0.

No new "load" formula is invented for this -- current_load is exactly
player_workload.py's pre-existing, documented composite, just summed to
match granularity instead of window granularity.

With a single real match in the dataset, acute_load == chronic_load for
every player (one data point), so acwr is trivially 1.0 -- mathematically
correct given the data, not a bug. It starts producing real signal the
moment a second match is ingested.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from analysis.player_trends import rolling_metric

ACUTE_WINDOW_MATCHES = 1
CHRONIC_WINDOW_MATCHES = 4

_LOAD_METRIC_COLUMNS = {
    "acceleration_load": "acceleration_load",
    "deceleration_load": "deceleration_load",
    "sprint_load": "sprint_load",
    "high_intensity_load": "high_intensity_load",
    "distance_covered": "distance_covered",
}


def _ordered_match_ids(match_loads_df: pd.DataFrame, matches_df: pd.DataFrame) -> list:
    """Chronological match_id order, by matches.parquet's 'date' column when
    parseable, else by match_id string -- same fallback convention
    analysis.player_trends.compute_player_trends() already uses."""
    ids = match_loads_df["match_id"].astype(str).unique().tolist()
    if matches_df is None or matches_df.empty or "date" not in matches_df.columns:
        return sorted(ids)
    dates = matches_df.set_index(matches_df["match_id"].astype(str))["date"]
    parsed = pd.to_datetime(dates, errors="coerce")
    if parsed.isna().any():
        return sorted(ids)
    order = parsed.sort_values().index.tolist()
    return [m for m in order if m in ids]


def compute_load_trends(match_loads_df: pd.DataFrame, matches_df: Optional[pd.DataFrame] = None) -> Dict[int, dict]:
    """Returns {player_id: {metrics: {...}, acute_load, chronic_load, acwr}}.

    match_loads_df: analysis.player_match_loads.build_player_match_loads_table()'s
    output (one row per match_id, player_id).
    """
    if match_loads_df is None or match_loads_df.empty:
        return {}

    match_order = _ordered_match_ids(match_loads_df, matches_df)
    order_index = {m: i for i, m in enumerate(match_order)}

    df = match_loads_df.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["_order"] = df["match_id"].map(order_index)
    df = df.dropna(subset=["_order"]).sort_values(["player_id", "_order"])

    out: Dict[int, dict] = {}
    for pid, g in df.groupby("player_id"):
        if pd.isna(pid):
            continue
        g = g.reset_index(drop=True)
        metrics = {}
        for metric_key, col in _LOAD_METRIC_COLUMNS.items():
            series = pd.to_numeric(g[col], errors="coerce").fillna(0.0).to_numpy()
            metrics[metric_key] = rolling_metric(series)

        current_load_series = (
            pd.to_numeric(g["acceleration_load"], errors="coerce").fillna(0.0)
            + pd.to_numeric(g["deceleration_load"], errors="coerce").fillna(0.0)
        ).to_numpy()

        acute_load = float(np.mean(current_load_series[-ACUTE_WINDOW_MATCHES:]))
        chronic_load = float(np.mean(current_load_series[-CHRONIC_WINDOW_MATCHES:]))
        acwr = round(acute_load / chronic_load, 3) if chronic_load != 0 else None

        out[int(pid)] = {
            "metrics": metrics,
            "acute_load": round(acute_load, 2),
            "chronic_load": round(chronic_load, 2),
            "acwr": acwr,
            "n_matches_with_load_data": int(len(g)),
        }
    return out
