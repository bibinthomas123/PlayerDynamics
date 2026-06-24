"""
Player Trend Engine — PlayerDynamics

Pure observable-data trend computation across MULTIPLE matches. Reads
data/processed/players.parquet (one row per (match_id, player_id), built by
ingestion/multi_match_pipeline.py from each match's statistics.csv) and
computes, per player, for distance/sprints/accelerations/decelerations/
high-intensity actions:

    last_match, last_5_matches (average), last_10_matches (average),
    season_average, trend (improving/declining/stable)

No model code involved -- this module does not import
analysis.anomaly_detection or analysis.orchestrator.

high_intensity_actions = "Distance (speed | High) (m)" + "Distance (speed |
Very high) (m)", summed directly from statistics.csv's own per-match
columns -- the same two columns KinexonResampler._session_summary_row()
already treats as "high speed distance" when computed from positions.csv;
here they are read straight from each match's own pre-aggregated total
instead of being recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_METRIC_COLUMNS = {
    "distance_m": "Distance (m)",
    "sprints": "Sprints",
    "accelerations": "Accelerations",
    "decelerations": "Decelerations",
}
_HIGH_INTENSITY_COLUMNS = ["Distance (speed | High) (m)", "Distance (speed | Very high) (m)"]


def _classify(recent_avg: float, season_avg: float, band: float = 0.05) -> str:
    if season_avg == 0:
        return "stable" if recent_avg == 0 else "improving"
    rel = (recent_avg - season_avg) / abs(season_avg)
    if rel > band:
        return "improving"
    if rel < -band:
        return "declining"
    return "stable"


def rolling_metric(series: np.ndarray) -> dict:
    """last_match/last_5/last_7/last_10/season_average/trend over one
    player's match-ordered series of a single metric. Extracted out of
    compute_player_trends()'s loop (same formula, unchanged) so
    analysis/player_load_trends.py can reuse it for workload-derived
    metrics instead of re-implementing the same rolling-window logic."""
    series = np.asarray(series, dtype=np.float64)
    last_match = float(series[-1])
    last_5 = float(np.mean(series[-5:]))
    last_7 = float(np.mean(series[-7:]))
    last_10 = float(np.mean(series[-10:]))
    season_avg = float(np.mean(series))
    return {
        "last_match": round(last_match, 2),
        "last_5_matches": round(last_5, 2),
        "last_7_matches": round(last_7, 2),
        "last_10_matches": round(last_10, 2),
        "season_average": round(season_avg, 2),
        "trend": _classify(last_5, season_avg),
    }


def compute_player_trends(players_df: pd.DataFrame) -> dict:
    """players_df: the concatenated statistics.csv rows (one per match_id,
    player_id) from players.parquet.

    Coach-facing output -- restricted to SC Magdeburg's own roster
    (Ownership == "SCM", persisted by ingestion/multi_match_pipeline.py).
    Every Kinexon export also contains the opposing team's full roster
    (Ownership == "OPPONENT"), kept in players.parquet for training/
    research but never surfaced here."""
    df = players_df.copy()
    if df.empty or "Player ID" not in df.columns:
        return {"n_matches_in_dataset": 0, "players": []}

    if "Ownership" in df.columns:
        df = df[df["Ownership"] == "SCM"]
        if df.empty:
            return {"n_matches_in_dataset": 0, "players": []}

    for col in _HIGH_INTENSITY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df["high_intensity_actions"] = (
        df[_HIGH_INTENSITY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    )

    sort_col = "Session begin (UTC)" if "Session begin (UTC)" in df.columns else "match_id"
    df = df.sort_values(["Player ID", sort_col])

    metric_cols = {**_METRIC_COLUMNS, "high_intensity_actions": "high_intensity_actions"}

    players_out = []
    for pid, g in df.groupby("Player ID"):
        g = g.reset_index(drop=True)
        if pd.isna(pid):
            continue
        entry = {
            "player_id": int(pid),
            "player_name": str(g["Name"].iloc[-1]) if "Name" in g.columns else None,
            "n_matches": int(len(g)),
            "metrics": {},
        }
        for metric_key, col in metric_cols.items():
            if col not in g.columns:
                continue
            series = pd.to_numeric(g[col], errors="coerce").fillna(0.0).to_numpy()
            entry["metrics"][metric_key] = rolling_metric(series)
        players_out.append(entry)

    return {
        "n_matches_in_dataset": int(df["match_id"].nunique()) if "match_id" in df.columns else 0,
        "players": players_out,
    }


def build_and_write_player_trends(processed_dir: Path, include_load_trends: bool = True) -> dict:
    """Read processed_dir/players.parquet, compute trends, write
    processed_dir/player_trends.json, and return the same dict.

    include_load_trends=True (default): additionally builds
    player_match_loads.parquet (analysis.player_match_loads) and merges
    Acute Load/Chronic Load/ACWR + Sprint/HighIntensity/Acceleration/
    Deceleration Load trends (analysis.player_load_trends) into each
    player's "metrics" dict. Re-runs the Kinexon resampler per match, so
    this is slower than the players.parquet-only path -- set False to skip
    it (e.g. in tests that only care about the statistics.csv-derived
    metrics already covered by compute_player_trends()).
    """
    players_path = processed_dir / "players.parquet"
    if not players_path.exists():
        result = {"n_matches_in_dataset": 0, "players": []}
    else:
        players_df = pd.read_parquet(players_path)
        result = compute_player_trends(players_df)

    if include_load_trends:
        matches_path = processed_dir / "matches.parquet"
        if matches_path.exists():
            from analysis.player_match_loads import build_player_match_loads_table
            from analysis.player_load_trends import compute_load_trends

            matches_df = pd.read_parquet(matches_path)
            match_loads_df = build_player_match_loads_table(matches_df)
            if not match_loads_df.empty:
                match_loads_df.to_parquet(processed_dir / "player_match_loads.parquet", index=False)
                load_trends = compute_load_trends(match_loads_df, matches_df)
                for entry in result["players"]:
                    extra = load_trends.get(entry["player_id"])
                    if extra:
                        entry["metrics"].update(extra["metrics"])
                        entry["acute_load"] = extra["acute_load"]
                        entry["chronic_load"] = extra["chronic_load"]
                        entry["acwr"] = extra["acwr"]
                        entry["n_matches_with_load_data"] = extra["n_matches_with_load_data"]

    out_path = processed_dir / "player_trends.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result
