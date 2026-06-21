"""
Player Match Load Aggregator — PlayerDynamics

Builds one row per (match_id, player_id) summing analysis.player_workload's
per-window Sprint/HighIntensity/Acceleration/Deceleration Load + distance
across a whole match. This is the per-player, per-match "load" record that
analysis/player_load_trends.py rolls up into Acute/Chronic Load, ACWR, and
cross-match trend metrics.

Deliberately reuses, unchanged:
    main._load_kinexon_frames()        -- KinexonAdapter -> KinexonResampler
                                           -> kinexon_events_features merge
                                           (the same loader
                                           scripts/publish_player_workload.py
                                           and `main.py train --data-source
                                           kinexon` already use)
    analysis.player_workload.compute_player_workload_windows()
                                        -- the SAME per-window
                                           acceleration_load/deceleration_load/
                                           sprint_load/high_intensity_load
                                           formulas already serving the live
                                           analytics.player_workload stream

No new load formula is introduced here -- this module only sums
already-approved per-window values across one match's windows, so e.g.
"Sprint Load" means exactly the same thing in the live in-match dashboard
and in the cross-match historical trend view.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

MATCH_LOAD_COLUMNS = [
    "match_id", "player_id",
    "acceleration_load", "deceleration_load",
    "sprint_load", "high_intensity_load", "distance_covered",
]


def compute_match_player_loads(match_dir: Path, match_id: str) -> pd.DataFrame:
    """One row per player for this match. Returns an empty (but correctly
    columned) DataFrame if this match's Kinexon export can't be loaded --
    callers iterating many matches should treat that the same as "0 rows
    for this match" rather than letting one bad match abort the whole run.
    """
    from main import _load_kinexon_frames
    from analysis.player_workload import compute_player_workload_windows
    from config.settings import CONFIG

    try:
        events_by_player, _sessions_df, _meta = _load_kinexon_frames(
            match_dir, match_id, use_event_features=True,
        )
    except SystemExit as exc:
        logger.warning(
            "compute_match_player_loads: skipping match_id=%s (%s) -- "
            "_load_kinexon_frames could not load it (exit code %s)",
            match_id, match_dir, exc.code,
        )
        return pd.DataFrame(columns=MATCH_LOAD_COLUMNS)

    hi_threshold = CONFIG.kinexon.high_intensity_threshold_ms
    rows: List[dict] = []
    for pid, df in events_by_player.items():
        windows = compute_player_workload_windows(pid, df, hi_threshold)
        if not windows:
            continue
        rows.append({
            "match_id": match_id,
            "player_id": pid,
            "acceleration_load": sum(w["acceleration_load"] for w in windows),
            "deceleration_load": sum(w["deceleration_load"] for w in windows),
            "sprint_load": sum(w["sprint_load"] for w in windows),
            "high_intensity_load": sum(w["high_intensity_load"] for w in windows),
            "distance_covered": sum(w["distance_covered"] for w in windows),
        })
    return pd.DataFrame(rows, columns=MATCH_LOAD_COLUMNS) if rows else pd.DataFrame(columns=MATCH_LOAD_COLUMNS)


def build_player_match_loads_table(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Runs compute_match_player_loads() for every match in matches_df
    (as produced by ingestion.multi_match_pipeline, read from
    matches.parquet) and concatenates the results. matches_df must have
    'match_id' and 'match_dir' columns.
    """
    frames = []
    for _, row in matches_df.iterrows():
        match_id = str(row["match_id"])
        match_dir = Path(row["match_dir"])
        if row.get("n_errors", 0):
            logger.info(
                "build_player_match_loads_table: skipping match_id=%s -- "
                "failed multi_match_pipeline validation", match_id,
            )
            continue
        frames.append(compute_match_player_loads(match_dir, match_id))
    if not frames:
        return pd.DataFrame(columns=MATCH_LOAD_COLUMNS)
    return pd.concat(frames, ignore_index=True)
