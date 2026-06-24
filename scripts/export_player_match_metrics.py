#!/usr/bin/env python
"""
Exports real per-(match_id, player_id) physical + workload metrics for
Backend's PlayerMatchHistory backfill -- one row per match per player,
not the cross-match rolling aggregates player_trends.json carries
(last_match/last_5_matches/season_average), which collapse to a single
"most recent" snapshot and cannot be used to backfill historical matches
correctly (every match would silently get the same "last_match" value).

Sources, both real:
  data/processed/players.parquet        -- per-match Distance (m)/Sprints/
    Accelerations/Decelerations/high-intensity columns (same formula
    analysis/player_trends.py uses: sum of "Distance (speed | High) (m)"
    + "Distance (speed | Very high) (m)"), deduplicated to the full-session
    row per (match_id, Player ID) exactly like export_match_roster.py.
  data/processed/player_match_loads.parquet -- per-match
    acceleration_load/deceleration_load/sprint_load/high_intensity_load/
    distance_covered (analysis/player_match_loads.py, built from
    positions.csv via the resampler -- a different real source than the
    statistics.csv-derived raw counts above, kept as separate fields).

Deliberately NOT included: acuteLoad/chronicLoad/acwr. Those are rolling
multi-match-history concepts that are only well-defined "as of" a specific
point in a player's match sequence -- player_trends.json's top-level
acute_load/chronic_load/acwr already represent that computed "as of right
now" (the player's most recent match), and recomputing a true point-in-time
value for every earlier historical match isn't implemented. Per the
project's missing-data principle, this is left for the backfill consumer
to handle explicitly (e.g. only attach acwr to a player's own chronologically
last match) rather than guessed here.

Usage: python scripts/export_player_match_metrics.py [--out path]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
_HIGH_INTENSITY_COLUMNS = ["Distance (speed | High) (m)", "Distance (speed | Very high) (m)"]


def _dedupe_full_session_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Same full-session-row selection as export_match_roster.py -- avoids
    double-counting a match's per-half statistics.csv rows."""
    is_full_session = df["Description"].fillna("").str.contains(" vs. ", regex=False)
    full_rows = df[is_full_session].drop_duplicates(subset=["match_id", "Player ID"], keep="first")
    half_rows = df[~is_full_session]
    covered = set(zip(full_rows["match_id"], full_rows["Player ID"]))
    half_only = half_rows[~half_rows.apply(lambda r: (r["match_id"], r["Player ID"]) in covered, axis=1)]
    if half_only.empty:
        return full_rows
    agg_spec = {c: "sum" for c in ["Distance (m)", "Sprints", "Accelerations", "Decelerations", *_HIGH_INTENSITY_COLUMNS]}
    if "Ownership" in df.columns:
        agg_spec["Ownership"] = "first"
    summed = half_only.groupby(["match_id", "Player ID"], as_index=False).agg(agg_spec)
    return pd.concat([full_rows, summed], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-parquet", default=str(PROCESSED_DIR / "players.parquet"))
    parser.add_argument("--player-match-loads-parquet", default=str(PROCESSED_DIR / "player_match_loads.parquet"))
    parser.add_argument("--out", default=str(PROCESSED_DIR / "player_match_metrics.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.players_parquet)
    df = df[df["Player ID"].notna()]
    for col in _HIGH_INTENSITY_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df = _dedupe_full_session_rows(df)

    df["high_intensity_actions"] = (
        df[_HIGH_INTENSITY_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    )

    records_by_key = {}
    for _, r in df.iterrows():
        pid = int(r["Player ID"])
        key = (str(r["match_id"]), pid)
        records_by_key[key] = {
            "match_id": str(r["match_id"]),
            "player_id": pid,
            "ownership": r.get("Ownership"),
            "distance_m": float(r["Distance (m)"]) if pd.notna(r.get("Distance (m)")) else None,
            "sprints": float(r["Sprints"]) if pd.notna(r.get("Sprints")) else None,
            "accelerations": float(r["Accelerations"]) if pd.notna(r.get("Accelerations")) else None,
            "decelerations": float(r["Decelerations"]) if pd.notna(r.get("Decelerations")) else None,
            "high_intensity_actions": float(r["high_intensity_actions"]),
            "acceleration_load": None,
            "deceleration_load": None,
            "sprint_load": None,
            "high_intensity_load": None,
            "distance_covered": None,
        }

    loads_path = Path(args.player_match_loads_parquet)
    if loads_path.exists():
        loads_df = pd.read_parquet(loads_path)
        for _, r in loads_df.iterrows():
            key = (str(r["match_id"]), int(r["player_id"]))
            if key in records_by_key:
                for field in ["acceleration_load", "deceleration_load", "sprint_load", "high_intensity_load", "distance_covered"]:
                    if field in r and pd.notna(r[field]):
                        records_by_key[key][field] = float(r[field])

    records = list(records_by_key.values())
    Path(args.out).write_text(json.dumps({"players": records}, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} per-match player metric rows to {args.out}")


if __name__ == "__main__":
    main()
