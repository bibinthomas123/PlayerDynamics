#!/usr/bin/env python
"""
Exports real per-player position + exposure (playing time) fields already
present in data/processed/players.parquet -- no new metric is computed here,
this only makes existing Kinexon export columns consumable by Backend
(Node can't read parquet without an extra dependency; this writes plain JSON
next to the other data/processed/*.json files Backend already reads).

Columns exported, all real and already in players.parquet:
  match_id, player_id (Kinexon "Player ID"), position (Kinexon "Position",
  e.g. "TW"/"KR"/"RM"/"RL"/"RR"/"LA"/"RA"), team (Kinexon "Group name"),
  time_on_field_s ("Time on Playing Field (s)" -- null if the player did
  not take the field / has zero tracked ticks for this match).

Backend's Match Intelligence Engine uses time_on_field_s to normalize
physical/workload metrics by playing time before any cross-player ranking
(per-minute rate instead of raw total) -- required so a player who simply
played more minutes doesn't automatically rank above a position peer who
played fewer but performed just as intensely.

Usage: python scripts/export_match_roster.py [--out path]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-parquet", default=str(PROCESSED_DIR / "players.parquet"))
    parser.add_argument("--out", default=str(PROCESSED_DIR / "match_roster.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.players_parquet)
    keep = df[["match_id", "Player ID", "Position", "Group name", "Time on Playing Field (s)"]].copy()
    keep.columns = ["match_id", "player_id", "position", "team", "time_on_field_s"]
    keep["player_id"] = keep["player_id"].astype(int)
    keep["time_on_field_s"] = keep["time_on_field_s"].astype(object).where(keep["time_on_field_s"].notna(), None)

    records = keep.to_dict(orient="records")
    Path(args.out).write_text(json.dumps({"players": records}, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} player-match roster rows to {args.out}")


if __name__ == "__main__":
    main()
