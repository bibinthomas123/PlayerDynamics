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
  not take the field / has zero tracked ticks for this match), ownership
  ("SCM"/"OPPONENT"/"BALL", players.parquet's Ownership column -- see
  config/settings.py::classify_ownership). Backend's MatchIntelligenceService
  filters on this field so opposing-team rosters (present in every Kinexon
  export alongside SC Magdeburg's own) never enter coach-facing ranking.

Backend's Match Intelligence Engine uses time_on_field_s to normalize
physical/workload metrics by playing time before any cross-player ranking
(per-minute rate instead of raw total) -- required so a player who simply
played more minutes doesn't automatically rank above a position peer who
played fewer but performed just as intensely.

DEDUPLICATION (statistics.csv carries multiple rows per player per match --
one "full session" row plus one row per half, e.g. "1. HZ"/"2.HZ"/
"1. Halbzeit"/"2. Halbzeit" -- see Types/Description columns):
  Without dedup, a (match_id, player_id) pair appears 1-3 times in
  players.parquet and whichever row happened to be written last would win
  in any downstream Map keyed by (match_id, player_id) -- in practice this
  silently picked the SECOND HALF's time_on_field_s instead of the full
  match's, for any match whose export includes half-rows.

  The full-session row is identified by its Description containing
  " vs. " (Kinexon always labels it "<team A> vs. <team B>" -- verified
  across all 4 ingested matches, present for every player in every match
  the same way regardless of whether the half rows additionally carry a
  Types="Half time" tag or just a bare Description like "1. Halbzeit").

  If a (match_id, player_id) pair has no such full-session row (a match
  export that only ever contains half rows -- not observed in any of the
  4 matches ingested so far, but not assumed impossible), the two halves'
  time_on_field_s are summed instead of arbitrarily picking one, and
  position/team are taken from whichever half row is non-null.

Usage: python scripts/export_match_roster.py [--out path]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _dedupe_to_one_row_per_player_match(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, Player ID): prefer the full-session row
    (Description contains " vs. "); otherwise sum half-rows' time on
    field and take position/team from the first non-null half row."""
    is_full_session = df["Description"].fillna("").str.contains(" vs. ", regex=False)
    full_rows = df[is_full_session].drop_duplicates(subset=["match_id", "Player ID"], keep="first")

    half_rows = df[~is_full_session]
    covered = set(zip(full_rows["match_id"], full_rows["Player ID"]))
    half_only = half_rows[~half_rows.apply(lambda r: (r["match_id"], r["Player ID"]) in covered, axis=1)]

    if half_only.empty:
        return full_rows

    summed = (
        half_only.groupby(["match_id", "Player ID"], as_index=False).agg(
            {
                "Position": "first",
                "Group name": "first",
                "Time on Playing Field (s)": "sum",
                **({"Ownership": "first"} if "Ownership" in df.columns else {}),
            }
        )
    )
    return pd.concat([full_rows, summed], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players-parquet", default=str(PROCESSED_DIR / "players.parquet"))
    parser.add_argument("--out", default=str(PROCESSED_DIR / "match_roster.json"))
    args = parser.parse_args()

    df = pd.read_parquet(args.players_parquet)
    df = df[df["Player ID"].notna()]
    df = _dedupe_to_one_row_per_player_match(df)

    cols = ["match_id", "Player ID", "Position", "Group name", "Time on Playing Field (s)"]
    out_cols = ["match_id", "player_id", "position", "team", "time_on_field_s"]
    if "Ownership" in df.columns:
        cols.append("Ownership")
        out_cols.append("ownership")
    keep = df[cols].copy()
    keep.columns = out_cols
    keep["player_id"] = keep["player_id"].astype(int)
    keep["time_on_field_s"] = keep["time_on_field_s"].astype(object).where(keep["time_on_field_s"].notna(), None)

    records = keep.to_dict(orient="records")
    Path(args.out).write_text(json.dumps({"players": records}, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} player-match roster rows to {args.out}")


if __name__ == "__main__":
    main()
