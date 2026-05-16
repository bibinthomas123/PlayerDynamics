"""
Create a globally time-ordered JSONL replay stream from events.csv.

Purpose
-------
Converts per-player event rows into a realistic multiplexed live stream:
    p003 @ t0
    p009 @ t0
    p006 @ t0
    p003 @ t15
    ...

instead of grouped player blocks.

This version also allows:
    - restricting replay to selected players
    - restricting replay duration
    - creating smaller debug streams

Usage
-----
python create_stream_jsonl.py \
    --events data/events.csv \
    --output test_anomalous.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────
# DEBUG / SMALL STREAM CONFIG
# ─────────────────────────────────────────────

# Keep only selected players
KEEP_PLAYERS = {
    "p003",
    "p007",
    "p014",
}

# Replay only first N elapsed seconds
MAX_ELAPSED_SECONDS = None   # 30 mins


def build_stream(events_csv: str, output_jsonl: str) -> None:

    # ─────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────
    df = pd.read_csv(events_csv)

    required = {
        "player_id",
        "ts",
        "elapsed_seconds",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # ─────────────────────────────────────────────
    # Ensure canonical player_external_id
    # ─────────────────────────────────────────────
    if "player_external_id" not in df.columns:
        df["player_external_id"] = df["player_id"].apply(
            lambda x: f"p{int(x):03d}"
        )

    # ─────────────────────────────────────────────
    # Filter players
    # ─────────────────────────────────────────────
    df = df[
        df["player_external_id"].isin(KEEP_PLAYERS)
    ].copy()

    # ─────────────────────────────────────────────
    # Filter replay duration
    # ─────────────────────────────────────────────
    if MAX_ELAPSED_SECONDS is not None:
        df = df[
            df["elapsed_seconds"] <= MAX_ELAPSED_SECONDS
        ].copy()

    # ─────────────────────────────────────────────
    # Global interleaving
    # ─────────────────────────────────────────────
    #
    # Critical fix:
    # sort ALL events globally by time before export
    #
    df = df.sort_values(
        by=[
            "elapsed_seconds",
            "ts",
            "player_external_id",
        ],
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    # ─────────────────────────────────────────────
    # Convert NaN → None for valid JSON
    # ─────────────────────────────────────────────
    records = (
        df.where(pd.notnull(df), None)
        .to_dict("records")
    )

    # ─────────────────────────────────────────────
    # Write JSONL
    # ─────────────────────────────────────────────
    out_path = Path(output_jsonl)

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in records:
            f.write(json.dumps(row))
            f.write("\n")

    # ─────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────
    print("=" * 60)
    print("STREAM JSONL CREATED")
    print("=" * 60)

    print(f"Input events : {events_csv}")
    print(f"Output jsonl : {output_jsonl}")
    print(f"Total rows   : {len(records):,}")

    print("\nPlayers kept:")
    print(sorted(KEEP_PLAYERS))

    print("\nReplay duration:")
    print(f"{MAX_ELAPSED_SECONDS} seconds")

    print("\nFirst 20 player IDs in stream order:")
    print(
        [
            r["player_external_id"]
            for r in records[:20]
        ]
    )

    print("\nFirst 5 timestamps:")
    for r in records[:5]:
        print(
            r["player_external_id"],
            r.get("elapsed_seconds"),
            r.get("ts"),
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--events",
        required=True,
        help="Path to events.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file",
    )

    args = parser.parse_args()

    build_stream(
        events_csv=args.events,
        output_jsonl=args.output,
    )


if __name__ == "__main__":
    main()