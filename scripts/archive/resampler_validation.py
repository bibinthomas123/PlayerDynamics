"""
resampler_validation.py

Validates KinexonResampler against real session 3387 data: raw rows,
resampled rows, buckets per player, windows per player, and confirms an
8-step window now spans ~120 seconds of real time (not ~0.4s).

Run:
    python scripts/resampler_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import CONFIG
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH = DATA_DIR / "statistics.csv"
SESSION_ID = "3387"

LINE = "-" * 78


def main() -> None:
    print(LINE)
    print("KinexonResampler Validation — real session 3387")
    print(LINE)

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(STATS_PATH)
    print(f"Players in statistics.csv: {len(meta)}")

    observations = list(
        adapter.stream_positions(POSITIONS_PATH, meta, session_id=SESSION_ID, match_id=SESSION_ID)
    )
    n_raw = len(observations)
    print(f"Raw rows streamed from positions.csv (excl. ball, excl. skipped): {n_raw}")

    resampler = KinexonResampler()
    print(f"\nConfig: event_interval_s={resampler.bucket_seconds}s, "
          f"window_steps={CONFIG.window.window_steps}, "
          f"window_seconds={CONFIG.window.window_seconds}s")

    events_by_player, sessions_df = resampler.resample(observations, session_id=SESSION_ID)

    n_players = len(events_by_player)
    n_resampled_total = sum(len(df) for df in events_by_player.values())
    print(f"\nPlayers with >=1 resampled bucket: {n_players}")
    print(f"Total resampled rows (all players): {n_resampled_total}")
    print(f"Reduction factor: {n_raw / n_resampled_total:.1f}x fewer rows")

    print(f"\n{'player_id':>10} {'raw_ticks':>10} {'buckets':>8} {'8-step windows':>15} {'span_check_s':>13}")
    total_windows = 0
    for player_id, df in sorted(events_by_player.items()):
        n_buckets = len(df)
        window_steps = CONFIG.window.window_steps
        n_windows = max(0, n_buckets - window_steps + 1)  # sliding, stride=window_steps default in build_from_session
        n_windows_stride1 = max(0, n_buckets - window_steps + 1)
        total_windows += max(0, n_buckets // window_steps)  # non-overlapping count, conservative
        raw_ticks_for_player = sum(df["n_raw_ticks"])
        # span check: take window_steps consecutive buckets, measure actual elapsed time
        if n_buckets >= window_steps:
            span = (df["ts"].iloc[window_steps - 1] - df["ts"].iloc[0]).total_seconds()
        else:
            span = float("nan")
        print(f"{player_id:>10} {raw_ticks_for_player:>10} {n_buckets:>8} {n_windows_stride1:>15} {span:>13.2f}")

    print(f"\nTotal raw ticks across all players: {sum(int(df['n_raw_ticks'].sum()) for df in events_by_player.values())}")
    print(f"Total buckets across all players: {n_resampled_total}")
    print(f"Total sliding 8-step windows (stride=1) across all players: "
          f"{sum(max(0, len(df) - CONFIG.window.window_steps + 1) for df in events_by_player.values())}")
    print(f"Total non-overlapping 8-step windows (stride=8) across all players: "
          f"{sum(len(df) // CONFIG.window.window_steps for df in events_by_player.values())}")

    # ── Example output: one player's first window, full detail ──────────
    example_pid = next(iter(events_by_player))
    example_df = events_by_player[example_pid]
    print(f"\n{LINE}\nExample resampled output -- player {example_pid}, first 8 buckets:\n{LINE}")
    print(example_df.head(8).to_string(index=False))

    window_steps = CONFIG.window.window_steps
    if len(example_df) >= window_steps:
        first_ts = example_df["ts"].iloc[0]
        last_ts = example_df["ts"].iloc[window_steps - 1]
        span_s = (last_ts - first_ts).total_seconds()
        print(f"\nFirst 8-step window literal span: {first_ts} -> {last_ts} = {span_s:.1f}s "
              f"(target: ~{CONFIG.window.window_seconds}s)")

    print(f"\n{LINE}\nsessions_df ({len(sessions_df)} rows):\n{LINE}")
    print(sessions_df.to_string(index=False))

    print(f"\n{LINE}\nVERDICT")
    print(f"  Raw rows                : {n_raw}")
    print(f"  Resampled rows          : {n_resampled_total}")
    print(f"  Players covered         : {n_players}")
    print(f"  8-step window real span : ~{span_s:.1f}s (target 120s)" if len(example_df) >= window_steps else "n/a")
    print(LINE)


if __name__ == "__main__":
    main()
