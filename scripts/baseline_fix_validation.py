"""
baseline_fix_validation.py

Task E validation for the MIN_EVENTS_PER_WINDOW fix: real
BaselineBuilder.compute_provisional() calls (actual production code, post-fix)
against real session 3387 data. Reports, per player: baseline built y/n,
n valid windows, and baseline quality metrics (distance/sprint/top_speed
mean+std). Also re-checks rejected players' raw candidate-window counts to
confirm rejection is due to genuinely insufficient tracked time (not a
remaining threshold miscalibration).

Run:
    python scripts/baseline_fix_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import CONFIG
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler
from analysis.baseline import BaselineBuilder

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH = DATA_DIR / "statistics.csv"
SESSION_ID = "3387"
LINE = "-" * 90


def main() -> None:
    print(LINE)
    print("Task E -- real-data validation of compute_provisional() post-fix (session 3387)")
    print(LINE)
    print(f"BaselineConfig.min_events_per_provisional_window = {CONFIG.baseline.min_events_per_provisional_window}")
    print(f"BaselineConfig.min_windows_for_provisional        = {CONFIG.baseline.min_windows_for_provisional} (unchanged)")

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(STATS_PATH)
    observations = list(
        adapter.stream_positions(POSITIONS_PATH, meta, session_id=SESSION_ID, match_id=SESSION_ID)
    )
    resampler = KinexonResampler()
    events_by_player, _ = resampler.resample(observations, session_id=SESSION_ID)

    builder = BaselineBuilder()
    built, rejected = [], []

    print(f"\n{'player_id':>10} {'mode':>12} {'dist_mean_m':>12} {'dist_std_m':>11} "
          f"{'sprint_mean':>11} {'top_speed_mean':>15} {'n_windows':>9}")
    for player_id, df in sorted(events_by_player.items()):
        profile = builder.compute_provisional(
            player_id=player_id, external_id=str(player_id), events_df=df, window_seconds=120
        )
        if profile is None:
            rejected.append(player_id)
            continue
        built.append(player_id)
        print(f"{player_id:>10} {profile.baseline_mode:>12} {profile.distance_mean:>12.1f} "
              f"{profile.distance_std:>11.1f} {profile.sprint_count_mean:>11.2f} "
              f"{profile.top_speed_mean:>15.2f} {profile.n_sessions:>9}")

    print(f"\n{LINE}\nSUMMARY\n{LINE}")
    print(f"Players with provisional baseline BEFORE fix (MIN_EVENTS_PER_WINDOW=30, prior run): 0 / {len(events_by_player)}")
    print(f"Players with provisional baseline AFTER fix  (min_events_per_provisional_window="
          f"{CONFIG.baseline.min_events_per_provisional_window}): {len(built)} / {len(events_by_player)}")
    print(f"Built:    {sorted(built)}")
    print(f"Rejected: {sorted(rejected)}")

    print(f"\n{LINE}\nRejection diagnostics (why are these {len(rejected)} still rejected?)\n{LINE}")
    for player_id in sorted(rejected):
        df = events_by_player[player_id]
        d = df.sort_values("ts").reset_index(drop=True)
        import pandas as pd
        elapsed = (pd.to_datetime(d["ts"], utc=True) - pd.to_datetime(d["ts"], utc=True).min()).dt.total_seconds()
        max_window = int(elapsed.max() // 120) + 1 if len(d) else 0
        valid = 0
        for w in range(max_window):
            seg = d[(elapsed >= w * 120) & (elapsed < (w + 1) * 120)]
            if len(seg) >= CONFIG.baseline.min_events_per_provisional_window:
                valid += 1
        print(f"  player {player_id}: {len(d)} total resampled rows, {valid} valid windows "
              f"(needs >= {CONFIG.baseline.min_windows_for_provisional}) -- "
              f"{'insufficient TOTAL tracked time' if valid < CONFIG.baseline.min_windows_for_provisional else 'unexpected'}")

    print(LINE)


if __name__ == "__main__":
    main()
