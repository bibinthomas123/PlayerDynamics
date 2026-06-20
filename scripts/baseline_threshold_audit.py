"""
baseline_threshold_audit.py

Pure audit (no production code changes here) of
BaselineBuilder.compute_provisional()'s MIN_EVENTS_PER_WINDOW=30 constant
against real session 3387 data, resampled to the actual event_interval_s=15
cadence (ingestion/kinexon_resampler.py).

Shows, for thresholds {4, 6, 8, 10, 12, 15, 20, 30}:
  - distribution of events per candidate 120s window (overall)
  - distribution of valid windows per player
  - resulting baseline "stability" (n windows retained, SEM of per-window
    distance) -- mirrors compute_provisional()'s own internal computation,
    parameterized by threshold, WITHOUT modifying analysis/baseline.py.

Run:
    python scripts/baseline_threshold_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config.settings import CONFIG
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH = DATA_DIR / "statistics.csv"
SESSION_ID = "3387"
LINE = "-" * 78

THRESHOLDS = [4, 6, 8, 10, 12, 15, 20, 30]
WINDOW_SECONDS = 120
MIN_WINDOWS_FOR_PROVISIONAL = CONFIG.baseline.min_windows_for_provisional  # 5, unchanged


def _candidate_windows(events_df: pd.DataFrame, window_seconds: int = WINDOW_SECONDS):
    """
    Mirrors compute_provisional()'s own elapsed_s binning EXACTLY (same
    floor-division into fixed window_seconds bins), independent of any
    threshold -- returns every candidate window's row-count and a few
    real telemetry stats, for sweeping thresholds against afterward
    without touching analysis/baseline.py.
    """
    d = events_df.sort_values("ts").reset_index(drop=True)
    ts = pd.to_datetime(d["ts"], utc=True)
    elapsed = (ts - ts.min()).dt.total_seconds()
    speed_cap = CONFIG.kinexon.max_speed_ms
    sprint_threshold = CONFIG.kinexon.sprint_threshold_ms

    max_window = int(elapsed.max() // window_seconds) + 1 if len(d) else 0
    windows = []
    for w in range(max_window):
        w_start, w_end = w * window_seconds, (w + 1) * window_seconds
        seg = d[(elapsed >= w_start) & (elapsed < w_end)]
        if len(seg) < 2:
            windows.append({"n_events": len(seg), "distance_m": 0.0, "top_speed": 0.0})
            continue
        speeds = seg["speed_ms"].fillna(0).clip(lower=0, upper=speed_cap).values
        seg_ts = pd.to_datetime(seg["ts"], utc=True)
        dt = seg_ts.diff().dt.total_seconds().values[1:]
        dt = np.clip(dt, 0, 5)
        dist = float(np.sum(speeds[:-1] * dt))
        windows.append({"n_events": len(seg), "distance_m": dist, "top_speed": float(speeds.max())})
    return windows


def main() -> None:
    print(LINE)
    print("BaselineBuilder.compute_provisional() MIN_EVENTS_PER_WINDOW audit -- session 3387")
    print(LINE)

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(STATS_PATH)
    observations = list(
        adapter.stream_positions(POSITIONS_PATH, meta, session_id=SESSION_ID, match_id=SESSION_ID)
    )
    resampler = KinexonResampler()
    events_by_player, _ = resampler.resample(observations, session_id=SESSION_ID)
    print(f"Players: {len(events_by_player)}  (resampled at event_interval_s="
          f"{CONFIG.window.event_interval_s}s, window_steps={CONFIG.window.window_steps} "
          f"-> hard ceiling of {CONFIG.window.window_steps} events per {WINDOW_SECONDS}s window)")

    per_player_windows = {pid: _candidate_windows(df) for pid, df in events_by_player.items()}

    # ── Distribution of events per candidate window (overall) ───────────
    all_counts = [w["n_events"] for windows in per_player_windows.values() for w in windows]
    all_counts = np.array(all_counts)
    print(f"\n{LINE}\nDistribution of events per 120s candidate window (all players, n={len(all_counts)})\n{LINE}")
    print(f"min={all_counts.min()}  p10={np.percentile(all_counts,10):.0f}  "
          f"p25={np.percentile(all_counts,25):.0f}  median={np.median(all_counts):.0f}  "
          f"p75={np.percentile(all_counts,75):.0f}  p90={np.percentile(all_counts,90):.0f}  "
          f"max={all_counts.max()}")
    print("Histogram (events -> count of windows):")
    for v in sorted(set(all_counts.tolist())):
        n = int((all_counts == v).sum())
        print(f"  {v:>3} events: {'#' * min(n, 60)} ({n})")

    # ── Sweep thresholds ──────────────────────────────────────────────────
    print(f"\n{LINE}\nThreshold sweep (min_windows_for_provisional={MIN_WINDOWS_FOR_PROVISIONAL}, unchanged)\n{LINE}")
    print(f"{'threshold':>10} {'total_valid_windows':>20} {'players>=1_window':>18} "
          f"{'players_with_baseline':>22} {'median_windows/player':>23} {'mean_rel_SEM_distance':>22}")

    sweep_rows = []
    for T in THRESHOLDS:
        total_valid = 0
        players_with_any = 0
        players_with_baseline = 0
        windows_per_player = []
        rel_sems = []

        for pid, windows in per_player_windows.items():
            valid = [w for w in windows if w["n_events"] >= T]
            n_valid = len(valid)
            total_valid += n_valid
            windows_per_player.append(n_valid)
            if n_valid >= 1:
                players_with_any += 1
            if n_valid >= MIN_WINDOWS_FOR_PROVISIONAL:
                players_with_baseline += 1
                dists = np.array([w["distance_m"] for w in valid])
                if dists.mean() > 0 and len(dists) > 1:
                    rel_sem = (dists.std() / np.sqrt(len(dists))) / dists.mean()
                    rel_sems.append(rel_sem)

        median_wpp = float(np.median(windows_per_player))
        mean_rel_sem = float(np.mean(rel_sems)) if rel_sems else float("nan")
        print(f"{T:>10} {total_valid:>20} {players_with_any:>18} {players_with_baseline:>22} "
              f"{median_wpp:>23.1f} {mean_rel_sem:>22.3f}")
        sweep_rows.append((T, total_valid, players_with_any, players_with_baseline, median_wpp, mean_rel_sem))

    # ── Per-player windows-per-player distribution at a few key thresholds ──
    print(f"\n{LINE}\nPer-player valid-window counts at T=6, T=8, T=30 (for comparison)\n{LINE}")
    print(f"{'player_id':>10} {'T=6':>6} {'T=8':>6} {'T=30':>6}")
    for pid, windows in sorted(per_player_windows.items()):
        c6 = sum(1 for w in windows if w["n_events"] >= 6)
        c8 = sum(1 for w in windows if w["n_events"] >= 8)
        c30 = sum(1 for w in windows if w["n_events"] >= 30)
        print(f"{pid:>10} {c6:>6} {c8:>6} {c30:>6}")

    print(f"\n{LINE}\nVERDICT\n{LINE}")
    print(f"Hard ceiling: a window can have AT MOST {CONFIG.window.window_steps} events "
          f"at this cadence (one per {CONFIG.window.event_interval_s}s bucket) -- any threshold "
          f"above {CONFIG.window.window_steps} rejects 100% of windows for every player, always.")
    for T, total, any_p, base_p, med, sem in sweep_rows:
        print(f"  T={T:>2}: {base_p}/{len(events_by_player)} players reach a baseline "
              f"(>= {MIN_WINDOWS_FOR_PROVISIONAL} valid windows), median {med:.0f} windows/player, "
              f"mean relative SEM={sem:.3f}" if not np.isnan(sem) else
              f"  T={T:>2}: {base_p}/{len(events_by_player)} players reach a baseline, "
              f"median {med:.0f} windows/player, no players to measure SEM")
    print(LINE)


if __name__ == "__main__":
    main()
