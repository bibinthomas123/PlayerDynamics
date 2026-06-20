"""
train_pilot_session_3387.py

First end-to-end real-data training attempt for PlayerDynamics' per-player
LSTM pipeline, using ONLY real session 3387 Kinexon data, via the validated
path:

    positions.csv
      -> KinexonResampler            (ingestion/kinexon_resampler.py)
      -> Gap-Aware Windowing         (analysis/gap_aware_windowing.py)
      -> BaselineBuilder.compute_provisional() / compute_with_fallback()
                                     (analysis/baseline.py, fixed threshold)
      -> PatternAnalysisEngine.build_training_sequences() [gap-aware variant]
      -> PlayersDataAnalysisPipeline.train_all_models()
                                     (analysis/orchestrator.py)

Two phases:

  PHASE 1 (dry run, torch-free): resample, compute provisional baselines,
  count gap-filtered windows -- entirely using modules with zero torch
  dependency (kinexon_resampler.py, baseline.py, gap_aware_windowing.py's
  detect_window_gaps()). Reports checkpoints A-D. Does NOT import
  analysis.orchestrator / analysis.anomaly_detection.

  PHASE 2 (real training attempt): only runs if Phase 1's checkpoints pass.
  Imports the REAL analysis.orchestrator.PlayersDataAnalysisPipeline (no
  torch stub -- if torch is genuinely unavailable, this import is allowed
  to fail / the training call is allowed to fail, and the exact exception
  is reported as the blocker rather than worked around).

Run:
    python scripts/train_pilot_session_3387.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from config.settings import CONFIG
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler
from analysis.baseline import BaselineBuilder
from analysis.gap_aware_windowing import detect_window_gaps

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH = DATA_DIR / "statistics.csv"
SESSION_ID = "3387"
LINE = "=" * 90
SUBLINE = "-" * 90

WINDOW_STEPS = CONFIG.window.window_steps
STRIDE = WINDOW_STEPS
GAP_THRESHOLD_S = CONFIG.window.gap_threshold_s
MIN_WINDOWS_FOR_PROVISIONAL = CONFIG.baseline.min_windows_for_provisional


def main() -> None:
    print(LINE)
    print("PILOT TRAINING RUN -- session 3387 (real Kinexon data only)")
    print(LINE)

    # =====================================================================
    # PHASE 1 -- dry run (torch-free)
    # =====================================================================
    print(f"\n{SUBLINE}\nPHASE 1 -- DRY RUN VALIDATION\n{SUBLINE}")

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(STATS_PATH)
    observations = list(
        adapter.stream_positions(POSITIONS_PATH, meta, session_id=SESSION_ID, match_id=SESSION_ID)
    )
    n_raw = len(observations)

    resampler = KinexonResampler()
    events_by_player, sessions_df = resampler.resample(observations, session_id=SESSION_ID)
    n_resampled = sum(len(df) for df in events_by_player.values())

    # ── A. Resampled dataset size ───────────────────────────────────────
    print(f"\n[A] Resampled dataset size")
    print(f"    Raw positions.csv rows streamed : {n_raw}")
    print(f"    Players discovered              : {len(events_by_player)}")
    print(f"    Resampled rows (all players)     : {n_resampled}")
    print(f"    Reduction factor                 : {n_raw / n_resampled:.1f}x")

    # ── B. Gap-filtered dataset size ────────────────────────────────────
    print(f"\n[B] Gap-filtered dataset size (window_steps={WINDOW_STEPS}, "
          f"stride={STRIDE}, gap_threshold_s={GAP_THRESHOLD_S})")
    total_windows_before = 0
    total_windows_after = 0
    windows_by_player = {}
    for pid, df in sorted(events_by_player.items()):
        gap_info = detect_window_gaps(df, WINDOW_STEPS, STRIDE, GAP_THRESHOLD_S)
        n_before = len(gap_info)
        n_after = sum(1 for ok, _ in gap_info if ok)
        windows_by_player[pid] = (n_before, n_after)
        total_windows_before += n_before
        total_windows_after += n_after
    print(f"    Windows generated (pre-filter)    : {total_windows_before}")
    print(f"    Windows dropped by gap filter      : {total_windows_before - total_windows_after}")
    print(f"    Windows retained (post-filter)     : {total_windows_after}")

    # ── C. Baselines generated ──────────────────────────────────────────
    print(f"\n[C] Baselines generated (compute_with_fallback, "
          f"min_events_per_provisional_window={CONFIG.baseline.min_events_per_provisional_window})")
    builder = BaselineBuilder()
    baselines = {}
    for pid, df in sorted(events_by_player.items()):
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        profile = builder.compute_with_fallback(
            player_id=pid,
            external_id=str(pid),
            sessions_df=player_sessions,
            events_df=df,
            window_days=28,
        )
        if profile is not None:
            baselines[pid] = profile
    print(f"    Players with a baseline (any mode) : {len(baselines)} / {len(events_by_player)}")
    n_provisional = sum(1 for b in baselines.values() if b.baseline_mode == "provisional")
    n_historical = sum(1 for b in baselines.values() if b.baseline_mode == "historical")
    print(f"      -- provisional : {n_provisional}")
    print(f"      -- historical  : {n_historical}  (expected 0: only 1 real session exists)")

    # ── D. Players eligible for training ────────────────────────────────
    # Eligible = has a baseline AND has >=1 gap-filtered window (matches
    # train_all_models()'s own two skip conditions: no baseline, or zero
    # sequences built).
    eligible = [
        pid for pid in events_by_player
        if pid in baselines and windows_by_player.get(pid, (0, 0))[1] > 0
    ]
    final_training_examples = sum(windows_by_player[pid][1] for pid in eligible)
    print(f"\n[D] Players eligible for training")
    print(f"    Eligible players                   : {len(eligible)} / {len(events_by_player)}")
    print(f"    Final training examples (windows)  : {final_training_examples}")
    rejected = sorted(set(events_by_player) - set(eligible))
    print(f"    Rejected: {rejected}")
    for pid in rejected:
        reason = "no baseline" if pid not in baselines else "0 gap-filtered windows"
        print(f"      player {pid}: {reason}")

    validation_passed = len(eligible) > 0 and final_training_examples > 0
    print(f"\n{'VALIDATION PASSED' if validation_passed else 'VALIDATION FAILED'} "
          f"-- {len(eligible)} eligible players, {final_training_examples} training examples")

    if not validation_passed:
        print("\nSTOPPING: dry-run validation did not pass. Not attempting training.")
        return

    # =====================================================================
    # PHASE 2 -- real training attempt (no mocks, no torch stub)
    # =====================================================================
    print(f"\n{SUBLINE}\nPHASE 2 -- REAL TRAINING ATTEMPT (no stubs, no mocks)\n{SUBLINE}")
    try:
        from analysis.orchestrator import PlayersDataAnalysisPipeline
    except Exception:
        print("\nBLOCKED at import of analysis.orchestrator / analysis.anomaly_detection:\n")
        traceback.print_exc()
        print(f"\n{LINE}\nRESULT: training cannot proceed -- exact blocker printed above.\n{LINE}")
        return

    pipeline = PlayersDataAnalysisPipeline()

    for pid in eligible:
        meta_row = meta.get(pid)
        pipeline.register_player(
            player_id=pid,
            external_id=str(pid),
            name=meta_row.player_name if meta_row else f"player_{pid}",
            position=meta_row.position_label if meta_row else "unknown",
            age=25,
        )
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        pipeline.load_historical_data(
            player_id=pid,
            sessions_df=player_sessions,
            events_df=events_by_player[pid],
        )

    print(f"Registered + loaded {len(eligible)} eligible players into pipeline.")

    computed_baselines = pipeline.compute_baselines(window_days=28, use_provisional_fallback=True)
    print(f"pipeline.compute_baselines(use_provisional_fallback=True): "
          f"{len(computed_baselines)} / {len(eligible)} players")

    if len(computed_baselines) == 0:
        print("\nSTOPPING: pipeline.compute_baselines() built 0 baselines -- cannot train.")
        return

    try:
        result = pipeline.train_all_models(use_gap_aware_windowing=True)
    except Exception:
        print("\nBLOCKED at pipeline.train_all_models(use_gap_aware_windowing=True):\n")
        traceback.print_exc()
        print(f"\n{LINE}\nRESULT: training cannot proceed -- exact blocker printed above.\n{LINE}")
        return

    print(f"\ntrain_all_models() result: {result}")

    if result.get("status") != "success":
        print(f"\n{LINE}\nRESULT: training did not complete successfully -- status={result.get('status')!r}")
        print(f"{LINE}")
        return

    print(f"\n{LINE}\nRESULT: PILOT TRAINING COMPLETE -- PILOT, SINGLE SESSION ONLY\n{LINE}")


if __name__ == "__main__":
    main()
