"""
compare_persistence.py

Measures (does not assume) which AlertManager.min_persistence value produces
the most useful pilot-mode anomaly signal on real session 3387: 1, 2, or the
production default 3 (baseline, for contrast).

IMPORTANT methodological note: SharedBackboneAutoencoder.train() does not
seed torch's RNG, so retraining from scratch for each persistence value
would compare apples to oranges (different weight init -> different raw
losses -> different breach pattern, confounding the persistence effect with
training randomness -- confirmed by an earlier draft of this script, which
retrained per persistence value and got inconsistent, non-monotonic results
for exactly this reason). Trains ONCE, holds the model/calibration fixed,
and replays the IDENTICAL window sequence through a fresh AlertManager
instance per persistence value -- isolating persistence as the only
variable. Also resets the other per-run streaming state (EMA smoothers,
position buffers) between replays, since analyze_window() mutates those
statefully across calls.

Run:
    python scripts/compare_persistence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from config.settings import CONFIG
from analysis.gap_aware_windowing import build_training_sequences_gap_aware
from utils.alert_manager import AlertManager
from evaluate_pilot_model import _build_pipeline_and_train, _windows_with_elapsed, _IDX  # noqa: E402

LINE = "=" * 90
SUB = "-" * 90


def _score_all_windows(engine, events_by_player, sessions_df, meta, eligible):
    rows = []
    for pid in eligible:
        df = events_by_player[pid].sort_values("ts").reset_index(drop=True)
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        windows, _audit = build_training_sequences_gap_aware(
            pattern_engine=engine, events_df=df, sessions_df=player_sessions,
        )
        elapsed_starts = _windows_with_elapsed(df)
        if len(elapsed_starts) != len(windows):
            elapsed_starts = [None] * len(windows)
        pos = meta[pid].position_label if meta.get(pid) else "unknown"

        for w_idx, ((seq, mask, _sid), elapsed) in enumerate(zip(windows, elapsed_starts)):
            last = seq[-1]
            live_event = {
                "x_pitch": float(last[_IDX["x_pitch"]]),
                "y_pitch": float(last[_IDX["y_pitch"]]),
                "elapsed_seconds": float(elapsed) if elapsed is not None else 0.0,
                "_tvl_confidence": 1.0,
            }
            result = engine.analyze_window(
                player_id=pid, sequence=seq, mask=mask,
                live_event=live_event, sessions_df=player_sessions,
            )
            rows.append({
                "player_id": pid, "position": pos, "window_idx": w_idx, "elapsed_s": elapsed,
                "anomaly_score": result.anomaly_score, "is_anomaly": result.is_anomaly,
                "confidence": result.confidence, "alert_level": str(result.alert_level),
                "persistence_windows": result.persistence_windows,
            })
    return pd.DataFrame(rows)


def main() -> None:
    print(LINE)
    print("AlertManager min_persistence comparison -- real session 3387")
    print("(model trained ONCE; only AlertManager.min_persistence varies between runs)")
    print(LINE)

    pipeline, events_by_player, sessions_df, meta, eligible, train_result = _build_pipeline_and_train()
    engine = pipeline.pattern_engine
    print(f"\nTrained once: {train_result['shared_model']}")

    results = {}
    for persistence in (1, 2, 3):
        # Reset every piece of per-run streaming state analyze_window()
        # mutates, WITHOUT touching the trained model or calibration
        # trackers (inference_engine, _threshold_trackers, _pooled_tracker,
        # _baselines are untouched -- analyze_window() only reads them).
        engine.alert_manager = AlertManager(min_persistence=persistence)
        engine._ema_smoothers.clear()
        engine._ema_scores.clear()
        engine._position_buffers.clear()

        print(f"\nScoring all 319 windows with min_persistence={persistence} ...")
        df = _score_all_windows(engine, events_by_player, sessions_df, meta, eligible)
        results[persistence] = df

    print(f"\n{SUB}\nComparison summary\n{SUB}")
    print(f"{'min_persistence':>16} {'n_anomaly_windows':>18} {'anomaly_rate_pct':>17} "
          f"{'affected_players':>17} {'alert_levels_seen':>40}")
    for persistence, df in results.items():
        n_anom = int(df["is_anomaly"].sum())
        rate = 100 * df["is_anomaly"].mean()
        affected = df.loc[df["is_anomaly"], "player_id"].nunique()
        levels = df.loc[df["is_anomaly"], "alert_level"].value_counts().to_dict()
        print(f"{persistence:>16} {n_anom:>18} {rate:>16.1f}% {affected:>17} {str(levels):>40}")

    print(f"\n{SUB}\nPer-player breakdown at each setting\n{SUB}")
    for persistence, df in results.items():
        n_anom = int(df["is_anomaly"].sum())
        if n_anom == 0:
            print(f"\nmin_persistence={persistence}: 0 alerts fired (no player affected)")
            continue
        print(f"\nmin_persistence={persistence}: {n_anom} alerts")
        per_player = df[df["is_anomaly"]].groupby("player_id").agg(
            n_alerts=("is_anomaly", "sum"),
            max_loss=("anomaly_score", "max"),
        )
        print(per_player.to_string())

    print(f"\n{LINE}\nDONE\n{LINE}")

    out = pd.concat(
        [df.assign(min_persistence=p) for p, df in results.items()], ignore_index=True
    )
    out_path = _ROOT / "scripts" / "_persistence_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\nFull comparison table written to: {out_path}")


if __name__ == "__main__":
    main()
