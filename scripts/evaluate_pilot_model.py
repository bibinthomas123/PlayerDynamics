"""
evaluate_pilot_model.py

Real-data evaluation of the first trained pilot model
(models/shared_backbone.pt, model_version=shared_20260620211924) BEFORE
wiring AnomalyResult -> analytics.players.

Re-runs the exact same training procedure as scripts/train_pilot_session_3387.py
(same data, same 21 players, same 319 windows) so the live
PatternAnalysisEngine/InferenceEngine object -- with its real, in-memory
per-player RegimeAwareThresholdStore calibration state -- is available for
evaluation. (That calibration state is NOT persisted in shared_backbone.pt --
see save()/load() in analysis/anomaly_detection.py -- so it must be
reproduced by training, not loaded from disk. This is itself one of this
script's findings, not a workaround.)

For every one of the 319 real training windows, calls the REAL
PatternAnalysisEngine.analyze_window() (the exact method process_live_event()
calls in production) to get a real AnomalyResult, built from a live_event
dict whose fields are read directly from that window's own last timestep
(no fabricated values).

Run:
    python scripts/evaluate_pilot_model.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config.settings import CONFIG, SEQUENCE_FEATURE_NAMES as _SFN
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler
from analysis.gap_aware_windowing import build_training_sequences_gap_aware, detect_window_gaps

# Pilot mode: pools calibration losses across all eligible players into one
# shared RegimeAwareThresholdStore (analysis/anomaly_detection.py
# InferenceEngine.train()/get_tracker()). Inert in production (default
# False) -- explicitly opted into here because this evaluation's whole
# purpose is to measure pilot-mode's effect on anomaly detection.
CONFIG.scoring.pilot_mode = True

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "statistics.csv"
SESSION_ID = "3387"
LINE = "=" * 90
SUB = "-" * 90

WINDOW_STEPS = CONFIG.window.window_steps
STRIDE = WINDOW_STEPS
GAP_THRESHOLD_S = CONFIG.window.gap_threshold_s

_IDX = {n: i for i, n in enumerate(_SFN)}


def _build_pipeline_and_train():
    """Mirrors scripts/train_pilot_session_3387.py Phase 2 exactly -- same
    data, same flags -- to obtain a live, trained pipeline object."""
    from analysis.orchestrator import PlayersDataAnalysisPipeline

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(DATA_DIR / "statistics.csv")
    observations = list(
        adapter.stream_positions(DATA_DIR / "positions.csv", meta, session_id=SESSION_ID, match_id=SESSION_ID)
    )
    resampler = KinexonResampler()
    events_by_player, sessions_df = resampler.resample(observations, session_id=SESSION_ID)

    from analysis.baseline import BaselineBuilder
    builder = BaselineBuilder()
    baselines = {}
    for pid, df in events_by_player.items():
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        profile = builder.compute_with_fallback(
            player_id=pid, external_id=str(pid), sessions_df=player_sessions, events_df=df, window_days=28,
        )
        if profile is not None:
            baselines[pid] = profile

    gap_counts = {}
    for pid, df in events_by_player.items():
        gap_info = detect_window_gaps(df, WINDOW_STEPS, STRIDE, GAP_THRESHOLD_S)
        gap_counts[pid] = sum(1 for ok, _ in gap_info if ok)

    eligible = [pid for pid in events_by_player if pid in baselines and gap_counts.get(pid, 0) > 0]

    pipeline = PlayersDataAnalysisPipeline()
    for pid in eligible:
        m = meta.get(pid)
        pipeline.register_player(
            player_id=pid, external_id=str(pid),
            name=m.player_name if m else f"player_{pid}",
            position=m.position_label if m else "unknown",
            age=25,
        )
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        pipeline.load_historical_data(player_id=pid, sessions_df=player_sessions, events_df=events_by_player[pid])

    pipeline.compute_baselines(window_days=28, use_provisional_fallback=True)
    result = pipeline.train_all_models(use_gap_aware_windowing=True)

    return pipeline, events_by_player, sessions_df, meta, eligible, result


def _windows_with_elapsed(df: pd.DataFrame):
    """Real elapsed_s (window start) for every GAP-FREE window, in the exact
    same order build_training_sequences_gap_aware() returns its (seq, mask)
    pairs -- verified identical ordering/count in the prior turn's audit."""
    sorted_df = df.sort_values("ts").reset_index(drop=True)
    n = len(sorted_df)
    elapsed_starts = []
    for start in range(0, n - WINDOW_STEPS + 1, STRIDE):
        end = start + WINDOW_STEPS
        ts = pd.to_datetime(sorted_df["ts"].iloc[start:end], utc=True)
        max_gap = ts.diff().dt.total_seconds().dropna().max() if end - start > 1 else 0.0
        max_gap = 0.0 if pd.isna(max_gap) else float(max_gap)
        if max_gap <= GAP_THRESHOLD_S:
            elapsed_starts.append(float(sorted_df["elapsed_s"].iloc[start]))
    return elapsed_starts


def main() -> None:
    print(LINE)
    print("PILOT MODEL EVALUATION -- session 3387 (real data, real trained model)")
    print(LINE)

    pipeline, events_by_player, sessions_df, meta, eligible, train_result = _build_pipeline_and_train()
    print(f"\nRetrained for evaluation: {train_result['status']}, "
          f"n_players={train_result['shared_model']['n_players']}, "
          f"n_windows={train_result['shared_model']['n_windows']}, "
          f"version={train_result['shared_model']['model_version']}")

    engine = pipeline.pattern_engine

    # =====================================================================
    # SECTION A -- calibration-state audit (run BEFORE scoring any window)
    # =====================================================================
    print(f"\n{SUB}\n[A] Calibration state per player (RegimeAwareThresholdStore)\n{SUB}")
    print(f"AnomalyScoringConfig.min_calibration_windows = {CONFIG.scoring.min_calibration_windows}")
    print(f"AnomalyScoringConfig.pilot_mode = {CONFIG.scoring.pilot_mode}")

    pooled = engine.inference_engine._pooled_tracker
    if pooled is not None:
        n_pooled = len(pooled._global._losses)
        print(f"\nPooled (pilot-mode) tracker: n_samples={n_pooled}  "
              f"is_calibrated={pooled.is_calibrated}  "
              f"global_threshold={pooled._global.threshold:.4f}")
        print(f"Pooled regime coverage: {pooled.regime_coverage()}")
    else:
        print("\nPooled (pilot-mode) tracker: not built (pilot_mode was False at train() time)")

    print(f"\n{'player_id':>10} {'calib_slice_n':>14} {'per_player_calibrated':>22} "
          f"{'tracker_source':>15} {'effective_threshold':>20}")
    n_calibrated = 0
    n_pilot_fallback = 0
    for pid in eligible:
        per_player = engine.inference_engine._threshold_trackers.get(pid)
        n_calib = len(per_player._global._losses) if per_player else 0
        per_player_calibrated = per_player.is_calibrated if per_player else False
        n_calibrated += int(per_player_calibrated)
        source = engine.inference_engine.get_tracker_source(pid)
        n_pilot_fallback += int(source == "pilot_pooled")
        eff_tracker = engine.inference_engine.get_tracker(pid)
        eff_thr = eff_tracker._global.threshold if eff_tracker else float("inf")
        print(f"{pid:>10} {n_calib:>14} {str(per_player_calibrated):>22} {source:>15} {eff_thr:>20.4f}")

    print(f"\nPlayers with their OWN calibrated tracker: {n_calibrated} / {len(eligible)}")
    print(f"Players using pilot-mode pooled fallback:   {n_pilot_fallback} / {len(eligible)}")
    print(f"Players with NO usable threshold:           "
          f"{len(eligible) - n_calibrated - n_pilot_fallback} / {len(eligible)}")

    # =====================================================================
    # Run real analyze_window() across all 319 real windows
    # =====================================================================
    rows = []
    for pid in eligible:
        df = events_by_player[pid].sort_values("ts").reset_index(drop=True)
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        windows, _audit = build_training_sequences_gap_aware(
            pattern_engine=engine, events_df=df, sessions_df=player_sessions,
        )
        elapsed_starts = _windows_with_elapsed(df)
        if len(elapsed_starts) != len(windows):
            print(f"WARNING: player {pid} elapsed/window count mismatch "
                  f"({len(elapsed_starts)} vs {len(windows)}) -- skipping elapsed-time fields")
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
            tracker = engine.inference_engine.get_tracker(pid)
            tracker_source = engine.inference_engine.get_tracker_source(pid)
            from analysis.regime import SessionRegimeClassifier
            regime_key = SessionRegimeClassifier().classify(seq).key
            raw_breach = None
            eff_threshold = float("inf")
            if tracker and tracker.is_calibrated:
                eff_threshold = tracker.threshold_for(regime_key)
                raw_breach = bool(result.anomaly_score > eff_threshold)

            rows.append({
                "player_id": pid, "position": pos, "window_idx": w_idx,
                "elapsed_s": elapsed, "anomaly_score": result.anomaly_score,
                "is_anomaly": result.is_anomaly, "confidence": result.confidence,
                "raw_threshold_breach": raw_breach, "regime": regime_key,
                "tracker_source": tracker_source, "effective_threshold": eff_threshold,
                "alert_level": str(result.alert_level), "model_type": result.model_type,
                "seq": seq, "mask": mask, "live_event": live_event,
            })

    df_rows = pd.DataFrame([{k: v for k, v in r.items() if k not in ("seq", "mask", "live_event")} for r in rows])

    # =====================================================================
    # SECTION B -- distributions
    # =====================================================================
    print(f"\n{SUB}\n[B] Reconstruction loss / anomaly score / confidence distributions (n={len(df_rows)})\n{SUB}")
    for col in ("anomaly_score", "confidence"):
        s = df_rows[col]
        print(f"{col}: min={s.min():.4f} p25={s.quantile(.25):.4f} median={s.median():.4f} "
              f"p75={s.quantile(.75):.4f} p90={s.quantile(.90):.4f} max={s.max():.4f} mean={s.mean():.4f}")

    print(f"\nis_anomaly (final, alert-gated) True count: {df_rows['is_anomaly'].sum()} / {len(df_rows)} "
          f"({100*df_rows['is_anomaly'].mean():.1f}%)")
    raw = df_rows["raw_threshold_breach"]
    print(f"raw_threshold_breach (pre-hysteresis) True count: {raw.sum()} / {raw.notna().sum()} non-null "
          f"({100*raw.mean():.1f}% of non-null)" if raw.notna().any() else
          "raw_threshold_breach: all null (no player tracker ever became calibrated)")

    print(f"\nWindows scored via tracker_source:")
    print(df_rows["tracker_source"].value_counts().to_string())

    # =====================================================================
    # SECTION B continued -- anomalies per player / position / match time
    # =====================================================================
    print(f"\n{SUB}\nAnomalies per player\n{SUB}")
    per_player = df_rows.groupby("player_id").agg(
        n_windows=("anomaly_score", "size"),
        mean_loss=("anomaly_score", "mean"),
        max_loss=("anomaly_score", "max"),
        n_anomaly=("is_anomaly", "sum"),
        tracker_source=("tracker_source", "first"),
    )
    print(per_player.to_string())

    print(f"\n{SUB}\nAnomalies per position\n{SUB}")
    per_pos = df_rows.groupby("position").agg(
        n_windows=("anomaly_score", "size"),
        mean_loss=("anomaly_score", "mean"),
        n_anomaly=("is_anomaly", "sum"),
    )
    print(per_pos.to_string())

    print(f"\n{SUB}\nAnomalies over match time (15-min buckets)\n{SUB}")
    df_rows["time_bucket_min"] = (df_rows["elapsed_s"].fillna(0) // 900 * 15).astype(int)
    per_time = df_rows.groupby("time_bucket_min").agg(
        n_windows=("anomaly_score", "size"),
        mean_loss=("anomaly_score", "mean"),
        n_anomaly=("is_anomaly", "sum"),
    )
    print(per_time.to_string())

    # =====================================================================
    # SECTION C -- threshold audit summary
    # =====================================================================
    print(f"\n{SUB}\n[C] Threshold audit summary\n{SUB}")
    print(f"Players with a calibrated per-player tracker: {n_calibrated} / {len(eligible)}")
    print(f"-> tracker.is_calibrated requires len(calibration_losses) >= "
          f"min_calibration_windows ({CONFIG.scoring.min_calibration_windows})")
    print(f"-> Each player's calibration slice = last 20% of THEIR OWN windows "
          f"(InferenceEngine.train()), and each player gets their OWN "
          f"RegimeAwareThresholdStore (not shared across players).")
    print(f"-> Pilot mode (CONFIG.scoring.pilot_mode={CONFIG.scoring.pilot_mode}) pools all "
          f"players' calibration losses into one additional shared store, used only as a "
          f"fallback for players whose own tracker is still uncalibrated.")

    # =====================================================================
    # SECTION D -- XAI on real windows (lowest / median / highest loss)
    # =====================================================================
    print(f"\n{SUB}\n[D] XAI examples (real SHAP via _explain_sequence_shap)\n{SUB}")
    row_lookup = {(r["player_id"], r["window_idx"]): r for r in rows}

    # NORMAL: lowest-loss window with is_anomaly False (should be almost all of them).
    normal_pool = df_rows[~df_rows["is_anomaly"]].sort_values("anomaly_score")
    normal_rec = normal_pool.iloc[0]

    # ANOMALOUS: highest-confidence row actually flagged is_anomaly True.
    # Falls back to the highest raw_threshold_breach, then highest loss
    # overall, with an explicit label if no real positive exists.
    flagged = df_rows[df_rows["is_anomaly"]].sort_values("confidence", ascending=False)
    if len(flagged) > 0:
        anomalous_rec = flagged.iloc[0]
        anomalous_label = "ANOMALOUS (real is_anomaly=True)"
    else:
        breached = df_rows[df_rows["raw_threshold_breach"] == True].sort_values("anomaly_score", ascending=False)  # noqa: E712
        if len(breached) > 0:
            anomalous_rec = breached.iloc[0]
            anomalous_label = "ANOMALOUS (raw threshold breach, pre-hysteresis)"
        else:
            anomalous_rec = df_rows.sort_values("anomaly_score").iloc[-1]
            anomalous_label = "ANOMALOUS (none flagged -- highest-loss window shown instead)"

    # BORDERLINE: among windows with a real (non-inf) effective threshold,
    # the one whose loss sits closest to its own threshold.
    has_threshold = df_rows[df_rows["effective_threshold"] < float("inf")].copy()
    if len(has_threshold) > 0:
        has_threshold["dist_to_threshold"] = (has_threshold["anomaly_score"] - has_threshold["effective_threshold"]).abs()
        borderline_rec = has_threshold.sort_values("dist_to_threshold").iloc[0]
        borderline_label = "BORDERLINE (closest to its effective threshold)"
    else:
        borderline_rec = df_rows.sort_values("anomaly_score").iloc[len(df_rows) // 2]
        borderline_label = "BORDERLINE (no real threshold exists -- median-loss window shown instead)"

    examples_recs = {"NORMAL (lowest loss, not flagged)": normal_rec,
                      borderline_label: borderline_rec,
                      anomalous_label: anomalous_rec}

    for label, rec in examples_recs.items():
        pid, w_idx = int(rec["player_id"]), int(rec["window_idx"])
        full_row = row_lookup[(pid, w_idx)]
        seq, mask = full_row["seq"], full_row["mask"]
        p = pipeline.registry.get(pid)
        model = engine._shared_model
        bg = p.get("sequence_background")
        print(f"\n--- {label} --- player={pid} window_idx={w_idx} "
              f"loss={rec['anomaly_score']:.4f} is_anomaly={rec['is_anomaly']} "
              f"confidence={rec['confidence']:.4f} regime={rec['regime']} "
              f"effective_threshold={rec['effective_threshold']:.4f} tracker_source={rec['tracker_source']}")
        if bg is None or len(bg) < 2:
            print("  (insufficient background samples for SHAP on this player -- skipped)")
            continue
        shap_dict, base_value, feature_values = pipeline.xai_layer._explain_sequence_shap(
            player_id=pid, model=model, sequence=seq, mask=mask, background=bg, extra_features={},
        )
        top = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        print(f"  base_value (background loss): {base_value:.4f}")
        print("  top contributing features (SHAP-style ablation delta):")
        for name, val in top:
            print(f"    {name:30s} {val:+.4f}  (raw value: {feature_values.get(name, float('nan')):.3f})")

    # =====================================================================
    # SECTION E -- 3 concrete window examples (full detail)
    # =====================================================================
    print(f"\n{SUB}\n[E] Three concrete window examples\n{SUB}")
    for label, rec in examples_recs.items():
        pid, w_idx = int(rec["player_id"]), int(rec["window_idx"])
        full_row = row_lookup[(pid, w_idx)]
        seq = full_row["seq"]
        print(f"\n{label}: player={pid} ({meta[pid].player_name if meta.get(pid) else '?'}, "
              f"{meta[pid].position_label if meta.get(pid) else '?'}) window_idx={w_idx} "
              f"elapsed_s={rec['elapsed_s']}")
        print(f"  loss={rec['anomaly_score']:.4f}  is_anomaly={rec['is_anomaly']}  "
              f"confidence={rec['confidence']:.4f}  regime={rec['regime']}  alert_level={rec['alert_level']}  "
              f"effective_threshold={rec['effective_threshold']:.4f}  tracker_source={rec['tracker_source']}")
        print(f"  last-timestep features: speed_ms={seq[-1][_IDX['speed_ms']]:.2f} "
              f"hr_bpm={seq[-1][_IDX['heart_rate_bpm']]:.1f} sprint_flag={seq[-1][_IDX['sprint_flag']]:.0f} "
              f"x_pitch={seq[-1][_IDX['x_pitch']]:.1f} y_pitch={seq[-1][_IDX['y_pitch']]:.1f}")

    print(f"\n{LINE}\nEVALUATION COMPLETE\n{LINE}")

    out_path = _ROOT / "scripts" / "_pilot_eval_windows.csv"
    df_rows.drop(columns=["regime"]).to_csv(out_path, index=False)
    print(f"\nFull per-window table written to: {out_path}")


if __name__ == "__main__":
    main()
