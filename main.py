"""
Players Data — IBM CIC Germany
Production ML Pipeline Entry Point

Five independent commands, each with a defined contract:

    generate   — synthesise or validate data in data/
    train      — fit shared backbone + per-player thresholds, save to models/
    evaluate   — score model against ground truth labels, write metrics JSON
    serve      — stream live events from stdin (newline-delimited JSON) and
                 emit alerts to stdout; runs until EOF or SIGTERM
    audit      — run fairness + recalibration checks against the inference log

Usage
─────
    python main.py generate [--seasons N] [--matchdays N] [--anomaly-rate F]
    python main.py train    [--data-dir PATH] [--model-dir PATH] [--sessions-per-player N]
    python main.py evaluate [--data-dir PATH] [--model-dir PATH] [--out PATH]
    python main.py serve    [--model-dir PATH] [--min-alert-windows N]
    python main.py audit    [--log PATH] [--out PATH]

Exit codes
──────────
    0   success
    1   validation / data error
    2   model error (not trained, corrupt checkpoint)
    3   evaluation failed (e.g. no anomalies in label set)
    4   serve / stream error
    5   audit found bias (non-zero biased groups)

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# Logging: structured to stderr, JSON in production; human-readable in dev
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FMT_HUMAN = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
_LOG_FMT_JSON  = None   # set below if JSON_LOGS=1

def _configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    if os.getenv("JSON_LOGS"):
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return _json.dumps({
                    "ts":      datetime.utcnow().isoformat() + "Z",
                    "level":   record.levelname,
                    "logger":  record.name,
                    "message": record.getMessage(),
                    **({"exc": self.formatException(record.exc_info)}
                       if record.exc_info else {}),
                })

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FMT_HUMAN))

    logging.root.setLevel(numeric)
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.getLogger("shap").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = logging.getLogger("players_data.main")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _exit(code: int, message: str) -> None:
    """Log a final message and exit with the given code."""
    fn = logger.error if code != 0 else logger.info
    fn(message)
    sys.exit(code)


def _load_csvs(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all five standard CSVs; exit(1) if any are missing."""
    required = ["players", "sessions", "events", "annotations", "ground_truth_labels"]
    frames: Dict[str, pd.DataFrame] = {}
    for name in required:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            _exit(1, f"Required file missing: {path}")
        frames[name] = pd.read_csv(path, low_memory=False)
        logger.info("Loaded %-25s  %d rows", f"{name}.csv", len(frames[name]))

    # Type coercions
    for col in ("ts", "started_at", "ended_at", "annotated_at"):
        for df in frames.values():
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    return frames


def _build_pipeline(model_dir: Path):
    """Construct a PlayersDataAnalysisPipeline pointed at model_dir."""
    from analysis.orchestrator import PlayersDataAnalysisPipeline
    from config.settings import CONFIG

    CONFIG.active_model = "lstm"
    pipeline = PlayersDataAnalysisPipeline()

    # Override model store so the pipeline loads/saves from the right place
    import analysis.anomaly_detection as _ad
    _ad.MODEL_STORE = model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    return pipeline


def _aggregate_session_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute window-level aggregates per session.
    Used when merging feature columns onto sessions_df for baseline computation.
    """
    from scipy.integrate import trapezoid

    if events_df.empty:
        return pd.DataFrame(columns=["session_id", "window_distance_m",
                                     "window_avg_speed_ms", "window_sprint_count",
                                     "heart_rate_bpm"])
    return (
        events_df
        .groupby("session_id")
        .agg(
            window_distance_m   =("speed_ms",       lambda x: trapezoid(x, dx=15)),
            window_avg_speed_ms =("speed_ms",        "mean"),
            window_sprint_count =("is_sprint",       "sum"),
            heart_rate_bpm      =("heart_rate_bpm",  "mean"),
        )
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: generate
# ─────────────────────────────────────────────────────────────────────────────
def cmd_generate(args: argparse.Namespace) -> None:
    """
    Generate (or re-generate) synthetic training data.

    Writes five CSVs to --data-dir.  Runs the v4 data generator with the
    requested parameters, then validates the output before exiting.

    Exits 1 if validation fails (e.g. zero anomalies seeded, missing columns).
    """
    logger.info(
        "generate | seasons=%d  matchdays=%d  anomaly_rate=%.3f  out=%s",
        args.seasons, args.matchdays, args.anomaly_rate, args.data_dir,
    )

    # Import here to keep other subcommands fast when data already exists
    try:
        import data_generator as dg
    except ImportError:
        _exit(1, "data_generator module not found — is it on PYTHONPATH?")

    t0 = time.perf_counter()
    data = dg.generate_dataset(
        n_seasons=args.seasons,
        n_matchdays_per_season=args.matchdays,
        anomaly_rate=args.anomaly_rate,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Generation complete in %.1f s", elapsed)

    # Validate before saving
    gt = pd.DataFrame(data.get("ground_truth_labels", []))
    if gt.empty or gt["is_anomaly"].sum() == 0:
        _exit(1, "Validation failed: no anomalies seeded — check anomaly_rate")

    sessions = pd.DataFrame(data["sessions"])
    events   = pd.DataFrame(data["events"])

    if events.empty:
        _exit(1, "Validation failed: events table is empty")

    # Ensure required feature columns are present
    required_cols = {"speed_ms", "heart_rate_bpm", "x_pitch", "y_pitch",
                     "is_sprint", "elapsed_seconds"}
    missing = required_cols - set(events.columns)
    if missing:
        _exit(1, f"Validation failed: events missing columns {missing}")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        dg.save_dataset(data, apply_corruption=not args.no_corruption)
    except Exception as exc:
        _exit(1, f"Save failed: {exc}")

    n_anomalies = int(gt["is_anomaly"].sum())
    anomaly_pct = float(gt["is_anomaly"].mean()) * 100
    logger.info(
        "Saved to %s | sessions=%d  events=%d  anomalies=%d (%.1f%%)",
        data_dir, len(sessions), len(events), n_anomalies, anomaly_pct,
    )

    if not args.quiet:
        dg.print_dataset_report(data)

    print(json.dumps({
        "status":         "ok",
        "data_dir":       str(data_dir.resolve()),
        "sessions":       len(sessions),
        "events":         len(events),
        "anomaly_count":  n_anomalies,
        "anomaly_pct":    round(anomaly_pct, 2),
        "elapsed_s":      round(elapsed, 2),
    }))


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: train
# ─────────────────────────────────────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> None:
    """
    Fit the shared LSTM backbone + per-player calibration thresholds.

    Data split is performed here, not inside the model:
      • sessions per player: most-recent N sessions → training
      • remaining sessions  → calibration (threshold fitting)

    All splits are deterministic: given the same data the output is identical.
    Saves shared_backbone.pt and writes train_summary.json to --model-dir.

    Exits 2 if training fails or produces a degenerate model.
    """
    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    logger.info("train | data=%s  model=%s  sessions_per_player=%d",
                data_dir, model_dir, args.sessions_per_player)

    frames   = _load_csvs(data_dir)
    pipeline = _build_pipeline(model_dir)

    players_df  = frames["players"]
    sessions_df = frames["sessions"]
    events_df   = frames["events"]
    annot_df    = frames["annotations"]

    # ── Register all players ──────────────────────────────────────────────────
    for _, row in players_df.iterrows():
        pipeline.register_player(
            player_id   = int(row["player_id"]),
            external_id = str(row["external_id"]),
            name        = str(row.get("full_name", row.get("name", f"Player {row['player_id']}"))),
            position    = str(row["position"]),
            age         = int(row.get("age", 25)),
            age_group   = str(row.get("age_group", "Senior")),
            nationality = str(row.get("nationality", "")),
        )
    logger.info("Registered %d players", len(players_df))

    # ── Load data per player ──────────────────────────────────────────────────
    features_df = _aggregate_session_features(events_df)
    sessions_with_features = sessions_df.merge(features_df, on="session_id", how="left")
    for col, fill in [("window_distance_m", 0), ("window_avg_speed_ms", 0),
                      ("window_sprint_count", 0), ("heart_rate_bpm", 120)]:
        sessions_with_features[col] = sessions_with_features[col].fillna(fill)

    n_loaded = 0
    for pid in players_df["player_id"].tolist():
        psessions = (
            sessions_with_features[sessions_with_features["player_id"] == pid]
            .sort_values("started_at")
            .tail(args.sessions_per_player)
        )
        if psessions.empty:
            logger.warning("Player %d: no sessions — skipping", pid)
            continue
        pevents = events_df[events_df["session_id"].isin(psessions["session_id"])]
        pannot  = annot_df[annot_df["session_id"].isin(psessions["session_id"])]
        pipeline.load_historical_data(
            player_id      = pid,
            sessions_df    = psessions,
            events_df      = pevents,
            annotations_df = pannot,
        )
        n_loaded += 1

    logger.info("Loaded data for %d / %d players", n_loaded, len(players_df))
    if n_loaded == 0:
        _exit(1, "No players with usable data — cannot train")

    # ── Compute baselines ─────────────────────────────────────────────────────
    logger.info("Computing personal baselines...")
    baselines = pipeline.compute_baselines(window_days=28)
    logger.info("Baselines computed for %d players", len(baselines))

    if len(baselines) == 0:
        _exit(1, "All baselines failed — check data quality / min_sessions_for_baseline")

    # Log a concise baseline table
    for pid, b in sorted(baselines.items()):
        logger.debug(
            "  p%03d  dist=%.0f±%.0f m  sprints=%.1f  fatigue_r2=%.3f",
            pid, b.distance_mean, b.distance_std,
            b.sprint_count_mean, b.fatigue_r_squared or 0,
        )

    # ── Train shared backbone ─────────────────────────────────────────────────
    logger.info("Training shared LSTM backbone...")
    t0 = time.perf_counter()
    result = pipeline.train_all_models()
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1f s | status=%s", elapsed, result.get("status"))

    # Orchestrator returns {"status": "success", "shared_model": {...}, "players": {...}}.
    # Accept any non-error status string so this doesn't break if the orchestrator
    # changes "success" → "ok" or "trained" in future.
    top_status = result.get("status", "")
    if not top_status or top_status.startswith("error") or top_status.startswith("fail"):
        _exit(2, f"Training returned error status: {result}")

    # n_windows lives under result["shared_model"]["n_windows"], not at the top level.
    shared_info = result.get("shared_model", {})
    n_windows   = shared_info.get("n_windows", 0)
    n_players_trained = shared_info.get("n_players", n_loaded)
    model_version     = shared_info.get("model_version", "unknown")

    if n_windows == 0:
        _exit(2, "Training produced 0 sequence windows — model is empty")

    # ── Verify checkpoint was written ─────────────────────────────────────────
    backbone_path = model_dir / "shared_backbone.pt"
    if not backbone_path.exists():
        _exit(2, f"Expected checkpoint not found at {backbone_path}")

    # ── Write training summary ────────────────────────────────────────────────
    summary = {
        "status":          "ok",
        "trained_at":      datetime.now(tz=timezone.utc).isoformat(),
        "model_dir":       str(model_dir.resolve()),
        "n_players":       n_players_trained,
        "n_windows":       n_windows,
        "n_baselines":     len(baselines),
        "elapsed_s":       round(elapsed, 2),
        "backbone_path":   str(backbone_path),
        "model_version":   model_version,
    }
    summary_path = model_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written → %s", summary_path)

    print(json.dumps(summary))

    # Always save serve state so `serve` can load baselines + thresholds without retraining.
    # --save-serve-state flag is kept for backwards-compat but is no longer required.
    serve_state_path = model_dir / "serve_state.json"
    _save_serve_state(pipeline, serve_state_path)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: evaluate
# ─────────────────────────────────────────────────────────────────────────────
def cmd_evaluate(args: argparse.Namespace) -> None:
    """
    Score the trained model against ground_truth_labels.csv.

    For each player:
      1. Load the held-out windows (sessions NOT used during training).
      2. Run the model.
      3. Compute ROC-AUC, PR-AUC, precision@k, FP-per-90min.

    Aggregates across all players and writes metrics JSON to --out.
    Exits 3 if no labeled windows can be evaluated.
    Exits 2 if the model checkpoint is missing.

    The evaluation intentionally does NOT re-generate or re-inject anomalies.
    It uses the ground_truth_labels seeded by the data generator.
    """
    # ── Progress-bar helper (graceful tqdm fallback) ──────────────────────────
    try:
        from tqdm import tqdm as _tqdm
        def _progress(it, **kw):
            return _tqdm(it, dynamic_ncols=True, **kw)
    except ImportError:
        def _progress(it, desc="", **kw):  # type: ignore[misc]
            if desc:
                logger.info("%s …", desc)
            return it

    data_dir  = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    logger.info("evaluate | data=%s  model=%s", data_dir, model_dir)

    backbone_path = model_dir / "shared_backbone.pt"
    if not backbone_path.exists():
        _exit(2, f"Model checkpoint not found at {backbone_path} — run train first")

    frames   = _load_csvs(data_dir)
    pipeline = _build_pipeline(model_dir)

    players_df  = frames["players"]
    sessions_df = frames["sessions"]
    events_df   = frames["events"]
    annot_df    = frames["annotations"]
    gt_df       = frames["ground_truth_labels"]

    # ── Pre-group for O(1) lookup — avoids O(n) DataFrame filter per player/session ──
    logger.info("Pre-grouping events and sessions …")
    session_events_map: Dict[int, pd.DataFrame] = {
        int(sid): grp for sid, grp in events_df.groupby("session_id")
    }
    player_sessions_map: Dict[int, pd.DataFrame] = {
        int(pid): grp for pid, grp in sessions_df.groupby("player_id")
    }
    player_annot_map: Dict[int, pd.DataFrame] = {
        int(pid): grp
        for pid, grp in annot_df.groupby(
            annot_df["session_id"].map(
                sessions_df.set_index("session_id")["player_id"]
            )
        )
    } if not annot_df.empty and "session_id" in annot_df.columns else {}

    # ── Load backbone ─────────────────────────────────────────────────────────
    from analysis.anomaly_detection import SharedBackboneAutoencoder
    shared = SharedBackboneAutoencoder.load(backbone_path)
    if shared is None or not shared.is_trained:
        _exit(2, "Could not load trained backbone from checkpoint")

    pipeline.pattern_engine._shared_model = shared
    pipeline.pattern_engine.inference_engine._shared_model = shared
    logger.info("Loaded backbone v=%s  players=%d",
                shared.model_version, shared.n_players)

    # ── Register players + baselines ─────────────────────────────────────────
    features_df = _aggregate_session_features(events_df)
    sessions_with_features = sessions_df.merge(features_df, on="session_id", how="left")
    for col, fill in [("window_distance_m", 0), ("window_avg_speed_ms", 0),
                      ("window_sprint_count", 0), ("heart_rate_bpm", 120)]:
        sessions_with_features[col] = sessions_with_features[col].fillna(fill)

    swf_by_player: Dict[int, pd.DataFrame] = {
        int(pid): grp for pid, grp in sessions_with_features.groupby("player_id")
    }

    for _, row in players_df.iterrows():
        pipeline.register_player(
            player_id   = int(row["player_id"]),
            external_id = str(row["external_id"]),
            name        = str(row.get("full_name", row.get("name", ""))),
            position    = str(row["position"]),
            age         = int(row.get("age", 25)),
            age_group   = str(row.get("age_group", "Senior")),
        )

    baselines = {}
    for pid in _progress(players_df["player_id"].tolist(),
                         desc="Loading player histories", unit="player"):
        psessions = swf_by_player.get(pid, pd.DataFrame())
        if psessions.empty:
            continue
        sids      = set(psessions["session_id"])
        pevents   = pd.concat(
            [session_events_map[sid] for sid in sids if sid in session_events_map],
            ignore_index=True,
        ) if sids else pd.DataFrame()
        pannot = player_annot_map.get(pid, pd.DataFrame())
        pipeline.load_historical_data(pid, psessions, pevents, pannot)

    baselines = pipeline.compute_baselines(window_days=28)
    logger.info("Baselines ready for %d players", len(baselines))

    # ── Rebuild threshold trackers — batched predict for speed ────────────────
    from analysis.regime import RegimeAwareThresholdStore, SessionRegimeClassifier
    from analysis.anomaly_detection import EMASmoother  # re-export via anomaly_detection
    from config.settings import CONFIG
    classifier = SessionRegimeClassifier()
    alpha      = CONFIG.scoring.score_ema_alpha

    for pid in _progress(sorted(baselines.keys()),
                         desc="Calibrating thresholds", unit="player"):
        psessions = player_sessions_map.get(pid, pd.DataFrame())
        if psessions.empty:
            continue
        psessions = psessions.sort_values("started_at")
        sids    = set(psessions["session_id"])
        pevents = pd.concat(
            [session_events_map[sid] for sid in sids if sid in session_events_map],
            ignore_index=True,
        ) if sids else pd.DataFrame()

        windows = pipeline.pattern_engine.build_training_sequences(pevents, psessions)
        if not windows:
            continue
        split = max(1, int(len(windows) * 0.8))
        calib = windows[split:]  # use held-out windows for calibration
        if len(calib) < 5:        
            calib = windows
        if not calib:
            continue

        # ── Batch predict — ONE forward pass for all calib windows ────────────
        seqs_arr = np.stack(
            [s for s, _, _ in calib]
        ).astype(np.float32)

        masks_arr = np.stack(
            [m for _, m, _ in calib]
        ).astype(bool)

        pids_arr = np.full(
            len(calib),
            pid,
            dtype=np.int64,
        )

        # pre-normalise ONCE outside model
        seqs_arr = shared.normaliser.transform(seqs_arr)

        losses = shared.predict_batch(
            player_ids=pids_arr,
            sequences=seqs_arr,
            masks=masks_arr,
            normalised=True,
        )

        print("POST NORM MEAN", seqs_arr.mean())
        print("POST NORM STD ", seqs_arr.std())
        print("MODEL OUTPUT  ", recon.mean(), recon.std())
        print("LOSS          ", loss.item())

        store   = RegimeAwareThresholdStore()
        smoother = EMASmoother(alpha)
        for loss, (seq, _, _) in zip(losses, calib):
            ema_val = smoother.update(float(loss))
            store.update(ema_val, classifier.classify(seq).key)
        pipeline.pattern_engine.inference_engine._threshold_trackers[pid] = store

        pipeline.pattern_engine._threshold_trackers[pid] = store

    # ── Build labeled window set ──────────────────────────────────────────────
    gt_map = dict(zip(gt_df["session_id"].astype(int),
                      gt_df["is_anomaly"].astype(bool)))

    all_player_metrics: List[dict] = []
    total_tp = total_fp = total_fn = total_tn = 0

    for pid in _progress(sorted(baselines.keys()),
                         desc="Evaluating players", unit="player"):
        psessions = player_sessions_map.get(pid, pd.DataFrame())
        if psessions.empty:
            continue
        psessions = psessions.sort_values("started_at")

        # Build labeled windows using pre-grouped session events
        labeled: List = []
        for sid, label in (
            (int(r.session_id), gt_map.get(int(r.session_id), False))
            for r in psessions.itertuples(index=False)
        ):
            sess_evs = session_events_map.get(sid, pd.DataFrame())
            if sess_evs.empty:
                continue
            for seq, mask in pipeline.pattern_engine.window_builder.build_from_session(sess_evs):
                labeled.append((seq, mask, label))

        if not labeled:
            logger.warning("Player %d: 0 labeled windows — skipping", pid)
            continue

        n_anom   = sum(1 for _, _, l in labeled if l)
        n_normal = sum(1 for _, _, l in labeled if not l)

        if n_anom == 0 or n_normal == 0:
            logger.info("Player %d: only one class present (anom=%d norm=%d) — skipping",
                        pid, n_anom, n_normal)
            continue

        metrics = pipeline.pattern_engine.evaluate_player(pid, labeled)

        if "error" in metrics:
            logger.warning("Player %d evaluation error: %s", pid, metrics["error"])
            continue

        metrics["player_id"] = pid
        all_player_metrics.append(metrics)
        total_tp += metrics.get("tp", 0)
        total_fp += metrics.get("fp", 0)
        total_fn += metrics.get("fn", 0)
        total_tn += metrics.get("tn", 0)

        logger.info(
            "p%03d | auc=%.3f  pr=%.3f  p@k=%.3f  fp/90=%.2f  tp=%d fp=%d fn=%d",
            pid,
            metrics.get("roc_auc", float("nan")),
            metrics.get("pr_auc",  float("nan")),
            metrics.get("precision_at_k", float("nan")),
            metrics.get("fp_per_90_min",  float("nan")),
            metrics.get("tp", 0), metrics.get("fp", 0), metrics.get("fn", 0),
        )

    if not all_player_metrics:
        _exit(3, "No players produced evaluable labeled windows "
                 "— check ground_truth_labels.csv and data split")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _nanmean(key: str) -> float:
        vals = [m[key] for m in all_player_metrics if key in m and m[key] is not None]
        return float(np.nanmean(vals)) if vals else float("nan")

    prec_global = total_tp / max(total_tp + total_fp, 1)
    rec_global  = total_tp / max(total_tp + total_fn, 1)

    aggregate = {
        "status":             "ok",
        "evaluated_at":       datetime.now(tz=timezone.utc).isoformat(),
        "n_players_evaluated":len(all_player_metrics),
        "mean_roc_auc":       round(_nanmean("roc_auc"), 4),
        "mean_pr_auc":        round(_nanmean("pr_auc"),  4),
        "mean_precision_at_k":round(_nanmean("precision_at_k"), 4),
        "mean_fp_per_90_min": round(_nanmean("fp_per_90_min"),  4),
        "global_precision":   round(prec_global, 4),
        "global_recall":      round(rec_global,  4),
        "global_tp":          total_tp,
        "global_fp":          total_fp,
        "global_fn":          total_fn,
        "global_tn":          total_tn,
        "per_player":         all_player_metrics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))
    logger.info("Metrics written → %s", out_path)

    print(json.dumps({k: v for k, v in aggregate.items() if k != "per_player"},
                     indent=2))

    # Fail-fast gate: exit non-zero if mean AUC is below minimum
    if aggregate["mean_roc_auc"] < args.min_auc:
        _exit(3, f"ROC-AUC {aggregate['mean_roc_auc']:.3f} < required {args.min_auc:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: serve
# ─────────────────────────────────────────────────────────────────────────────
def cmd_serve(args: argparse.Namespace) -> None:
    """
    Production inference loop.
    """

    model_dir = Path(args.model_dir)
    backbone_path = model_dir / "shared_backbone.pt"

    if not backbone_path.exists():
        _exit(
            2, f"Model checkpoint not found: {backbone_path} — run train first"
        )

    logger.info(
        "serve | model=%s  min_alert_windows=%d  max_latency_ms=%d",
        model_dir,
        args.min_alert_windows,
        args.max_latency_ms,
    )

    # ─────────────────────────────────────────────
    # Build pipeline
    # ─────────────────────────────────────────────
    pipeline = _build_pipeline(model_dir)

    from analysis.anomaly_detection import SharedBackboneAutoencoder

    shared = SharedBackboneAutoencoder.load(backbone_path)

    if shared is None or not shared.is_trained:
        _exit(2, "Backbone checkpoint is present but model is not trained")

    pipeline.pattern_engine.inference_engine._shared_model = shared

    logger.info(
        "Backbone loaded (v=%s  players=%d)",
        shared.model_version,
        shared.n_players,
    )

    # ─────────────────────────────────────────────
    # Restore serve state
    # ─────────────────────────────────────────────
    serve_state_path = model_dir / "serve_state.json"

    if serve_state_path.exists():

        _restore_serve_state(
            pipeline,
            serve_state_path,
        )

        print(
            "\nRESTORED PLAYERS:",
            len(pipeline.registry._players),
        )

        # IMPORTANT:
        # inject model AFTER restore
        for player in pipeline.registry._players.values():
            player["model"] = shared

        logger.info(
            "Serve state restored from %s",
            serve_state_path,
        )

    else:

        logger.warning(
            "No serve_state.json found — "
            "player baselines and thresholds not loaded. "
            "Run train first or provide serve_state.json."
        )

    # ─────────────────────────────────────────────
    # Alert gate
    # ─────────────────────────────────────────────
    gate_counts: Dict[str, int] = defaultdict(int)
    gate_last: Dict[str, str] = {}

    def _gate_fire(ext_id: str, rec_type: str) -> bool:

        if gate_last.get(ext_id) != rec_type:

            gate_counts[ext_id] = 1
            gate_last[ext_id] = rec_type

        else:
            gate_counts[ext_id] += 1

        return gate_counts[ext_id] >= args.min_alert_windows

    def _gate_reset(ext_id: str) -> None:

        gate_counts[ext_id] = 0
        gate_last.pop(ext_id, None)

    # ─────────────────────────────────────────────
    # Enrichment
    # ─────────────────────────────────────────────
    def _enrich(event: dict) -> dict:

        ext_id = event.get("player_external_id", "")

        player = pipeline.registry.get_by_external_id(ext_id)

        baseline = player["baseline"] if player else None

        if baseline is None:
            return event

        speed = float(event.get("speed_ms") or 0.0)

        elapsed_min = float(event.get("elapsed_seconds", 0)) / 60.0

        alpha = baseline.fatigue_alpha or 0.005

        beta = baseline.fatigue_beta or speed * 1.3

        expected_spd = beta * np.exp(-alpha * elapsed_min)

        start_speed = float(
            event.get("session_start_speed_ms")
            or (baseline.distance_mean / (90 * 60) * 1.3)
        )

        event["fatigue_decay_residual"] = round(
            (speed - expected_spd) * 15.0,
            2,
        )

        event["speed_drop_pct"] = round(
            (1.0 - speed / max(start_speed, 0.1)) * 100.0,
            1,
        )

        return event

    # ─────────────────────────────────────────────
    # Inference log
    # ─────────────────────────────────────────────
    inference_log_path = Path("logs") / "inference_log.jsonl"

    inference_log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    _inference_log_fh = open(
        inference_log_path,
        "a",
        buffering=1,
    )

    logger.info(
        "Inference log → %s",
        inference_log_path,
    )

    # ─────────────────────────────────────────────
    # Stream loop
    # ─────────────────────────────────────────────
    n_events = 0
    n_alerts = 0
    sla_violations = 0

    logger.info("Serving — reading newline-delimited JSON from stdin")

    try:

        for raw_line in sys.stdin:

            raw_line = raw_line.strip()

            if not raw_line:
                continue

            t_start = time.perf_counter()

            n_events += 1

            try:
                event = json.loads(raw_line)

            except json.JSONDecodeError as exc:

                logger.warning(
                    "Line %d: invalid JSON — %s",
                    n_events,
                    exc,
                )

                continue

            ext_id = event.get(
                "player_external_id",
                "",
            )

            if not ext_id:

                logger.debug(
                    "Line %d: missing player_external_id — skipped",
                    n_events,
                )

                continue

            event = _enrich(event)

            try:

                result = pipeline.process_live_event(
                    event,
                    segment_index=0,
                )

            except Exception as exc:

                logger.warning(
                    "Inference error for %s: %s",
                    ext_id,
                    exc,
                )

                continue

            latency_ms = (time.perf_counter() - t_start) * 1000

            if latency_ms > args.max_latency_ms:

                sla_violations += 1

                logger.warning(
                    "SLA breach: player=%s latency=%.1f ms > %d ms (total=%d)",
                    ext_id,
                    latency_ms,
                    args.max_latency_ms,
                    sla_violations,
                )

            # ─────────────────────────────
            # Persist inference
            # ─────────────────────────────
            if result is not None:

                _inference_log_fh.write(
                    json.dumps(
                        {
                            "inference_id": n_events,
                            "player_id": result.player_id,
                            "external_id": ext_id,
                            "session_id": event.get(
                                "session_id",
                                0,
                            ),
                            "recommendation_type": result.recommendation_type,
                            "is_anomaly": result.recommendation_type
                            is not None,
                            "anomaly_score": round(
                                float(result.anomaly_score or 0.0),
                                6,
                            ),
                            "confidence": round(
                                float(result.confidence or 0.0),
                                4,
                            ),
                            "fatigue_flag": bool(result.fatigue_flag),
                            "drift_flag": bool(result.positional_drift_flag),
                            "workload_flag": bool(result.workload_flag),
                            "nlg_summary": getattr(
                                result,
                                "nlg_summary",
                                "",
                            ),
                            "ts": datetime.now(tz=timezone.utc).isoformat(),
                        }
                    )
                    + "\n"
                )

            if result is None or result.recommendation_type is None:

                _gate_reset(ext_id)

                continue

            # ─────────────────────────────
            # Consecutive alert gate
            # ─────────────────────────────
            if not _gate_fire(
                ext_id,
                result.recommendation_type,
            ):
                continue

            n_alerts += 1

            alert_payload = {
                "player_id": result.player_id,
                "external_id": result.external_id,
                "recommendation_type": result.recommendation_type,
                "confidence": round(result.confidence, 4),
                "anomaly_score": round(result.anomaly_score, 6),
                "fatigue_flag": result.fatigue_flag,
                "drift_flag": result.positional_drift_flag,
                "workload_flag": result.workload_flag,
                "workload_status": result.workload_status,
                "nlg_summary": getattr(
                    result,
                    "nlg_summary",
                    "",
                ),
                "counterfactual": getattr(
                    result,
                    "counterfactual",
                    "",
                ),
                "shap_values": getattr(
                    result,
                    "shap_values",
                    {},
                ),
                "top_features": [
                    {
                        "feature": c.feature_name,
                        "shap": round(
                            c.shap_value,
                            6,
                        ),
                        "value": round(
                            c.feature_value,
                            4,
                        ),
                        "label": c.human_label,
                    }
                    for c in (
                        result.top_contributions[:5]
                        if hasattr(
                            result,
                            "top_contributions",
                        )
                        else []
                    )
                ],
                "latency_ms": round(
                    latency_ms,
                    2,
                ),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "gate_windows": gate_counts[ext_id],
            }

            print(
                json.dumps(alert_payload),
                flush=True,
            )

            logger.info(
                "ALERT player=%-6s type=%-20s conf=%.2f latency=%.1f ms",
                ext_id,
                result.recommendation_type,
                result.confidence,
                latency_ms,
            )

    except KeyboardInterrupt:

        logger.info("SIGINT received — shutting down gracefully")

    except BrokenPipeError:

        logger.info("Pipe closed — exiting")

    except Exception as exc:

        logger.exception(
            "Unhandled stream error: %s",
            exc,
        )

        _inference_log_fh.close()

        _exit(4, f"Serve exited with unhandled error: {exc}")

    finally:

        _inference_log_fh.close()

        logger.info(
            "Inference log closed → %s",
            inference_log_path,
        )

    logger.info(
        "Serve complete | events=%d alerts=%d sla_violations=%d",
        n_events,
        n_alerts,
        sla_violations,
    )


def _restore_serve_state(pipeline, path: Path) -> None:
    """
    Restore per-player baselines and thresholds from the serialised serve-state
    written at the end of a train run.

    The serve-state is written by cmd_train when --save-serve-state is passed.
    Without it, the serve command has no baselines and cannot compute drift /
    workload / fatigue flags — it can still run the LSTM but SHAP features
    will default to zero.
    """
    try:
        state = json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Could not load serve_state.json: %s", exc)
        return

    from analysis.baseline import PlayerBaselineProfile
    from analysis.regime import RegimeAwareThresholdStore
    from analysis.anomaly_detection import DynamicThresholdTracker
    from datetime import timezone as _tz

    for pid_str, ps in state.get("players", {}).items():
        pid = int(pid_str)
        # Re-register player
        pipeline.register_player(
            player_id   = pid,
            external_id = ps["external_id"],
            name        = ps.get("name", ""),
            position    = ps.get("position", "CM"),
            age         = ps.get("age", 25),
            age_group   = ps.get("age_group", "Senior"),
        )
        # Restore baseline
        b = ps.get("baseline")
        if b:
            baseline = PlayerBaselineProfile(
                player_id           = pid,
                external_id         = ps["external_id"],
                window_days         = b.get("window_days", 28),
                computed_at         = datetime.now(tz=_tz.utc),
                n_sessions          = b.get("n_sessions", 0),
                distance_mean       = b.get("distance_mean", 0.0),
                distance_std        = b.get("distance_std", 1.0),
                sprint_count_mean   = b.get("sprint_count_mean", 0.0),
                sprint_count_std    = b.get("sprint_count_std", 1.0),
                top_speed_mean      = b.get("top_speed_mean", 0.0),
                top_speed_std       = b.get("top_speed_std", 1.0),
                high_speed_dist_mean= b.get("high_speed_dist_mean", 0.0),
                high_speed_dist_std = b.get("high_speed_dist_std", 1.0),
                fatigue_alpha       = b.get("fatigue_alpha"),
                fatigue_beta        = b.get("fatigue_beta"),
                fatigue_r_squared   = b.get("fatigue_r_squared"),
                avg_x               = b.get("avg_x"),
                avg_y               = b.get("avg_y"),
                position_std_radius = b.get("position_std_radius"),
            )
            pipeline.pattern_engine._baselines[pid] = baseline
            pipeline.pattern_engine._position_buffers[pid] = []
            p_reg = pipeline.registry.get(pid)
            if p_reg:
                p_reg["baseline"] = baseline
                p_reg["model"] = pipeline.pattern_engine._shared_model

        # Restore threshold tracker (best-effort)
        t = ps.get("threshold_tracker")
        if t:
            try:
                store = RegimeAwareThresholdStore.from_state_dict(
                    t, inner_tracker_cls=DynamicThresholdTracker
                )
                pipeline.pattern_engine._threshold_trackers[pid] = store
                pipeline.pattern_engine.inference_engine._threshold_trackers[pid] = store
            except Exception as exc:
                logger.debug("Could not restore threshold for player %d: %s", pid, exc)


def _save_serve_state(pipeline, path: Path) -> None:
    """
    Save per-player baselines and thresholds to serve_state.json.

    This enables faster serving by avoiding retraining/recibration.
    """
    try:
        from analysis.baseline import PlayerBaselineProfile
        from analysis.regime import RegimeAwareThresholdStore

        state = {"players": {}}

        # Save state for all registered players
        for pid in pipeline.registry.all_player_ids():
            player_info = pipeline.registry.get(pid)
            ps = {
                "external_id": player_info["external_id"],
                "name": player_info.get("name", ""),
                "position": player_info.get("position", "CM"),
                "age": player_info.get("age", 25),
                "age_group": player_info.get("age_group", "Senior"),
            }

            # Save baseline if available
            baseline = pipeline.pattern_engine._baselines.get(pid)
            if baseline and isinstance(baseline, PlayerBaselineProfile):
                ps["baseline"] = {
                    "window_days": baseline.window_days,
                    "n_sessions": baseline.n_sessions,
                    "distance_mean": baseline.distance_mean,
                    "distance_std": baseline.distance_std,
                    "sprint_count_mean": baseline.sprint_count_mean,
                    "sprint_count_std": baseline.sprint_count_std,
                    "top_speed_mean": baseline.top_speed_mean,
                    "top_speed_std": baseline.top_speed_std,
                    "high_speed_dist_mean": baseline.high_speed_dist_mean,
                    "high_speed_dist_std": baseline.high_speed_dist_std,
                    "fatigue_alpha": baseline.fatigue_alpha,
                    "fatigue_beta": baseline.fatigue_beta,
                    "fatigue_r_squared": baseline.fatigue_r_squared,
                    "avg_x": baseline.avg_x,
                    "avg_y": baseline.avg_y,
                    "position_std_radius": baseline.position_std_radius,
                }

            # Save threshold tracker if available
            # Must read from inference_engine — that is where InferenceEngine.train()
            # writes calibrated thresholds. pattern_engine._threshold_trackers is
            # never populated during training so it is always empty at save time.
            tracker = pipeline.pattern_engine.inference_engine._threshold_trackers.get(pid)
            if tracker and isinstance(tracker, RegimeAwareThresholdStore):
                try:
                    ps["threshold_tracker"] = tracker.state_dict()
                except Exception as exc:
                    logger.debug("Could not serialize threshold tracker for player %d: %s", pid, exc)

            state["players"][str(pid)] = ps

        # Write to file
        path.write_text(json.dumps(state, indent=2))
        logger.info("Serve state saved to %s", path)

    except Exception as exc:
        logger.warning("Failed to save serve state: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: audit
# ─────────────────────────────────────────────────────────────────────────────
def cmd_audit(args: argparse.Namespace) -> None:
    """
    Fairness audit + recalibration check against the saved inference log.

    Reads the inference log written by serve (or by a previous train run).
    Runs FairnessMonitor and RecalibrationPipeline.
    Writes a structured audit report to --out.

    Exits 5 if bias is detected in any protected attribute group.
    Exits 1 if the inference log is missing or empty.
    """
    log_path = Path(args.log)
    if not log_path.exists():
        _exit(1, f"Inference log not found: {log_path}")

    logger.info("audit | log=%s  out=%s", log_path, args.out)

    try:
        inference_df = pd.read_json(log_path, lines=True)
    except Exception:
        try:
            inference_df = pd.read_json(log_path)
        except Exception as exc:
            _exit(1, f"Cannot parse inference log: {exc}")

    if inference_df.empty:
        _exit(1, "Inference log is empty — nothing to audit")

    logger.info("Inference log loaded: %d records", len(inference_df))

    # Required columns for fairness audit
    if "is_anomaly" not in inference_df.columns and "recommendation_type" in inference_df.columns:
        # Derive is_anomaly from recommendation_type (any non-null = anomaly)
        inference_df["is_anomaly"] = inference_df["recommendation_type"].notna()

    if "player_id" not in inference_df.columns:
        _exit(1, "Inference log missing 'player_id' column")

    # Load player metadata for protected attributes
    data_dir = Path(args.data_dir)
    players_path = data_dir / "players.csv"
    if not players_path.exists():
        _exit(1, f"players.csv not found at {data_dir}")
    players_df = pd.read_csv(players_path)

    # Rename to match FairnessMonitor expectation
    if "full_name" in players_df.columns and "name" not in players_df.columns:
        players_df = players_df.rename(columns={"full_name": "name"})

    from feedback.recalibration import FairnessMonitor, FeedbackStore, RecalibrationPipeline
    from config.settings import CONFIG

    monitor = FairnessMonitor()
    audit_results = monitor.audit(inference_df, players_df)
    report_text   = monitor.generate_audit_report(audit_results)

    print(report_text)

    # Recalibration check
    feedback = FeedbackStore()
    recal_pipeline = RecalibrationPipeline()

    # Populate feedback from inference log if override info is present
    override_count = 0
    if "decision" in inference_df.columns:
        from feedback.recalibration import OverrideRecord
        for _, row in inference_df[inference_df["decision"] == "override"].iterrows():
            from datetime import timezone as _tz
            record = OverrideRecord(
                inference_id         = int(row.get("inference_id", 0)),
                player_id            = int(row["player_id"]),
                player_external_id   = str(row.get("external_id", "")),
                session_id           = int(row.get("session_id", 0)),
                recommendation_type  = str(row.get("recommendation_type", "")),
                decision             = "override",
                coach_id             = str(row.get("coach_id", "unknown")),
                coach_note           = str(row.get("coach_note", "")),
                overridden_at        = datetime.now(tz=_tz.utc),
                context_snapshot     = row.get("feature_values", {}),
                position             = str(row.get("position", "")),
                age_group            = str(row.get("age_group", "")),
            )
            feedback.log_override(record)
            override_count += 1

    logger.info("Override records in log: %d", override_count)

    recal_results = recal_pipeline.run(feedback, player_models={}, trigger_reason="audit")
    if recal_results:
        logger.info("Recalibration recommended:")
        for r in recal_results:
            logger.info("  player=%s  adjustments=%s", r.player_id, r.adjustments)
    else:
        logger.info("No recalibration needed")

    # Build structured output
    audit_output = {
        "audited_at":       datetime.now(tz=timezone.utc).isoformat(),
        "n_inferences":     len(inference_df),
        "n_overrides":      override_count,
        "recalibration_needed": len(recal_results) > 0,
        "bias_detected":    any(r.is_biased for r in audit_results),
        "biased_groups":    [g for r in audit_results for g in r.biased_groups],
        "audit_results":    [
            {
                "attribute":          r.attribute,
                "is_biased":          r.is_biased,
                "squad_avg_flag_rate": r.squad_avg_flag_rate,
                "biased_groups":      r.biased_groups,
                "group_results":      r.group_results,
                "action_recommended": r.action_recommended,
            }
            for r in audit_results
        ],
        "recalibration_results": [
            {
                "player_id":   r.player_id,
                "reason":      r.trigger_reason,
                "adjustments": r.adjustments,
                "notes":       r.notes,
            }
            for r in recal_results
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit_output, indent=2))
    logger.info("Audit report written → %s", out_path)

    if audit_output["bias_detected"]:
        biased = audit_output["biased_groups"]
        _exit(5, f"Bias detected in groups: {biased}  — see {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI definition
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Players Data — IBM CIC Germany production ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (stderr)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Synthesise training data")
    p_gen.add_argument("--data-dir",    default="data",   help="Output directory for CSVs")
    p_gen.add_argument("--seasons",     type=int,   default=2)
    p_gen.add_argument("--matchdays",   type=int,   default=38)
    p_gen.add_argument("--anomaly-rate",type=float, default=0.05)
    p_gen.add_argument("--no-corruption", action="store_true",
                       help="Skip sensor corruption layer (faster, cleaner data)")
    p_gen.add_argument("--quiet", action="store_true",
                       help="Suppress per-position stats table")

    # ── train ─────────────────────────────────────────────────────────────────
    p_tr = sub.add_parser("train", help="Train model and save checkpoint")
    p_tr.add_argument("--data-dir",            default="data",   help="CSV source directory")
    p_tr.add_argument("--model-dir",           default="models", help="Checkpoint output directory")
    p_tr.add_argument("--sessions-per-player", type=int, default=60,
                      help="Max sessions per player loaded for training")
    p_tr.add_argument("--save-serve-state",    action="store_true",
                      help="Save serve state (baselines, thresholds) for faster serving")

    # ── evaluate ──────────────────────────────────────────────────────────────
    p_ev = sub.add_parser("evaluate", help="Score model against ground truth")
    p_ev.add_argument("--data-dir",  default="data",              help="CSV source directory")
    p_ev.add_argument("--model-dir", default="models",            help="Checkpoint directory")
    p_ev.add_argument("--out",       default="metrics/eval.json", help="Metrics output file")
    p_ev.add_argument("--min-auc",   type=float, default=0.60,
                      help="Minimum acceptable mean ROC-AUC (exit 3 if below)")

    # ── serve ─────────────────────────────────────────────────────────────────
    p_sv = sub.add_parser("serve",
                          help="Stream inference: stdin→events, stdout→alerts (NDJSON)")
    p_sv.add_argument("--model-dir",         default="models", help="Checkpoint directory")
    p_sv.add_argument("--min-alert-windows", type=int, default=3,
                      help="Consecutive anomalous windows before emitting alert")
    p_sv.add_argument("--max-latency-ms",    type=int, default=200,
                      help="SLA threshold; violations are logged as warnings")

    # ── audit ─────────────────────────────────────────────────────────────────
    p_au = sub.add_parser("audit", help="Fairness audit + recalibration check")
    p_au.add_argument("--log",      default="logs/inference_log.jsonl",
                      help="Path to inference log (NDJSON or JSON array)")
    p_au.add_argument("--data-dir", default="data",             help="CSV directory for player metadata")
    p_au.add_argument("--out",      default="metrics/audit.json", help="Audit report output")

    return parser


def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()

    _configure_logging(args.log_level)
    logger.info("Players Data pipeline | command=%s", args.command)

    dispatch = {
        "generate": cmd_generate,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "serve":    cmd_serve,
        "audit":    cmd_audit,
    }

    try:
        dispatch[args.command](args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unhandled exception in command '%s': %s", args.command, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()