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
import concurrent.futures
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from time import monotonic
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Logging: structured to stderr, JSON in production; human-readable in dev
# ─────────────────────────────────────────────────────────────────────────────
_LOG_FMT_HUMAN = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
_LOG_FMT_JSON = None  # set below if JSON_LOGS=1
ALERT_COOLDOWN_S = 20

ALERT_FAMILY = {
    "anomaly_flag": "physiological_instability",
    "substitution": "physiological_instability",
    "fatigue_alert": "physiological_instability",
    "positional_drift": "tactical_instability",
    "workload_alert": "workload_instability",
}


def _configure_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    if os.getenv("JSON_LOGS"):
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return _json.dumps(
                    {
                        "ts": datetime.utcnow().isoformat() + "Z",
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                        **(
                            {"exc": self.formatException(record.exc_info)}
                            if record.exc_info
                            else {}
                        ),
                    }
                )

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


def _build_pipeline(model_dir: Path, replay_mode: bool = False):
    """Construct a PlayersDataAnalysisPipeline pointed at model_dir."""
    from analysis.orchestrator import PlayersDataAnalysisPipeline
    from config.settings import CONFIG

    CONFIG.active_model = "lstm"
    pipeline = PlayersDataAnalysisPipeline(replay_mode=replay_mode)

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
        return pd.DataFrame(
            columns=[
                "session_id",
                "window_distance_m",
                "window_avg_speed_ms",
                "window_sprint_count",
                "heart_rate_bpm",
            ]
        )
    return (
        events_df.groupby("session_id")
        .agg(
            window_distance_m=("speed_ms", lambda x: trapezoid(x, dx=15)),
            window_avg_speed_ms=("speed_ms", "mean"),
            window_sprint_count=("is_sprint", "sum"),
            heart_rate_bpm=("heart_rate_bpm", "mean"),
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
        args.seasons,
        args.matchdays,
        args.anomaly_rate,
        args.data_dir,
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
    events = pd.DataFrame(data["events"])

    if events.empty:
        _exit(1, "Validation failed: events table is empty")

    # Ensure required feature columns are present
    required_cols = {
        "speed_ms",
        "heart_rate_bpm",
        "x_pitch",
        "y_pitch",
        "is_sprint",
        "elapsed_seconds",
    }
    missing = required_cols - set(events.columns)
    if missing:
        _exit(1, f"Validation failed: events missing columns {missing}")

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        dg.save_dataset(data, apply_corruption=not args.no_corruption, output_dir=data_dir)
    except Exception as exc:
        _exit(1, f"Save failed: {exc}")

    n_anomalies = int(gt["is_anomaly"].sum())
    anomaly_pct = float(gt["is_anomaly"].mean()) * 100
    logger.info(
        "Saved to %s | sessions=%d  events=%d  anomalies=%d (%.1f%%)",
        data_dir,
        len(sessions),
        len(events),
        n_anomalies,
        anomaly_pct,
    )

    if not args.quiet:
        dg.print_dataset_report(data)

    print(
        json.dumps(
            {
                "status": "ok",
                "data_dir": str(data_dir.resolve()),
                "sessions": len(sessions),
                "events": len(events),
                "anomaly_count": n_anomalies,
                "anomaly_pct": round(anomaly_pct, 2),
                "elapsed_s": round(elapsed, 2),
            }
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kinexon (real-data) loading — shared by train/evaluate --data-source kinexon
# ─────────────────────────────────────────────────────────────────────────────
def _load_kinexon_frames(data_dir: Path, session_id: str, use_event_features: bool = False):
    """
    Load one Kinexon session via the validated real-data path:
    KinexonAdapter.stream_positions() -> KinexonResampler.resample().

    Consolidates loading logic that was previously duplicated between
    scripts/train_pilot_session_3387.py and scripts/evaluate_pilot_model.py
    (both standalone validation scripts; neither is modified by this
    function's introduction). Used by both _cmd_train_kinexon and
    _cmd_evaluate_kinexon below so the production CLI has exactly one
    Kinexon-loading code path.

    use_event_features=False (default): unchanged behaviour -- returns
    exactly the 8-column-feature events_by_player KinexonResampler produces.

    use_event_features=True: additionally merges the 24 window-aggregated
    events.csv features (ingestion/kinexon_events_features.py) onto each
    player's events_df, bucket-aligned to KinexonResampler's own elapsed_s
    boundaries. Does not change KinexonResampler's own output or behaviour.

    Returns (events_by_player, sessions_df, meta). Exits 1 if required
    Kinexon export files are missing or produce zero usable data.
    """
    from ingestion.kinexon_adapter import KinexonAdapter
    from ingestion.kinexon_resampler import KinexonResampler
    from config.settings import CONFIG

    positions_path = data_dir / CONFIG.kinexon.positions_file
    statistics_path = data_dir / CONFIG.kinexon.statistics_file
    if not positions_path.exists():
        _exit(1, f"Required file missing: {positions_path}")
    if not statistics_path.exists():
        _exit(1, f"Required file missing: {statistics_path}")

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(statistics_path)
    observations = list(
        adapter.stream_positions(positions_path, meta, session_id=session_id, match_id=session_id)
    )
    if not observations:
        _exit(1, f"No valid position observations parsed from {positions_path}")

    resampler = KinexonResampler()
    events_by_player, sessions_df = resampler.resample(observations, session_id=session_id)
    if not events_by_player:
        _exit(1, "KinexonResampler produced 0 players with usable resampled events")

    logger.info(
        "Loaded Kinexon session %s: %d players, %d raw positions",
        session_id, len(events_by_player), len(observations),
    )

    if use_event_features:
        from ingestion.kinexon_events_features import merge_event_features

        events_csv_path = data_dir / CONFIG.kinexon.events_file
        events_by_player = merge_event_features(
            events_by_player=events_by_player,
            events_csv_path=events_csv_path,
            real_player_ids=meta.keys(),
            bucket_seconds=resampler.bucket_seconds,
        )
        logger.info(
            "Merged events.csv window features onto %d players' events_df "
            "(events_csv=%s)", len(events_by_player), events_csv_path,
        )

    return events_by_player, sessions_df, meta


def _register_and_load_kinexon_players(pipeline, events_by_player, sessions_df, meta) -> None:
    """Shared player-registration loop for the Kinexon train/evaluate paths."""
    for pid, df in events_by_player.items():
        m = meta.get(pid)
        pipeline.register_player(
            player_id=pid,
            external_id=str(pid),
            name=m.player_name if m else f"player_{pid}",
            position=m.position_label if m else "unknown",
            age=25,
        )
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        pipeline.load_historical_data(player_id=pid, sessions_df=player_sessions, events_df=df)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: train
# ─────────────────────────────────────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> None:
    """
    Dispatches to the synthetic or real-data (Kinexon) training path based
    on --data-source. Both paths produce the same artifacts (shared_backbone.pt,
    train_summary.json, serve_state.json) so cmd_serve and downstream
    consumers are unaffected by which path trained the model.
    """
    if args.data_source == "kinexon":
        _cmd_train_kinexon(args)
    else:
        _cmd_train_synthetic(args)


def _cmd_train_kinexon(args: argparse.Namespace) -> None:
    """
    Real-data training path, preferred for production runs:

        KinexonAdapter -> KinexonResampler -> gap-aware windowing ->
        BaselineBuilder.compute_with_fallback() -> train_all_models()

    This is the validated path from scripts/train_pilot_session_3387.py,
    wired into the production CLI contract (exit codes, train_summary.json,
    serve_state.json) instead of a standalone print-only script. Does not
    touch model architecture, thresholds, EMA, or alert logic -- it only
    selects the existing use_gap_aware_windowing=True /
    use_provisional_fallback=True opt-in flags that orchestrator.py's
    compute_baselines()/train_all_models() already support, exactly as the
    validated standalone script does.

    Exits 1 on missing/empty Kinexon data, 2 on training failure.
    """
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    session_id = args.session_id
    use_event_features = getattr(args, "use_event_features", False)

    logger.info(
        "train | data_source=kinexon  data=%s  model=%s  session_id=%s  use_event_features=%s",
        data_dir, model_dir, session_id, use_event_features,
    )

    events_by_player, sessions_df, meta = _load_kinexon_frames(
        data_dir, session_id, use_event_features=use_event_features
    )
    n_raw_players = len(events_by_player)

    pipeline = _build_pipeline(model_dir)
    _register_and_load_kinexon_players(pipeline, events_by_player, sessions_df, meta)
    logger.info("Registered + loaded %d players into pipeline", n_raw_players)

    logger.info("Computing baselines (historical-first, provisional fallback)...")
    baselines = pipeline.compute_baselines(window_days=28, use_provisional_fallback=True)
    logger.info("Baselines computed for %d / %d players", len(baselines), n_raw_players)
    if len(baselines) == 0:
        _exit(1, "All baselines failed — compute_with_fallback() produced 0 profiles")

    logger.info("Training shared LSTM backbone (gap-aware windowing)...")
    t0 = time.perf_counter()
    result = pipeline.train_all_models(use_gap_aware_windowing=True)
    elapsed = time.perf_counter() - t0
    logger.info("Training complete in %.1f s | status=%s", elapsed, result.get("status"))

    top_status = result.get("status", "")
    if not top_status or top_status.startswith("error") or top_status.startswith("fail"):
        _exit(2, f"Training returned error status: {result}")

    shared_info = result.get("shared_model", {})
    n_windows = shared_info.get("n_windows", 0)
    n_players_trained = shared_info.get("n_players", len(baselines))
    model_version = shared_info.get("model_version", "unknown")

    if n_windows == 0:
        _exit(2, "Training produced 0 sequence windows — model is empty")

    backbone_path = model_dir / "shared_backbone.pt"
    if not backbone_path.exists():
        _exit(2, f"Expected checkpoint not found at {backbone_path}")

    from config.settings import N_SEQUENCE_FEATURES as _n_feat

    summary = {
        "status": "ok",
        "data_source": "kinexon",
        "session_id": session_id,
        "use_event_features": use_event_features,
        "n_sequence_features": _n_feat,
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_dir": str(model_dir.resolve()),
        "n_players": n_players_trained,
        "n_windows": n_windows,
        "n_baselines": len(baselines),
        "elapsed_s": round(elapsed, 2),
        "backbone_path": str(backbone_path),
        "model_version": model_version,
    }
    summary_path = model_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary written → %s", summary_path)
    print(json.dumps(summary))

    serve_state_path = model_dir / "serve_state.json"
    _save_serve_state(pipeline, serve_state_path)


def _cmd_train_synthetic(args: argparse.Namespace) -> None:
    """
    Fit the shared LSTM backbone + per-player calibration thresholds.

    Data split is performed here, not inside the model:
      • sessions per player: most-recent N sessions → training
      • remaining sessions  → calibration (threshold fitting)

    All splits are deterministic: given the same data the output is identical.
    Saves shared_backbone.pt and writes train_summary.json to --model-dir.

    Exits 2 if training fails or produces a degenerate model.
    """
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    logger.info(
        "train | data=%s  model=%s  sessions_per_player=%d",
        data_dir,
        model_dir,
        args.sessions_per_player,
    )

    frames = _load_csvs(data_dir)
    pipeline = _build_pipeline(model_dir)

    players_df = frames["players"]
    sessions_df = frames["sessions"]
    events_df = frames["events"]
    annot_df = frames["annotations"]

    # ── Register all players ──────────────────────────────────────────────────
    for _, row in players_df.iterrows():
        pipeline.register_player(
            player_id=int(row["player_id"]),
            external_id=str(row["external_id"]),
            name=str(
                row.get("full_name", row.get("name", f"Player {row['player_id']}"))
            ),
            position=str(row["position"]),
            age=int(row.get("age", 25)),
            age_group=str(row.get("age_group", "Senior")),
            nationality=str(row.get("nationality", "")),
        )
    logger.info("Registered %d players", len(players_df))

    # ── Load data per player ──────────────────────────────────────────────────
    features_df = _aggregate_session_features(events_df)
    sessions_with_features = sessions_df.merge(features_df, on="session_id", how="left")
    for col, fill in [
        ("window_distance_m", 0),
        ("window_avg_speed_ms", 0),
        ("window_sprint_count", 0),
        ("heart_rate_bpm", 120),
    ]:
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
        pannot = annot_df[annot_df["session_id"].isin(psessions["session_id"])]
        pipeline.load_historical_data(
            player_id=pid,
            sessions_df=psessions,
            events_df=pevents,
            annotations_df=pannot,
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
        _exit(
            1, "All baselines failed — check data quality / min_sessions_for_baseline"
        )

    # Log a concise baseline table
    for pid, b in sorted(baselines.items()):
        logger.debug(
            "  p%03d  dist=%.0f±%.0f m  sprints=%.1f  fatigue_r2=%.3f",
            pid,
            b.distance_mean,
            b.distance_std,
            b.sprint_count_mean,
            b.fatigue_r_squared or 0,
        )

    # ── Train shared backbone ─────────────────────────────────────────────────
    logger.info("Training shared LSTM backbone...")
    t0 = time.perf_counter()
    result = pipeline.train_all_models()
    elapsed = time.perf_counter() - t0
    logger.info(
        "Training complete in %.1f s | status=%s", elapsed, result.get("status")
    )

    # Orchestrator returns {"status": "success", "shared_model": {...}, "players": {...}}.
    # Accept any non-error status string so this doesn't break if the orchestrator
    # changes "success" → "ok" or "trained" in future.
    top_status = result.get("status", "")
    if (
        not top_status
        or top_status.startswith("error")
        or top_status.startswith("fail")
    ):
        _exit(2, f"Training returned error status: {result}")

    # n_windows lives under result["shared_model"]["n_windows"], not at the top level.
    shared_info = result.get("shared_model", {})
    n_windows = shared_info.get("n_windows", 0)
    n_players_trained = shared_info.get("n_players", n_loaded)
    model_version = shared_info.get("model_version", "unknown")

    if n_windows == 0:
        _exit(2, "Training produced 0 sequence windows — model is empty")

    # ── Verify checkpoint was written ─────────────────────────────────────────
    backbone_path = model_dir / "shared_backbone.pt"
    if not backbone_path.exists():
        _exit(2, f"Expected checkpoint not found at {backbone_path}")

    # ── Write training summary ────────────────────────────────────────────────
    summary = {
        "status": "ok",
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "model_dir": str(model_dir.resolve()),
        "n_players": n_players_trained,
        "n_windows": n_windows,
        "n_baselines": len(baselines),
        "elapsed_s": round(elapsed, 2),
        "backbone_path": str(backbone_path),
        "model_version": model_version,
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
    """Dispatches to the synthetic or real-data (Kinexon) evaluation path."""
    if args.data_source == "kinexon":
        _cmd_evaluate_kinexon(args)
    else:
        _cmd_evaluate_synthetic(args)


def _cmd_evaluate_kinexon(args: argparse.Namespace) -> None:
    """
    Real-data evaluation path.

    Real Kinexon sessions carry no ground_truth_labels — no human or system
    has ever labeled a window as anomalous for session 3387 (or any other
    real session collected so far). Computing ROC-AUC / PR-AUC /
    precision@k against labels that do not exist would mean fabricating
    them, which this task explicitly rules out ("do not invent metrics that
    are not supported by the available data"). This path instead reports
    the descriptive statistics that ARE honestly computable from real model
    output: loss/confidence distributions, calibration coverage (per-player
    vs pilot-pooled-fallback vs uncalibrated), and the raw (pre-EMA,
    pre-persistence) threshold-breach rate.

    Reuses the exact validated procedure from scripts/evaluate_pilot_model.py:
    retrains once (SharedBackboneAutoencoder.train() does not persist
    per-player calibration state to shared_backbone.pt — see that script's
    module docstring — so calibration state must be reproduced via training,
    not loaded from a checkpoint) and scores every real window through the
    real PatternAnalysisEngine.analyze_window().

    --min-auc is ignored on this path (logged, not silently dropped) — there
    is no AUC to gate on without ground truth.
    """
    data_dir = Path(args.data_dir)
    session_id = args.session_id

    logger.info("evaluate | data_source=kinexon  data=%s  session_id=%s", data_dir, session_id)
    if args.min_auc != 0.60:  # the argparse default; a non-default value means the caller set it
        logger.warning("--min-auc has no effect for --data-source kinexon (no ground truth labels exist)")

    # Pilot-mode pooled calibration: a single real session gives most
    # players far too few windows for their OWN RegimeAwareThresholdStore
    # to reach min_calibration_windows, which otherwise leaves confidence at
    # 0.0 / tracker_source "none" for everyone (the exact issue solved by
    # pilot_mode earlier in this project -- see AnomalyScoringConfig in
    # config/settings.py). scripts/evaluate_pilot_model.py sets this same
    # flag for the same reason; omitting it here would silently diverge
    # from "the exact validated procedure" this function's docstring claims
    # to reuse.
    from config.settings import CONFIG as _CONFIG
    _CONFIG.scoring.pilot_mode = True

    use_event_features = getattr(args, "use_event_features", False)
    events_by_player, sessions_df, meta = _load_kinexon_frames(
        data_dir, session_id, use_event_features=use_event_features
    )

    pipeline = _build_pipeline(Path(args.model_dir))
    _register_and_load_kinexon_players(pipeline, events_by_player, sessions_df, meta)

    baselines = pipeline.compute_baselines(window_days=28, use_provisional_fallback=True)
    if not baselines:
        _exit(3, "No baselines computed — nothing to evaluate")

    result = pipeline.train_all_models(use_gap_aware_windowing=True)
    if result.get("status") != "success" or result.get("shared_model", {}).get("n_windows", 0) == 0:
        _exit(2, f"Training (required to reproduce calibration state) failed: {result}")

    engine = pipeline.pattern_engine
    from analysis.gap_aware_windowing import build_training_sequences_gap_aware
    from analysis.regime import SessionRegimeClassifier
    from config.settings import SEQUENCE_FEATURE_NAMES as _SFN

    idx = {n: i for i, n in enumerate(_SFN)}
    classifier = SessionRegimeClassifier()

    rows: List[dict] = []
    for pid in baselines:
        df = events_by_player[pid].sort_values("ts").reset_index(drop=True)
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        windows, _audit = build_training_sequences_gap_aware(
            pattern_engine=engine, events_df=df, sessions_df=player_sessions,
        )
        for seq, mask, _sid in windows:
            last = seq[-1]
            live_event = {
                "x_pitch": float(last[idx["x_pitch"]]),
                "y_pitch": float(last[idx["y_pitch"]]),
                "elapsed_seconds": 0.0,
                "_tvl_confidence": 1.0,
            }
            res = engine.analyze_window(
                player_id=pid, sequence=seq, mask=mask,
                live_event=live_event, sessions_df=player_sessions,
            )
            tracker = engine.inference_engine.get_tracker(pid)
            tracker_source = engine.inference_engine.get_tracker_source(pid)
            regime_key = classifier.classify(seq).key
            eff_threshold = float("inf")
            raw_breach = None
            if tracker and tracker.is_calibrated:
                eff_threshold = tracker.threshold_for(regime_key)
                raw_breach = bool(res.anomaly_score > eff_threshold)
            rows.append({
                "player_id": pid,
                "anomaly_score": res.anomaly_score,
                "confidence": res.confidence,
                "tracker_source": tracker_source,
                "raw_threshold_breach": raw_breach,
            })

    if not rows:
        _exit(3, "0 evaluable windows — check gap-aware windowing output")

    df_rows = pd.DataFrame(rows)
    n_calibrated = sum(
        1 for pid in baselines
        if (t := engine.inference_engine._threshold_trackers.get(pid)) and t.is_calibrated
    )
    breach = df_rows["raw_threshold_breach"]

    def _dist(s: pd.Series) -> dict:
        return {
            "min": float(s.min()), "p25": float(s.quantile(.25)), "median": float(s.median()),
            "p75": float(s.quantile(.75)), "p90": float(s.quantile(.90)), "max": float(s.max()),
            "mean": float(s.mean()),
        }

    aggregate = {
        "status": "ok",
        "data_source": "kinexon",
        "session_id": session_id,
        "use_event_features": use_event_features,
        "n_sequence_features": len(_SFN),
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_players_evaluated": len(baselines),
        "n_windows_evaluated": len(df_rows),
        "anomaly_score_distribution": _dist(df_rows["anomaly_score"]),
        "confidence_distribution": _dist(df_rows["confidence"]),
        "tracker_source_counts": df_rows["tracker_source"].value_counts().to_dict(),
        "n_players_with_own_calibration": n_calibrated,
        "raw_threshold_breach_rate": (
            float(breach.dropna().mean()) if breach.notna().any() else None
        ),
        "raw_threshold_breach_count": int(breach.fillna(False).sum()),
        "note": (
            "No roc_auc/pr_auc/precision_at_k: real Kinexon sessions carry no "
            "ground_truth_labels. These are descriptive statistics over real "
            "model output, not a classification-accuracy evaluation."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))
    logger.info("Metrics written → %s", out_path)
    print(json.dumps(aggregate, indent=2))


def _cmd_evaluate_synthetic(args: argparse.Namespace) -> None:
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

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    logger.info("evaluate | data=%s  model=%s", data_dir, model_dir)

    backbone_path = model_dir / "shared_backbone.pt"
    if not backbone_path.exists():
        _exit(2, f"Model checkpoint not found at {backbone_path} — run train first")

    frames = _load_csvs(data_dir)
    pipeline = _build_pipeline(model_dir)

    players_df = frames["players"]
    sessions_df = frames["sessions"]
    events_df = frames["events"]
    annot_df = frames["annotations"]
    gt_df = frames["ground_truth_labels"]

    # ── Pre-group for O(1) lookup — avoids O(n) DataFrame filter per player/session ──
    logger.info("Pre-grouping events and sessions …")
    session_events_map: Dict[int, pd.DataFrame] = {
        int(sid): grp for sid, grp in events_df.groupby("session_id")
    }
    player_sessions_map: Dict[int, pd.DataFrame] = {
        int(pid): grp for pid, grp in sessions_df.groupby("player_id")
    }
    player_annot_map: Dict[int, pd.DataFrame] = (
        {
            int(pid): grp
            for pid, grp in annot_df.groupby(
                annot_df["session_id"].map(
                    sessions_df.set_index("session_id")["player_id"]
                )
            )
        }
        if not annot_df.empty and "session_id" in annot_df.columns
        else {}
    )

    # ── Load backbone ─────────────────────────────────────────────────────────
    from analysis.anomaly_detection import SharedBackboneAutoencoder

    shared = SharedBackboneAutoencoder.load(backbone_path)
    if shared is None or not shared.is_trained:
        _exit(2, "Could not load trained backbone from checkpoint")

    pipeline.pattern_engine._shared_model = shared
    pipeline.pattern_engine.inference_engine._shared_model = shared
    logger.info(
        "Loaded backbone v=%s  players=%d", shared.model_version, shared.n_players
    )

    # ── Register players + baselines ─────────────────────────────────────────
    features_df = _aggregate_session_features(events_df)
    sessions_with_features = sessions_df.merge(features_df, on="session_id", how="left")
    for col, fill in [
        ("window_distance_m", 0),
        ("window_avg_speed_ms", 0),
        ("window_sprint_count", 0),
        ("heart_rate_bpm", 120),
    ]:
        sessions_with_features[col] = sessions_with_features[col].fillna(fill)

    swf_by_player: Dict[int, pd.DataFrame] = {
        int(pid): grp for pid, grp in sessions_with_features.groupby("player_id")
    }

    for _, row in players_df.iterrows():
        pipeline.register_player(
            player_id=int(row["player_id"]),
            external_id=str(row["external_id"]),
            name=str(row.get("full_name", row.get("name", ""))),
            position=str(row["position"]),
            age=int(row.get("age", 25)),
            age_group=str(row.get("age_group", "Senior")),
        )

    baselines = {}
    for pid in _progress(
        players_df["player_id"].tolist(), desc="Loading player histories", unit="player"
    ):
        psessions = swf_by_player.get(pid, pd.DataFrame())
        if psessions.empty:
            continue
        sids = set(psessions["session_id"])
        pevents = (
            pd.concat(
                [session_events_map[sid] for sid in sids if sid in session_events_map],
                ignore_index=True,
            )
            if sids
            else pd.DataFrame()
        )
        pannot = player_annot_map.get(pid, pd.DataFrame())
        pipeline.load_historical_data(pid, psessions, pevents, pannot)

    baselines = pipeline.compute_baselines(window_days=28)
    logger.info("Baselines ready for %d players", len(baselines))

    # ── Rebuild threshold trackers — batched predict for speed ────────────────
    from analysis.regime import RegimeAwareThresholdStore, SessionRegimeClassifier
    from analysis.anomaly_detection import (
        EMASmoother,
    )  # re-export via anomaly_detection
    from config.settings import CONFIG

    classifier = SessionRegimeClassifier()
    alpha = CONFIG.scoring.score_ema_alpha

    for pid in _progress(
        sorted(baselines.keys()), desc="Calibrating thresholds", unit="player"
    ):
        psessions = player_sessions_map.get(pid, pd.DataFrame())
        if psessions.empty:
            continue
        psessions = psessions.sort_values("started_at")
        sids = set(psessions["session_id"])
        pevents = (
            pd.concat(
                [session_events_map[sid] for sid in sids if sid in session_events_map],
                ignore_index=True,
            )
            if sids
            else pd.DataFrame()
        )

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
        seqs_arr = np.stack([s for s, _, _ in calib]).astype(np.float32)

        masks_arr = np.stack([m for _, m, _ in calib]).astype(bool)

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

        store = RegimeAwareThresholdStore()
        smoother = EMASmoother(alpha)
        for loss, (seq, _, _) in zip(losses, calib):
            ema_val = smoother.update(float(loss))
            store.update(ema_val, classifier.classify(seq).key)
        pipeline.pattern_engine.inference_engine._threshold_trackers[pid] = store

        pipeline.pattern_engine._threshold_trackers[pid] = store

    # ── Build labeled window set ──────────────────────────────────────────────
    gt_map = dict(
        zip(gt_df["session_id"].astype(int), gt_df["is_anomaly"].astype(bool))
    )

    all_player_metrics: List[dict] = []
    total_tp = total_fp = total_fn = total_tn = 0

    for pid in _progress(
        sorted(baselines.keys()), desc="Evaluating players", unit="player"
    ):
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
            for seq, mask in pipeline.pattern_engine.window_builder.build_from_session(
                sess_evs
            ):
                labeled.append((seq, mask, label))

        if not labeled:
            logger.warning("Player %d: 0 labeled windows — skipping", pid)
            continue

        n_anom = sum(1 for _, _, l in labeled if l)
        n_normal = sum(1 for _, _, l in labeled if not l)

        if n_anom == 0 or n_normal == 0:
            logger.info(
                "Player %d: only one class present (anom=%d norm=%d) — skipping",
                pid,
                n_anom,
                n_normal,
            )
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
            metrics.get("pr_auc", float("nan")),
            metrics.get("precision_at_k", float("nan")),
            metrics.get("fp_per_90_min", float("nan")),
            metrics.get("tp", 0),
            metrics.get("fp", 0),
            metrics.get("fn", 0),
        )

    if not all_player_metrics:
        _exit(
            3,
            "No players produced evaluable labeled windows "
            "— check ground_truth_labels.csv and data split",
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _nanmean(key: str) -> float:
        vals = [m[key] for m in all_player_metrics if key in m and m[key] is not None]
        return float(np.nanmean(vals)) if vals else float("nan")

    prec_global = total_tp / max(total_tp + total_fp, 1)
    rec_global = total_tp / max(total_tp + total_fn, 1)

    aggregate = {
        "status": "ok",
        "evaluated_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_players_evaluated": len(all_player_metrics),
        "mean_roc_auc": round(_nanmean("roc_auc"), 4),
        "mean_pr_auc": round(_nanmean("pr_auc"), 4),
        "mean_precision_at_k": round(_nanmean("precision_at_k"), 4),
        "mean_fp_per_90_min": round(_nanmean("fp_per_90_min"), 4),
        "global_precision": round(prec_global, 4),
        "global_recall": round(rec_global, 4),
        "global_tp": total_tp,
        "global_fp": total_fp,
        "global_fn": total_fn,
        "global_tn": total_tn,
        "per_player": all_player_metrics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))
    logger.info("Metrics written → %s", out_path)

    print(
        json.dumps({k: v for k, v in aggregate.items() if k != "per_player"}, indent=2)
    )

    # Fail-fast gate: exit non-zero if mean AUC is below minimum
    if aggregate["mean_roc_auc"] < args.min_auc:
        _exit(
            3, f"ROC-AUC {aggregate['mean_roc_auc']:.3f} < required {args.min_auc:.3f}"
        )


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
        _exit(2, f"Model checkpoint not found: {backbone_path} — run train first")

    logger.info(
        "serve | model=%s  min_alert_windows=%d  max_latency_ms=%d",
        model_dir,
        args.min_alert_windows,
        args.max_latency_ms,
    )

    # ─────────────────────────────────────────────
    # Replay mode
    # ─────────────────────────────────────────────
    _replay_mode = args.replay_mode or (
        getattr(args, "ignore_time_gaps", False)
        and getattr(args, "ignore_session_boundaries", False)
    )

    pipeline = _build_pipeline(
        model_dir,
        replay_mode=_replay_mode,
    )

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

        for player in pipeline.registry._players.values():
            player["model"] = shared

        logger.info(
            "Serve state restored from %s",
            serve_state_path,
        )

    else:

        logger.warning(
            "No serve_state.json found — " "player baselines and thresholds not loaded."
        )

    # ─────────────────────────────────────────────
    # Alert gate
    # ─────────────────────────────────────────────
    gate_last: Dict[str, str] = {}
    gate_last_emit_ts: Dict[int, int] = {}
    gate_counts: Dict[str, int] = {}

    def _gate_fire(ext_id: str, rec_type: str) -> bool:

        family = ALERT_FAMILY.get(rec_type, rec_type)

        previous_family = gate_last.get(ext_id)

        if previous_family != family:
            gate_last[ext_id] = family
            gate_last_emit_ts.pop(ext_id, None)

        now = monotonic()

        last_emit = gate_last_emit_ts.get(ext_id, 0)

        if now - last_emit < ALERT_COOLDOWN_S:
            return False

        gate_last_emit_ts[ext_id] = now

        return True

    def _gate_reset(ext_id: str) -> None:
        gate_last.pop(ext_id, None)
        gate_last_emit_ts.pop(ext_id, None)

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
    # Ollama timeout
    # ─────────────────────────────────────────────
    _async_nlg_timeout = float(os.getenv("OLLAMA_NLG_TIMEOUT_S", "60.0"))

    os.environ["OLLAMA_NLG_TIMEOUT_S"] = str(_async_nlg_timeout)

    try:
        pipeline.xai_layer._llm_nlg._timeout_s = _async_nlg_timeout

        if pipeline.xai_layer._llm_nlg._client is not None:
            pipeline.xai_layer._llm_nlg._client.timeout_s = _async_nlg_timeout

    except Exception:
        pass

    # ─────────────────────────────────────────────
    # NLG executor
    # ─────────────────────────────────────────────
    _nlg_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="nlg",
    )

    # ─────────────────────────────────────────────
    # Match setup
    # ─────────────────────────────────────────────
    n_events = 0
    n_alerts = 0
    sla_violations = 0

    match_id = datetime.now(timezone.utc).strftime("serve_match_%Y%m%d_%H%M%S")

    pipeline.start_match(match_id)

    logger.info(
        "Started serve session: %s",
        match_id,
    )

    # ─────────────────────────────────────────────
    # Warmup
    # ─────────────────────────────────────────────
    try:
        pipeline.xai_layer.warmup_nlg()
    except Exception as exc:
        logger.warning(
            "NLG warmup failed: %s",
            exc,
        )

    # ─────────────────────────────────────────────
    # Window accumulator
    # ─────────────────────────────────────────────
    WINDOW_SIZE = 24
    STRIDE = 24

    from analysis.live_window_accumulator import (
        LiveWindowAccumulator,
    )

    accumulator = LiveWindowAccumulator(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        ignore_time_gaps=args.ignore_time_gaps,
        ignore_session_boundaries=args.ignore_session_boundaries,
    )

    logger.info(
        "LiveWindowAccumulator ready (window_size=%d  stride=%d)",
        WINDOW_SIZE,
        STRIDE,
    )

    def _process_window(window):

        latest_event = window[-1]

        ext_id = latest_event.get(
            "player_external_id",
            "<missing>",
        )

        player = pipeline.registry.get_by_external_id(ext_id)

        if player is None:

            logger.warning(
                "_process_window: no registry entry for ext_id=%r",
                ext_id,
            )

            return None

        return pipeline.process_window_direct(
            window_events=window,
            player_id=player["player_id"],
            replay_mode=_replay_mode,
            nlg_async=False,
        )

    logger.info("Serving — reading newline-delimited JSON from stdin")

    try:

        for raw_line in sys.stdin:

            raw_line = raw_line.strip()

            if not raw_line:
                continue

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
                continue

            event = _enrich(event)

            window = accumulator.push(
                player_id=ext_id,
                event=event,
            )

            if accumulator.consume_reset_flag(ext_id):

                try:

                    player_id_int = int(ext_id.replace("p", ""))

                    pipeline.pattern_engine.reset_ema_state(player_id_int)

                    pipeline.pattern_engine.alert_manager.clear_player(player_id_int)

                    pipeline.tvl.reset_player(player_id_int)

                    _gate_reset(ext_id)

                except Exception as exc:

                    logger.warning(
                        "Session reset failed for %s: %s",
                        ext_id,
                        exc,
                    )

            if window is None:
                continue

            t_start = time.perf_counter()

            try:

                result = _process_window(window)

            except Exception as exc:

                logger.exception(
                    "Inference error for %s: %s",
                    ext_id,
                    exc,
                )

                continue

            # Inference latency: model forward pass + SHAP + symbolic reasoning.
            # This is the SLA-gated path — NLG is excluded intentionally because
            # LLM generation can take 8–11 s and must never block the alert decision.
            inference_latency_ms = (time.perf_counter() - t_start) * 1000
            nlg_latency_ms = 0.0

            if inference_latency_ms > args.max_latency_ms:

                sla_violations += 1

                logger.warning(
                    "SLA breach: player=%s inference_latency=%.1f ms > %d ms",
                    ext_id,
                    inference_latency_ms,
                    args.max_latency_ms,
                )

            # ─────────────────────────────────────────────
            # FORCE synchronous LLM generation
            # Timed separately — NLG latency is tracked but
            # does NOT count against the inference SLA.
            # ─────────────────────────────────────────────
            if result is not None:

                try:

                    _base_expl = getattr(
                        result,
                        "base_explanation",
                        None,
                    )

                    _sem_state = getattr(result, "semantic_state", None)
                    _compressed_ctx = getattr(result, "compressed_context", None)

                    if _base_expl is not None:

                        logger.info(
                            "Generating LLM summary | player=%s",
                            ext_id,
                        )

                        t_nlg_start = time.perf_counter()


                        logger.warning(
            "RESULT DEBUG | has_ctx=%s ctx=%r",
            hasattr(result, "compressed_context"),
            getattr(result, "compressed_context", None),
        )
                                

                        expl = pipeline.xai_layer.generate_nlg(
                            base=_base_expl,
                            match_context=_sem_state,
                            compressed_context=_compressed_ctx,
                        )

                    
                        nlg_latency_ms = (time.perf_counter() - t_nlg_start) * 1000

                        result.nlg_summary = expl.nlg_summary

                        result.nlg_engine = expl.nlg_engine

                        result.shap_values = expl.shap_values

                        if hasattr(
                            expl,
                            "top_contributions",
                        ):
                            result.top_contributions = expl.top_contributions

                        logger.info(
                            "LLM success | player=%s engine=%s len=%d nlg_ms=%.0f",
                            ext_id,
                            expl.nlg_engine,
                            len(expl.nlg_summary or ""),
                            nlg_latency_ms,
                        )

                except Exception as exc:

                    logger.exception(
                        "LLM generation failed for %s: %s",
                        ext_id,
                        exc,
                    )

                    result.nlg_engine = "template_fallback"

            # ─────────────────────────────────────────────
            # Persist
            # ─────────────────────────────────────────────
            if result is None:
                continue

            if result.recommendation_type is None:

                _gate_reset(ext_id)

                continue

            if not _gate_fire(
                ext_id,
                result.recommendation_type,
            ):
                continue

            n_alerts += 1

            gate_counts[ext_id] = gate_counts.get(ext_id, 0) + 1

            alert_payload = {
                "player_id": result.player_id,
                "external_id": result.external_id,
                "recommendation_type": result.recommendation_type,
                "confidence": round(
                    result.confidence,
                    4,
                ),
                "anomaly_score": round(
                    result.anomaly_score,
                    6,
                ),
                "fatigue_flag": result.fatigue_flag,
                "drift_flag": result.positional_drift_flag,
                "workload_flag": result.workload_flag,
                "workload_status": result.workload_status,
                "nlg_engine": getattr(
                    result,
                    "nlg_engine",
                    "unknown",
                ),
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
                            10,
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
                    inference_latency_ms,
                    2,
                ),
                "inference_latency_ms": round(inference_latency_ms, 2),
                "nlg_latency_ms": round(nlg_latency_ms, 2),
                "end_to_end_latency_ms": round(
                    inference_latency_ms + nlg_latency_ms, 2
                ),
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "gate_windows": gate_counts[ext_id],
            }

            print(
                json.dumps(alert_payload),
                flush=True,
            )

            _inference_log_fh.write(json.dumps(alert_payload) + "\n")

            logger.info(
                "ALERT player=%-6s type=%-20s conf=%.2f infer=%.1f ms nlg=%.0f ms engine=%s",
                ext_id,
                result.recommendation_type,
                result.confidence,
                inference_latency_ms,
                nlg_latency_ms,
                getattr(
                    result,
                    "nlg_engine",
                    "unknown",
                ),
            )

    except KeyboardInterrupt:

        logger.info("SIGINT received — shutting down gracefully")

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
            player_id=pid,
            external_id=ps["external_id"],
            name=ps.get("name", ""),
            position=ps.get("position", "CM"),
            age=ps.get("age", 25),
            age_group=ps.get("age_group", "Senior"),
        )
        # Restore baseline
        b = ps.get("baseline")
        if b:
            baseline = PlayerBaselineProfile(
                player_id=pid,
                external_id=ps["external_id"],
                window_days=b.get("window_days", 28),
                computed_at=datetime.now(tz=_tz.utc),
                n_sessions=b.get("n_sessions", 0),
                distance_mean=b.get("distance_mean", 0.0),
                distance_std=b.get("distance_std", 1.0),
                sprint_count_mean=b.get("sprint_count_mean", 0.0),
                sprint_count_std=b.get("sprint_count_std", 1.0),
                top_speed_mean=b.get("top_speed_mean", 0.0),
                top_speed_std=b.get("top_speed_std", 1.0),
                high_speed_dist_mean=b.get("high_speed_dist_mean", 0.0),
                high_speed_dist_std=b.get("high_speed_dist_std", 1.0),
                fatigue_alpha=b.get("fatigue_alpha"),
                fatigue_beta=b.get("fatigue_beta"),
                fatigue_r_squared=b.get("fatigue_r_squared"),
                avg_x=b.get("avg_x"),
                avg_y=b.get("avg_y"),
                position_std_radius=b.get("position_std_radius"),
            )
            pipeline.pattern_engine._baselines[pid] = baseline
            pipeline.pattern_engine._position_buffers[pid] = []
            p_reg = pipeline.registry.get(pid)
            if p_reg:
                p_reg["baseline"] = baseline
                p_reg["model"] = pipeline.pattern_engine._shared_model

                seq_bg = ps.get("sequence_background")

                if seq_bg is not None:
                    p_reg["sequence_background"] = np.asarray(
                        seq_bg,
                        dtype=np.float32,
                    )

        # Restore threshold tracker (best-effort)
        t = ps.get("threshold_tracker")
        if t:
            try:
                store = RegimeAwareThresholdStore.from_state_dict(
                    t, inner_tracker_cls=DynamicThresholdTracker
                )
                pipeline.pattern_engine._threshold_trackers[pid] = store
                pipeline.pattern_engine.inference_engine._threshold_trackers[pid] = (
                    store
                )
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
            tracker = pipeline.pattern_engine.inference_engine._threshold_trackers.get(
                pid
            )

            # Save SHAP background windows for AE explainability
            seq_bg = player_info.get("sequence_background")

            if seq_bg is not None:
                try:
                    ps["sequence_background"] = np.asarray(seq_bg, dtype=np.float32)[
                        :50
                    ].tolist()
                except Exception as exc:
                    logger.debug(
                        "Could not serialize sequence background for player %d: %s",
                        pid,
                        exc,
                    )

            if tracker and isinstance(tracker, RegimeAwareThresholdStore):
                try:
                    ps["threshold_tracker"] = tracker.state_dict()
                except Exception as exc:
                    logger.debug(
                        "Could not serialize threshold tracker for player %d: %s",
                        pid,
                        exc,
                    )

            state["players"][str(pid)] = ps

        # Write to file
        path.write_text(json.dumps(state, indent=2))
        logger.info("Serve state saved to %s", path)

    except Exception as exc:
        logger.warning("Failed to save serve state: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: ingest
# ─────────────────────────────────────────────────────────────────────────────
def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Multi-match dataset pipeline: scan --data-dir for match_<id>/ subdirectories,
    validate each export, build the match metadata index, and write/update
    matches.parquet, players.parquet, events.parquet, positions.parquet under
    --output-dir. Incremental: a match directory whose files are unchanged
    since the last run is not re-parsed.

    Does not touch model code, training, or calibration. Writes three JSON
    reports (dataset_summary.json, data_quality_report.json,
    match_inventory.json) into --output-dir and prints the dataset summary.

    Exits 1 if 0 match_* directories are found.
    """
    from ingestion.multi_match_pipeline import MultiMatchDatasetBuilder
    from analysis.player_trends import build_and_write_player_trends

    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    builder = MultiMatchDatasetBuilder(data_root=data_root, output_dir=output_dir)
    match_dirs = builder.scan()
    if not match_dirs:
        _exit(1, f"No match_* directories found under {data_root}")

    result = builder.run()

    trends = build_and_write_player_trends(output_dir)
    logger.info(
        "Player trends written -> %s (n_matches_in_dataset=%d, n_players=%d)",
        output_dir / "player_trends.json", trends["n_matches_in_dataset"], len(trends["players"]),
    )

    print(json.dumps(result["dataset_summary"], indent=2))


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
    if (
        "is_anomaly" not in inference_df.columns
        and "recommendation_type" in inference_df.columns
    ):
        # Derive is_anomaly from recommendation_type (any non-null = anomaly)
        inference_df["is_anomaly"] = inference_df["recommendation_type"].notna()

    if "player_id" not in inference_df.columns:
        _exit(1, "Inference log missing 'player_id' column")

    # Load player metadata for protected attributes
    data_dir = Path(args.data_dir)
    players_path = data_dir / "players.csv"
    if players_path.exists():
        players_df = pd.read_csv(players_path)
        # Rename to match FairnessMonitor expectation
        if "full_name" in players_df.columns and "name" not in players_df.columns:
            players_df = players_df.rename(columns={"full_name": "name"})
    else:
        # Real Kinexon sessions have no players.csv roster export (that file
        # is synthetic-only). Fall back to statistics.csv via KinexonAdapter
        # instead of failing outright -- this gives FairnessMonitor a real
        # player_id/name/position frame. age_group and nationality are
        # genuinely absent from the Kinexon export; left out rather than
        # fabricated, so FairnessMonitor simply finds no groups for those
        # attributes instead of being fed invented values.
        from config.settings import CONFIG as _CONFIG
        stats_path = data_dir / _CONFIG.kinexon.statistics_file
        if not stats_path.exists():
            _exit(1, f"Neither players.csv nor {stats_path.name} found at {data_dir}")
        from ingestion.kinexon_adapter import KinexonAdapter
        kinexon_meta = KinexonAdapter().load_player_meta(stats_path)
        players_df = pd.DataFrame([
            {"player_id": pid, "name": m.player_name, "position": m.position_label}
            for pid, m in kinexon_meta.items()
        ])
        logger.warning(
            "players.csv not found — built player metadata from %s instead "
            "(position only; age_group/nationality unavailable for real Kinexon data)",
            stats_path.name,
        )

    from feedback.recalibration import (
        FairnessMonitor,
        FeedbackStore,
        RecalibrationPipeline,
    )
    from config.settings import CONFIG

    monitor = FairnessMonitor()
    audit_results = monitor.audit(inference_df, players_df)
    report_text = monitor.generate_audit_report(audit_results)

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
                inference_id=int(row.get("inference_id", 0)),
                player_id=int(row["player_id"]),
                player_external_id=str(row.get("external_id", "")),
                session_id=int(row.get("session_id", 0)),
                recommendation_type=str(row.get("recommendation_type", "")),
                decision="override",
                coach_id=str(row.get("coach_id", "unknown")),
                coach_note=str(row.get("coach_note", "")),
                overridden_at=datetime.now(tz=_tz.utc),
                context_snapshot=row.get("feature_values", {}),
                position=str(row.get("position", "")),
                age_group=str(row.get("age_group", "")),
            )
            feedback.log_override(record)
            override_count += 1

    logger.info("Override records in log: %d", override_count)

    recal_results = recal_pipeline.run(
        feedback, player_models={}, trigger_reason="audit"
    )
    if recal_results:
        logger.info("Recalibration recommended:")
        for r in recal_results:
            logger.info("  player=%s  adjustments=%s", r.player_id, r.adjustments)
    else:
        logger.info("No recalibration needed")

    # Build structured output
    audit_output = {
        "audited_at": datetime.now(tz=timezone.utc).isoformat(),
        "n_inferences": len(inference_df),
        "n_overrides": override_count,
        "recalibration_needed": len(recal_results) > 0,
        "bias_detected": any(r.is_biased for r in audit_results),
        "biased_groups": [g for r in audit_results for g in r.biased_groups],
        "audit_results": [
            {
                "attribute": r.attribute,
                "is_biased": r.is_biased,
                "squad_avg_flag_rate": r.squad_avg_flag_rate,
                "biased_groups": r.biased_groups,
                "group_results": r.group_results,
                "action_recommended": r.action_recommended,
            }
            for r in audit_results
        ],
        "recalibration_results": [
            {
                "player_id": r.player_id,
                "reason": r.trigger_reason,
                "adjustments": r.adjustments,
                "notes": r.notes,
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (stderr)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate", help="Synthesise training data")
    p_gen.add_argument("--data-dir", default="data", help="Output directory for CSVs")
    p_gen.add_argument("--seasons", type=int, default=2)
    p_gen.add_argument("--matchdays", type=int, default=38)
    p_gen.add_argument("--anomaly-rate", type=float, default=0.05)
    p_gen.add_argument(
        "--no-corruption",
        action="store_true",
        help="Skip sensor corruption layer (faster, cleaner data)",
    )
    p_gen.add_argument(
        "--quiet", action="store_true", help="Suppress per-position stats table"
    )

    # ── train ─────────────────────────────────────────────────────────────────
    p_tr = sub.add_parser("train", help="Train model and save checkpoint")
    p_tr.add_argument(
        "--data-source",
        choices=["synthetic", "kinexon"],
        default="synthetic",
        help="synthetic: five-CSV generated dataset (regression-testing path, "
             "unchanged default). kinexon: real UWB tracking export via "
             "KinexonAdapter -> KinexonResampler -> gap-aware windowing -> "
             "BaselineBuilder.compute_with_fallback() (preferred for production "
             "runs against real session data).",
    )
    p_tr.add_argument("--data-dir", default="data", help="CSV/Kinexon-export source directory")
    p_tr.add_argument(
        "--model-dir", default="models", help="Checkpoint output directory"
    )
    p_tr.add_argument(
        "--sessions-per-player",
        type=int,
        default=60,
        help="Max sessions per player loaded for training (synthetic path only)",
    )
    p_tr.add_argument(
        "--session-id",
        default="3387",
        help="Kinexon session identifier to train on (kinexon path only; "
             "validated against the real session 3387 export so far)",
    )
    p_tr.add_argument(
        "--use-event-features",
        action="store_true",
        default=False,
        help="kinexon path only. Extend the 8 positions.csv-derived sequence "
             "features with 24 window-aggregated events.csv features "
             "(acceleration/deceleration/sprint/jump/change-of-direction/"
             "possession/pass/shot). Default False keeps the original "
             "8-feature model byte-for-byte reproducible.",
    )
    p_tr.add_argument(
        "--save-serve-state",
        action="store_true",
        help="Save serve state (baselines, thresholds) for faster serving",
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    p_ev = sub.add_parser("evaluate", help="Score model against ground truth / real-data diagnostics")
    p_ev.add_argument(
        "--data-source",
        choices=["synthetic", "kinexon"],
        default="synthetic",
        help="synthetic: ROC-AUC/PR-AUC/precision@k against ground_truth_labels.csv "
             "(unchanged default). kinexon: descriptive loss/confidence/calibration "
             "diagnostics over real model output -- no classification metrics, since "
             "real Kinexon sessions carry no ground-truth anomaly labels.",
    )
    p_ev.add_argument("--data-dir", default="data", help="CSV/Kinexon-export source directory")
    p_ev.add_argument("--model-dir", default="models", help="Checkpoint directory")
    p_ev.add_argument("--out", default="metrics/eval.json", help="Metrics output file")
    p_ev.add_argument(
        "--session-id",
        default="3387",
        help="Kinexon session identifier to evaluate (kinexon path only)",
    )
    p_ev.add_argument(
        "--use-event-features",
        action="store_true",
        default=False,
        help="kinexon path only. Must match the value used for --use-event-features "
             "at train time -- evaluating an 8-feature model with this on (or "
             "vice versa) will mismatch the trained input dimensionality.",
    )
    p_ev.add_argument(
        "--min-auc",
        type=float,
        default=0.60,
        help="Minimum acceptable mean ROC-AUC (exit 3 if below; synthetic path only)",
    )

    # ── serve ─────────────────────────────────────────────────────────────────
    p_sv = sub.add_parser(
        "serve", help="Stream inference: stdin→events, stdout→alerts (NDJSON)"
    )
    p_sv.add_argument("--model-dir", default="models", help="Checkpoint directory")
    p_sv.add_argument(
        "--min-alert-windows",
        type=int,
        default=3,
        help="Consecutive anomalous windows before emitting alert",
    )
    p_sv.add_argument(
        "--max-latency-ms",
        type=int,
        default=200,
        help="SLA threshold; violations are logged as warnings",
    )
    p_sv.add_argument(
        "--ignore-time-gaps",
        action="store_true",
        default=False,
        help="Disable temporal gap reset in the accumulator. Use for batch/replay data "
        "where inter-event gaps are expected (e.g. one row per session).",
    )
    p_sv.add_argument(
        "--ignore-session-boundaries",
        action="store_true",
        default=False,
        help="Disable session-boundary buffer reset in the accumulator. Use for "
        "historical replay where events from multiple sessions are interleaved.",
    )
    p_sv.add_argument(
        "--replay-mode",
        action="store_true",
        default=False,
        help="Replay-safe mode. Implies --ignore-time-gaps and --ignore-session-boundaries. "
        "Historical replay streams frequently interleave events from distinct source "
        "sessions; raw session_id transitions in these streams do not represent live "
        "continuity boundaries and must not trigger accumulator resets.",
    )

    # ── audit ─────────────────────────────────────────────────────────────────
    p_au = sub.add_parser("audit", help="Fairness audit + recalibration check")
    p_au.add_argument(
        "--log",
        default="logs/inference_log.jsonl",
        help="Path to inference log (NDJSON or JSON array)",
    )
    p_au.add_argument(
        "--data-dir", default="data", help="CSV directory for player metadata"
    )
    p_au.add_argument("--out", default="metrics/audit.json", help="Audit report output")

    # ── ingest ────────────────────────────────────────────────────────────────
    p_in = sub.add_parser(
        "ingest", help="Multi-match dataset pipeline: scan data/match_*/, validate, build unified Parquet datasets"
    )
    p_in.add_argument("--data-dir", default="data", help="Root directory containing match_<id>/ subdirectories")
    p_in.add_argument(
        "--output-dir", default="data/processed",
        help="Output directory for matches/players/events/positions.parquet and reports",
    )

    return parser


def main() -> None:
    # Windows consoles often default stdout to a legacy codepage (cp1252),
    # which raises UnicodeEncodeError on any non-Latin-1 character -- e.g.
    # FairnessMonitor.generate_audit_report()'s "⚠" warning marker, printed
    # by cmd_audit whenever a fairness alert actually fires (encountered
    # while verifying `audit` against real-data metadata; unrelated to
    # data-source and not previously surfaced because `audit` had
    # apparently never been run to completion with an active alert before).
    # reconfigure() is a no-op-safe best effort -- swallow failures (e.g.
    # stdout already redirected to a pipe that doesn't support it).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(args.log_level)
    logger.info("Players Data pipeline | command=%s", args.command)

    dispatch = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
        "audit": cmd_audit,
        "ingest": cmd_ingest,
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
