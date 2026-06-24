"""
pilot_pipeline.py

Shared pilot-checkpoint pipeline logic for real session 3387 (Kinexon).
Single home for the loading/training/per-window-scoring code that used to
be duplicated across scripts/evaluate_pilot_model.py,
scripts/publish_pilot_analytics.py, and scripts/run_live_player_analytics.py.

Two ways to get a live, scoreable pipeline:
  build_pipeline_and_train() -- fits a fresh SharedBackboneAutoencoder
      (genuine retraining; used by `main.py train`'s pilot path and by
      evaluation/comparison tools that need a freshly-trained model).
  build_pipeline_and_load()  -- loads an already-promoted checkpoint with
      no fitting (used by `main.py publish`; the backbone's weights are
      the checkpoint's, unmodified -- only per-player threshold
      calibration is recomputed, since that state isn't persisted to disk).

Both return a pipeline with identical player registration / baseline /
eligibility logic, so callers get the same player set regardless of which
path they use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import CONFIG, SEQUENCE_FEATURE_NAMES as _SFN
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler
from analysis.gap_aware_windowing import detect_window_gaps
from analysis.player_analytics_event import to_pilot_player_analytics_event

_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "statistics.csv"
SESSION_ID = "3387"
SESSION_MATCH_ID = "3387"
TOP_N_SHAP = 3

WINDOW_STEPS = CONFIG.window.window_steps
STRIDE = WINDOW_STEPS
GAP_THRESHOLD_S = CONFIG.window.gap_threshold_s

IDX = {n: i for i, n in enumerate(_SFN)}


def load_session_data_and_pipeline(use_event_features: bool = False):
    """Loads session 3387's real Kinexon data, computes baselines, determines
    eligible players, and returns a pipeline with players/historical data
    registered but no shared model trained or loaded yet.

    use_event_features=False (default, unchanged): events_by_player has only
    the 8 columns KinexonResampler.resample() produces directly.

    use_event_features=True: additionally merges the 24 window-aggregated
    events.csv features (ingestion/kinexon_events_features.py), matching the
    full N_SEQUENCE_FEATURES input the promoted checkpoint is trained on.
    """
    from analysis.orchestrator import PlayersDataAnalysisPipeline

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(DATA_DIR / "statistics.csv")
    observations = list(
        adapter.stream_positions(DATA_DIR / "positions.csv", meta, session_id=SESSION_ID, match_id=SESSION_ID)
    )
    resampler = KinexonResampler()
    events_by_player, sessions_df = resampler.resample(observations, session_id=SESSION_ID)

    if use_event_features:
        from ingestion.kinexon_events_features import merge_event_features
        events_by_player = merge_event_features(
            events_by_player=events_by_player,
            events_csv_path=DATA_DIR / "events.csv",
            real_player_ids=meta.keys(),
            bucket_seconds=resampler.bucket_seconds,
        )

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

    return pipeline, events_by_player, sessions_df, meta, eligible


def discover_match_dirs(data_dir: Optional[Path] = None) -> list:
    """Auto-discovers every data/match_<id>/ directory with both
    positions.csv and statistics.csv -- no hardcoded session IDs. Used by
    load_multi_match_pipeline() so training automatically picks up every
    match main.py ingest has organized, including ones added after this
    code was written."""
    data_dir = data_dir or DATA_DIR
    return sorted(
        p for p in data_dir.glob("match_*")
        if p.is_dir() and (p / "positions.csv").exists() and (p / "statistics.csv").exists()
    )


def load_multi_match_pipeline(
    match_dirs: Optional[list] = None, use_event_features: bool = False,
    include_ownership: tuple = None,
):
    """Loads and merges EVERY discovered real Kinexon match (own roster +
    opponents) into one pipeline -- the model-training path.

    Each player's events/sessions are concatenated across every match they
    appear in: SC Magdeburg's own roster keeps the same Kinexon player_id
    match-to-match (real multi-session history), while each opponent
    club's roster occupies its own distinct player_id range (verified
    empirically -- no cross-match identity collisions), so merging by
    player_id is safe without any team-aware special-casing.

    include_ownership (e.g. (OWNERSHIP_SCM,)) restricts the returned
    eligible list to that subset -- training itself always passes None
    (no restriction: ALL players, by design, maximizes real training data).

    Returns (pipeline, events_by_player, sessions_df, meta, eligible,
    ownership, match_ids_used) where ownership is {player_id: "SCM"|
    "OPPONENT"} and match_ids_used is the list of session_ids actually
    loaded."""
    from collections import defaultdict
    from analysis.orchestrator import PlayersDataAnalysisPipeline
    from analysis.baseline import BaselineBuilder
    from config.settings import classify_ownership

    match_dirs = match_dirs if match_dirs is not None else discover_match_dirs()
    if not match_dirs:
        raise RuntimeError(
            "No data/match_<id>/ directories with positions.csv + statistics.csv found -- "
            "run `python main.py ingest` first."
        )

    per_player_events: dict = defaultdict(list)
    per_player_sessions: dict = defaultdict(list)
    meta_by_pid: dict = {}
    ownership: dict = {}
    match_ids_used: list = []

    for match_dir in match_dirs:
        match_id = match_dir.name[len("match_"):]
        adapter = KinexonAdapter()
        m = adapter.load_player_meta(match_dir / "statistics.csv")
        observations = list(
            adapter.stream_positions(match_dir / "positions.csv", m, session_id=match_id, match_id=match_id)
        )
        if not observations:
            continue

        resampler = KinexonResampler()
        events_by_player, sessions_df = resampler.resample(observations, session_id=match_id)

        if use_event_features and (match_dir / "events.csv").exists():
            from ingestion.kinexon_events_features import merge_event_features
            events_by_player = merge_event_features(
                events_by_player=events_by_player, events_csv_path=match_dir / "events.csv",
                real_player_ids=m.keys(), bucket_seconds=resampler.bucket_seconds,
            )

        match_ids_used.append(match_id)
        for pid, df in events_by_player.items():
            per_player_events[pid].append(df)
            meta_by_pid.setdefault(pid, m.get(pid))
            pm = m.get(pid)
            ownership[pid] = pm.ownership if pm and pm.ownership else classify_ownership(pm.group_name if pm else None)
        if "player_id" in sessions_df.columns:
            for pid in sessions_df["player_id"].unique():
                per_player_sessions[pid].append(sessions_df[sessions_df["player_id"] == pid])

    merged_events = {
        pid: pd.concat(dfs, ignore_index=True).sort_values("ts").reset_index(drop=True)
        for pid, dfs in per_player_events.items()
    }
    merged_sessions = (
        pd.concat([pd.concat(v, ignore_index=True) for v in per_player_sessions.values()], ignore_index=True)
        if per_player_sessions else pd.DataFrame()
    )

    builder = BaselineBuilder()
    baselines = {}
    for pid, df in merged_events.items():
        player_sessions = merged_sessions[merged_sessions["player_id"] == pid] if not merged_sessions.empty else merged_sessions
        profile = builder.compute_with_fallback(
            player_id=pid, external_id=str(pid), sessions_df=player_sessions, events_df=df, window_days=28,
        )
        if profile is not None:
            baselines[pid] = profile

    gap_counts = {}
    for pid, df in merged_events.items():
        gap_info = detect_window_gaps(df, WINDOW_STEPS, STRIDE, GAP_THRESHOLD_S)
        gap_counts[pid] = sum(1 for ok, _ in gap_info if ok)

    eligible = [pid for pid in merged_events if pid in baselines and gap_counts.get(pid, 0) > 0]
    if include_ownership is not None:
        eligible = [pid for pid in eligible if ownership.get(pid) in include_ownership]

    pipeline = PlayersDataAnalysisPipeline()
    for pid in eligible:
        pm = meta_by_pid.get(pid)
        pipeline.register_player(
            player_id=pid, external_id=str(pid),
            name=pm.player_name if pm and pm.player_name else f"player_{pid}",
            position=pm.position_label if pm else "unknown",
            age=25,
        )
        player_sessions = merged_sessions[merged_sessions["player_id"] == pid] if not merged_sessions.empty else merged_sessions
        pipeline.load_historical_data(player_id=pid, sessions_df=player_sessions, events_df=merged_events[pid])

    pipeline.compute_baselines(window_days=28, use_provisional_fallback=True)

    return pipeline, merged_events, merged_sessions, meta_by_pid, eligible, ownership, match_ids_used


def build_pipeline_and_train(use_event_features: bool = False):
    """Fits a fresh SharedBackboneAutoencoder on real session 3387 data
    (genuine retraining -- saves+promotes to models/shared_backbone.pt).

    use_event_features: False trains on the original 8 resampled columns
    only; True merges the 24 event-derived columns first, so the resulting
    checkpoint's encoder is N_SEQUENCE_FEATURES (32) wide."""
    pipeline, events_by_player, sessions_df, meta, eligible = load_session_data_and_pipeline(use_event_features)
    result = pipeline.train_all_models(use_gap_aware_windowing=True)
    return pipeline, events_by_player, sessions_df, meta, eligible, result


def build_multi_match_pipeline_and_train(
    match_dirs: Optional[list] = None, use_event_features: bool = False, include_ownership: tuple = None,
):
    """Multi-match counterpart of build_pipeline_and_train(): fits a fresh
    SharedBackboneAutoencoder across EVERY discovered real match (own +
    opponents, by default -- see load_multi_match_pipeline()). Saves +
    promotes to models/shared_backbone.pt exactly like the single-match
    path; only the training data's breadth changes.

    Returns (pipeline, events_by_player, sessions_df, meta, eligible,
    ownership, match_ids_used, result)."""
    pipeline, events_by_player, sessions_df, meta, eligible, ownership, match_ids_used = load_multi_match_pipeline(
        match_dirs, use_event_features, include_ownership,
    )
    result = pipeline.train_all_models(use_gap_aware_windowing=True)
    return pipeline, events_by_player, sessions_df, meta, eligible, ownership, match_ids_used, result


def build_pipeline_and_load(backbone_path=None, use_event_features: bool = False):
    """Loads the promoted shared backbone checkpoint (analysis.anomaly_
    detection.SharedBackboneAutoencoder.load(), no fitting) instead of
    retraining. Per-player threshold calibration is still recomputed against
    the loaded model -- it is genuinely not retraining: the backbone's
    weights come unmodified from backbone_path (default
    models/shared_backbone.pt, the promoted checkpoint), only the per-player
    RegimeAwareThresholdStore calibration state is (re)computed, exactly as
    it would have to be even right after a real train() call -- see
    InferenceEngine.load_and_calibrate()'s docstring."""
    backbone_path = Path(backbone_path) if backbone_path else _ROOT / "models" / "shared_backbone.pt"
    pipeline, events_by_player, sessions_df, meta, eligible = load_session_data_and_pipeline(use_event_features)
    result = pipeline.load_shared_model(backbone_path, use_gap_aware_windowing=True)
    return pipeline, events_by_player, sessions_df, meta, eligible, result


def windows_with_elapsed(df: pd.DataFrame):
    """Real elapsed_s (window start) for every GAP-FREE window, in the exact
    same order build_training_sequences_gap_aware() returns its (seq, mask)
    pairs."""
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


def score_window_and_build_event(
    *, pipeline, engine, clf, pid: int, seq, mask, elapsed_s: Optional[float],
    player_sessions: pd.DataFrame, meta: dict, model_version: str,
    match_id: str = SESSION_MATCH_ID, top_n_shap: int = TOP_N_SHAP,
):
    """Runs the real analyze_window() + SHAP pipeline on one already-built
    window and returns a ready-to-publish PilotPlayerAnalyticsEvent.

    Single home for the per-window scoring/event-construction logic
    previously duplicated between scripts/publish_pilot_analytics.py
    (historical batch replay) and scripts/run_live_player_analytics.py
    (continuous paced replay) -- both now call this via `main.py publish`'s
    --historical-replay / --continuous modes. Same analyze_window() call,
    same SHAP call, same z-score/event-field formulas as both originals;
    only the call site moved.
    """
    last = seq[-1]
    live_event = {
        "x_pitch": float(last[IDX["x_pitch"]]),
        "y_pitch": float(last[IDX["y_pitch"]]),
        "elapsed_seconds": float(elapsed_s) if elapsed_s is not None else 0.0,
        "_tvl_confidence": 1.0,
    }
    result = engine.analyze_window(
        player_id=pid, sequence=seq, mask=mask,
        live_event=live_event, sessions_df=player_sessions,
    )

    regime_key = clf.classify(seq).key
    tracker = engine.inference_engine.get_tracker(pid)
    tracker_source = engine.inference_engine.get_tracker_source(pid)
    threshold = (
        tracker.threshold_for(regime_key) if tracker and tracker.is_calibrated else float("inf")
    )

    p_record = pipeline.registry.get(pid)
    background = p_record.get("sequence_background") if p_record else None
    top_shap_features = []
    if background is not None and len(background) >= 2:
        shap_dict, _base_value, feature_values = pipeline.xai_layer._explain_sequence_shap(
            player_id=pid, model=engine._shared_model, sequence=seq, mask=mask,
            background=background, extra_features={},
        )
        top = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n_shap]
        top_shap_features = [
            {"feature": name, "value": round(float(val), 4),
             "raw_value": round(float(feature_values.get(name, 0.0)), 4)}
            for name, val in top
        ]

    window_distance_m = float(seq[:, IDX["distance_delta_m"]].sum())
    window_avg_speed_ms = float(seq[:, IDX["speed_ms"]].mean())
    window_sprint_ticks = int(seq[:, IDX["sprint_flag"]].sum())
    baseline = engine._baselines.get(pid)
    baseline_distance_z = baseline.zscore("distance", window_distance_m) if baseline else 0.0
    baseline_speed_z = baseline.zscore("top_speed", window_avg_speed_ms) if baseline else 0.0
    baseline_sprint_z = baseline.zscore("sprint_count", window_sprint_ticks) if baseline else 0.0

    session_total_distance_m = 0.0
    session_high_speed_distance_m = 0.0
    if not player_sessions.empty:
        session_row = player_sessions.iloc[0]
        session_total_distance_m = float(session_row.get("total_distance_m", 0.0))
        session_high_speed_distance_m = float(session_row.get("high_speed_distance_m", 0.0))

    m = meta.get(pid)
    player_name = m.player_name if m else f"player_{pid}"
    position = m.position_label if m else "unknown"

    return to_pilot_player_analytics_event(
        result, player_name=player_name, position=position, threshold=threshold,
        top_shap_features=top_shap_features, model_version=model_version, regime=regime_key,
        tracker_source=tracker_source, match_id=match_id,
        window_distance_m=window_distance_m, window_avg_speed_ms=window_avg_speed_ms,
        window_sprint_ticks=window_sprint_ticks, baseline_distance_z=baseline_distance_z,
        baseline_speed_z=baseline_speed_z, baseline_sprint_z=baseline_sprint_z,
        session_total_distance_m=session_total_distance_m,
        session_high_speed_distance_m=session_high_speed_distance_m,
    )
