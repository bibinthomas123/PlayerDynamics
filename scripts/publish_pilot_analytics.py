"""
publish_pilot_analytics.py

Publishes real PlayerDynamics pilot outputs (session 3387, the PROMOTED
checkpoint at models/shared_backbone.pt) onto the analytics.players Redis
Stream as PilotPlayerAnalyticsEvent entries, for Backend's existing
AnalyticsBridgeService / SSE relay / Frontend "Player Analytics" tab to
consume -- no Backend or Frontend-side publishing code involved; this is
the sole producer.

Does NOT publish is_anomaly or alert_level (see PilotPlayerAnalyticsEvent's
docstring) -- those remain experimental. Publishes, per real window, per
eligible player: reconstruction_loss, confidence, threshold,
raw_threshold_breach, and the real top-3 SHAP features for that window.

LOADS the promoted checkpoint (scripts/evaluate_pilot_model.py's
_build_pipeline_and_load(use_event_features=True), which calls
PlayersDataAnalysisPipeline.load_shared_model() -> SharedBackboneAutoencoder.
load() -- no fitting). use_event_features=True merges the 24 event-derived
columns onto the resampled data, matching the model's actual 32-feature
input (same loader scripts/run_live_player_analytics.py uses) -- previously
this script ran the promoted checkpoint on only 8 of its 32 trained inputs.
Per-player threshold calibration is still recomputed against the loaded
model, since that state is not persisted to disk, but the backbone's
weights are the promoted checkpoint's, unmodified. This script remains a
pure publisher: AnomalyResult/SHAP -> PilotPlayerAnalyticsEvent -> XADD.

Run:
    python scripts/publish_pilot_analytics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import CONFIG
from config.redis_client import RedisStreamProducer, StreamTopics
from analysis.gap_aware_windowing import build_training_sequences_gap_aware
from analysis.regime import SessionRegimeClassifier
from analysis.player_analytics_event import to_pilot_player_analytics_event
from evaluate_pilot_model import _build_pipeline_and_load, _windows_with_elapsed, _IDX  # noqa: E402

LINE = "=" * 90
SUB = "-" * 90
SESSION_MATCH_ID = "3387"
TOP_N_SHAP = 3


def main() -> None:
    print(LINE)
    print("Publishing PILOT player analytics to analytics.players -- real session 3387")
    print(LINE)

    # The promoted checkpoint's LSTM input is 32 features (8 resampled +
    # 24 event-derived) -- matches scripts/run_live_player_analytics.py.
    pipeline, events_by_player, sessions_df, meta, eligible, load_result = _build_pipeline_and_load(
        use_event_features=True,
    )
    engine = pipeline.pattern_engine
    model_version = engine._shared_model.model_version if engine._shared_model else "unknown"
    print(f"\nLoaded promoted checkpoint (not retrained): {load_result['shared_model']} "
          f"(model_version={model_version})")

    producer = RedisStreamProducer()
    producer.ensure_stream(StreamTopics.ANALYTICS_PLAYERS)

    clf = SessionRegimeClassifier()
    n_published = 0
    n_failed = 0

    for pid in eligible:
        df = events_by_player[pid].sort_values("ts").reset_index(drop=True)
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        windows, _audit = build_training_sequences_gap_aware(
            pattern_engine=engine, events_df=df, sessions_df=player_sessions,
        )
        elapsed_starts = _windows_with_elapsed(df)
        if len(elapsed_starts) != len(windows):
            elapsed_starts = [None] * len(windows)

        m = meta.get(pid)
        player_name = m.player_name if m else f"player_{pid}"
        position = m.position_label if m else "unknown"

        p_record = pipeline.registry.get(pid)
        background = p_record.get("sequence_background") if p_record else None
        baseline = engine._baselines.get(pid)

        # Real session-level totals (KinexonResampler._session_summary_row()),
        # constant across this player's events for this session -- not a
        # per-window measurement, just session context repeated on the wire.
        session_total_distance_m = 0.0
        session_high_speed_distance_m = 0.0
        if not player_sessions.empty:
            session_row = player_sessions.iloc[0]
            session_total_distance_m = float(session_row.get("total_distance_m", 0.0))
            session_high_speed_distance_m = float(session_row.get("high_speed_distance_m", 0.0))

        for w_idx, ((seq, mask, _sid), elapsed) in enumerate(zip(windows, elapsed_starts)):
            last = seq[-1]
            live_event = {
                "x_pitch": float(last[_IDX["x_pitch"]]),
                "y_pitch": float(last[_IDX["y_pitch"]]),
                "elapsed_seconds": float(elapsed) if elapsed is not None else 0.0,
                "_tvl_confidence": 1.0,
            }

            # Real per-window telemetry: sum/mean of the REAL per-tick values
            # across all window_steps timesteps of this window's own sequence
            # array (not just the last tick, and not a new model output --
            # plain arithmetic over already-real window data).
            window_distance_m = float(seq[:, _IDX["distance_delta_m"]].sum())
            window_avg_speed_ms = float(seq[:, _IDX["speed_ms"]].mean())
            window_sprint_ticks = int(seq[:, _IDX["sprint_flag"]].sum())

            baseline_distance_z = baseline.zscore("distance", window_distance_m) if baseline else 0.0
            baseline_speed_z = baseline.zscore("top_speed", window_avg_speed_ms) if baseline else 0.0
            baseline_sprint_z = baseline.zscore("sprint_count", window_sprint_ticks) if baseline else 0.0

            try:
                result = engine.analyze_window(
                    player_id=pid, sequence=seq, mask=mask,
                    live_event=live_event, sessions_df=player_sessions,
                )

                regime_key = clf.classify(seq).key
                tracker = engine.inference_engine.get_tracker(pid)
                tracker_source = engine.inference_engine.get_tracker_source(pid)
                threshold = (
                    tracker.threshold_for(regime_key)
                    if tracker and tracker.is_calibrated
                    else float("inf")
                )

                top_shap_features = []
                if background is not None and len(background) >= 2:
                    shap_dict, _base_value, feature_values = pipeline.xai_layer._explain_sequence_shap(
                        player_id=pid, model=engine._shared_model, sequence=seq, mask=mask,
                        background=background, extra_features={},
                    )
                    top = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N_SHAP]
                    top_shap_features = [
                        {"feature": name, "value": round(float(val), 4),
                         "raw_value": round(float(feature_values.get(name, 0.0)), 4)}
                        for name, val in top
                    ]

                event = to_pilot_player_analytics_event(
                    result,
                    player_name=player_name,
                    position=position,
                    threshold=threshold,
                    top_shap_features=top_shap_features,
                    model_version=model_version,
                    regime=regime_key,
                    tracker_source=tracker_source,
                    match_id=SESSION_MATCH_ID,
                    window_distance_m=window_distance_m,
                    window_avg_speed_ms=window_avg_speed_ms,
                    window_sprint_ticks=window_sprint_ticks,
                    baseline_distance_z=baseline_distance_z,
                    baseline_speed_z=baseline_speed_z,
                    baseline_sprint_z=baseline_sprint_z,
                    session_total_distance_m=session_total_distance_m,
                    session_high_speed_distance_m=session_high_speed_distance_m,
                )
                producer.publish_dataclass(StreamTopics.ANALYTICS_PLAYERS, event)
                n_published += 1
            except Exception as exc:
                n_failed += 1
                print(f"  FAILED player={pid} window={w_idx}: {exc}")

        print(f"Player {pid} ({player_name}, {position}): {len(windows)} windows published")

    print(f"\n{SUB}\nDONE: {n_published} events published to {StreamTopics.ANALYTICS_PLAYERS}, "
          f"{n_failed} failed\n{SUB}")


if __name__ == "__main__":
    main()
