"""
publish_pilot_analytics.py

Publishes real PlayerDynamics pilot outputs (session 3387, model_version
shared_*) onto the analytics.players Redis Stream as PilotPlayerAnalyticsEvent
entries, for Backend's existing AnalyticsBridgeService / SSE relay /
Frontend "Player Analytics" tab to consume -- no Backend or Frontend-side
publishing code involved; this is the sole producer.

Does NOT publish is_anomaly or alert_level (see PilotPlayerAnalyticsEvent's
docstring) -- those remain experimental. Publishes, per real window, per
eligible player: reconstruction_loss, confidence, threshold,
raw_threshold_breach, and the real top-3 SHAP features for that window.

Reuses the exact training/scoring path validated over the last several
turns (scripts/evaluate_pilot_model.py) -- trains the model once, computes
nothing new, introduces no new threshold or calibration logic. This script
is a pure publisher: AnomalyResult/SHAP -> PilotPlayerAnalyticsEvent -> XADD.

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
from evaluate_pilot_model import _build_pipeline_and_train, _windows_with_elapsed, _IDX  # noqa: E402

LINE = "=" * 90
SUB = "-" * 90
SESSION_MATCH_ID = "3387"
TOP_N_SHAP = 3


def main() -> None:
    print(LINE)
    print("Publishing PILOT player analytics to analytics.players -- real session 3387")
    print(LINE)

    pipeline, events_by_player, sessions_df, meta, eligible, train_result = _build_pipeline_and_train()
    engine = pipeline.pattern_engine
    model_version = engine._shared_model.model_version if engine._shared_model else "unknown"
    print(f"\nTrained: {train_result['shared_model']} (model_version={model_version})")

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

        for w_idx, ((seq, mask, _sid), elapsed) in enumerate(zip(windows, elapsed_starts)):
            last = seq[-1]
            live_event = {
                "x_pitch": float(last[_IDX["x_pitch"]]),
                "y_pitch": float(last[_IDX["y_pitch"]]),
                "elapsed_seconds": float(elapsed) if elapsed is not None else 0.0,
                "_tvl_confidence": 1.0,
            }
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
