"""
run_live_player_analytics.py

Continuously-running production process for analytics.players. Replaces
scripts/publish_pilot_analytics.py's one-shot batch pattern (load everything,
score all 319 windows, publish all of them, exit) with an incremental,
continuously-running pipeline: real per-tick rows from session 3387 are fed
one at a time, in chronological order across all players -- exactly as a
live Kinexon feed would arrive -- through the SAME LiveWindowAccumulator
class main.py's cmd_serve already uses for live inference. Each time a
player's window completes, this process runs ONE real inference through the
already-loaded model and publishes immediately, instead of waiting for an
entire batch to finish.

No real Kinexon hardware feed exists yet in this codebase (see PHASE 1 of
the architecture audit this script was built from -- analytics.tracking
events has no producer). This script is therefore a paced REPLAY of real
recorded session data, not a connection to a stadium feed -- but it is a
genuine long-running process, not a batch script: the checkpoint loads once
at startup, inference runs incrementally as windows complete, and nothing
is retrained or reloaded per event.

Also publishes analytics.player_workload per tick (scripts/
publish_player_workload.py's existing, unchanged, model-free aggregation
logic) immediately before the same tick is pushed into the window
accumulator -- mirroring the "new workload event arrives -> model runs"
causal chain requested for verification, within one process instead of two
independently-paced batch scripts racing over the same source data.

Run:
    python scripts/run_live_player_analytics.py [--tick-interval-seconds 0.2] [--max-ticks N]

Stop with Ctrl+C (SIGINT) -- finishes the current tick, then exits cleanly.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import CONFIG
from config.redis_client import RedisStreamProducer, StreamTopics
from analysis.regime import SessionRegimeClassifier
from analysis.player_workload import compute_player_workload_windows, assign_workload_status
from analysis.player_workload_event import PlayerWorkloadEvent
from analysis.player_analytics_event import to_pilot_player_analytics_event
from evaluate_pilot_model import _build_pipeline_and_load, _IDX  # noqa: E402

LINE = "=" * 90
SUB = "-" * 90
SESSION_MATCH_ID = "3387"
TOP_N_SHAP = 3

_running = True


def _shutdown(signum, _frame):
    global _running
    print(f"\nReceived signal {signum} -- finishing current tick then shutting down.")
    _running = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tick-interval-seconds", type=float, default=0.2,
                         help="Pacing between ticks -- this is a paced replay, not an instant batch dump.")
    parser.add_argument("--max-ticks", type=int, default=None,
                         help="Stop after N ticks (for verification runs). Default: process all available ticks.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(LINE)
    print("LIVE player analytics pipeline -- real session 3387, promoted checkpoint, no retraining")
    print(LINE)

    # ── Startup: load real data (FULL 32-feature set) + the PROMOTED ─────────
    # checkpoint, ONCE. Never reloaded, never retrained for the rest of this
    # process's lifetime.
    pipeline, events_by_player, sessions_df, meta, eligible, load_result = _build_pipeline_and_load(
        use_event_features=True,
    )
    engine = pipeline.pattern_engine
    model_version = engine._shared_model.model_version if engine._shared_model else "unknown"
    print(f"\nLoaded promoted checkpoint (once, not retrained): {load_result['shared_model']} "
          f"(model_version={model_version})")

    # ── Per-player workload rows, precomputed once (pure aggregation over ──
    # already-real data, identical to publish_player_workload.py -- only the
    # PACING of publishing them is new, not the computation).
    hi_threshold = CONFIG.kinexon.high_intensity_threshold_ms
    dfs_by_player = {pid: events_by_player[pid].sort_values("ts").reset_index(drop=True) for pid in eligible}
    workload_rows_by_player = {}
    for pid, df in dfs_by_player.items():
        rows = compute_player_workload_windows(pid, df, hi_threshold)
        if rows:
            workload_rows_by_player[pid] = rows
    assign_workload_status(workload_rows_by_player)

    # ── Build the global, chronologically-ordered tick sequence across all ──
    # players -- this is what "ticks arriving live, interleaved across
    # players" looks like when replayed from real recorded data.
    ticks = []
    for pid, df in dfs_by_player.items():
        for row_idx in range(len(df)):
            ticks.append((df["ts"].iloc[row_idx], pid, row_idx))
    ticks.sort(key=lambda t: t[0])
    if args.max_ticks:
        ticks = ticks[: args.max_ticks]
    print(f"Replaying {len(ticks)} real ticks across {len(eligible)} players, "
          f"tick_interval={args.tick_interval_seconds}s")

    # ── The dedicated player-sequence buffer (PHASE 3 finding: reused as-is, ─
    # no new buffering logic) -- configured to the model's REAL window_steps
    # (8), not cmd_serve's mismatched hardcoded 24/24.
    from analysis.live_window_accumulator import LiveWindowAccumulator
    accumulator = LiveWindowAccumulator(
        window_size=CONFIG.window.window_steps,
        stride=CONFIG.window.window_steps,
    )

    producer = RedisStreamProducer()
    producer.ensure_stream(StreamTopics.ANALYTICS_PLAYER_WORKLOAD)
    producer.ensure_stream(StreamTopics.ANALYTICS_PLAYERS)
    clf = SessionRegimeClassifier()

    n_workload_published = 0
    n_players_published = 0

    for ts, pid, row_idx in ticks:
        if not _running:
            break

        m = meta.get(pid)
        player_name = m.player_name if m else f"player_{pid}"
        position = m.position_label if m else "unknown"
        df = dfs_by_player[pid]
        row = df.iloc[row_idx]
        player_sessions = sessions_df[sessions_df["player_id"] == pid]

        # ── Step 1: publish the model-free workload tick (unchanged logic) ──
        wl_row = workload_rows_by_player.get(pid, [None] * len(df))[row_idx] if pid in workload_rows_by_player else None
        if wl_row is not None:
            wl_ts = wl_row["ts"]
            if hasattr(wl_ts, "to_pydatetime"):
                wl_ts = wl_ts.to_pydatetime()
            if wl_ts.tzinfo is None:
                wl_ts = wl_ts.replace(tzinfo=timezone.utc)
            workload_event = PlayerWorkloadEvent(
                player_id=pid, external_id=str(pid), player_name=player_name, position=position,
                match_id=SESSION_MATCH_ID, timestamp=wl_ts, elapsed_s=wl_row["elapsed_s"],
                current_load=wl_row["current_load"], load_trend=wl_row["load_trend"],
                acceleration_load=wl_row["acceleration_load"], deceleration_load=wl_row["deceleration_load"],
                sprint_load=wl_row["sprint_load"], high_intensity_load=wl_row["high_intensity_load"],
                distance_covered=wl_row["distance_covered"], performance_trend=wl_row["performance_trend"],
                workload_status=wl_row["workload_status"],
            )
            producer.publish_dataclass(StreamTopics.ANALYTICS_PLAYER_WORKLOAD, workload_event)
            n_workload_published += 1

        # ── Step 2: feed the SAME tick into the player's window buffer ──────
        live_tick = row.to_dict()
        window = accumulator.push(player_id=str(pid), event=live_tick)

        if window is not None:
            # ── Step 3: a window just completed -- run REAL inference now, ──
            # using the model loaded once at startup. No reload, no retrain.
            seq, mask = engine.window_builder.build_live_window(window)

            last = seq[-1]
            live_event = {
                "x_pitch": float(last[_IDX["x_pitch"]]),
                "y_pitch": float(last[_IDX["y_pitch"]]),
                "elapsed_seconds": float(window[-1].get("elapsed_s", 0.0)),
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
                top = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N_SHAP]
                top_shap_features = [
                    {"feature": name, "value": round(float(val), 4),
                     "raw_value": round(float(feature_values.get(name, 0.0)), 4)}
                    for name, val in top
                ]

            window_distance_m = float(seq[:, _IDX["distance_delta_m"]].sum())
            window_avg_speed_ms = float(seq[:, _IDX["speed_ms"]].mean())
            window_sprint_ticks = int(seq[:, _IDX["sprint_flag"]].sum())
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

            event = to_pilot_player_analytics_event(
                result, player_name=player_name, position=position, threshold=threshold,
                top_shap_features=top_shap_features, model_version=model_version, regime=regime_key,
                tracker_source=tracker_source, match_id=SESSION_MATCH_ID,
                window_distance_m=window_distance_m, window_avg_speed_ms=window_avg_speed_ms,
                window_sprint_ticks=window_sprint_ticks, baseline_distance_z=baseline_distance_z,
                baseline_speed_z=baseline_speed_z, baseline_sprint_z=baseline_sprint_z,
                session_total_distance_m=session_total_distance_m,
                session_high_speed_distance_m=session_high_speed_distance_m,
            )
            producer.publish_dataclass(StreamTopics.ANALYTICS_PLAYERS, event)
            n_players_published += 1

            top_feat = top_shap_features[0]["feature"] if top_shap_features else "n/a"
            print(f"[{result.ts.isoformat()}] player={pid} ({player_name}) window_end_ts={ts} "
                  f"model_version={model_version} reconstruction_loss={result.anomaly_score:.4f} "
                  f"confidence={result.confidence:.4f} top_shap={top_feat} -> analytics.players "
                  f"(published #{n_players_published})")

        time.sleep(args.tick_interval_seconds)

    print(f"\n{SUB}\nSTOPPED: {n_workload_published} workload ticks published, "
          f"{n_players_published} model predictions published to {StreamTopics.ANALYTICS_PLAYERS}\n{SUB}")


if __name__ == "__main__":
    main()
