"""
publish_player_workload.py

Publishes coach-facing, MODEL-FREE player workload metrics onto the
analytics.player_workload Redis Stream, for Backend's existing
AnalyticsBridgeService / SSE relay / Frontend "Player Analytics" dashboard
to consume.

This is a pure data-pipeline publisher: it loads real Kinexon session data
(KinexonAdapter -> KinexonResampler -> kinexon_events_features merge) and
computes analysis/player_workload.py's metrics. It does NOT train, load, or
call SharedBackboneAutoencoder, does NOT compute reconstruction_loss /
confidence / SHAP / anomaly scores, and does NOT touch calibration. Compare
with scripts/publish_pilot_analytics.py, which is the model-output publisher
for the separate analytics.players stream -- this script and that one never
publish to the same topic.

Run:
    python scripts/publish_player_workload.py [--data-dir data] [--session-id 3387]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import CONFIG, OWNERSHIP_SCM
from config.redis_client import RedisStreamProducer, StreamTopics
from analysis.player_workload import compute_player_workload_windows, assign_workload_status
from analysis.player_workload_event import PlayerWorkloadEvent
from main import _load_kinexon_frames  # noqa: E402

LINE = "=" * 90
SUB = "-" * 90


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--session-id", default="3387")
    args = parser.parse_args()

    print(LINE)
    print(f"Publishing coach-facing player workload to {StreamTopics.ANALYTICS_PLAYER_WORKLOAD} "
          f"-- real session {args.session_id}")
    print(LINE)

    events_by_player, sessions_df, meta = _load_kinexon_frames(
        Path(args.data_dir), args.session_id, use_event_features=True,
    )

    # Coach-facing stream -- SC Magdeburg's own roster only. Every Kinexon
    # export also carries the opposing team's full roster (meta[pid].
    # ownership == "OPPONENT"); never published here.
    n_opponent = sum(1 for pid in events_by_player if meta.get(pid) and meta[pid].ownership != OWNERSHIP_SCM)
    events_by_player = {
        pid: df for pid, df in events_by_player.items()
        if meta.get(pid) and meta[pid].ownership == OWNERSHIP_SCM
    }
    print(f"Roster: {len(events_by_player)} SCM players, {n_opponent} opponent players excluded")

    hi_threshold = CONFIG.kinexon.high_intensity_threshold_ms
    rows_by_player = {}
    for pid, df in events_by_player.items():
        rows = compute_player_workload_windows(pid, df, hi_threshold)
        if rows:
            rows_by_player[pid] = rows

    assign_workload_status(rows_by_player)

    producer = RedisStreamProducer()
    producer.ensure_stream(StreamTopics.ANALYTICS_PLAYER_WORKLOAD)

    n_published = 0
    n_failed = 0
    for pid, rows in rows_by_player.items():
        m = meta.get(pid)
        player_name = m.player_name if m else f"player_{pid}"
        position = m.position_label if m else "unknown"

        for row in rows:
            try:
                ts = row["ts"]
                if hasattr(ts, "to_pydatetime"):
                    ts = ts.to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                event = PlayerWorkloadEvent(
                    player_id=pid,
                    external_id=str(pid),
                    player_name=player_name,
                    position=position,
                    match_id=args.session_id,
                    timestamp=ts,
                    elapsed_s=row["elapsed_s"],
                    current_load=row["current_load"],
                    load_trend=row["load_trend"],
                    acceleration_load=row["acceleration_load"],
                    deceleration_load=row["deceleration_load"],
                    sprint_load=row["sprint_load"],
                    high_intensity_load=row["high_intensity_load"],
                    distance_covered=row["distance_covered"],
                    performance_trend=row["performance_trend"],
                    workload_status=row["workload_status"],
                )
                producer.publish_dataclass(StreamTopics.ANALYTICS_PLAYER_WORKLOAD, event)
                n_published += 1
            except Exception as exc:
                n_failed += 1
                print(f"  FAILED player={pid} elapsed_s={row.get('elapsed_s')}: {exc}")

        print(f"Player {pid} ({player_name}, {position}): {len(rows)} windows published")

    print(f"\n{SUB}\nDONE: {n_published} events published to "
          f"{StreamTopics.ANALYTICS_PLAYER_WORKLOAD}, {n_failed} failed\n{SUB}")


if __name__ == "__main__":
    main()
