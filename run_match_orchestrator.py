"""
run_match_orchestrator.py — PlayerDynamics

Closes the gap identified in PRODUCTION_READINESS_AUDIT.md §2 ("Runtime
Verification"): MatchOrchestrator existed as a fully tested class, but
nothing in this codebase ever instantiated it in a running process. This
script is that process -- the actual entrypoint that wires
MatchOrchestrator to Redis Streams and runs it continuously for one match.

Per the ownership rule (see BACKEND_INTEGRATION_IMPLEMENTATION.md §1):
  - PlayerDynamics owns Kinexon ingestion directly. This script never reads
    Kinexon data from Backend/Redis -- only from a local file (--events-csv),
    via the existing KinexonTacticalEventAdapter. (A live Kinexon feed
    adapter publishing onto the PlayerDynamics-internal tracking.events
    stream is a separate, not-yet-built piece of work -- this script
    already supports consuming that stream too, the moment one exists.)
  - match.events / match.context are consumed FROM Backend, never computed
    here -- see MatchOrchestrator.consume_match_events()/consume_match_context().
  - analytics.* are published TO Backend, on every tick.

Usage:
    python run_match_orchestrator.py --match-id 3387 \
        [--events-csv data/events.csv] [--player-meta data/statistics.csv] \
        [--tick-interval-seconds 5] [--consumer-name playerdynamics-1]

Stop with Ctrl+C (SIGINT) or SIGTERM -- both trigger finalize() + a final
publish before exiting, so nothing still "tail"/provisional is lost.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from analysis.match_orchestrator import MatchOrchestrator
from config.redis_client import RedisStreamConsumer, RedisStreamProducer, StreamTopics, check_redis_connection

logger = logging.getLogger("run_match_orchestrator")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True, help="Match identifier shared with Backend.")
    parser.add_argument("--events-csv", default=None, help="Optional Kinexon events.csv to backfill the TacticalEvent buffer from at startup.")
    parser.add_argument("--player-meta", default=None, help="Optional Kinexon statistics.csv for team_id resolution (passed to KinexonTacticalEventAdapter).")
    parser.add_argument("--tick-interval-seconds", type=float, default=5.0, help="How often to recompute the pipeline and publish results.")
    parser.add_argument("--consumer-name", default="playerdynamics-1", help="Stable consumer name for Redis consumer groups -- MUST stay the same across restarts for crash-recovery replay to work.")
    parser.add_argument("--read-count", type=int, default=200, help="Max entries to read per stream per tick.")
    return parser.parse_args()


def _load_backfill_events(events_csv: Optional[str], player_meta_csv: Optional[str], match_id: str) -> list:
    if not events_csv:
        return []
    from ingestion.tactical_event import KinexonTacticalEventAdapter

    player_meta = None
    if player_meta_csv:
        from ingestion.kinexon_adapter import KinexonAdapter
        player_meta = KinexonAdapter().load_player_meta(Path(player_meta_csv))

    adapter = KinexonTacticalEventAdapter()
    events = list(adapter.parse(Path(events_csv), player_meta=player_meta, match_id=match_id))
    logger.info("Backfilled %d TacticalEvents from %s", len(events), events_csv)
    return events


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _parse_args()

    if not check_redis_connection():
        logger.error("Cannot reach Redis (REDIS_HOST/REDIS_PORT) -- aborting startup. "
                     "This process has nothing useful to do without a broker.")
        return 1

    orchestrator = MatchOrchestrator(match_id=args.match_id)
    for event in _load_backfill_events(args.events_csv, args.player_meta, args.match_id):
        orchestrator.ingest_event(event)

    producer = RedisStreamProducer()
    group = "playerdynamics-runtime"
    tracking_consumer = RedisStreamConsumer(StreamTopics.TRACKING_EVENTS, group=group, consumer_name=args.consumer_name)
    match_events_consumer = RedisStreamConsumer(StreamTopics.MATCH_EVENTS, group=group, consumer_name=args.consumer_name)
    match_context_consumer = RedisStreamConsumer(StreamTopics.MATCH_CONTEXT, group=group, consumer_name=args.consumer_name)

    running = True

    def _shutdown(signum, _frame):
        nonlocal running
        logger.info("Received signal %s -- finishing current tick then shutting down.", signum)
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("MatchOrchestrator runtime started for match_id=%s", args.match_id)
    while running:
        n_tracking = orchestrator.consume_tracking_events(tracking_consumer, count=args.read_count)
        n_match = orchestrator.consume_match_events(match_events_consumer, count=args.read_count)
        n_context = orchestrator.consume_match_context(match_context_consumer, count=args.read_count)
        if n_tracking or n_match or n_context:
            logger.info("Consumed: tracking=%d match_events=%d match_context=%d", n_tracking, n_match, n_context)

        new_objects = orchestrator.tick()
        published = MatchOrchestrator.publish(producer, new_objects)
        if published:
            logger.info("Published %d analytics objects across %d streams", published, len(StreamTopics.OUTBOUND))

        time.sleep(args.tick_interval_seconds)

    logger.info("Finalizing match_id=%s before exit...", args.match_id)
    final_objects = orchestrator.finalize()
    published = MatchOrchestrator.publish(producer, final_objects)
    logger.info("Final publish: %d analytics objects. Shutdown complete.", published)
    return 0


if __name__ == "__main__":
    sys.exit(main())
