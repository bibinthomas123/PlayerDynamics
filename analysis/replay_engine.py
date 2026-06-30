"""
replay_engine.py — PlayerDynamics

ReplayEngine: replays a previously ingested Kinexon match into the live
analytics pipeline, producing Redis stream output that is bit-for-bit
identical to what a live match would produce. The frontend never knows
whether it is watching live or replay data.

Two concurrent sub-pipelines run in separate threads:

  Tactical pipeline (events.csv → MatchOrchestrator):
    TacticalEvent stream → Possession / TeamState / Trend / Insight / Situation
    → analytics.possessions / .teamstate / .trends / .insights / .situations

  LSTM / Workload pipeline (positions.csv → LiveWindowAccumulator → model):
    Per-tick resampled positions → workload aggregation + LSTM scoring
    → analytics.player_workload / analytics.players

Both threads share a ReplayTimer so timing is derived from the original
match's event timestamps, not wall-clock time.

Speed control
─────────────
  speed=1.0  — realtime (events replayed at their original pace)
  speed=5.0  — 5× accelerated (a 90-min match finishes in 18 min)
  speed=0    — instant (no sleeping; both threads run as fast as possible)

Both threads use the same speed setting. A threading.Event is used for
coordinated shutdown (Ctrl+C from cmd_replay sets it).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import timezone as _tz
from pathlib import Path
from typing import Optional

from analysis.dataset_manager import MatchDataset

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

class ReplayTimer:
    """
    Paces replay events against the original match timeline.

    Usage
    -----
        timer = ReplayTimer(speed=2.0)
        for event in sorted_events:
            timer.wait_until(event.ts_ms)   # sleep if ahead of schedule
            process(event)
    """

    def __init__(self, speed: float) -> None:
        self._speed = max(0.0, speed)
        self._wall_start: Optional[float] = None
        self._event_start_ms: Optional[float] = None

    def calibrate(self, first_event_ts_ms: float) -> None:
        """Call once with the timestamp of the very first event."""
        self._wall_start = time.monotonic()
        self._event_start_ms = float(first_event_ts_ms)

    def wait_until(self, event_ts_ms: float) -> None:
        """Sleep until it is time to process the event at event_ts_ms."""
        if self._speed == 0 or self._wall_start is None:
            return
        elapsed_match_s = (float(event_ts_ms) - self._event_start_ms) / 1000.0
        target_wall = self._wall_start + elapsed_match_s / self._speed
        sleep_s = target_wall - time.monotonic()
        if sleep_s > 0.001:
            time.sleep(sleep_s)


# ─────────────────────────────────────────────────────────────────────────────
# Replay Engine
# ─────────────────────────────────────────────────────────────────────────────

class ReplayEngine:
    """
    Replays one ingested match through both analytics pipelines.

    Parameters
    ----------
    dataset     : MatchDataset resolved by MatchDatasetManager.get(match_id)
    model_dir   : directory containing shared_backbone.pt (for LSTM scoring)
    speed       : replay speed multiplier (0 = instant)
    tick_interval_match_s : how often (in match seconds) to call
                  MatchOrchestrator.tick() in the tactical thread.
    """

    def __init__(
        self,
        dataset: MatchDataset,
        model_dir: Optional[Path] = None,
        speed: float = 1.0,
        tick_interval_match_s: float = 30.0,
    ) -> None:
        self._dataset = dataset
        self._model_dir = Path(model_dir) if model_dir else _ROOT / "models"
        self._speed = speed
        self._tick_interval_match_s = tick_interval_match_s

    def run(self, stop: threading.Event) -> dict[str, int]:
        """
        Start both sub-pipelines concurrently. Blocks until both finish or
        stop is set (Ctrl+C / SIGTERM from the caller).

        Returns a summary dict: {tactical_published, lstm_published, workload_published}.
        """
        from config.redis_client import RedisStreamProducer

        producer = RedisStreamProducer()
        results: dict[str, int] = {}
        errors: list[Exception] = []

        def run_tactical():
            try:
                n = self._run_tactical(producer, stop)
                results["tactical_published"] = n
            except Exception as exc:
                logger.exception("ReplayEngine: tactical thread failed")
                errors.append(exc)

        def run_lstm():
            try:
                n_workload, n_players = self._run_lstm(producer, stop)
                results["workload_published"] = n_workload
                results["lstm_published"] = n_players
            except Exception as exc:
                logger.exception("ReplayEngine: LSTM thread failed")
                errors.append(exc)

        t_tactical = threading.Thread(target=run_tactical, name="replay-tactical", daemon=True)
        t_lstm = threading.Thread(target=run_lstm, name="replay-lstm", daemon=True)

        logger.info(
            "ReplayEngine: starting replay of %s (speed=%.1f×)",
            self._dataset.label, self._speed if self._speed > 0 else float("inf"),
        )

        t_tactical.start()
        t_lstm.start()
        t_tactical.join()
        t_lstm.join()

        if errors:
            logger.warning("ReplayEngine: %d thread(s) encountered errors", len(errors))

        logger.info(
            "ReplayEngine: replay complete | tactical=%d workload=%d lstm=%d",
            results.get("tactical_published", 0),
            results.get("workload_published", 0),
            results.get("lstm_published", 0),
        )
        return results

    # ── Tactical pipeline ─────────────────────────────────────────────────────

    def _run_tactical(self, producer, stop: threading.Event) -> int:
        """
        Loads events.csv, replays TacticalEvents chronologically into
        MatchOrchestrator, ticks every tick_interval_match_s of match time,
        and publishes to analytics.* streams. Returns the count of published
        analytics objects.
        """
        from ingestion.kinexon_adapter import KinexonAdapter
        from ingestion.tactical_event import KinexonTacticalEventAdapter
        from analysis.match_orchestrator import MatchOrchestrator
        from config.redis_client import StreamTopics

        dataset = self._dataset

        if dataset.events_path is None:
            logger.warning(
                "ReplayEngine: no events.csv for match %s — tactical pipeline skipped",
                dataset.match_id,
            )
            return 0

        logger.info("ReplayEngine [tactical]: loading %s", dataset.events_path)
        adapter = KinexonAdapter()
        player_meta = adapter.load_player_meta(dataset.statistics_path)

        tactical_adapter = KinexonTacticalEventAdapter()
        events = sorted(
            tactical_adapter.parse(dataset.events_path, player_meta=player_meta, match_id=dataset.match_id),
            key=lambda e: e.timestamp,
        )
        logger.info("ReplayEngine [tactical]: %d events loaded", len(events))

        if not events:
            return 0

        orchestrator = MatchOrchestrator(match_id=dataset.match_id, player_meta=player_meta)
        timer = ReplayTimer(self._speed)

        def _ts_ms(e) -> float:
            return e.timestamp.timestamp() * 1000.0

        timer.calibrate(_ts_ms(events[0]))

        last_tick_match_ms = _ts_ms(events[0])
        tick_interval_ms = self._tick_interval_match_s * 1000.0
        n_published = 0

        for event in events:
            if stop.is_set():
                break
            event_ms = _ts_ms(event)
            timer.wait_until(event_ms)
            orchestrator.ingest_event(event)

            if (event_ms - last_tick_match_ms) >= tick_interval_ms:
                new_objects = orchestrator.tick()
                published = MatchOrchestrator.publish(producer, new_objects)
                if published:
                    n_published += published
                    logger.debug(
                        "ReplayEngine [tactical]: tick at match_ms=%.0f → %d objects published",
                        event_ms, published,
                    )
                last_tick_match_ms = event_ms

        # Final tick + finalize
        if not stop.is_set():
            final_objects = orchestrator.finalize()
            n_published += MatchOrchestrator.publish(producer, final_objects)
            logger.info("ReplayEngine [tactical]: finalized — %d total objects published", n_published)

        return n_published

    # ── LSTM / Workload pipeline ──────────────────────────────────────────────

    def _run_lstm(self, producer, stop: threading.Event) -> tuple[int, int]:
        """
        Loads position/statistics data, resamples, and replays each tick
        through LiveWindowAccumulator. On each completed window, scores with
        the LSTM and publishes to analytics.players and analytics.player_workload.

        Returns (n_workload_published, n_players_published).
        """
        from datetime import timezone as _timezone
        from config.redis_client import StreamTopics
        from config.settings import CONFIG, OWNERSHIP_SCM
        from analysis.regime import SessionRegimeClassifier
        from analysis.player_workload import compute_player_workload_windows, assign_workload_status
        from analysis.player_workload_event import PlayerWorkloadEvent
        from analysis.live_window_accumulator import LiveWindowAccumulator
        from analysis import pilot_pipeline as pp
        from ingestion.kinexon_adapter import KinexonAdapter
        from ingestion.kinexon_resampler import KinexonResampler

        backbone_path = self._model_dir / "shared_backbone.pt"
        if not backbone_path.exists():
            logger.warning(
                "ReplayEngine: no checkpoint at %s — LSTM pipeline skipped. "
                "Run `python main.py train --data-source kinexon --use-event-features` first.",
                backbone_path,
            )
            return 0, 0

        dataset = self._dataset
        match_id = dataset.match_id

        logger.info("ReplayEngine [lstm]: loading checkpoint + session data for match %s", match_id)

        # Load data for this specific match
        adapter = KinexonAdapter()
        meta = adapter.load_player_meta(dataset.statistics_path)
        observations = list(
            adapter.stream_positions(dataset.positions_path, meta, session_id=match_id, match_id=match_id)
        )
        resampler = KinexonResampler()
        events_by_player, sessions_df = resampler.resample(observations, session_id=match_id)

        # Merge event features if available (matches the 32-feature checkpoint)
        if dataset.events_path is not None and dataset.events_path.exists():
            try:
                from ingestion.kinexon_events_features import merge_event_features
                events_by_player = merge_event_features(
                    events_by_player=events_by_player,
                    events_csv_path=dataset.events_path,
                    real_player_ids=meta.keys(),
                    bucket_seconds=resampler.bucket_seconds,
                )
                logger.debug("ReplayEngine [lstm]: merged 32 event features")
            except Exception as exc:
                logger.warning("ReplayEngine [lstm]: event feature merge failed (%s) — using 8-feature mode", exc)

        # Load the promoted checkpoint (no retraining)
        pipeline, _ebp, _sdf, _meta_pp, eligible, load_result = pp.build_pipeline_and_load(
            backbone_path=backbone_path,
            use_event_features=True,
        )
        engine = pipeline.pattern_engine
        model_version = engine._shared_model.model_version if engine._shared_model else "unknown"
        logger.info("ReplayEngine [lstm]: checkpoint loaded (model_version=%s)", model_version)

        # Restrict to SCM players only (no opponent feed in replay)
        scm_eligible = [
            pid for pid in eligible
            if meta.get(pid) and getattr(meta.get(pid), "ownership", None) == OWNERSHIP_SCM
        ]
        if not scm_eligible:
            # Fallback: use all eligible if ownership unknown for this match
            scm_eligible = [pid for pid in eligible if pid in events_by_player]
            logger.warning(
                "ReplayEngine [lstm]: ownership unknown for match %s — using all %d eligible players",
                match_id, len(scm_eligible),
            )

        logger.info(
            "ReplayEngine [lstm]: %d SCM players will be replayed for match %s",
            len(scm_eligible), match_id,
        )

        # Pre-compute workload rows
        hi_threshold = CONFIG.kinexon.high_intensity_threshold_ms
        dfs_by_player = {
            pid: events_by_player[pid].sort_values("ts").reset_index(drop=True)
            for pid in scm_eligible
            if pid in events_by_player
        }
        workload_rows_by_player = {}
        for pid, df in dfs_by_player.items():
            rows = compute_player_workload_windows(pid, df, hi_threshold)
            if rows:
                workload_rows_by_player[pid] = rows
        assign_workload_status(workload_rows_by_player)

        # Build interleaved tick list (all players sorted by timestamp)
        ticks: list[tuple[float, int, int]] = []
        for pid, df in dfs_by_player.items():
            for row_idx in range(len(df)):
                ts_val = df["ts"].iloc[row_idx]
                ts_float = float(ts_val.value / 1e6) if hasattr(ts_val, "value") else float(ts_val)
                ticks.append((ts_float, pid, row_idx))
        ticks.sort(key=lambda t: t[0])

        producer.ensure_stream(StreamTopics.ANALYTICS_PLAYER_WORKLOAD)
        producer.ensure_stream(StreamTopics.ANALYTICS_PLAYERS)

        accumulator = LiveWindowAccumulator(
            window_size=CONFIG.window.window_steps,
            stride=CONFIG.window.window_steps,
        )
        clf = SessionRegimeClassifier()

        timer = ReplayTimer(self._speed)
        if ticks:
            timer.calibrate(ticks[0][0] * 1000.0)  # convert s → ms for timer

        n_workload = 0
        n_players = 0

        for ts_s, pid, row_idx in ticks:
            if stop.is_set():
                break

            timer.wait_until(ts_s * 1000.0)

            m = meta.get(pid)
            player_name = m.player_name if m else f"player_{pid}"
            position = m.position_label if m else "unknown"
            df = dfs_by_player[pid]
            row = df.iloc[row_idx]
            player_sessions = sessions_df[sessions_df["player_id"] == pid]

            # Publish workload tick
            wl_list = workload_rows_by_player.get(pid)
            if wl_list and row_idx < len(wl_list) and wl_list[row_idx] is not None:
                wl_row = wl_list[row_idx]
                wl_ts = wl_row["ts"]
                if hasattr(wl_ts, "to_pydatetime"):
                    wl_ts = wl_ts.to_pydatetime()
                if wl_ts.tzinfo is None:
                    wl_ts = wl_ts.replace(tzinfo=_timezone.utc)
                workload_event = PlayerWorkloadEvent(
                    player_id=pid, external_id=str(pid), player_name=player_name,
                    position=position, match_id=match_id, timestamp=wl_ts,
                    elapsed_s=wl_row["elapsed_s"],
                    current_load=wl_row["current_load"], load_trend=wl_row["load_trend"],
                    acceleration_load=wl_row["acceleration_load"],
                    deceleration_load=wl_row["deceleration_load"],
                    sprint_load=wl_row["sprint_load"],
                    high_intensity_load=wl_row["high_intensity_load"],
                    distance_covered=wl_row["distance_covered"],
                    performance_trend=wl_row["performance_trend"],
                    workload_status=wl_row["workload_status"],
                )
                producer.publish_dataclass(StreamTopics.ANALYTICS_PLAYER_WORKLOAD, workload_event)
                n_workload += 1

            # Push tick through accumulator → score when window completes
            live_tick = row.to_dict()
            window = accumulator.push(player_id=str(pid), event=live_tick)
            if window is not None:
                seq, mask = engine.window_builder.build_live_window(window)
                elapsed_s = float(window[-1].get("elapsed_s", 0.0))
                event = pp.score_window_and_build_event(
                    pipeline=pipeline, engine=engine, clf=clf, pid=pid,
                    seq=seq, mask=mask, elapsed_s=elapsed_s,
                    player_sessions=player_sessions, meta=meta,
                    model_version=model_version, match_id=match_id,
                )
                producer.publish_dataclass(StreamTopics.ANALYTICS_PLAYERS, event)
                n_players += 1

        logger.info(
            "ReplayEngine [lstm]: complete — %d workload ticks, %d LSTM windows published",
            n_workload, n_players,
        )
        return n_workload, n_players
