"""
Kinexon Resampler — PlayerDynamics

Closes the granularity gap identified in the real-data training audit:
Kinexon's positions.csv delivers one tick per player every 0.05s (20 Hz),
but SequenceWindowConfig (config/settings.py) assumes one event every
event_interval_s=15s, with an 8-step window meant to span window_seconds=120s.
Feeding 20 Hz ticks straight into the existing pipeline as if each row were
already one "event" would make an 8-step window span ~0.4s, not 120s.

This module sits strictly BETWEEN ingestion/kinexon_adapter.py and the
existing, UNMODIFIED PlayerDynamics pipeline:

    KinexonAdapter.stream_positions()  (20 Hz KinexonObservation)
                |
                v
    KinexonResampler.resample()        (this module — NEW)
                |
                v
    events_df / sessions_df  (event_interval_s-resolution, exact shape
                               PatternAnalysisEngine.build_training_sequences()
                               and PlayersDataAnalysisPipeline.load_historical_data()
                               already expect)
                |
                v
    SequenceWindowBuilder / BaselineBuilder / SharedBackboneAutoencoder
    (none of these are modified by this module)

No new computation is invented for the model-facing columns: ts, speed_ms,
x_pitch, y_pitch, heart_rate_bpm are exactly the fields
analysis/anomaly_detection.py's _extract() already reads, and is_sprint is
exactly the field analysis/baseline.py's BaselineBuilder.compute_provisional()
already reads. This module only changes WHICH ROW RATE those fields arrive
at -- one row per event_interval_s bucket instead of one row per 20 Hz tick.

Bucket aggregation (per event_interval_s-second bucket, per player)
------------------------------------------------------------------------
speed_ms             -- mean speed across the bucket's raw ticks (the
                         representative value _extract()/compute_provisional()
                         read as "the" speed for this tick)
speed_ms_max         -- max speed across the bucket (auxiliary; not read by
                         the existing pipeline today, kept for inspection and
                         future use)
acceleration_ms2     -- mean acceleration across the bucket (auxiliary;
                         _extract() computes its own delta-based acceleration
                         from consecutive bucket speed_ms values instead)
x_pitch / y_pitch    -- mean normalised pitch position across the bucket
                         ("representative position")
distance_traveled_m  -- true intra-bucket path length: sum of consecutive
                         raw-tick Euclidean displacements (metres), NOT the
                         straight-line distance between bucket endpoints.
                         Named distance_traveled_m (not distance_delta_m) so
                         it is never confused with _extract()'s own
                         inter-bucket distance_delta, which it recomputes
                         itself from consecutive bucket x_pitch/y_pitch.
                         The displacement leading INTO a tick is attributed
                         to that tick's own bucket (simple, deterministic
                         convention; the one displacement that straddles a
                         bucket boundary is attributed to the later bucket).
is_sprint            -- True if ANY raw tick in the bucket reached
                         KinexonConfig.sprint_threshold_ms (sprint is a peak
                         phenomenon -- a mean-speed gate would wash it out)
heart_rate_bpm       -- mean of whatever HR values are present in the
                         bucket; None if none are (always None for the
                         current Kinexon export -- see KinexonConfig.hr_sensor_present)
n_raw_ticks          -- how many 20 Hz rows fed this bucket (data-quality /
                         validation column, not read by the existing pipeline)

Anchoring
----------
Buckets are anchored to EACH PLAYER'S OWN first tracked tick (elapsed_s = 0
at that player's first observation), matching
BaselineBuilder.compute_provisional()'s own elapsed_s convention. This is
deliberate, not an oversight: every downstream consumer here
(SequenceWindowBuilder, BaselineBuilder) reasons per-player and never
compares cross-player wall-clock alignment, so anchoring per-player avoids
sparse leading buckets for players who enter the match late (substitutes).

Resolved gap (was open as of this module's introduction, fixed in the
baseline-threshold audit -- see scripts/archive/baseline_threshold_audit.py)
-------------------------------------------------------------------------------
BaselineBuilder.compute_provisional() used to reject every 120s window
because of a hardcoded MIN_EVENTS_PER_WINDOW=30, copy-pasted from a
different function's 900s-window threshold. At this module's required
event_interval_s=15 cadence, a 120s window has exactly 8 rows by
construction (window_seconds // event_interval_s) -- always below 30,
rejecting 27/27 players unconditionally. Fixed by adding
BaselineConfig.min_events_per_provisional_window (config/settings.py,
empirically set to 6 against real session 3387) and having
compute_provisional() read it instead of the hardcoded literal.
_compute_fatigue_curve's own (differently-scoped, 900s-window)
MIN_EVENTS_PER_SEGMENT=30 was left untouched.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config.settings import CONFIG, KinexonConfig, SequenceWindowConfig
from ingestion.kinexon_adapter import KinexonObservation

logger = logging.getLogger(__name__)


@dataclass
class _Bucket:
    """Accumulator for one event_interval_s-second bucket, one player."""
    bucket_index: int
    start_ts: datetime
    speeds: List[float] = field(default_factory=list)
    accels: List[float] = field(default_factory=list)
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    hrs: List[float] = field(default_factory=list)
    sprint_hit: bool = False
    distance_m: float = 0.0


class KinexonResampler:
    """
    Usage
    -----
        adapter = KinexonAdapter()
        resampler = KinexonResampler()
        meta = adapter.load_player_meta(stats_path)
        observations = adapter.stream_positions(positions_path, meta, session_id="3387", match_id="3387")
        events_by_player, sessions_df = resampler.resample(observations, session_id="3387")
    """

    def __init__(
        self,
        window_config: Optional[SequenceWindowConfig] = None,
        kinexon_config: Optional[KinexonConfig] = None,
    ) -> None:
        self.window_config = window_config or CONFIG.window
        self.kinexon_config = kinexon_config or CONFIG.kinexon
        self.bucket_seconds = self.window_config.event_interval_s

    # ─────────────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────────────

    def resample(
        self,
        observations: Iterable[KinexonObservation],
        session_id: str,
    ) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
        """
        Consumes KinexonObservation ticks (any order; sorted internally per
        player) and returns:

            events_by_player: {player_id: events_df}
                Columns: session_id, match_id, player_id, ts, elapsed_s,
                         speed_ms, speed_ms_max, acceleration_ms2, x_pitch,
                         y_pitch, distance_traveled_m, is_sprint,
                         heart_rate_bpm, n_raw_ticks
                One row per event_interval_s-second bucket, sorted by ts --
                exactly what build_training_sequences()/build_from_session()
                and BaselineBuilder.compute_provisional() read.

            sessions_df: one row per player, matching
                BaselineBuilder.compute()'s documented sessions_df contract
                (session_id, started_at, total_distance_m, sprint_count,
                max_speed_ms, high_speed_distance_m) plus player_id and a
                diagnostic n_buckets column. Unreachable today (only one
                real session exists, below min_sessions_for_baseline) but
                shaped correctly for when more sessions accumulate.
        """
        # Normalise to int here, once, at the ingestion boundary.
        # PatternAnalysisEngine.build_training_sequences() (and the gap-aware
        # mirror of it) both do `session_id = int(session.session_id)` when
        # grouping events_df by session before windowing -- an assumption
        # that holds for the synthetic generator's int session IDs but not
        # for Kinexon's string ones ("3387"). Comparing
        # events_df["session_id"] (str) against that int() result always
        # came back False, silently producing 0 windows for every player.
        # Casting once here, so every events_df/sessions_df column this
        # resampler produces is already int, makes that downstream
        # comparison match correctly without touching build_training_sequences()
        # itself or the synthetic pipeline's behaviour.
        session_id = int(session_id)

        raw_by_player: Dict[int, List[KinexonObservation]] = {}
        n_raw_total = 0
        for obs in observations:
            n_raw_total += 1
            raw_by_player.setdefault(obs.player_id, []).append(obs)

        events_by_player: Dict[int, pd.DataFrame] = {}
        session_rows: List[dict] = []

        for player_id, obs_list in raw_by_player.items():
            obs_list.sort(key=lambda o: o.ts)
            df = self._resample_one_player(obs_list, session_id)
            if df.empty:
                continue
            events_by_player[player_id] = df
            session_rows.append(self._session_summary_row(player_id, session_id, df))

        sessions_df = pd.DataFrame(session_rows)

        logger.info(
            "KinexonResampler: %d raw ticks -> %d players, %d total buckets "
            "(bucket=%ds, window_steps=%d -> %ds window)",
            n_raw_total, len(events_by_player),
            sum(len(d) for d in events_by_player.values()),
            self.bucket_seconds, self.window_config.window_steps,
            self.window_config.window_steps * self.bucket_seconds,
        )
        return events_by_player, sessions_df

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def _resample_one_player(
        self, obs_list: List[KinexonObservation], session_id: str
    ) -> pd.DataFrame:
        if not obs_list:
            return pd.DataFrame()

        t0 = obs_list[0].ts
        sprint_threshold = self.kinexon_config.sprint_threshold_ms
        buckets: Dict[int, _Bucket] = {}
        prev_obs: Optional[KinexonObservation] = None

        for obs in obs_list:
            elapsed = (obs.ts - t0).total_seconds()
            idx = int(elapsed // self.bucket_seconds)

            b = buckets.get(idx)
            if b is None:
                b = _Bucket(bucket_index=idx, start_ts=t0 + timedelta(seconds=idx * self.bucket_seconds))
                buckets[idx] = b

            if obs.speed_ms is not None:
                b.speeds.append(obs.speed_ms)
                if obs.speed_ms >= sprint_threshold:
                    b.sprint_hit = True
            if obs.acceleration_ms2 is not None:
                b.accels.append(obs.acceleration_ms2)
            b.xs.append(obs.x_pitch)
            b.ys.append(obs.y_pitch)
            if obs.heart_rate_bpm is not None:
                b.hrs.append(float(obs.heart_rate_bpm))

            # True intra-bucket path length: displacement leading INTO this
            # tick is attributed to this tick's own bucket.
            if prev_obs is not None:
                dx = obs.x_m - prev_obs.x_m
                dy = obs.y_m - prev_obs.y_m
                b.distance_m += math.sqrt(dx * dx + dy * dy)
            prev_obs = obs

        rows = []
        for idx in sorted(buckets.keys()):
            b = buckets[idx]
            if not b.speeds:
                continue  # bucket had only missing-speed ticks -- not usable
            rows.append({
                "session_id": session_id,
                "match_id": obs_list[0].match_id,
                "player_id": obs_list[0].player_id,
                "ts": b.start_ts,
                "elapsed_s": float(idx * self.bucket_seconds),
                "speed_ms": float(sum(b.speeds) / len(b.speeds)),
                "speed_ms_max": float(max(b.speeds)),
                "acceleration_ms2": float(sum(b.accels) / len(b.accels)) if b.accels else 0.0,
                "x_pitch": float(sum(b.xs) / len(b.xs)) if b.xs else 50.0,
                "y_pitch": float(sum(b.ys) / len(b.ys)) if b.ys else 50.0,
                "distance_traveled_m": round(b.distance_m, 4),
                "is_sprint": bool(b.sprint_hit),
                "heart_rate_bpm": (float(sum(b.hrs) / len(b.hrs)) if b.hrs else None),
                "n_raw_ticks": len(b.speeds),
            })

        return pd.DataFrame(rows)

    def _session_summary_row(self, player_id: int, session_id: str, df: pd.DataFrame) -> dict:
        hi_threshold = self.kinexon_config.high_intensity_threshold_ms
        high_speed_distance_m = float(
            df.loc[df["speed_ms_max"] >= hi_threshold, "distance_traveled_m"].sum()
        )
        return {
            "session_id": session_id,
            "player_id": player_id,
            "started_at": df["ts"].min(),
            "ended_at": df["ts"].max(),
            "total_distance_m": float(df["distance_traveled_m"].sum()),
            "sprint_count": int(df["is_sprint"].sum()),
            "max_speed_ms": float(df["speed_ms_max"].max()),
            "high_speed_distance_m": high_speed_distance_m,
            "n_buckets": int(len(df)),
        }
