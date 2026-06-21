"""
Kinexon Events Feature Engineering — PlayerDynamics

Converts the discrete Kinexon events.csv export (Acceleration, Deceleration,
Sprint, Jump, Change of Direction, Ball Possession, Pass, Shots, ...) into
per-player, per-window aggregate features, using the EXACT same bucket
boundaries KinexonResampler.resample() already produces for positions.csv
(bucket_seconds = CONFIG.window.event_interval_s, anchored to each player's
own first tracked position tick).

This module is intentionally separate from ingestion/kinexon_adapter.py and
ingestion/kinexon_resampler.py (neither is modified here). Its only output
is a per-player DataFrame keyed by ``elapsed_s`` that callers left-merge onto
KinexonResampler's existing per-player events_df, filling unmatched buckets
with 0.0 -- the correct semantic for "zero events of that type occurred in
this window", not a missing-data placeholder.

events.csv structure
---------------------
Row 0: 5 fixed columns (Timestamp (ms), Timestamp in local format,
Player ID, Name, Event type) followed by up to 7 generic value slots whose
MEANING depends on the row's Event type.
Rows 1-12: one legend row per Event type, naming those generic slots.
Rows 13+: ragged data rows (column count varies by Event type).

Event type strings in the data are singular ("Acceleration", "Sprint", ...)
while several legend rows are plural ("Accelerations", "Sprints", ...);
this module hardcodes the data-row (singular) spelling actually observed in
session 3387's export, not the legend spelling.

Ball pseudo-entities
---------------------
Kinexon tracks the ball itself as a trackable id (e.g. "Ball1 Ball",
"Ball3 Ball") and attributes some Shots rows to it. These ids are never
present in statistics.csv's real roster, so they are filtered out by
restricting to ``real_player_ids`` (KinexonAdapter.load_player_meta()'s
keys) before aggregating.
"""
from __future__ import annotations

import csv
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from config.settings import EVENT_DERIVED_FEATURE_NAMES

logger = logging.getLogger(__name__)

_HEADER_ROWS = 13  # 1 fixed-column header row + 12 per-event-type legend rows

_KMH_TO_MS = 1.0 / 3.6


def _f(value: Optional[str], default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class _Acc:
    """Per-(player, bucket) raw-value accumulator, one event type's worth at a time."""

    __slots__ = (
        "accel_max", "accel_avg",
        "decel_max", "decel_avg",
        "sprint_dist", "sprint_dur", "sprint_speed_max",
        "jump_height",
        "cod_angle",
        "possession_dur",
        "pass_dist", "pass_speed",
        "shot_speed", "shot_dist",
        "n_accel", "n_decel", "n_sprint", "n_jump", "n_cod", "n_possession", "n_pass", "n_shot",
    )

    def __init__(self) -> None:
        self.accel_max: list = []
        self.accel_avg: list = []
        self.decel_max: list = []
        self.decel_avg: list = []
        self.sprint_dist: list = []
        self.sprint_dur: list = []
        self.sprint_speed_max: list = []
        self.jump_height: list = []
        self.cod_angle: list = []
        self.possession_dur: list = []
        self.pass_dist: list = []
        self.pass_speed: list = []
        self.shot_speed: list = []
        self.shot_dist: list = []
        self.n_accel = 0
        self.n_decel = 0
        self.n_sprint = 0
        self.n_jump = 0
        self.n_cod = 0
        self.n_possession = 0
        self.n_pass = 0
        self.n_shot = 0


def _aggregate_bucket(acc: _Acc) -> dict:
    def mean(lst):
        return float(sum(lst) / len(lst)) if lst else 0.0

    def mx(lst):
        return float(max(lst)) if lst else 0.0

    def total(lst):
        return float(sum(lst)) if lst else 0.0

    return {
        "accel_event_count": float(acc.n_accel),
        "accel_mean_ms2": mean(acc.accel_avg),
        "accel_max_ms2": mx(acc.accel_max),
        "decel_event_count": float(acc.n_decel),
        "decel_mean_ms2": mean(acc.decel_avg),
        "decel_max_ms2": mx(acc.decel_max),
        "sprint_event_count": float(acc.n_sprint),
        "sprint_distance_m": total(acc.sprint_dist),
        "sprint_duration_s": total(acc.sprint_dur),
        "sprint_max_speed_ms": mx(acc.sprint_speed_max),
        "jump_event_count": float(acc.n_jump),
        "jump_avg_height_m": mean(acc.jump_height),
        "jump_max_height_m": mx(acc.jump_height),
        "cod_event_count": float(acc.n_cod),
        "cod_angle_sum_deg": total(acc.cod_angle),
        "cod_angle_mean_deg": mean(acc.cod_angle),
        "possession_event_count": float(acc.n_possession),
        "possession_duration_s": total(acc.possession_dur),
        "pass_event_count": float(acc.n_pass),
        "pass_avg_distance_m": mean(acc.pass_dist),
        "pass_avg_ball_speed_ms": mean(acc.pass_speed),
        "shot_event_count": float(acc.n_shot),
        "shot_avg_speed_ms": mean(acc.shot_speed),
        "shot_avg_distance_m": mean(acc.shot_dist),
    }


def build_event_window_features(
    events_csv_path: Path,
    real_player_ids: Iterable[int],
    t0_by_player: Dict[int, "pd.Timestamp"],
    bucket_seconds: float,
) -> Dict[int, pd.DataFrame]:
    """
    Parse events.csv and aggregate into per-player, per-bucket window features.

    Parameters
    ----------
    events_csv_path:
        Path to events.csv (polymorphic Kinexon event log).
    real_player_ids:
        Player IDs known from statistics.csv's roster (KinexonAdapter.
        load_player_meta()). Rows whose Player ID is not in this set
        (ball pseudo-entities, or any other non-roster id) are dropped.
    t0_by_player:
        Each player's first tracked position tick (KinexonResampler's own
        per-player anchor: ``events_by_player[pid]["ts"].min()``). Event
        buckets are computed relative to this SAME t0 so bucket indices
        line up exactly with KinexonResampler's positions.csv-derived buckets.
    bucket_seconds:
        Bucket width in seconds (CONFIG.window.event_interval_s) -- must
        match the value KinexonResampler used.

    Returns
    -------
    {player_id: DataFrame[elapsed_s, <24 EVENT_DERIVED_FEATURE_NAMES columns>]}
        One row per (player, bucket) that had at least one qualifying event.
        Buckets with zero events of every type simply do not appear --
        callers left-merge onto the existing per-player events_df and
        fillna(0.0), which is the correct semantic (no events occurred),
        not a missing-data gap.
    """
    real_ids = set(int(p) for p in real_player_ids)
    buckets: Dict[int, Dict[int, _Acc]] = {}  # player_id -> bucket_index -> _Acc
    n_rows_total = 0
    n_rows_kept = 0
    n_rows_dropped_not_roster = 0
    n_rows_dropped_no_t0 = 0
    n_rows_dropped_negative_elapsed = 0

    with open(events_csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    for row in rows[_HEADER_ROWS:]:
        if len(row) < 5:
            continue
        n_rows_total += 1

        pid_raw = row[2].strip()
        if not pid_raw:
            continue
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        if pid not in real_ids:
            n_rows_dropped_not_roster += 1
            continue

        t0 = t0_by_player.get(pid)
        if t0 is None:
            n_rows_dropped_no_t0 += 1
            continue

        try:
            ts_ms = int(row[0])
        except (ValueError, IndexError):
            continue
        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

        elapsed = (pd.Timestamp(ts) - pd.Timestamp(t0)).total_seconds()
        if elapsed < 0:
            n_rows_dropped_negative_elapsed += 1
            continue

        idx = int(elapsed // bucket_seconds)
        event_type = row[4]

        player_buckets = buckets.setdefault(pid, {})
        acc = player_buckets.setdefault(idx, _Acc())

        if event_type == "Acceleration":
            # row[5..11]: Duration(s), Distance(m), Speed_max(km/h),
            # Acceleration_max(m/s2), Acceleration_avg(m/s2), SpeedChange(km/h), Category
            acc.n_accel += 1
            if len(row) > 8:
                acc.accel_max.append(_f(row[8]))
            if len(row) > 9:
                acc.accel_avg.append(_f(row[9]))
        elif event_type == "Deceleration":
            # row[5..11]: Duration(s), Distance(m), Speed_max(km/h),
            # Deceleration_max(m/s2), Deceleration_avg(m/s2), SpeedChange(km/h), Category
            acc.n_decel += 1
            if len(row) > 8:
                acc.decel_max.append(abs(_f(row[8])))
            if len(row) > 9:
                acc.decel_avg.append(abs(_f(row[9])))
        elif event_type == "Sprint":
            # row[5..9]: Duration(s), Distance(m), Speed_max(km/h), Speed_avg(km/h), Category
            acc.n_sprint += 1
            if len(row) > 5:
                acc.sprint_dur.append(_f(row[5]))
            if len(row) > 6:
                acc.sprint_dist.append(_f(row[6]))
            if len(row) > 7:
                acc.sprint_speed_max.append(_f(row[7]) * _KMH_TO_MS)
        elif event_type == "Jump":
            # row[5..9]: Airtime(s), Height(m), Distance(m), JumpRatioMax, Category
            acc.n_jump += 1
            if len(row) > 6:
                acc.jump_height.append(_f(row[6]))
        elif event_type == "Change of Direction":
            # row[5..8]: Magnitude(deg), Deceleration_max(m/s2), Acceleration_max(m/s2), Direction
            acc.n_cod += 1
            if len(row) > 5:
                acc.cod_angle.append(abs(_f(row[5])))
        elif event_type == "Ball Possession":
            # row[5]: Duration(s)
            acc.n_possession += 1
            if len(row) > 5:
                acc.possession_dur.append(_f(row[5]))
        elif event_type == "Pass":
            # row[5..9]: Distance(m), BallSpeed(km/h), OutplayedOpp, ReceivingPlayer, Type
            acc.n_pass += 1
            if len(row) > 5:
                acc.pass_dist.append(_f(row[5]))
            if len(row) > 6:
                acc.pass_speed.append(_f(row[6]) * _KMH_TO_MS)
        elif event_type == "Shots":
            # row[5..6]: Distance(m), BallSpeed(km/h)
            acc.n_shot += 1
            if len(row) > 5:
                acc.shot_dist.append(_f(row[5]))
            if len(row) > 6:
                acc.shot_speed.append(_f(row[6]) * _KMH_TO_MS)
        else:
            # Exertion, Impact, Ball Possession Lost/Recovery -- not in the
            # requested 24-feature set; row is attributed to roster filtering
            # stats above but contributes no aggregate.
            continue

        n_rows_kept += 1

    logger.info(
        "kinexon_events_features: %d data rows | kept=%d | dropped(not_roster)=%d "
        "dropped(no_t0)=%d dropped(negative_elapsed)=%d | players_with_buckets=%d",
        n_rows_total, n_rows_kept, n_rows_dropped_not_roster,
        n_rows_dropped_no_t0, n_rows_dropped_negative_elapsed, len(buckets),
    )

    result: Dict[int, pd.DataFrame] = {}
    for pid, player_buckets in buckets.items():
        recs = []
        for idx in sorted(player_buckets.keys()):
            row_dict = {"elapsed_s": float(idx * bucket_seconds)}
            row_dict.update(_aggregate_bucket(player_buckets[idx]))
            recs.append(row_dict)
        result[pid] = pd.DataFrame(recs)

    return result


def merge_event_features(
    events_by_player: Dict[int, pd.DataFrame],
    events_csv_path: Path,
    real_player_ids: Iterable[int],
    bucket_seconds: float,
) -> Dict[int, pd.DataFrame]:
    """
    Left-merge the 24 event-derived window features onto KinexonResampler's
    existing per-player events_df (keyed on the shared ``elapsed_s`` bucket
    boundary), filling unmatched buckets with 0.0.

    Does not mutate KinexonResampler or its output's existing columns --
    only appends EVENT_DERIVED_FEATURE_NAMES columns.

    If events_csv_path does not exist, returns events_by_player unchanged
    except for the 24 new columns added as all-zero (graceful no-op, same
    convention as "0 events of this type occurred").
    """
    if not events_csv_path.exists():
        logger.warning(
            "merge_event_features: %s not found -- event-derived features "
            "will be all-zero for this session", events_csv_path,
        )
        feat_by_player: Dict[int, pd.DataFrame] = {}
    else:
        t0_by_player = {pid: df["ts"].min() for pid, df in events_by_player.items() if not df.empty}
        feat_by_player = build_event_window_features(
            events_csv_path=events_csv_path,
            real_player_ids=real_player_ids,
            t0_by_player=t0_by_player,
            bucket_seconds=bucket_seconds,
        )

    merged: Dict[int, pd.DataFrame] = {}
    for pid, df in events_by_player.items():
        feat_df = feat_by_player.get(pid)
        if feat_df is None or feat_df.empty:
            out = df.copy()
            for col in EVENT_DERIVED_FEATURE_NAMES:
                out[col] = 0.0
        else:
            out = df.merge(feat_df, on="elapsed_s", how="left")
            for col in EVENT_DERIVED_FEATURE_NAMES:
                out[col] = out[col].fillna(0.0)
        merged[pid] = out

    return merged
