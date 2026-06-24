"""
Kinexon UWB Tracking Adapter — PlayerDynamics

Converts Kinexon CSV exports into RawPlayerObservation records for the
existing PlayerDynamics analytics pipeline.

Data Source
-----------
Kinexon uses Ultra-Wideband (UWB) tracking — NOT GPS.
Positions arrive in metres in a pitch-centred coordinate system:

    (0, 0) = pitch centre
    X axis = long axis of the court (handball: ±20 m, with run-off to ±22 m)
    Y axis = short axis of the court (handball: ±10 m, with run-off to ±13 m)

Player Identity
---------------
Kinexon assigns each tracked entity a 'mapped id' (integer). This identifier
appears consistently across all four export files:

    positions.csv  → 'mapped id'  column
    Inertial.csv   → 'mapped id'  column
    events.csv     → 'Player ID'  column
    statistics.csv → 'Player ID'  column

This adapter uses 'mapped id' directly as player_id in PlayerDynamics. It is
NOT the same namespace as the backend database Player.id (autoincrement from 1).
A cross-system mapping table is required if Kinexon data must be linked to the
backend Player table. That mapping is OUT OF SCOPE for this adapter.

Files
-----
    positions.csv  — PRIMARY  — continuous 20 Hz: position + kinematics per player
    statistics.csv — METADATA — session aggregates: positions, HR, distance zones
    Inertial.csv   — UNUSED   — IMU orientation only; x/y sparse; HR absent
    events.csv     — UNUSED   — discrete event aggregates; not a continuous stream

Coordinate Normalisation
------------------------
Pipeline expects x_pitch, y_pitch ∈ [0, 100]:

    x_pitch = clamp((x_m + pitch_length / 2) / pitch_length × 100, 0, 100)
    y_pitch = clamp((y_m + pitch_width  / 2) / pitch_width  × 100, 0, 100)

Heart Rate
----------
HR is absent from the provided Kinexon export. heart_rate_bpm is set to None
for all observations. The SequenceWindowBuilder.build_live_window() requires
both speed_ms and heart_rate_bpm to be non-None to treat an event as 'real'
(is_real flag). With HR=None every event is treated as padding — anomaly
inference will not produce meaningful scores until HR data is available.

distance_delta_m
----------------
'total distance in m' in positions.csv is always 0. The adapter computes
distance_delta_m as the Euclidean distance between consecutive (x, y) positions
for each player.
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from config.settings import KinexonConfig, classify_ownership
from ingestion.pipeline import RawPlayerObservation

logger = logging.getLogger(__name__)

_BALL_GROUP = "Ball"

# IHF position code → canonical role string
_POSITION_LABELS: Dict[str, str] = {
    "TW": "goalkeeper",
    "KR": "pivot",
    "RM": "centre_back",
    "RR": "right_back",
    "RL": "left_back",
    "RA": "right_wing",
    "LA": "left_wing",
}


# ─────────────────────────────────────────────────────────────────────────────
# Rich per-tick representation (internal; does not replace RawPlayerObservation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KinexonObservation:
    """
    Single 20 Hz position tick from positions.csv.

    player_id = Kinexon 'mapped id' (int).  This is the canonical Kinexon
    identity and is used as player_id throughout PlayerDynamics for Kinexon
    sessions.  It is NOT the backend DB Player.id.
    """
    # Identity
    player_id: int
    player_name: str
    jersey_number: int
    group_name: str           # team label ("SC Magdeburg", "HSG Wetzlar")

    # Time
    timestamp_ms: int         # epoch ms (Kinexon clock)
    ts: datetime              # UTC datetime

    # Raw Kinexon coordinates (pitch-centred, metres)
    x_m: float
    y_m: float

    # Normalised pitch coordinates [0, 100]
    x_pitch: float
    y_pitch: float

    # Kinematics (Kinexon-computed from UWB positions)
    speed_ms: Optional[float]        # None if unparseable or negative
    acceleration_ms2: Optional[float]  # None if unparseable or outlier

    # Euclidean displacement from previous tick (computed in adapter; 0.0 for first tick)
    distance_delta_m: float

    # Physiological — None when wearable sensor absent or not exported
    heart_rate_bpm: Optional[int]

    # Derived
    sprint_flag: bool             # speed_ms >= KinexonConfig.sprint_threshold_ms

    # Session context
    session_id: Optional[str]
    match_id: Optional[str]

    # Validity — False if any field failed plausibility check
    valid: bool
    issues: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Per-player session metadata from statistics.csv
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KinexonPlayerMeta:
    """Per-player session context loaded from statistics.csv."""
    player_id: int
    player_name: str       # populated from positions.csv (stats Name col is empty)
    jersey_number: int
    position_code: str     # IHF code: TW / KR / RM / RR / RL / RA / LA
    position_label: str    # human-readable English role
    group_name: str        # team name
    session_id: str        # Kinexon session identifier (use as match_id context)
    ownership: str = ""    # "SCM" | "OPPONENT" -- see config.settings.classify_ownership(); Ball entities are excluded before this dataclass is built


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────

class KinexonAdapter:
    """
    Converts Kinexon CSV exports into the PlayerDynamics pipeline's data contract.

    Usage
    -----
        adapter = KinexonAdapter()
        meta    = adapter.load_player_meta(stats_path)
        for obs in adapter.stream_positions(positions_path, meta):
            raw = adapter.to_raw_observation(obs)
            evt = adapter.to_event_dict(obs)
    """

    def __init__(self, config: Optional[KinexonConfig] = None) -> None:
        self.config = config or KinexonConfig()
        self._half_length = self.config.pitch_length_m / 2.0
        self._half_width  = self.config.pitch_width_m  / 2.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public — load player metadata from statistics.csv
    # ──────────────────────────────────────────────────────────────────────────

    def load_player_meta(self, path: Path) -> Dict[int, KinexonPlayerMeta]:
        """
        Read per-player session context from statistics.csv.

        Returns a dict keyed by Kinexon player_id (int).
        Ball entities (group_name == 'Ball') are excluded.
        """
        meta: Dict[int, KinexonPlayerMeta] = {}
        try:
            with open(path, encoding="latin-1", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=";")
                for row in reader:
                    pid_raw = row.get("Player ID", "").strip()
                    if not pid_raw:
                        continue
                    try:
                        pid = int(pid_raw)
                    except ValueError:
                        continue

                    group = row.get("Group name", "").strip()
                    if group == _BALL_GROUP:
                        continue

                    pos_code = row.get("Position", "").strip()
                    meta[pid] = KinexonPlayerMeta(
                        player_id=pid,
                        player_name="",   # filled from positions.csv
                        jersey_number=0,
                        position_code=pos_code,
                        position_label=_POSITION_LABELS.get(pos_code, pos_code.lower()),
                        group_name=group,
                        session_id=row.get("Session ID", "").strip(),
                        ownership=classify_ownership(group, self.config.scm_team_name),
                    )
        except FileNotFoundError:
            logger.warning("statistics.csv not found at %s — player meta unavailable", path)

        logger.info("Loaded metadata for %d players from %s", len(meta), path.name)
        return meta

    # ──────────────────────────────────────────────────────────────────────────
    # Public — stream positions from positions.csv (generator, memory-efficient)
    # ──────────────────────────────────────────────────────────────────────────

    def stream_positions(
        self,
        path: Path,
        meta: Optional[Dict[int, KinexonPlayerMeta]] = None,
        session_id: Optional[str] = None,
        match_id: Optional[str] = None,
    ) -> Generator[KinexonObservation, None, None]:
        """
        Yield one KinexonObservation per player row in positions.csv.

        Validation rules
        ----------------
        Silently dropped (no observation yielded):
          - Ball entities (group_name == 'Ball')
          - Rows with missing mapped_id or timestamp

        Yielded with valid=False (observation produced; issues list populated):
          - Missing x/y coordinates
          - Negative speed
          - Speed > KinexonConfig.max_speed_ms (capped; flagged)
          - |Acceleration| > KinexonConfig.max_accel_ms2 (set to None; flagged)
          - HR outside [20, 250] bpm (set to None; not flagged — expected absent)
        """
        _prev_x: Dict[int, float] = {}
        _prev_y: Dict[int, float] = {}
        n_valid = n_invalid = n_skipped = 0

        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            for lineno, row in enumerate(reader, start=2):

                # ── Exclude ball entities ──────────────────────────────────
                if row.get("group name", "").strip() == _BALL_GROUP:
                    continue

                # ── Require player_id and timestamp ────────────────────────
                pid_raw = row.get("mapped id", "").strip()
                ts_raw  = row.get("ts in ms", "").strip()
                if not pid_raw or not ts_raw:
                    n_skipped += 1
                    continue
                try:
                    pid   = int(pid_raw)
                    ts_ms = int(ts_raw)
                except ValueError:
                    n_skipped += 1
                    continue

                ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                issues: List[str] = []

                # ── Parse coordinates ─────────────────────────────────────
                x_m_raw = _safe_float(row.get("x in m"))
                y_m_raw = _safe_float(row.get("y in m"))
                if x_m_raw is None or y_m_raw is None:
                    issues.append("missing_coordinates")
                x_m = x_m_raw if x_m_raw is not None else 0.0
                y_m = y_m_raw if y_m_raw is not None else 0.0

                # ── Parse speed ───────────────────────────────────────────
                speed_raw = _safe_float(row.get("speed in m/s"))
                speed_ms: Optional[float] = None
                if speed_raw is not None:
                    if speed_raw < 0.0:
                        issues.append("negative_speed")
                    elif speed_raw > self.config.max_speed_ms:
                        issues.append(f"speed_capped({speed_raw:.1f}_to_{self.config.max_speed_ms})")
                        speed_ms = self.config.max_speed_ms
                    else:
                        speed_ms = speed_raw

                # ── Parse acceleration ────────────────────────────────────
                accel_raw = _safe_float(row.get("acceleration in m/s2"))
                accel_ms2: Optional[float] = None
                if accel_raw is not None:
                    if abs(accel_raw) > self.config.max_accel_ms2:
                        issues.append(f"accel_outlier({accel_raw:.1f})")
                    else:
                        accel_ms2 = accel_raw

                # ── Parse heart rate (absent in this export) ───────────────
                hr_raw = _safe_float(row.get("heart rate in bpm"))
                heart_rate: Optional[int] = None
                if hr_raw is not None and 20.0 <= hr_raw <= 250.0:
                    heart_rate = int(hr_raw)

                # ── Normalise pitch coordinates ────────────────────────────
                if x_m_raw is not None and y_m_raw is not None:
                    x_pitch, y_pitch = self._normalise_coords(x_m, y_m)
                else:
                    x_pitch, y_pitch = 50.0, 50.0   # centre as safe default

                # ── Distance delta from consecutive positions ───────────────
                distance_delta = 0.0
                if x_m_raw is not None and y_m_raw is not None:
                    if pid in _prev_x:
                        dx = x_m - _prev_x[pid]
                        dy = y_m - _prev_y[pid]
                        distance_delta = math.sqrt(dx * dx + dy * dy)
                    _prev_x[pid] = x_m
                    _prev_y[pid] = y_m

                # ── Sprint flag ────────────────────────────────────────────
                sprint = speed_ms is not None and speed_ms >= self.config.sprint_threshold_ms

                # ── Player name / jersey from this row ─────────────────────
                name = row.get("full name", "").strip()
                try:
                    number = int(row.get("number", "").strip())
                except (ValueError, AttributeError):
                    number = 0

                # Back-fill meta that statistics.csv couldn't provide
                if meta and pid in meta:
                    pm = meta[pid]
                    if not pm.player_name:
                        pm.player_name = name
                    if not pm.jersey_number:
                        pm.jersey_number = number

                sid = session_id or (meta[pid].session_id if (meta and pid in meta) else None)

                valid = not bool(issues)
                if valid:
                    n_valid += 1
                else:
                    n_invalid += 1

                yield KinexonObservation(
                    player_id=pid,
                    player_name=name,
                    jersey_number=number,
                    group_name=row.get("group name", "").strip(),
                    timestamp_ms=ts_ms,
                    ts=ts,
                    x_m=x_m,
                    y_m=y_m,
                    x_pitch=x_pitch,
                    y_pitch=y_pitch,
                    speed_ms=speed_ms,
                    acceleration_ms2=accel_ms2,
                    distance_delta_m=distance_delta,
                    heart_rate_bpm=heart_rate,
                    sprint_flag=sprint,
                    session_id=sid,
                    match_id=match_id,
                    valid=valid,
                    issues=issues,
                )

        logger.info(
            "%s: %d valid, %d flagged-invalid, %d skipped",
            path.name, n_valid, n_invalid, n_skipped,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public — convert to pipeline contract types
    # ──────────────────────────────────────────────────────────────────────────

    def to_raw_observation(self, obs: KinexonObservation) -> RawPlayerObservation:
        """
        Map a KinexonObservation to the pipeline's RawPlayerObservation.

        latitude/longitude are None — Kinexon uses UWB, not GPS.
        Pitch coordinates and Kinexon-specific fields are preserved in raw_payload
        so downstream consumers can access them without touching the canonical fields.

        heart_rate_bpm is None when HR wearable is absent (expected for this export).
        """
        return RawPlayerObservation(
            source=self.config.source,
            player_external_id=str(obs.player_id),
            ts=obs.ts,
            latitude=None,            # UWB system — no GPS coordinates
            longitude=None,
            speed_ms=obs.speed_ms,
            acceleration_ms2=obs.acceleration_ms2,
            heart_rate_bpm=obs.heart_rate_bpm,
            hr_recovery_time_s=None,
            event_type="sprint" if obs.sprint_flag else None,
            match_id=obs.match_id,
            raw_payload={
                "kinexon_player_id": obs.player_id,
                "player_name":       obs.player_name,
                "jersey_number":     obs.jersey_number,
                "group_name":        obs.group_name,
                "timestamp_ms":      obs.timestamp_ms,
                "x_m":               obs.x_m,
                "y_m":               obs.y_m,
                "x_pitch":           obs.x_pitch,
                "y_pitch":           obs.y_pitch,
                "distance_delta_m":  obs.distance_delta_m,
                "sprint_flag":       int(obs.sprint_flag),
                "session_id":        obs.session_id,
                "valid":             obs.valid,
                "issues":            obs.issues,
            },
        )

    def to_event_dict(self, obs: KinexonObservation, elapsed_s: float = 0.0) -> dict:
        """
        Produce a normalised event dict compatible with process_window_direct().

        SequenceWindowBuilder.build_live_window() reads these keys from each event:
            speed_ms        — used directly
            heart_rate_bpm  — None → event treated as padding (is_real=False)
            x_pitch         — used directly; defaults to 50.0 inside builder
            y_pitch         — used directly; defaults to 50.0 inside builder

        TVL (TelemetryValidityLayer) completeness check requires:
            is_sprint       — TVL canonical name; sprint_flag is also emitted
                              for other consumers that expect the internal name

        LIMITATION: When heart_rate_bpm is None (this export), every event is
        marked as padding by build_live_window (is_real=False). Anomaly inference
        requires HR data to produce meaningful reconstruction errors.
        """
        return {
            "player_external_id": str(obs.player_id),
            "ts":                 obs.ts.isoformat(),
            "source":             self.config.source,
            "speed_ms":           obs.speed_ms,
            "acceleration_ms2":   obs.acceleration_ms2,
            "heart_rate_bpm":     obs.heart_rate_bpm,
            "x_pitch":            obs.x_pitch,
            "y_pitch":            obs.y_pitch,
            "distance_delta_m":   obs.distance_delta_m,
            "is_sprint":          int(obs.sprint_flag),   # TVL completeness key
            "sprint_flag":        int(obs.sprint_flag),   # retained for other consumers
            "elapsed_seconds":    elapsed_s,
            "match_id":           obs.match_id,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _normalise_coords(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Convert Kinexon centred-origin metre coordinates to [0, 100].

        X = long axis (handball: ±20 m court + run-off to ±22 m)
        Y = short axis (handball: ±10 m court + run-off to ±13 m)
        Positions outside court boundaries are clamped to [0, 100].
        """
        x_pitch = (x_m + self._half_length) / self.config.pitch_length_m * 100.0
        y_pitch = (y_m + self._half_width)  / self.config.pitch_width_m  * 100.0
        return (
            max(0.0, min(100.0, x_pitch)),
            max(0.0, min(100.0, y_pitch)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value: Optional[str]) -> Optional[float]:
    """Parse float from a CSV string cell; return None on empty or non-numeric."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None
