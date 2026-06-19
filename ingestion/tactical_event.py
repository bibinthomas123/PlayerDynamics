"""
TacticalEvent — PlayerDynamics

Normalizes discrete match events into a single typed stream.

    TacticalEvent                  -- canonical dataclass
    KinexonTacticalEventAdapter    -- parses Kinexon events.csv -> TacticalEvent

Explicitly out of scope for this module (see the architecture doc):
    TeamState, Coach Insights, Frontend, PostgreSQL `match_events` ingestion.

Source data quirk
------------------
events.csv's own header rows (0-12) declare a taxonomy using plural /
parenthetical labels ("Accelerations", "Ball Possession (lost)", "Passes",
"Sprints", "Changes of Direction", ...) that do NOT match the actual
singular labels used in the data rows themselves ("Acceleration",
"Ball Possession Lost", "Pass", "Sprint", "Change of Direction", ...).
_ROW_PARSERS below is keyed on the verified actual data-row labels, not the
header declarations.

Tier-1 event taxonomy (directly recorded by Kinexon; confidence=1.0)
----------------------------------------------------------------------
    possession, turnover, pass, shot, sprint_event, acceleration_event,
    deceleration_event, change_of_direction, exertion_event, impact_event,
    jump_event

"Ball Possession" and "Ball Possession Recovery" both map to event_type=
"possession" (metadata.gained_via distinguishes "start" vs "recovery").
"Ball Possession Lost" maps to event_type="turnover". These three CSV
labels do NOT reliably pair 1:1 by timestamp in session 3387 (only ~42% of
"Lost" rows have a matching "Recovery" row at the same timestamp -- e.g. a
turnover recovered by the goalkeeper, or a ball that goes out of bounds,
produces no "Recovery" row at all) -- so each CSV row is treated as an
independent atomic event rather than fused into a single possession-change
event.

Team resolution
----------------
team_id is resolved via the existing player metadata mapping produced by
KinexonAdapter.load_player_meta() (statistics.csv -> KinexonPlayerMeta.group_name,
e.g. "SC Magdeburg" / "HSG Wetzlar"). No numeric team_id scheme exists yet
anywhere in the codebase, so team_id is the raw team-name string. If no
player_meta is supplied, or a player_id is absent from it, team_id is None
(the event is still emitted -- never dropped for this reason alone).

Ball-entity filtering
----------------------
Kinexon tags the match ball itself as a tracked "player" (e.g. "Ball1 Ball",
"Ball3 Ball" in this export). These rows are filtered out by name pattern
(name ends with " Ball"), independent of whether player_meta is supplied --
mirroring the precedent already established for positions.csv in
ingestion/kinexon_adapter.py (_BALL_GROUP filtering).
"""
from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from config.settings import KinexonConfig

logger = logging.getLogger(__name__)

# Kinexon's own taxonomy-declaration rows occupy lines 0-12; real data starts at 13.
_HEADER_ROWS = 13


def _looks_like_ball_entity(name: str) -> bool:
    return name.strip().endswith(" Ball")


@dataclass
class TacticalEvent:
    """
    Canonical normalized match event (see TACTICAL_EVENT_ARCHITECTURE.md §2).

    event_id is deterministic: sha256 of the identifying fields plus a
    per-key occurrence index. A handful of Kinexon rows share an identical
    (match_id, source, timestamp_ms, player_id, event_type) key (e.g. two
    simultaneous Impact readings logged in the same second) -- the
    occurrence index disambiguates those without breaking reproducibility,
    since it is derived purely from row order in the source file.
    """
    event_id: str
    timestamp: datetime
    match_id: Optional[str]
    team_id: Optional[str]
    player_id: Optional[int]
    event_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "kinexon"
    confidence: float = 1.0


def make_event_id(
    match_id: Optional[str],
    source: str,
    timestamp: datetime,
    player_id: Optional[int],
    event_type: str,
    occurrence: int = 0,
) -> str:
    """sha256(match_id|source|timestamp_ms|player_id|event_type|occurrence)."""
    ts_ms = int(timestamp.timestamp() * 1000)
    key = f"{match_id}|{source}|{ts_ms}|{player_id}|{event_type}|{occurrence}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Per-label row parsers
# Column indices verified directly against data/events.csv data rows (not the
# header taxonomy rows, which use different field names/order in places).
# Columns 0-4 are always: timestamp_ms, timestamp_local, player_id, name, label.
# ─────────────────────────────────────────────────────────────────────────────

def _f(row: List[str], idx: int) -> Optional[float]:
    try:
        v = row[idx].strip()
        return float(v) if v else None
    except (IndexError, ValueError):
        return None


def _i(row: List[str], idx: int) -> Optional[int]:
    v = _f(row, idx)
    return int(v) if v is not None else None


def _s(row: List[str], idx: int) -> Optional[str]:
    try:
        v = row[idx].strip()
        return v if v else None
    except IndexError:
        return None


def _parse_acceleration(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "acceleration_event", {
        "duration_s": _f(row, 5),
        "distance_m": _f(row, 6),
        "max_speed_kmh": _f(row, 7),
        "max_accel_ms2": _f(row, 8),
        "avg_accel_ms2": _f(row, 9),
        "speed_change_kmh": _f(row, 10),
        "category": _s(row, 11),
    }


def _parse_deceleration(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "deceleration_event", {
        "duration_s": _f(row, 5),
        "distance_m": _f(row, 6),
        "max_speed_kmh": _f(row, 7),
        "max_decel_ms2": _f(row, 8),
        "avg_decel_ms2": _f(row, 9),
        "speed_change_kmh": _f(row, 10),
        "category": _s(row, 11),
    }


def _parse_possession_start(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "possession", {
        "duration_s": _f(row, 5),
        "gained_via": "start",
    }


def _parse_possession_lost(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "turnover", {
        "opponent_id": _i(row, 5),
    }


def _parse_possession_recovery(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "possession", {
        "gained_via": "recovery",
        "opponent_id": _i(row, 5),
    }


def _parse_change_of_direction(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "change_of_direction", {
        "magnitude_deg": _f(row, 5),
        "max_decel_ms2": _f(row, 6),
        "max_accel_ms2": _f(row, 7),
        "direction": _s(row, 8),
    }


def _parse_exertion(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "exertion_event", {
        "duration_s": _f(row, 5),
        "accel_load_avg": _f(row, 6),
        "accel_load_max": _f(row, 7),
        "distance_m": _f(row, 8),
        "max_speed_kmh": _f(row, 9),
        "exertion_level": _s(row, 10),
    }


def _parse_impact(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "impact_event", {
        "magnitude_g": _f(row, 5),
        "speed_kmh": _f(row, 6),
    }


def _parse_jump(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "jump_event", {
        "airtime_s": _f(row, 5),
        "height_m": _f(row, 6),
        "distance_m": _f(row, 7),
        "jump_ratio_max": _f(row, 8),
        "category": _s(row, 9),
    }


def _parse_pass(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "pass", {
        "distance_m": _f(row, 5),
        "ball_speed_kmh": _f(row, 6),
        "outplayed_opponents": _i(row, 7),
        "receiving_player_id": _i(row, 8),
        "pass_type": _s(row, 9),
    }


def _parse_shot(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "shot", {
        "distance_m": _f(row, 5),
        "ball_speed_kmh": _f(row, 6),
    }


def _parse_sprint(row: List[str]) -> Tuple[str, Dict[str, Any]]:
    return "sprint_event", {
        "duration_s": _f(row, 5),
        "distance_m": _f(row, 6),
        "max_speed_kmh": _f(row, 7),
        "avg_speed_kmh": _f(row, 8),
        "category": _s(row, 9),
    }


_ROW_PARSERS: Dict[str, Callable[[List[str]], Tuple[str, Dict[str, Any]]]] = {
    "Acceleration": _parse_acceleration,
    "Ball Possession": _parse_possession_start,
    "Ball Possession Lost": _parse_possession_lost,
    "Ball Possession Recovery": _parse_possession_recovery,
    "Change of Direction": _parse_change_of_direction,
    "Deceleration": _parse_deceleration,
    "Exertion": _parse_exertion,
    "Impact": _parse_impact,
    "Jump": _parse_jump,
    "Pass": _parse_pass,
    "Shots": _parse_shot,
    "Sprint": _parse_sprint,
}


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────

class KinexonTacticalEventAdapter:
    """
    Parses Kinexon's events.csv export into a stream of TacticalEvent records.

    Usage
    -----
        kinexon  = KinexonAdapter()
        meta     = kinexon.load_player_meta(stats_path)     # existing adapter
        tactical = KinexonTacticalEventAdapter()
        for event in tactical.parse(events_path, player_meta=meta, match_id="3387"):
            ...
        stats = tactical.stats()   # counts_by_type, n_skipped_ball, ...
    """

    def __init__(self, config: Optional[KinexonConfig] = None) -> None:
        self.config = config or KinexonConfig()
        self.n_parsed = 0
        self.n_skipped_ball = 0
        self.n_unresolved_team = 0
        self.n_unknown_type = 0
        self.n_parse_errors = 0
        self.counts_by_type: Dict[str, int] = {}
        self.unknown_labels: Dict[str, int] = {}

    def parse(
        self,
        path: Path,
        player_meta: Optional[Dict[int, Any]] = None,
        match_id: Optional[str] = None,
    ) -> Iterator[TacticalEvent]:
        """
        Yield one TacticalEvent per parseable, non-ball data row in events.csv.

        player_meta: output of KinexonAdapter.load_player_meta() (dict keyed
        by Kinexon player_id -> KinexonPlayerMeta, with .group_name as the
        team label). If omitted, every event's team_id is None.
        """
        team_lookup: Dict[int, Optional[str]] = {
            pid: getattr(m, "group_name", None) for pid, m in (player_meta or {}).items()
        }
        occurrence: Dict[tuple, int] = {}

        with open(path, encoding="latin-1", newline="") as fh:
            reader = csv.reader(fh, delimiter=";")
            rows = list(reader)

        for lineno, row in enumerate(rows[_HEADER_ROWS:], start=_HEADER_ROWS + 1):
            if len(row) < 6:
                self.n_parse_errors += 1
                logger.warning("events.csv line %d: too few columns (%d), skipping", lineno, len(row))
                continue

            ts_raw, _ts_local, pid_raw, name, label = row[0], row[1], row[2], row[3], row[4]

            if _looks_like_ball_entity(name):
                self.n_skipped_ball += 1
                continue

            try:
                ts_ms = int(ts_raw.strip())
                player_id = int(pid_raw.strip())
            except (ValueError, AttributeError):
                self.n_parse_errors += 1
                logger.warning("events.csv line %d: unparseable timestamp/player_id, skipping", lineno)
                continue

            ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

            parser = _ROW_PARSERS.get(label)
            if parser is None:
                self.n_unknown_type += 1
                self.unknown_labels[label] = self.unknown_labels.get(label, 0) + 1
                logger.warning("events.csv line %d: unknown event label '%s', skipping", lineno, label)
                continue

            try:
                event_type, metadata = parser(row)
            except (IndexError, ValueError) as exc:
                self.n_parse_errors += 1
                logger.warning("events.csv line %d: failed to parse '%s' row: %s", lineno, label, exc)
                continue

            team_id = team_lookup.get(player_id)
            if team_id is None:
                self.n_unresolved_team += 1

            key = (match_id, self.config.source, ts_ms, player_id, event_type)
            occ = occurrence.get(key, 0)
            occurrence[key] = occ + 1

            event = TacticalEvent(
                event_id=make_event_id(match_id, self.config.source, ts, player_id, event_type, occ),
                timestamp=ts,
                match_id=match_id,
                team_id=team_id,
                player_id=player_id,
                event_type=event_type,
                metadata=metadata,
                source=self.config.source,
                confidence=1.0,
            )
            self.n_parsed += 1
            self.counts_by_type[event_type] = self.counts_by_type.get(event_type, 0) + 1
            yield event

    def stats(self) -> Dict[str, Any]:
        """Parsing summary, populated after parse() has been fully consumed."""
        return {
            "n_parsed": self.n_parsed,
            "counts_by_type": dict(self.counts_by_type),
            "n_skipped_ball": self.n_skipped_ball,
            "n_unresolved_team": self.n_unresolved_team,
            "n_unknown_type": self.n_unknown_type,
            "unknown_labels": dict(self.unknown_labels),
            "n_parse_errors": self.n_parse_errors,
        }
