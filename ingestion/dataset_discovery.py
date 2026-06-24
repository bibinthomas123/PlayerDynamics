"""
dataset_discovery.py

Automated discovery, classification, metadata extraction, pairing, and
organization of Kinexon match exports dropped into data/incoming/ -- no
filename conventions, no hardcoded session IDs, no manual registration.

Real Kinexon exports come as three independent CSVs per match, each with an
unpredictable, team-name-embedded filename (e.g.
"Bergischer_HC_vs._SC_Magdeburg_Match_positions.csv",
"-Overview-Match_THW_Kiel_vs__SC_Magdeburg-hz_01_hz_02.csv"). Classification
and pairing therefore use column-header inspection and in-file metadata
(Session ID, timestamp ranges, player rosters) exclusively -- never the
filename.

File-type signatures (deterministic, verified against every real export in
this repo's data/ -- zero false positives):
    positions:  header has both "ts in ms" and "x in m"
    statistics: header has "Session ID" and a "Distance (m)"-style column
    events:     header has "Timestamp (ms)", "Player ID", and "Event type"

Pairing: statistics.csv is the only file type carrying an explicit Session
ID in its data rows. positions.csv and events.csv carry no session_id --
they're paired to a statistics bundle by timestamp-range overlap with that
session's Session begin/end window, tie-broken by player-roster overlap
(full names for positions, Player IDs for events). This is genuinely
metadata-based pairing, not filename matching.
"""
from __future__ import annotations

import csv
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

from config.settings import CONFIG, classify_ownership, OWNERSHIP_SCM, OWNERSHIP_OPPONENT

logger = logging.getLogger(__name__)

POSITIONS = "positions"
STATISTICS = "statistics"
EVENTS = "events"
UNSUPPORTED = "unsupported"

_EVENTS_HEADER_ROWS = 13  # 1 fixed-column header + 12 per-event-type legend rows (see ingestion/kinexon_events_features.py)
_BALL_GROUP = "Ball"
_POSITIONS_SAMPLE_ROWS = 20_000  # enough to see every rostered player at 20 Hz without loading 180MB files


def _read_header(path: Path) -> List[str]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f, delimiter=";"))


def classify_file(path: Path) -> str:
    """Column-header-only classification -- never inspects the filename."""
    try:
        header = _read_header(path)
    except Exception as exc:
        logger.warning("Could not read header of %s: %s", path, exc)
        return UNSUPPORTED

    header_set = set(header)
    has_distance_col = any(h.startswith("Distance (m)") or h.startswith("Distance / min") for h in header)

    if "ts in ms" in header_set and "x in m" in header_set:
        return POSITIONS
    if "Session ID" in header_set and has_distance_col:
        return STATISTICS
    if "Timestamp (ms)" in header_set and "Player ID" in header_set and "Event type" in header_set:
        return EVENTS
    return UNSUPPORTED


def _parse_kinexon_datetime(raw: str) -> Optional[datetime]:
    """Parses Kinexon's "MM/DD/YYYY H:MM:SS AM/PM"-style timestamp strings."""
    raw = raw.strip().strip('"')
    if not raw:
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y, %I:%M:%S.%f %p"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _extract_statistics_metadata(path: Path) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)

    if not rows:
        return {"session_id": None, "date": None, "player_count": None,
                "team_name": None, "opponent_name": None, "roster_names": set(), "roster_player_ids": set()}

    session_ids = {r.get("Session ID") for r in rows if r.get("Session ID")}
    session_id = int(next(iter(session_ids))) if len(session_ids) == 1 else None

    real_rows = [r for r in rows if r.get("Group name") and r.get("Group name") != _BALL_GROUP]
    player_count = len({r.get("Player ID") for r in real_rows if r.get("Player ID")}) or None
    roster_names = {r.get("Name", "").strip().strip('"') for r in real_rows if r.get("Name")}
    roster_player_ids = {r.get("Player ID") for r in real_rows if r.get("Player ID")}

    ownership_by_pid = {
        r.get("Player ID"): classify_ownership(r.get("Group name"), CONFIG.kinexon.scm_team_name)
        for r in real_rows if r.get("Player ID")
    }
    scm_player_count = sum(1 for o in ownership_by_pid.values() if o == OWNERSHIP_SCM)
    opponent_player_count = sum(1 for o in ownership_by_pid.values() if o == OWNERSHIP_OPPONENT)

    description = next((r.get("Description") for r in rows if r.get("Description")), None)
    team_name = opponent_name = None
    if description:
        description = description.strip().strip('"')
        for sep in (" vs. ", " vs "):
            if sep in description:
                parts = description.split(sep, 1)
                team_name, opponent_name = parts[0].strip(), parts[1].strip()
                break

    begin_raw = next((r.get("Session begin (Local timezone)") for r in rows if r.get("Session begin (Local timezone)")), None)
    begin_dt = _parse_kinexon_datetime(begin_raw) if begin_raw else None
    end_raw = next((r.get("Session end (Local timezone)") for r in rows if r.get("Session end (Local timezone)")), None)
    end_dt = _parse_kinexon_datetime(end_raw) if end_raw else None

    return {
        "session_id": session_id,
        "date": begin_dt.strftime("%m/%d/%Y") if begin_dt else None,
        "session_begin": begin_dt,
        "session_end": end_dt,
        "player_count": player_count,
        "scm_player_count": scm_player_count or None,
        "opponent_player_count": opponent_player_count or None,
        "team_name": team_name,
        "opponent_name": opponent_name,
        "roster_names": roster_names,
        "roster_player_ids": roster_player_ids,
    }


def _tail_line(path: Path, chunk_size: int = 8192) -> Optional[str]:
    """Reads the last complete data line of a large file without loading it fully."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - chunk_size))
        tail = f.read().decode("utf-8", errors="ignore")
    lines = [ln for ln in tail.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def _extract_positions_metadata(path: Path) -> dict:
    header = _read_header(path)
    idx = {name: i for i, name in enumerate(header)}
    ts_i, name_i, group_i, mid_i = idx.get("ts in ms"), idx.get("full name"), idx.get("group name"), idx.get("mapped id")

    roster_names: set = set()
    roster_ids: set = set()
    start_ts = None
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # header
        for n, row in enumerate(reader):
            if n >= _POSITIONS_SAMPLE_ROWS:
                break
            if group_i is not None and group_i < len(row) and row[group_i] == _BALL_GROUP:
                continue
            if start_ts is None and ts_i is not None and ts_i < len(row) and row[ts_i]:
                start_ts = row[ts_i]
            if name_i is not None and name_i < len(row) and row[name_i]:
                roster_names.add(row[name_i].strip().strip('"'))
            if mid_i is not None and mid_i < len(row) and row[mid_i]:
                roster_ids.add(row[mid_i])

    end_ts = None
    tail = _tail_line(path)
    if tail and ts_i is not None:
        cols = next(csv.reader([tail], delimiter=";"))
        if ts_i < len(cols) and cols[ts_i]:
            end_ts = cols[ts_i]

    def _ms_to_dt(ms: Optional[str]) -> Optional[datetime]:
        if not ms:
            return None
        try:
            return datetime.fromtimestamp(int(ms) / 1000.0)
        except (ValueError, OSError):
            return None

    start_dt, end_dt = _ms_to_dt(start_ts), _ms_to_dt(end_ts)
    return {
        "session_id": None,
        "date": start_dt.strftime("%m/%d/%Y") if start_dt else None,
        "session_begin": start_dt,
        "session_end": end_dt,
        "player_count": len(roster_ids) or None,
        "team_name": None,
        "opponent_name": None,
        "roster_names": roster_names,
        "roster_player_ids": set(),
    }


def _extract_events_metadata(path: Path) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    header = rows[0] if rows else []
    idx = {name: i for i, name in enumerate(header)}
    ts_i, pid_i = idx.get("Timestamp (ms)"), idx.get("Player ID")

    roster_ids: set = set()
    timestamps: List[int] = []
    for row in rows[_EVENTS_HEADER_ROWS:]:
        if ts_i is None or ts_i >= len(row) or not row[ts_i]:
            continue
        try:
            timestamps.append(int(row[ts_i]))
        except ValueError:
            continue
        if pid_i is not None and pid_i < len(row) and row[pid_i]:
            roster_ids.add(row[pid_i])

    def _ms_to_dt(ms: Optional[int]) -> Optional[datetime]:
        if ms is None:
            return None
        try:
            return datetime.fromtimestamp(ms / 1000.0)
        except (ValueError, OSError):
            return None

    start_dt = _ms_to_dt(min(timestamps)) if timestamps else None
    end_dt = _ms_to_dt(max(timestamps)) if timestamps else None
    return {
        "session_id": None,
        "date": start_dt.strftime("%m/%d/%Y") if start_dt else None,
        "session_begin": start_dt,
        "session_end": end_dt,
        "player_count": len(roster_ids) or None,
        "team_name": None,
        "opponent_name": None,
        "roster_names": set(),
        "roster_player_ids": roster_ids,
    }


def extract_metadata(path: Path, file_type: str) -> dict:
    """Extracts whatever metadata is genuinely present in the file's own
    contents -- never inferred, never read from the filename. Unavailable
    fields are None."""
    if file_type == STATISTICS:
        return _extract_statistics_metadata(path)
    if file_type == POSITIONS:
        return _extract_positions_metadata(path)
    if file_type == EVENTS:
        return _extract_events_metadata(path)
    return {"session_id": None, "date": None, "player_count": None,
            "team_name": None, "opponent_name": None, "roster_names": set(), "roster_player_ids": set()}


@dataclass
class DiscoveredFile:
    path: Path
    file_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class MatchBundle:
    session_id: Optional[int]
    date: Optional[str] = None
    player_count: Optional[int] = None
    scm_player_count: Optional[int] = None
    opponent_player_count: Optional[int] = None
    team_name: Optional[str] = None
    opponent_name: Optional[str] = None
    positions_file: Optional[Path] = None
    statistics_file: Optional[Path] = None
    events_file: Optional[Path] = None
    status: str = "incomplete"  # ready | incomplete | duplicate | orphaned

    @property
    def has_positions(self) -> bool:
        return self.positions_file is not None

    @property
    def has_statistics(self) -> bool:
        return self.statistics_file is not None

    @property
    def has_events(self) -> bool:
        return self.events_file is not None


def _windows_overlap(a_start, a_end, b_start, b_end, tolerance_s: float = 7200.0) -> bool:
    if a_start is None or b_start is None:
        return False
    a_end = a_end or a_start
    b_end = b_end or b_start
    tol = timedelta(seconds=tolerance_s)
    return (a_start - tol) <= b_end and (a_end + tol) >= b_start


def pair_bundles(discovered: List[DiscoveredFile]) -> List[MatchBundle]:
    """Anchors one bundle per statistics file (the only file type carrying
    an explicit session_id), then assigns each positions/events file to the
    bundle whose timestamp window it overlaps -- tie-broken by roster
    overlap (player names for positions, player IDs for events). Never
    matches on filename."""
    stats_files = [d for d in discovered if d.file_type == STATISTICS]
    other_files = [d for d in discovered if d.file_type in (POSITIONS, EVENTS)]

    bundles: Dict[Union[int, str], MatchBundle] = {}
    bundle_meta: Dict[int, dict] = {}
    for d in stats_files:
        sid = d.metadata.get("session_id")
        if sid is None:
            logger.warning("Statistics file %s has no resolvable Session ID -- skipped", d.path)
            continue
        bundles[sid] = MatchBundle(
            session_id=sid, date=d.metadata.get("date"), player_count=d.metadata.get("player_count"),
            scm_player_count=d.metadata.get("scm_player_count"),
            opponent_player_count=d.metadata.get("opponent_player_count"),
            team_name=d.metadata.get("team_name"), opponent_name=d.metadata.get("opponent_name"),
            statistics_file=d.path,
        )
        bundle_meta[sid] = d.metadata

    orphaned: List[DiscoveredFile] = []
    for d in other_files:
        candidates = []
        for sid, meta in bundle_meta.items():
            if _windows_overlap(meta.get("session_begin"), meta.get("session_end"),
                                 d.metadata.get("session_begin"), d.metadata.get("session_end")):
                if d.file_type == POSITIONS:
                    overlap = len(d.metadata.get("roster_names", set()) & meta.get("roster_names", set()))
                else:
                    overlap = len(d.metadata.get("roster_player_ids", set()) & meta.get("roster_player_ids", set()))
                candidates.append((overlap, sid))
        if not candidates:
            orphaned.append(d)
            continue
        candidates.sort(reverse=True)
        _, best_sid = candidates[0]
        if d.file_type == POSITIONS:
            bundles[best_sid].positions_file = d.path
        else:
            bundles[best_sid].events_file = d.path

    for d in orphaned:
        bundles[f"orphaned:{d.path.name}"] = MatchBundle(
            session_id=None, date=d.metadata.get("date"), player_count=d.metadata.get("player_count"),
            positions_file=d.path if d.file_type == POSITIONS else None,
            events_file=d.path if d.file_type == EVENTS else None,
            status="orphaned",
        )

    return list(bundles.values())


def detect_duplicates(bundles: List[MatchBundle], raw_matches_dir: Path) -> List[MatchBundle]:
    """Sets each bundle's final status. Never overwrites an existing
    data/raw_matches/<session_id>/ directory -- if one exists, the bundle
    is marked "duplicate" and organize_dataset() skips it entirely."""
    for b in bundles:
        if b.status == "orphaned":
            continue
        if b.session_id is not None and (raw_matches_dir / str(b.session_id)).exists():
            b.status = "duplicate"
        elif b.has_positions and b.has_statistics:
            b.status = "ready"
        else:
            b.status = "incomplete"
    return bundles


def build_inventory(bundles: List[MatchBundle]) -> List[dict]:
    return [
        {
            "session_id": b.session_id,
            "date": b.date,
            "player_count": b.player_count,
            "scm_player_count": b.scm_player_count,
            "opponent_player_count": b.opponent_player_count,
            "team_name": b.team_name,
            "opponent_name": b.opponent_name,
            "has_positions": b.has_positions,
            "has_statistics": b.has_statistics,
            "has_events": b.has_events,
            "status": b.status,
        }
        for b in bundles
    ]


def organize_dataset(bundles: List[MatchBundle], raw_matches_dir: Path) -> List[int]:
    """Moves every "ready" bundle's files into
    data/raw_matches/<session_id>/{positions.csv,statistics.csv,events.csv}.
    Duplicates and incomplete/orphaned bundles are left untouched -- no
    overwrite, no partial directories. Returns the list of session_ids
    actually organized this run."""
    organized = []
    for b in bundles:
        if b.status != "ready":
            continue
        dest_dir = raw_matches_dir / str(b.session_id)
        if dest_dir.exists():
            continue  # belt-and-suspenders -- detect_duplicates() should already have caught this
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(b.positions_file), str(dest_dir / "positions.csv"))
        shutil.move(str(b.statistics_file), str(dest_dir / "statistics.csv"))
        if b.events_file:
            shutil.move(str(b.events_file), str(dest_dir / "events.csv"))
        organized.append(b.session_id)
        logger.info("Organized session %s -> %s", b.session_id, dest_dir)
    return organized


class DatasetDiscoveryService:
    """Drop files -> run ingest -> everything updates automatically.

    scan() -> classify_file()/extract_metadata() -> pair_bundles() ->
    detect_duplicates() -> build_inventory() -> organize_dataset().
    No filename conventions, no hardcoded session IDs, no manual
    match registration anywhere in this chain.
    """

    def __init__(self, incoming_dir: Path, raw_matches_dir: Path, processed_dir: Path):
        self.incoming_dir = Path(incoming_dir)
        self.raw_matches_dir = Path(raw_matches_dir)
        self.processed_dir = Path(processed_dir)

    def scan(self) -> List[DiscoveredFile]:
        if not self.incoming_dir.exists():
            return []
        discovered = []
        for path in sorted(self.incoming_dir.rglob("*.csv")):
            file_type = classify_file(path)
            metadata = extract_metadata(path, file_type)
            discovered.append(DiscoveredFile(path=path, file_type=file_type, metadata=metadata))
        return discovered

    def run(self) -> dict:
        """Full discovery pipeline. Returns a JSON-serializable summary and
        writes data/processed/discovery_inventory.json (the pre-ingestion
        file-presence/readiness inventory -- distinct from multi_match_
        pipeline.py's post-ingestion match_inventory.json, which has a
        different schema and is written only after parquet processing)."""
        discovered = self.scan()
        unsupported = [d for d in discovered if d.file_type == UNSUPPORTED]
        classified = [d for d in discovered if d.file_type != UNSUPPORTED]

        bundles = pair_bundles(classified)
        bundles = detect_duplicates(bundles, self.raw_matches_dir)
        inventory = build_inventory(bundles)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        inventory_path = self.processed_dir / "discovery_inventory.json"
        inventory_path.write_text(__import__("json").dumps(inventory, indent=2, default=str))

        organized_sessions = organize_dataset(bundles, self.raw_matches_dir)

        return {
            "discovered_files": len(discovered),
            "unsupported_files": [str(d.path) for d in unsupported],
            "paired_bundles": len(bundles),
            "ready": sum(1 for b in bundles if b.status == "ready"),
            "duplicate": sum(1 for b in bundles if b.status == "duplicate"),
            "incomplete": sum(1 for b in bundles if b.status == "incomplete"),
            "orphaned": sum(1 for b in bundles if b.status == "orphaned"),
            "organized_sessions": organized_sessions,
            "inventory_path": str(inventory_path),
            "inventory": inventory,
        }
