"""
Multi-Match Dataset Pipeline — PlayerDynamics

Production ingestion layer for multiple Kinexon match exports, laid out as

    data/match_<id>/positions.csv
    data/match_<id>/events.csv
    data/match_<id>/statistics.csv

Scans every ``match_*`` directory, validates each export, builds a match
metadata index, and writes four unified Parquet datasets:

    matches.parquet    -- one row per match (metadata index)
    players.parquet    -- one row per (match_id, player_id): the
                           statistics.csv whole-session aggregate row,
                           concatenated across matches
    events.parquet      -- tidy long-format concatenation of every match's
                           raw events.csv data rows (event_type-specific
                           generic value slots kept as value_1..value_7,
                           undecoded -- semantic decoding is the job of
                           ingestion/kinexon_events_features.py, not this
                           layer)
    positions.parquet   -- concatenation of every match's positions.csv
                           20 Hz ticks

Incremental: a manifest (``_manifest.json`` in the output directory) records
each match directory's file sizes + mtimes. A match already in the manifest
with unchanged files is skipped entirely (no re-parsing) on subsequent runs.
This module does not modify, decode, or use any of the four PlayerDynamics
model classes -- it has no import of analysis.anomaly_detection or its
training/calibration code.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ingestion.kinexon_adapter import _BALL_GROUP

logger = logging.getLogger(__name__)

REQUIRED_FILES = ("positions.csv", "events.csv", "statistics.csv")
_EVENTS_HEADER_ROWS = 13
_MANIFEST_NAME = "_manifest.json"


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MatchValidation:
    match_dir: str
    match_id: Optional[str] = None
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _fingerprint(match_dir: Path) -> dict:
    """File-level fingerprint (size + mtime) used for incremental skip detection."""
    fp = {}
    for name in REQUIRED_FILES:
        p = match_dir / name
        if p.exists():
            st = p.stat()
            fp[name] = {"size": st.st_size, "mtime": st.st_mtime}
    return fp


def validate_match_directory(match_dir: Path) -> MatchValidation:
    """Validate one match_* directory. Does not raise -- all problems are
    collected into the returned MatchValidation (errors = hard failures that
    exclude the match from the unified datasets; warnings = data-quality
    issues that do not).
    """
    v = MatchValidation(match_dir=str(match_dir))

    for fname in REQUIRED_FILES:
        if not (match_dir / fname).exists():
            v.errors.append(f"missing required file: {fname}")
    if v.errors:
        v.ok = False
        return v

    # ── statistics.csv: parse roster + match metadata, check malformed rows ──
    try:
        stats_df = pd.read_csv(match_dir / "statistics.csv", sep=";", encoding="utf-8-sig")
    except Exception as exc:
        v.errors.append(f"statistics.csv unreadable/corrupted: {exc}")
        v.ok = False
        return v

    # Kinexon tracks the ball itself as a trackable "player" row in
    # statistics.csv (Group name == "Ball", e.g. "Ball1 Ball" / "Ball2 Ball" /
    # "Ball3 Ball"). Excluded here using the SAME convention
    # ingestion/kinexon_adapter.py's load_player_meta() already applies to
    # the live Kinexon path -- so this dataset's player_count/players.parquet
    # rows match what the live analytics.player_workload stream already
    # shows, instead of separately re-leaking ball entities into the
    # multi-match/trend path only.
    if "Group name" in stats_df.columns:
        is_ball = stats_df["Group name"].fillna("").str.strip() == _BALL_GROUP
        n_ball = int(is_ball.sum())
        if n_ball:
            v.warnings.append(
                f"Excluded {n_ball} ball pseudo-entit{'y' if n_ball == 1 else 'ies'} "
                f"from statistics.csv roster (Group name == '{_BALL_GROUP}')"
            )
            stats_df = stats_df[~is_ball].reset_index(drop=True)

    if "Session ID" not in stats_df.columns or stats_df.empty:
        v.errors.append("statistics.csv missing 'Session ID' column or has 0 rows")
        v.ok = False
        return v

    session_ids = stats_df["Session ID"].dropna().unique().tolist()
    if len(session_ids) != 1:
        v.warnings.append(f"statistics.csv has {len(session_ids)} distinct Session IDs (expected 1): {session_ids}")
    v.match_id = str(session_ids[0]) if session_ids else match_dir.name

    dir_id_match = re.match(r"^match_(.+)$", match_dir.name)
    if dir_id_match and dir_id_match.group(1) != v.match_id:
        v.warnings.append(
            f"directory name implies match_id={dir_id_match.group(1)!r} but "
            f"statistics.csv Session ID={v.match_id!r}"
        )

    roster_ids = set(stats_df["Player ID"].dropna().astype(int).tolist()) if "Player ID" in stats_df.columns else set()

    # ── positions.csv: schema + timestamp sanity + roster cross-check ────────
    try:
        pos_df = pd.read_csv(match_dir / "positions.csv", sep=";", encoding="utf-8-sig")
    except Exception as exc:
        v.errors.append(f"positions.csv unreadable/corrupted: {exc}")
        v.ok = False
        return v

    pos_required_cols = {"ts in ms", "mapped id", "x in m", "y in m"}
    missing_cols = pos_required_cols - set(pos_df.columns)
    if missing_cols:
        v.errors.append(f"positions.csv missing expected columns: {missing_cols}")
        v.ok = False
        return v

    n_position_rows = len(pos_df)
    bad_ts = pos_df["ts in ms"].isna() | (pos_df["ts in ms"] <= 0)
    if bad_ts.any():
        v.warnings.append(f"positions.csv: {int(bad_ts.sum())} rows with malformed/non-positive timestamps")

    position_ids = set(pos_df["mapped id"].dropna().astype(int).tolist())
    missing_from_roster = position_ids - roster_ids
    if missing_from_roster:
        v.warnings.append(
            f"positions.csv has {len(missing_from_roster)} player id(s) absent from "
            f"statistics.csv roster: {sorted(missing_from_roster)[:10]}"
        )
    missing_positions = roster_ids - position_ids
    if missing_positions:
        v.warnings.append(
            f"{len(missing_positions)} roster player(s) have zero position ticks "
            f"(DNP or substitute): {sorted(missing_positions)[:10]}"
        )

    # ── events.csv: polymorphic structure + timestamp sanity ────────────────
    try:
        with open(match_dir / "events.csv", encoding="utf-8-sig", newline="") as f:
            events_rows = list(csv.reader(f, delimiter=";"))
    except Exception as exc:
        v.errors.append(f"events.csv unreadable/corrupted: {exc}")
        v.ok = False
        return v

    if len(events_rows) < _EVENTS_HEADER_ROWS:
        v.errors.append(
            f"events.csv has only {len(events_rows)} rows -- expected >= "
            f"{_EVENTS_HEADER_ROWS} header/legend rows"
        )
        v.ok = False
        return v

    data_rows = events_rows[_EVENTS_HEADER_ROWS:]
    n_event_rows = 0
    n_bad_ts = 0
    n_bad_player = 0
    for row in data_rows:
        if len(row) < 5:
            n_bad_player += 1
            continue
        n_event_rows += 1
        try:
            ts_ms = int(row[0])
            if ts_ms <= 0:
                n_bad_ts += 1
        except (ValueError, IndexError):
            n_bad_ts += 1
        pid_raw = row[2].strip()
        if pid_raw:
            try:
                int(pid_raw)
            except ValueError:
                n_bad_player += 1

    if n_bad_ts:
        v.warnings.append(f"events.csv: {n_bad_ts} rows with malformed timestamps")
    if n_bad_player:
        v.warnings.append(f"events.csv: {n_bad_player} rows with malformed/short rows")

    v._stats_df = stats_df          # type: ignore[attr-defined]
    v._pos_df = pos_df              # type: ignore[attr-defined]
    v._events_data_rows = data_rows  # type: ignore[attr-defined]
    v._n_event_rows = n_event_rows   # type: ignore[attr-defined]
    v._n_position_rows = n_position_rows  # type: ignore[attr-defined]
    return v


def _parse_team_names(description: str) -> tuple:
    """'HSG Wetzlar vs. SC Magdeburg' -> ('HSG Wetzlar', 'SC Magdeburg')."""
    m = re.match(r"^(.*?)\s+vs\.?\s+(.*)$", str(description).strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return str(description).strip(), ""


def build_match_metadata(v: MatchValidation) -> dict:
    """Build the match_id/team_a/team_b/date/player_count/event_count/position_count row."""
    stats_df = v._stats_df  # type: ignore[attr-defined]
    description = stats_df["Description"].iloc[0] if "Description" in stats_df.columns and len(stats_df) else ""
    team_a, team_b = _parse_team_names(description)
    date_col = "Session begin date (Local timezone)" if "Session begin date (Local timezone)" in stats_df.columns else None
    date = str(stats_df[date_col].iloc[0]) if date_col and len(stats_df) else None
    player_count = int(stats_df["Player ID"].nunique()) if "Player ID" in stats_df.columns else 0

    return {
        "match_id": v.match_id,
        "team_a": team_a,
        "team_b": team_b,
        "date": date,
        "player_count": player_count,
        "event_count": int(v._n_event_rows),       # type: ignore[attr-defined]
        "position_count": int(v._n_position_rows),  # type: ignore[attr-defined]
        "match_dir": v.match_dir,
        "n_errors": len(v.errors),
        "n_warnings": len(v.warnings),
        "errors": v.errors,
        "warnings": v.warnings,
    }


def _events_to_long_df(v: MatchValidation, match_id: str) -> pd.DataFrame:
    """Tidy long-format events table: undecoded generic value slots (value_1..value_7)."""
    recs = []
    for row in v._events_data_rows:  # type: ignore[attr-defined]
        ts_ms = row[0] if len(row) > 0 else None
        ts_local = row[1] if len(row) > 1 else None
        player_id = row[2] if len(row) > 2 else None
        player_name = row[3] if len(row) > 3 else None
        event_type = row[4] if len(row) > 4 else None
        values = (row[5:12] + [None] * 7)[:7]
        recs.append({
            "match_id": match_id,
            "timestamp_ms": ts_ms,
            "timestamp_local": ts_local,
            "player_id": player_id,
            "player_name": player_name,
            "event_type": event_type,
            "value_1": values[0], "value_2": values[1], "value_3": values[2],
            "value_4": values[3], "value_5": values[4], "value_6": values[5],
            "value_7": values[6],
        })
    return pd.DataFrame(recs)


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────
class MultiMatchDatasetBuilder:
    """
    Usage
    -----
        builder = MultiMatchDatasetBuilder(data_root=Path("data"), output_dir=Path("data/processed"))
        report = builder.run()
    """

    def __init__(self, data_root: Path, output_dir: Path):
        self.data_root = data_root
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / _MANIFEST_NAME
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except Exception:
                logger.warning("Manifest %s unreadable -- starting fresh", self.manifest_path)
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def scan(self) -> List[Path]:
        if not self.data_root.exists():
            return []
        return sorted(p for p in self.data_root.glob("match_*") if p.is_dir())

    def run(self) -> dict:
        match_dirs = self.scan()
        match_rows: List[dict] = []
        new_players_frames: List[pd.DataFrame] = []
        new_events_frames: List[pd.DataFrame] = []
        new_positions_frames: List[pd.DataFrame] = []

        n_skipped_unchanged = 0
        n_processed = 0
        n_failed_validation = 0
        seen_match_ids: Dict[str, str] = {}
        duplicate_match_ids: List[dict] = []
        reprocessed_match_ids: set = set()

        for match_dir in match_dirs:
            dir_key = match_dir.name
            fp = _fingerprint(match_dir)
            cached = self.manifest.get(dir_key)

            if cached and cached.get("fingerprint") == fp and cached.get("ok"):
                n_skipped_unchanged += 1
                match_rows.append(cached["metadata"])
                seen_match_ids.setdefault(cached["metadata"]["match_id"], dir_key)
                continue

            v = validate_match_directory(match_dir)
            if not v.ok:
                n_failed_validation += 1
                self.manifest[dir_key] = {"fingerprint": fp, "ok": False, "errors": v.errors}
                match_rows.append({
                    "match_id": v.match_id or dir_key, "team_a": None, "team_b": None,
                    "date": None, "player_count": 0, "event_count": 0, "position_count": 0,
                    "match_dir": str(match_dir), "n_errors": len(v.errors), "n_warnings": 0,
                    "errors": v.errors, "warnings": [],
                })
                continue

            if v.match_id in seen_match_ids:
                duplicate_match_ids.append({
                    "match_id": v.match_id,
                    "directories": [seen_match_ids[v.match_id], dir_key],
                })
                logger.warning(
                    "Duplicate match_id=%s: %s and %s both claim it -- keeping the "
                    "first, skipping %s for the unified datasets",
                    v.match_id, seen_match_ids[v.match_id], dir_key, dir_key,
                )
                self.manifest[dir_key] = {"fingerprint": fp, "ok": False, "errors": [f"duplicate match_id {v.match_id}"]}
                continue
            seen_match_ids[v.match_id] = dir_key
            reprocessed_match_ids.add(v.match_id)

            metadata = build_match_metadata(v)
            match_rows.append(metadata)

            players_df = v._stats_df.copy()           # type: ignore[attr-defined]
            players_df.insert(0, "match_id", v.match_id)
            new_players_frames.append(players_df)

            new_events_frames.append(_events_to_long_df(v, v.match_id))

            pos_df = v._pos_df.copy()                  # type: ignore[attr-defined]
            pos_df.insert(0, "match_id", v.match_id)
            new_positions_frames.append(pos_df)

            self.manifest[dir_key] = {"fingerprint": fp, "ok": True, "metadata": metadata}
            n_processed += 1

        # reprocessed_match_ids covers BOTH brand-new matches (no-op: they
        # can't already be in the existing parquet) and matches whose files
        # changed since the last run (the actual reason this filter exists --
        # without it, a changed match's rows would be appended a second time
        # alongside its old, now-stale rows instead of replacing them).
        self._append_and_write("players.parquet", new_players_frames, reprocessed_match_ids)
        self._append_and_write("events.parquet", new_events_frames, reprocessed_match_ids)
        self._append_and_write("positions.parquet", new_positions_frames, reprocessed_match_ids)

        matches_df = pd.DataFrame(match_rows).drop_duplicates(subset=["match_id"], keep="first") if match_rows else pd.DataFrame()
        if not matches_df.empty:
            matches_df.to_parquet(self.output_dir / "matches.parquet", index=False)

        self._save_manifest()

        dataset_summary = {
            "matches_total": len(match_dirs),
            "matches_processed_this_run": n_processed,
            "matches_skipped_unchanged": n_skipped_unchanged,
            "matches_failed_validation": n_failed_validation,
            "duplicate_match_ids": duplicate_match_ids,
            "output_dir": str(self.output_dir.resolve()),
        }
        data_quality = {
            "matches_with_warnings": [
                {"match_id": r["match_id"], "warnings": r["warnings"]}
                for r in match_rows if r.get("warnings")
            ],
            "matches_with_errors": [
                {"match_id": r["match_id"], "match_dir": r.get("match_dir"), "errors": r["errors"]}
                for r in match_rows if r.get("errors")
            ],
        }
        match_inventory = {
            "matches": [
                {k: r[k] for k in ("match_id", "team_a", "team_b", "date", "player_count", "event_count", "position_count")}
                for r in match_rows
            ],
        }

        for name, payload in (
            ("dataset_summary.json", dataset_summary),
            ("data_quality_report.json", data_quality),
            ("match_inventory.json", match_inventory),
        ):
            (self.output_dir / name).write_text(json.dumps(payload, indent=2, default=str))

        logger.info(
            "MultiMatchDatasetBuilder: %d matches total | %d processed | %d skipped (unchanged) | "
            "%d failed validation | %d duplicate match_id pairs",
            len(match_dirs), n_processed, n_skipped_unchanged, n_failed_validation, len(duplicate_match_ids),
        )
        return {
            "dataset_summary": dataset_summary,
            "data_quality_report": data_quality,
            "match_inventory": match_inventory,
        }

    def _append_and_write(self, filename: str, new_frames: List[pd.DataFrame], reprocessed_match_ids: set) -> None:
        path = self.output_dir / filename
        if not new_frames:
            return  # nothing new -- existing parquet (if any) is left untouched
        new_df = pd.concat(new_frames, ignore_index=True)
        if path.exists():
            existing = pd.read_parquet(path)
            # Drop this run's match_ids from the existing data before
            # re-appending their freshly-parsed rows -- otherwise a
            # changed-and-reprocessed match ends up duplicated (old rows +
            # new rows) instead of replaced.
            existing = existing[~existing["match_id"].astype(str).isin(reprocessed_match_ids)]
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_parquet(path, index=False)
