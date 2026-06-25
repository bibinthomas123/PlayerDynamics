"""
dataset_manager.py — PlayerDynamics

Single source of truth for resolving a match_id to its CSV file paths.
No caller ever needs to know where files live — just supply a match_id.

Resolution order for each match_id:
  1. data/raw_matches/<match_id>/  — canonical organized home (ingest writes here)
  2. data/match_<match_id>/        — symlink directories (also written by ingest)
  If neither has positions.csv + statistics.csv, raises KeyError with a clear
  message listing available match IDs.

Match metadata (team names, date, player/event counts) comes from
data/processed/match_inventory.json (written by `main.py ingest`).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _ROOT / "data"
_DEFAULT_INVENTORY = _ROOT / "data" / "processed" / "match_inventory.json"


@dataclass(frozen=True)
class MatchDataset:
    """Resolved file paths + metadata for one ingested match."""

    match_id: str
    positions_path: Path
    statistics_path: Path
    events_path: Optional[Path]   # None if events.csv is absent (older exports)
    team_a: Optional[str] = None
    team_b: Optional[str] = None
    date: Optional[str] = None
    player_count: Optional[int] = None
    event_count: Optional[int] = None

    @property
    def label(self) -> str:
        parts = []
        if self.team_a:
            parts.append(self.team_a)
        if self.team_b:
            parts.append(self.team_b)
        name = " vs ".join(parts) if parts else f"match_{self.match_id}"
        return f"{name} ({self.date})" if self.date else name

    def __str__(self) -> str:
        ev = str(self.events_path) if self.events_path else "(no events.csv)"
        return (
            f"MatchDataset({self.match_id!r}: {self.label}\n"
            f"  positions  = {self.positions_path}\n"
            f"  statistics = {self.statistics_path}\n"
            f"  events     = {ev})"
        )


class MatchDatasetManager:
    """
    Resolves match IDs to file paths. Reads match_inventory.json for
    metadata. Never requires callers to supply file paths.

    Usage
    -----
        manager = MatchDatasetManager()
        dataset = manager.get("3387")
        print(dataset.label)             # "HSG Wetzlar vs SC Magdeburg (06/07/2026)"
        print(dataset.positions_path)    # Path(...)/data/raw_matches/3387/positions.csv
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        inventory_path: Optional[Path] = None,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._inventory_path = Path(inventory_path) if inventory_path else _DEFAULT_INVENTORY
        self._inventory: Optional[dict[str, dict]] = None

    # ── inventory ──────────────────────────────────────────────────────────────

    def _load_inventory(self) -> dict[str, dict]:
        if self._inventory is not None:
            return self._inventory
        if not self._inventory_path.exists():
            logger.warning(
                "match_inventory.json not found at %s — run `python main.py ingest` first.",
                self._inventory_path,
            )
            self._inventory = {}
            return self._inventory
        with open(self._inventory_path, encoding="utf-8") as f:
            data = json.load(f)
        self._inventory = {str(m["match_id"]): m for m in data.get("matches", [])}
        logger.debug("MatchDatasetManager: loaded inventory with %d matches", len(self._inventory))
        return self._inventory

    def available_match_ids(self) -> list[str]:
        """Return all match IDs known to the inventory."""
        return list(self._load_inventory().keys())

    def list_matches(self) -> list[MatchDataset]:
        """Return a MatchDataset for every known match."""
        return [self.get(mid) for mid in self.available_match_ids()]

    # ── resolution ─────────────────────────────────────────────────────────────

    def get(self, match_id: str) -> MatchDataset:
        """
        Resolve match_id to a MatchDataset. Raises KeyError with a helpful
        message (including available IDs) if the match cannot be found.
        """
        match_id = str(match_id)
        inv = self._load_inventory()
        meta = inv.get(match_id, {})

        match_dir = self._resolve_dir(match_id)
        if match_dir is None:
            available = self.available_match_ids()
            raise KeyError(
                f"Match {match_id!r} not found on disk. "
                f"Run `python main.py ingest` to import match data. "
                f"Available match IDs: {available or '(none — run ingest first)'}"
            )

        events_path = match_dir / "events.csv"
        return MatchDataset(
            match_id=match_id,
            positions_path=match_dir / "positions.csv",
            statistics_path=match_dir / "statistics.csv",
            events_path=events_path if events_path.exists() else None,
            team_a=meta.get("team_a"),
            team_b=meta.get("team_b"),
            date=meta.get("date"),
            player_count=meta.get("player_count"),
            event_count=meta.get("event_count"),
        )

    def _resolve_dir(self, match_id: str) -> Optional[Path]:
        """Find the directory that has both positions.csv and statistics.csv."""
        candidates = [
            self._data_dir / "raw_matches" / match_id,
            self._data_dir / f"match_{match_id}",
        ]
        for candidate in candidates:
            if (
                candidate.exists()
                and (candidate / "positions.csv").exists()
                and (candidate / "statistics.csv").exists()
            ):
                logger.debug("MatchDatasetManager: resolved %r → %s", match_id, candidate)
                return candidate
        return None
