"""
match_resolver.py — PlayerDynamics

MatchResolver: automatically determines which match to operate on, so
callers never need to supply a match ID during normal operation.

Resolution priority
───────────────────
orchestrate (live mode):
  1. match.context Redis stream (Backend writes the running match_id here)
  2. match.events  Redis stream (first entry's match_id)
  3. active_match.json          (last-used match, written on each run)
  4. most recent in match_inventory.json
  5. interactive selection menu (if multiple equally valid)

replay mode:
  1. active_match.json          (most recently replayed/orchestrated match)
  2. most recent in match_inventory.json
  3. interactive selection menu

After resolving, the chosen match_id is persisted to active_match.json so
the next invocation of either command picks it up without user input.

active_match.json lives at data/active_match.json (next to
data/processed/match_inventory.json). It is written after every successful
resolution. Format:
    { "match_id": "3387", "label": "HSG Wetzlar vs SC Magdeburg (06/07/2026)", "resolved_via": "redis" }
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _ROOT / "data"
_ACTIVE_MATCH_FILE = _DEFAULT_DATA_DIR / "active_match.json"
_INVENTORY_PATH = _DEFAULT_DATA_DIR / "processed" / "match_inventory.json"

# Date format used in match_inventory.json
_DATE_FMT = "%m/%d/%Y"


def _parse_date(date_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(date_str.strip(), _DATE_FMT)
    except (ValueError, AttributeError):
        return None


class MatchResolver:
    """
    Resolves which match to use for orchestrate or replay.

    Usage
    -----
        resolver = MatchResolver()

        # orchestrate (live mode) — checks Redis first
        match_id = resolver.resolve(hint=args.match_id, prefer_live=True)

        # replay — skips Redis, uses local state
        match_id = resolver.resolve(hint=args.match_id, prefer_live=False)

    If hint is provided (--match-id override), it is returned immediately
    without any discovery — backward compatible with scripts/debugging.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        active_match_file: Optional[Path] = None,
        inventory_path: Optional[Path] = None,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self._active_file = Path(active_match_file) if active_match_file else _ACTIVE_MATCH_FILE
        self._inventory_path = Path(inventory_path) if inventory_path else _INVENTORY_PATH
        # Set after resolve() so callers can log the actual source
        self.last_resolved_via: Optional[str] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def resolve(self, hint: Optional[str], prefer_live: bool = False) -> str:
        """
        Return the match_id to use. Raises RuntimeError if nothing is
        resolvable (no inventory, no Redis, no active_match.json).

        hint        : value of --match-id if the user supplied one, else None
        prefer_live : True for `orchestrate`, False for `replay`
        """
        if hint:
            mid = str(hint).strip()
            label = self._label_for(mid)
            self._persist(mid, label, "cli_override")
            self.last_resolved_via = "cli --match-id override"
            logger.info("MatchResolver: using CLI override match_id=%s", mid)
            return mid

        if prefer_live:
            redis_result = self._from_redis()
            if redis_result:
                mid, label = redis_result
                self._persist(mid, label, "redis")
                self.last_resolved_via = "Redis (match.context)"
                self._announce(mid, label, "Redis (match.context)")
                return mid

        # Check active_match.json
        saved = self._from_active_file()
        if saved:
            mid, label = saved
            self._persist(mid, label, "active_match_file")
            self.last_resolved_via = "active_match.json"
            self._announce(mid, label, "active_match.json")
            return mid

        # Fall back to inventory
        inventory = self._sorted_inventory()
        if not inventory:
            raise RuntimeError(
                "No matches found. Run `python main.py ingest` to import match data first."
            )

        if len(inventory) == 1:
            m = inventory[0]
            mid, label = m["match_id"], self._format_label(m)
            self._persist(mid, label, "inventory_only")
            self.last_resolved_via = "match_inventory.json (only match)"
            self._announce(mid, label, "match_inventory.json (only match)")
            return mid

        # Most recent match
        most_recent = inventory[0]
        # Check if the top two have the same date (ambiguous)
        if len(inventory) >= 2:
            top_date = _parse_date(most_recent.get("date", ""))
            second_date = _parse_date(inventory[1].get("date", ""))
            if top_date and second_date and top_date == second_date:
                # Ambiguous — prompt
                mid = self._select_interactively(inventory)
                label = self._label_for(mid)
                self._persist(mid, label, "user_selection")
                self.last_resolved_via = "interactive selection"
                return mid

        mid = most_recent["match_id"]
        label = self._format_label(most_recent)
        self._persist(mid, label, "inventory_recent")
        self.last_resolved_via = "match_inventory.json (most recent)"
        self._announce(mid, label, "match_inventory.json (most recent)")
        return mid

    # ── Resolution sources ─────────────────────────────────────────────────────

    def _from_redis(self) -> Optional[tuple[str, str]]:
        """Read the most recent match.context entry from Redis. Returns (match_id, label) or None."""
        try:
            from config.redis_client import RedisConnectionPool, check_redis_connection, StreamTopics
            if not check_redis_connection():
                return None
            client = RedisConnectionPool.client()

            # Try match.context first (most authoritative — Backend writes it
            # with every clock tick during a live match)
            for stream in (StreamTopics.MATCH_CONTEXT, StreamTopics.MATCH_EVENTS):
                try:
                    entries = client.xrevrange(stream, "+", "-", "COUNT", 1)
                    if not entries:
                        continue
                    _entry_id, fields = entries[0]
                    mid = fields.get("match_id") or fields.get("b'match_id'")
                    if mid and str(mid).strip() and str(mid).strip() != "":
                        mid = str(mid).strip()
                        label = self._label_for(mid)
                        logger.info("MatchResolver: found match_id=%s in %s", mid, stream)
                        return mid, label
                except Exception as exc:
                    logger.debug("MatchResolver: could not read %s: %s", stream, exc)

        except Exception as exc:
            logger.debug("MatchResolver: Redis lookup failed: %s", exc)
        return None

    def _from_active_file(self) -> Optional[tuple[str, str]]:
        """Read active_match.json. Returns (match_id, label) or None."""
        if not self._active_file.exists():
            return None
        try:
            with open(self._active_file, encoding="utf-8") as f:
                data = json.load(f)
            mid = str(data.get("match_id", "")).strip()
            label = str(data.get("label", mid))
            if mid:
                logger.info("MatchResolver: found match_id=%s in %s", mid, self._active_file)
                return mid, label
        except Exception as exc:
            logger.debug("MatchResolver: could not read %s: %s", self._active_file, exc)
        return None

    def _sorted_inventory(self) -> list[dict]:
        """Load match_inventory.json sorted newest-first by date."""
        if not self._inventory_path.exists():
            return []
        try:
            with open(self._inventory_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("MatchResolver: could not read inventory: %s", exc)
            return []

        matches = data.get("matches", [])
        # Sort by parsed date descending (newest first)
        def _sort_key(m: dict) -> datetime:
            parsed = _parse_date(m.get("date", ""))
            return parsed if parsed else datetime.min

        return sorted(matches, key=_sort_key, reverse=True)

    # ── Interactive selection ──────────────────────────────────────────────────

    def _select_interactively(self, matches: list[dict]) -> str:
        """
        Print a numbered menu to stderr and read the user's choice from stdin.
        Falls back to the first (most recent) match if stdin is not a terminal.
        """
        if not sys.stdin.isatty():
            m = matches[0]
            logger.info(
                "MatchResolver: non-interactive stdin, auto-selecting most recent match_id=%s",
                m["match_id"],
            )
            self._announce(m["match_id"], self._format_label(m), "most recent (non-interactive)")
            return m["match_id"]

        print("\nMultiple matches available — select one:\n", file=sys.stderr)
        for i, m in enumerate(matches, start=1):
            label = self._format_label(m)
            print(f"  [{i}] {label}", file=sys.stderr)
        print(f"  [Enter] {self._format_label(matches[0])} (default)\n", file=sys.stderr)

        while True:
            try:
                raw = input("Match number [1]: ").strip()
            except (EOFError, KeyboardInterrupt):
                raw = ""
            if raw == "":
                return matches[0]["match_id"]
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(matches):
                    return matches[idx]["match_id"]
            except ValueError:
                pass
            print(f"  Please enter a number between 1 and {len(matches)}.", file=sys.stderr)

    # ── Persistence ────────────────────────────────────────────────────────────

    def persist(self, match_id: str, label: str, resolved_via: str) -> None:
        """
        Public entry point for external callers (e.g. the live orchestrate loop)
        to update active_match.json when the active match changes at runtime.
        """
        self._persist(match_id, label, resolved_via)
        self.last_resolved_via = resolved_via

    def _persist(self, match_id: str, label: str, resolved_via: str) -> None:
        """Write active_match.json so the next invocation picks this up."""
        try:
            self._active_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "match_id": match_id,
                "label": label,
                "resolved_via": resolved_via,
                "updated_at": datetime.utcnow().isoformat(),
            }
            with open(self._active_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.debug("MatchResolver: could not write %s: %s", self._active_file, exc)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _label_for(self, match_id: str) -> str:
        """Look up a human label from the inventory, fall back to the raw ID."""
        for m in self._sorted_inventory():
            if str(m["match_id"]) == str(match_id):
                return self._format_label(m)
        return f"match {match_id}"

    @staticmethod
    def _format_label(m: dict) -> str:
        parts = []
        if m.get("team_a"):
            parts.append(m["team_a"])
        if m.get("team_b"):
            parts.append(m["team_b"])
        name = " vs ".join(parts) if parts else f"match_{m['match_id']}"
        date = m.get("date", "")
        return f"{name} ({date})" if date else name

    @staticmethod
    def _announce(match_id: str, label: str, source: str) -> None:
        print(f"Auto-selected match: {label} [ID: {match_id}] (via {source})", flush=True)
