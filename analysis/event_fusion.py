"""
EventFusionEngine — PlayerDynamics

Produces one canonical TacticalEvent stream from two complementary sources:

    Kinexon TacticalEvents  — continuous positional/physical tracking; every
                               possession boundary, pass, shot, and sprint is
                               captured automatically with millisecond precision.

    Backend MatchEvents     — coach-confirmed outcome annotations entered live
                               via the frontend (GOAL / FEHLWURF / PARADE /
                               BALL-VERLUST); higher semantic precision, lower
                               temporal precision (human reaction latency 1-3 s).

Neither source alone is sufficient:
    • Kinexon records that a shot occurred but cannot determine whether it went
      in, was saved, or missed (no video integration yet).
    • Coach events carry the outcome label but lack the sub-second timing and
      physical context (ball speed, distance) that Kinexon provides.

Fusion rules
------------
For every Backend coach shot event (goal / missed_shot / save / turnover):

  1. Search the Kinexon stream for a matching event:
       - same canonical team_id (normalised via EventFusionConfig.team_id_aliases)
       - event_type matches the expected Kinexon type ("shot" or "turnover")
       - |timestamp_delta| ≤ config.dedup_window_seconds
       - nearest unmatched match within that window wins

  2. If a Kinexon match is found:
       - Enrich that Kinexon event in-place with metadata["shot_outcome"]
         (GOAL / SAVED / MISSED / UNKNOWN).  The Kinexon event_id, timestamp,
         and all other fields are preserved.  No new event is inserted.

  3. If no Kinexon match is found:
       - Create a synthetic TacticalEvent(event_type="shot" | "turnover",
         source="backend") with the coach-provided timestamp and outcome.

Each Kinexon event is claimed by at most one coach event (earliest delta wins
within the window; ties resolved by coach-event order). This guarantees that
one physical action never appears twice in the canonical stream.

Outcome labels
--------------
All shot events keep event_type="shot". The shot_outcome metadata field
distinguishes them:

    "GOAL"    — coach confirmed: shot scored
    "SAVED"   — coach confirmed: shot on target, stopped by goalkeeper
    "missed"  — coach confirmed: shot off target / blocked by field player
    "UNKNOWN" — Kinexon detected a shot but no coach annotation arrived
                within the dedup window

Coach vocabulary → canonical mapping
-------------------------------------
    Backend event_type   →  canonical event_type   shot_outcome  team search
    "goal"               →  "shot"                  GOAL          same team as coach
    "shot" (FEHLWURF)    →  "shot"                  MISSED        same team as coach
    "save" (PARADE)      →  "shot"                  SAVED         opponent team *
    "turnover"           →  "turnover"              (none)        same team as coach

* PARADE is entered by the defending team's coach. The shot being saved belongs
  to the attacking (opponent) team. Team inversion is only applied when
  opponent_team_id is supplied deterministically. If it cannot be resolved,
  save events are silently dropped rather than guessed.

Non-tactical event types (timeout, substitution, card, score_update,
coach_annotation) are ignored by the fusion engine and never appear in the
canonical stream — they carry no possession-boundary or shot-outcome semantics.

Determinism
-----------
fuse() is a pure function. Same inputs → same output always.
No random selection, no mutable module-level state, no timestamps from "now".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config.settings import CONFIG, EventFusionConfig
from ingestion.match_event import MatchEvent
from ingestion.tactical_event import TacticalEvent, make_event_id

logger = logging.getLogger(__name__)

# Backend event_type values that carry shot-outcome semantics.
# Mapped to: (kinexon_search_type, shot_outcome_label, use_opponent_team)
_BACKEND_SHOT_VOCABULARY: Dict[str, Tuple[str, str, bool]] = {
    "goal":     ("shot",     "GOAL",    False),
    "shot":     ("shot",     "MISSED",  False),  # FEHLWURF from frontend
    "save":     ("shot",     "SAVED",   True),   # PARADE — team inversion required
    "turnover": ("turnover", "",        False),  # passthrough, no shot_outcome
}

# Backend event types to silently ignore (not tactical terminators)
_IGNORED_BACKEND_TYPES = frozenset({
    "timeout", "substitution", "card", "score_update", "coach_annotation",
})


class EventFusionEngine:
    """
    Merges Kinexon TacticalEvents and Backend MatchEvents into a single
    canonical TacticalEvent stream.

    Usage
    -----
        engine = EventFusionEngine()  # or EventFusionEngine(config=my_config)
        canonical = engine.fuse(kinexon_events, match_events, opponent_team_id)
        possessions = possession_engine.generate(canonical)
    """

    def __init__(self, config: Optional[EventFusionConfig] = None) -> None:
        self.config = config or CONFIG.event_fusion

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def fuse(
        self,
        kinexon_events: List[TacticalEvent],
        match_events: List[MatchEvent],
        opponent_team_id: Optional[str] = None,
    ) -> List[TacticalEvent]:
        """
        Return the canonical event stream for a given snapshot of both sources.

        Parameters
        ----------
        kinexon_events  : All Kinexon TacticalEvents accumulated so far.
        match_events    : All Backend MatchEvents accumulated so far.
        opponent_team_id: Canonical team_id of the opposing team. Required for
                          correct handling of PARADE (save) events; if None,
                          save events are dropped (never guessed).

        Returns
        -------
        Sorted (by timestamp, terminators first) list of TacticalEvent.
        """
        if not match_events:
            return list(kinexon_events)

        # Work on a mutable copy — each element is either the original or a
        # replace()-enriched version (dataclass immutability enforced by replace).
        canonical: List[Optional[TacticalEvent]] = list(kinexon_events)
        claimed: set = set()   # indices of already-matched Kinexon slots
        synthetics: List[TacticalEvent] = []

        for me in match_events:
            if me.event_type in _IGNORED_BACKEND_TYPES:
                continue

            row = _BACKEND_SHOT_VOCABULARY.get(me.event_type)
            if row is None:
                logger.debug("EventFusionEngine: unrecognised MatchEvent type %r, skipping", me.event_type)
                continue

            kinexon_type, shot_outcome, use_opponent = row

            # Determine which team's Kinexon event to search for
            if use_opponent:
                if opponent_team_id is None:
                    logger.debug(
                        "EventFusionEngine: save event at %s dropped — opponent_team_id unknown",
                        me.timestamp,
                    )
                    continue
                search_team = self._normalize(opponent_team_id)
            else:
                if me.team_id is None:
                    logger.debug(
                        "EventFusionEngine: %r event at %s has no team_id, skipping",
                        me.event_type, me.timestamp,
                    )
                    continue
                search_team = self._normalize(me.team_id)

            best_idx, best_delta = self._find_best_match(
                canonical, claimed, kinexon_type, search_team, me.timestamp
            )

            if best_idx is not None:
                # Enrich the Kinexon event with the coach-confirmed outcome label
                existing = canonical[best_idx]
                new_meta = dict(existing.metadata) if existing.metadata else {}
                if shot_outcome:
                    new_meta["shot_outcome"] = shot_outcome
                canonical[best_idx] = replace(existing, metadata=new_meta)
                claimed.add(best_idx)
                logger.debug(
                    "EventFusionEngine: enriched Kinexon %s@%s with shot_outcome=%r (delta=%.2fs)",
                    kinexon_type, existing.timestamp, shot_outcome, best_delta,
                )
            else:
                # No matching Kinexon event — create a synthetic one so the
                # coach's observation still enters the possession pipeline.
                synthetic = self._make_synthetic(me, kinexon_type, shot_outcome, search_team)
                synthetics.append(synthetic)
                logger.debug(
                    "EventFusionEngine: synthetic %s created from %r at %s (no Kinexon match)",
                    kinexon_type, me.event_type, me.timestamp,
                )

        all_events = [e for e in canonical if e is not None] + synthetics
        _TERMINATOR_TYPES = ("shot", "turnover")
        return sorted(
            all_events,
            key=lambda e: (e.timestamp, 0 if e.event_type in _TERMINATOR_TYPES else 1),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _normalize(self, team_id: str) -> str:
        """Apply team_id_aliases; return team_id unchanged if not in aliases."""
        return self.config.team_id_aliases.get(team_id, team_id)

    def _find_best_match(
        self,
        events: List[Optional[TacticalEvent]],
        claimed: set,
        kinexon_type: str,
        canonical_team: str,
        target_ts: datetime,
    ) -> Tuple[Optional[int], float]:
        """
        Search for the nearest unclaimed Kinexon event matching:
            event_type == kinexon_type
            normalize(team_id) == canonical_team
            |delta| <= dedup_window_seconds

        Returns (index, delta_seconds) of the best match, or (None, inf).
        """
        window = self.config.dedup_window_seconds
        best_idx: Optional[int] = None
        best_delta: float = float("inf")

        for i, ev in enumerate(events):
            if ev is None or i in claimed:
                continue
            if ev.event_type != kinexon_type:
                continue
            if ev.team_id is None:
                continue
            if self._normalize(ev.team_id) != canonical_team:
                continue
            delta = abs((ev.timestamp - target_ts).total_seconds())
            if delta <= window and delta < best_delta:
                best_idx, best_delta = i, delta

        return best_idx, best_delta

    def _make_synthetic(
        self,
        me: MatchEvent,
        event_type: str,
        shot_outcome: str,
        team_id: str,
    ) -> TacticalEvent:
        """
        Build a synthetic TacticalEvent from a coach MatchEvent that had no
        matching Kinexon counterpart within the dedup window.

        The synthetic event carries source="backend" so downstream consumers
        can distinguish it from hardware-tracked events if needed.
        """
        meta: Dict = dict(me.metadata) if me.metadata else {}
        if shot_outcome:
            meta["shot_outcome"] = shot_outcome

        return TacticalEvent(
            event_id=make_event_id(
                me.match_id, "backend", me.timestamp, me.player_id, event_type
            ),
            timestamp=me.timestamp,
            match_id=me.match_id,
            team_id=team_id,
            player_id=me.player_id,
            event_type=event_type,
            metadata=meta,
            source="backend",
            confidence=1.0,
        )
