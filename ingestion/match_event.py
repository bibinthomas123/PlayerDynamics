"""
MatchEvent / MatchContext — PlayerDynamics

PlayerDynamics-side decoders for the two streams Backend owns and
publishes (see BACKEND_INTEGRATION_IMPLEMENTATION.md §1 for the full
ownership rule):

    match.events   -- discrete match actions (shot, goal, save, turnover,
                       timeout, substitution, card, coach annotation, ...)
    match.context  -- running match state (score, clock, period)

Backend is the source of truth for both. PlayerDynamics does not compute,
validate business rules for, or own either schema -- these dataclasses
exist only so PlayerDynamics can decode what Backend publishes and make it
available as additional context alongside its own Kinexon-derived
TacticalEvent stream. Mirrors ingestion/tactical_event.py's shape
deliberately (event_id, timestamp, match_id, team_id, player_id,
event_type, metadata) so the two streams are easy to reason about
side-by-side, but MatchEvent.source is always "backend", never "kinexon".

Explicitly out of scope for this module (per the ownership rule):
    Recomputing score, clock, or period state. Validating that a
    substitution/card/etc. is legal. Any business logic at all -- this is
    a pure wire-format decoder, same role stream_codec.py plays for the
    analytics dataclasses, just for the two inbound-from-Backend streams.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class MatchEvent:
    """One discrete Backend-sourced match action (match.events)."""
    event_id: str
    timestamp: datetime
    match_id: Optional[str]
    team_id: Optional[str]
    player_id: Optional[int]
    event_type: str  # e.g. "shot","goal","save","turnover","timeout","substitution","card","coach_annotation","score_update"
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "backend"


@dataclass
class MatchContext:
    """A running-match-state snapshot from Backend (match.context)."""
    timestamp: datetime
    match_id: Optional[str]
    home_score: int
    away_score: int
    period: str  # e.g. "first_half","halftime","second_half","overtime","finished"
    game_clock_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "backend"
