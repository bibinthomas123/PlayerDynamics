"""
Unit tests for EventFusionEngine (analysis/event_fusion.py).

All tests are deterministic: same inputs always produce the same outputs.
No mocks, no randomness, no I/O.

Test matrix
-----------
TC-01  Coach-only replay     — no Kinexon events; coach events become synthetics
TC-02  Kinexon-only replay   — no coach events; stream passes through unchanged
TC-03  Mixed replay          — coach goal enriches matching Kinexon shot
TC-04  Goal enrichment       — shot_outcome=GOAL set on matched Kinexon shot
TC-05  Miss enrichment       — shot_outcome=MISSED set on FEHLWURF match
TC-06  Turnover passthrough  — coach turnover deduplicates Kinexon turnover
TC-07  Duplicate prevention  — one coach event claims at most one Kinexon slot
TC-08  Save inversion        — PARADE enriches opponent's shot; drops when unknown
TC-09  Deterministic replay  — identical inputs produce identical ordered output
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import List

import pytest

from analysis.event_fusion import EventFusionEngine
from config.settings import EventFusionConfig
from ingestion.match_event import MatchEvent
from ingestion.tactical_event import TacticalEvent, make_event_id


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts(offset_s: float) -> datetime:
    """Return a fixed UTC datetime offset by `offset_s` seconds."""
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).replace(
        second=0, microsecond=0
    ).replace(microsecond=int((offset_s % 1) * 1_000_000)).replace(
        second=int(offset_s) % 60,
        minute=(int(offset_s) // 60) % 60,
        hour=12 + (int(offset_s) // 3600),
    )


def _kinexon_shot(
    team_id: str = "SC Magdeburg",
    offset_s: float = 0.0,
    match_id: str = "M1",
    player_id: int = 1,
) -> TacticalEvent:
    ts = _ts(offset_s)
    return TacticalEvent(
        event_id=make_event_id(match_id, "kinexon", ts, player_id, "shot"),
        timestamp=ts,
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
        event_type="shot",
        metadata={"distance_m": 9.0},
        source="kinexon",
        confidence=1.0,
    )


def _kinexon_turnover(
    team_id: str = "SC Magdeburg",
    offset_s: float = 0.0,
    match_id: str = "M1",
    player_id: int = 1,
) -> TacticalEvent:
    ts = _ts(offset_s)
    return TacticalEvent(
        event_id=make_event_id(match_id, "kinexon", ts, player_id, "turnover"),
        timestamp=ts,
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
        event_type="turnover",
        metadata={},
        source="kinexon",
        confidence=1.0,
    )


def _backend_event(
    event_type: str,
    team_id: str = "SCM",
    offset_s: float = 0.0,
    match_id: str = "M1",
    player_id: int = 99,
) -> MatchEvent:
    ts = _ts(offset_s)
    return MatchEvent(
        event_id=f"be-{event_type}-{offset_s}",
        timestamp=ts,
        match_id=match_id,
        team_id=team_id,
        player_id=player_id,
        event_type=event_type,
        metadata={},
        source="backend",
    )


def _engine(window: float = 5.0) -> EventFusionEngine:
    return EventFusionEngine(
        config=EventFusionConfig(
            dedup_window_seconds=window,
            team_id_aliases={"SCM": "SC Magdeburg"},
        )
    )


def _shot_outcome(events: List[TacticalEvent], idx: int = 0) -> str:
    shots = [e for e in events if e.event_type == "shot"]
    return (shots[idx].metadata or {}).get("shot_outcome", "UNKNOWN")


# ─────────────────────────────────────────────────────────────────────────────
# TC-01  Coach-only replay
# ─────────────────────────────────────────────────────────────────────────────

def test_tc01_coach_only_replay():
    """With no Kinexon events, coach events become synthetic TacticalEvents."""
    match_events = [
        _backend_event("goal",     offset_s=10.0),
        _backend_event("turnover", offset_s=30.0),
    ]
    result = _engine().fuse([], match_events, opponent_team_id=None)

    assert len(result) == 2
    shot_events = [e for e in result if e.event_type == "shot"]
    turnover_events = [e for e in result if e.event_type == "turnover"]
    assert len(shot_events) == 1
    assert len(turnover_events) == 1
    assert shot_events[0].source == "backend"
    assert shot_events[0].metadata.get("shot_outcome") == "GOAL"
    assert result == sorted(result, key=lambda e: e.timestamp)


# ─────────────────────────────────────────────────────────────────────────────
# TC-02  Kinexon-only replay
# ─────────────────────────────────────────────────────────────────────────────

def test_tc02_kinexon_only():
    """With no match events, Kinexon stream is returned unchanged."""
    kinexon = [_kinexon_shot(offset_s=5.0), _kinexon_turnover(offset_s=20.0)]
    result = _engine().fuse(kinexon, [], opponent_team_id=None)

    assert len(result) == 2
    assert all(e.source == "kinexon" for e in result)
    # metadata must not have been modified
    assert "shot_outcome" not in result[0].metadata


# ─────────────────────────────────────────────────────────────────────────────
# TC-03  Mixed replay — coach goal enriches Kinexon shot
# ─────────────────────────────────────────────────────────────────────────────

def test_tc03_mixed_replay_enrichment():
    """
    Kinexon shot at T=0 + coach GOAL at T=2 → exactly one shot in output,
    enriched with shot_outcome=GOAL, Kinexon metadata preserved.
    """
    kinexon = [_kinexon_shot(offset_s=0.0)]
    match_events = [_backend_event("goal", offset_s=2.0)]

    result = _engine().fuse(kinexon, match_events, opponent_team_id=None)
    shots = [e for e in result if e.event_type == "shot"]

    assert len(shots) == 1, "Must produce exactly one shot (no duplicate)"
    assert shots[0].metadata.get("shot_outcome") == "GOAL"
    assert shots[0].metadata.get("distance_m") == 9.0, "Original Kinexon metadata preserved"
    assert shots[0].source == "kinexon", "Enriched event keeps kinexon source"


# ─────────────────────────────────────────────────────────────────────────────
# TC-04  Goal enrichment — shot_outcome=GOAL on matched event
# ─────────────────────────────────────────────────────────────────────────────

def test_tc04_goal_enrichment():
    kinexon = [_kinexon_shot(offset_s=0.0)]
    match_events = [_backend_event("goal", offset_s=1.5)]

    result = _engine().fuse(kinexon, match_events, opponent_team_id=None)
    assert _shot_outcome(result) == "GOAL"


# ─────────────────────────────────────────────────────────────────────────────
# TC-05  Miss enrichment — FEHLWURF (backend event_type="shot") → MISSED
# ─────────────────────────────────────────────────────────────────────────────

def test_tc05_miss_enrichment():
    """Backend event_type='shot' (FEHLWURF) matches Kinexon shot → MISSED."""
    kinexon = [_kinexon_shot(offset_s=0.0)]
    match_events = [_backend_event("shot", offset_s=1.0)]

    result = _engine().fuse(kinexon, match_events, opponent_team_id=None)
    shots = [e for e in result if e.event_type == "shot"]
    assert len(shots) == 1
    assert _shot_outcome(result) == "MISSED"


# ─────────────────────────────────────────────────────────────────────────────
# TC-06  Turnover passthrough — coach turnover deduplicates Kinexon turnover
# ─────────────────────────────────────────────────────────────────────────────

def test_tc06_turnover_dedup():
    """
    Kinexon turnover at T=0 + coach turnover at T=2 → exactly one turnover.
    The Kinexon event is kept (enriched, claimed); no synthetic added.
    """
    kinexon = [_kinexon_turnover(offset_s=0.0)]
    match_events = [_backend_event("turnover", offset_s=2.0)]

    result = _engine().fuse(kinexon, match_events, opponent_team_id=None)
    turnovers = [e for e in result if e.event_type == "turnover"]
    assert len(turnovers) == 1, "Duplicate turnover must be suppressed"
    assert turnovers[0].source == "kinexon"


# ─────────────────────────────────────────────────────────────────────────────
# TC-07  Duplicate prevention — one coach event per Kinexon slot
# ─────────────────────────────────────────────────────────────────────────────

def test_tc07_no_double_claim():
    """
    Two coach GOAL events close together can each claim at most one Kinexon shot.
    With only one Kinexon shot available the second coach event creates a synthetic.
    """
    kinexon = [_kinexon_shot(offset_s=0.0)]
    match_events = [
        _backend_event("goal", offset_s=1.0),  # claims kinexon shot
        _backend_event("goal", offset_s=2.0),  # no kinexon match → synthetic
    ]

    result = _engine().fuse(kinexon, match_events, opponent_team_id=None)
    shots = [e for e in result if e.event_type == "shot"]
    assert len(shots) == 2, "Second goal must produce a synthetic shot, not overwrite"
    sources = {e.source for e in shots}
    assert "kinexon" in sources and "backend" in sources


# ─────────────────────────────────────────────────────────────────────────────
# TC-08  Save inversion — PARADE enriches opponent's shot; drops when unknown
# ─────────────────────────────────────────────────────────────────────────────

def test_tc08_save_inversion_with_opponent():
    """
    Coach (SCM) enters PARADE (save). Shot belongs to opponent HSG Wetzlar.
    EventFusionEngine inverts team and enriches the opponent's Kinexon shot.
    """
    kinexon = [_kinexon_shot(team_id="HSG Wetzlar", offset_s=0.0)]
    match_events = [_backend_event("save", team_id="SCM", offset_s=1.5)]

    result = _engine().fuse(kinexon, match_events, opponent_team_id="HSG Wetzlar")
    shots = [e for e in result if e.event_type == "shot"]

    assert len(shots) == 1
    assert shots[0].metadata.get("shot_outcome") == "SAVED"
    assert shots[0].team_id == "HSG Wetzlar"


def test_tc08b_save_dropped_without_opponent():
    """Without opponent_team_id, PARADE events are dropped (never guessed)."""
    kinexon = [_kinexon_shot(team_id="HSG Wetzlar", offset_s=0.0)]
    match_events = [_backend_event("save", team_id="SCM", offset_s=1.5)]

    result = _engine().fuse(kinexon, match_events, opponent_team_id=None)
    shots = [e for e in result if e.event_type == "shot"]

    assert len(shots) == 1
    assert shots[0].metadata.get("shot_outcome", "UNKNOWN") == "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# TC-09  Deterministic replay — same inputs → same output always
# ─────────────────────────────────────────────────────────────────────────────

def test_tc09_deterministic():
    """fuse() is a pure function: calling it twice with the same inputs yields
    identical results (same event_ids, same metadata, same order)."""
    kinexon = [
        _kinexon_shot(offset_s=0.0),
        _kinexon_turnover(offset_s=15.0),
        _kinexon_shot(offset_s=30.0),
    ]
    match_events = [
        _backend_event("goal",     offset_s=1.0),
        _backend_event("turnover", offset_s=14.0),
        _backend_event("shot",     offset_s=31.0),  # FEHLWURF
    ]
    engine = _engine()

    result_a = engine.fuse(copy.deepcopy(kinexon), copy.deepcopy(match_events), "HSG Wetzlar")
    result_b = engine.fuse(copy.deepcopy(kinexon), copy.deepcopy(match_events), "HSG Wetzlar")

    assert len(result_a) == len(result_b)
    for a, b in zip(result_a, result_b):
        assert a.event_id == b.event_id
        assert a.event_type == b.event_type
        assert a.metadata == b.metadata
        assert a.timestamp == b.timestamp


# ─────────────────────────────────────────────────────────────────────────────
# TC-10  Outside dedup window — no merging when coach event is too late
# ─────────────────────────────────────────────────────────────────────────────

def test_tc10_outside_window_no_merge():
    """
    Kinexon shot at T=0, coach GOAL at T=10 (beyond 5 s window) →
    both appear as separate events in the canonical stream.
    """
    kinexon = [_kinexon_shot(offset_s=0.0)]
    match_events = [_backend_event("goal", offset_s=10.0)]

    result = _engine(window=5.0).fuse(kinexon, match_events, opponent_team_id=None)
    shots = [e for e in result if e.event_type == "shot"]

    assert len(shots) == 2, "Events outside the window must not be merged"
    sources = {e.source for e in shots}
    assert "kinexon" in sources and "backend" in sources
    # The Kinexon shot remains unenriched (no coach annotation within window)
    kinexon_shot = next(e for e in shots if e.source == "kinexon")
    assert kinexon_shot.metadata.get("shot_outcome", "UNKNOWN") == "UNKNOWN"
