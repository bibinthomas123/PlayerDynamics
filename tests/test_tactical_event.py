"""
tests/test_tactical_event.py

Validates TacticalEvent ingestion v1 (ingestion/tactical_event.py):

    TacticalEvent                  -- canonical dataclass
    KinexonTacticalEventAdapter    -- parses Kinexon events.csv -> TacticalEvent

Covers:
  A. Event parsing per Tier-1 type, against synthetic rows matching the
     exact column layout verified in data/events.csv
  B. Event type mapping (CSV label -> canonical event_type, incl. the
     Ball Possession / Lost / Recovery split)
  C. Team resolution via player_meta, including the unresolved case
  D. Deterministic event IDs (same input -> same ID; differing inputs,
     including the occurrence-index tie-breaker for duplicate keys, ->
     different IDs)
  E. Ball-entity filtering
  F. Real-data validation against data/events.csv (session 3387)

Run:
    pytest tests/test_tactical_event.py -v
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from ingestion.tactical_event import (
    KinexonTacticalEventAdapter,
    TacticalEvent,
    make_event_id,
)

DATA_DIR = _ROOT / "data"
EVENTS_PATH = DATA_DIR / "events.csv"

_HEADER_ROWS = 13


# ---------------------------------------------------------------------------
# Minimal stand-in for KinexonPlayerMeta (only .group_name is used)
# ---------------------------------------------------------------------------

@dataclass
class _FakeMeta:
    group_name: str


def _write_events_csv(tmp_path: Path, data_rows: list[list[str]]) -> Path:
    """Writes a minimal events.csv with 13 dummy header rows + given data rows."""
    path = tmp_path / "events.csv"
    with open(path, "w", encoding="latin-1", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        for _ in range(_HEADER_ROWS):
            writer.writerow(["header"])
        for row in data_rows:
            writer.writerow(row)
    return path


@pytest.fixture()
def adapter() -> KinexonTacticalEventAdapter:
    return KinexonTacticalEventAdapter()


# ---------------------------------------------------------------------------
# A. Event parsing per Tier-1 type (synthetic rows, exact real column layout)
# ---------------------------------------------------------------------------

class TestEventParsingPerType:

    def test_acceleration(self, adapter, tmp_path):
        row = ["1780837279000", "06/07/2026, 03:01:19 PM", "2057", "Daniel Pettersson",
               "Acceleration", "1.696", "5.89", "19.82", "3.25", "2.76", "16.87", "High"]
        path = _write_events_csv(tmp_path, [row])
        events = list(adapter.parse(path, match_id="3387"))
        assert len(events) == 1
        e = events[0]
        assert e.event_type == "acceleration_event"
        assert e.player_id == 2057
        assert e.metadata == {
            "duration_s": 1.696, "distance_m": 5.89, "max_speed_kmh": 19.82,
            "max_accel_ms2": 3.25, "avg_accel_ms2": 2.76, "speed_change_kmh": 16.87,
            "category": "High",
        }

    def test_deceleration(self, adapter, tmp_path):
        row = ["1780837278000", "06/07/2026, 03:01:18 PM", "1824", "Felix Claar",
               "Deceleration", "0.894", "3.12", "17.53", "-2.82", "-2.74", "-8.83", "Medium"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "deceleration_event"
        assert e.metadata["max_decel_ms2"] == -2.82
        assert e.metadata["category"] == "Medium"

    def test_change_of_direction(self, adapter, tmp_path):
        row = ["1780837290000", "06/07/2026, 03:01:30 PM", "1164", "Magnus Saugstrup",
               "Change of Direction", "114.8", "-2.05", "1.85", "Right"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "change_of_direction"
        assert e.metadata == {
            "magnitude_deg": 114.8, "max_decel_ms2": -2.05,
            "max_accel_ms2": 1.85, "direction": "Right",
        }

    def test_exertion(self, adapter, tmp_path):
        row = ["1780837242000", "06/07/2026, 03:00:42 PM", "2407", "Andreas Palicka",
               "Exertion", "2.052", "5.62", "6.54", "0.51", "2.17", "very high"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "exertion_event"
        assert e.metadata["exertion_level"] == "very high"
        assert e.metadata["distance_m"] == 0.51

    def test_impact(self, adapter, tmp_path):
        row = ["1780837276000", "06/07/2026, 03:01:16 PM", "732", "Gisli Thorgeir Kristjansson",
               "Impact", "6.8", "0"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "impact_event"
        assert e.metadata == {"magnitude_g": 6.8, "speed_kmh": 0.0}

    def test_jump(self, adapter, tmp_path):
        row = ["1780837242000", "06/07/2026, 03:00:42 PM", "2407", "Andreas Palicka",
               "Jump", "0.6", "0", "0.11", "0", "Low"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "jump_event"
        assert e.metadata["airtime_s"] == 0.6
        assert e.metadata["category"] == "Low"

    def test_pass(self, adapter, tmp_path):
        row = ["1780837241000", "06/07/2026, 03:00:41 PM", "1164", "Magnus Saugstrup",
               "Pass", "1.23", "9.77", "0", "1824", "successful"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "pass"
        assert e.metadata == {
            "distance_m": 1.23, "ball_speed_kmh": 9.77, "outplayed_opponents": 0,
            "receiving_player_id": 1824, "pass_type": "successful",
        }

    def test_shot(self, adapter, tmp_path):
        row = ["1780837280000", "06/07/2026, 03:01:20 PM", "1796", "Albin Lagergren",
               "Shots", "6.92", "85.42"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "shot"
        assert e.metadata == {"distance_m": 6.92, "ball_speed_kmh": 85.42}

    def test_sprint(self, adapter, tmp_path):
        row = ["1780837281000", "06/07/2026, 03:01:21 PM", "2057", "Daniel Pettersson",
               "Sprint", "1.629", "11.21", "25.35", "24.4", "Medium"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "sprint_event"
        assert e.metadata["max_speed_kmh"] == 25.35

    def test_ball_possession_start(self, adapter, tmp_path):
        row = ["1780836882000", "06/07/2026, 02:54:42 PM", "1796", "Albin Lagergren",
               "Ball Possession", "17.45"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "possession"
        assert e.metadata == {"duration_s": 17.45, "gained_via": "start"}

    def test_ball_possession_lost(self, adapter, tmp_path):
        row = ["1780837248000", "06/07/2026, 03:00:48 PM", "2057", "Daniel Pettersson",
               "Ball Possession Lost", "2260"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "turnover"
        assert e.metadata == {"opponent_id": 2260}

    def test_ball_possession_recovery(self, adapter, tmp_path):
        row = ["1780837248000", "06/07/2026, 03:00:48 PM", "2260", "Ahmed Nafea",
               "Ball Possession Recovery", "2057"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.event_type == "possession"
        assert e.metadata == {"gained_via": "recovery", "opponent_id": 2057}


# ---------------------------------------------------------------------------
# B. Event type mapping
# ---------------------------------------------------------------------------

class TestEventTypeMapping:

    def test_unknown_label_is_skipped_and_counted(self, adapter, tmp_path):
        row = ["1780837241000", "06/07/2026, 03:00:41 PM", "1164", "Magnus Saugstrup",
               "Totally Unknown Type", "1", "2"]
        path = _write_events_csv(tmp_path, [row])
        events = list(adapter.parse(path))
        assert events == []
        assert adapter.n_unknown_type == 1
        assert adapter.unknown_labels == {"Totally Unknown Type": 1}

    def test_too_few_columns_is_a_parse_error_not_a_crash(self, adapter, tmp_path):
        row = ["1780837241000", "06/07/2026, 03:00:41 PM", "1164"]  # < 6 cols
        path = _write_events_csv(tmp_path, [row])
        events = list(adapter.parse(path))
        assert events == []
        assert adapter.n_parse_errors == 1

    def test_all_twelve_real_labels_map_to_eleven_tier1_types(self, adapter, tmp_path):
        rows = [
            ["1000", "t", "1", "Player A", "Acceleration", "1", "1", "1", "1", "1", "1", "High"],
            ["1001", "t", "1", "Player A", "Ball Possession", "1"],
            ["1002", "t", "1", "Player A", "Ball Possession Lost", "2"],
            ["1003", "t", "1", "Player A", "Ball Possession Recovery", "2"],
            ["1004", "t", "1", "Player A", "Change of Direction", "1", "1", "1", "Left"],
            ["1005", "t", "1", "Player A", "Deceleration", "1", "1", "1", "1", "1", "1", "Low"],
            ["1006", "t", "1", "Player A", "Exertion", "1", "1", "1", "1", "1", "low"],
            ["1007", "t", "1", "Player A", "Impact", "1", "1"],
            ["1008", "t", "1", "Player A", "Jump", "1", "1", "1", "1", "Low"],
            ["1009", "t", "1", "Player A", "Pass", "1", "1", "0", "2", "successful"],
            ["1010", "t", "1", "Player A", "Shots", "1", "1"],
            ["1011", "t", "1", "Player A", "Sprint", "1", "1", "1", "1", "Low"],
        ]
        path = _write_events_csv(tmp_path, rows)
        events = list(adapter.parse(path))
        assert len(events) == 12
        types = {e.event_type for e in events}
        assert types == {
            "acceleration_event", "possession", "turnover", "change_of_direction",
            "deceleration_event", "exertion_event", "impact_event", "jump_event",
            "pass", "shot", "sprint_event",
        }
        assert adapter.n_unknown_type == 0
        assert adapter.n_parse_errors == 0


# ---------------------------------------------------------------------------
# C. Team resolution
# ---------------------------------------------------------------------------

class TestTeamResolution:

    def test_team_id_resolved_from_player_meta(self, adapter, tmp_path):
        row = ["1000", "t", "1164", "Magnus Saugstrup", "Sprint", "1", "1", "1", "1", "Low"]
        path = _write_events_csv(tmp_path, [row])
        meta = {1164: _FakeMeta(group_name="SC Magdeburg")}
        e = list(adapter.parse(path, player_meta=meta))[0]
        assert e.team_id == "SC Magdeburg"
        assert adapter.n_unresolved_team == 0

    def test_team_id_none_when_player_not_in_meta(self, adapter, tmp_path):
        row = ["1000", "t", "9999", "Unknown Player", "Sprint", "1", "1", "1", "1", "Low"]
        path = _write_events_csv(tmp_path, [row])
        meta = {1164: _FakeMeta(group_name="SC Magdeburg")}
        e = list(adapter.parse(path, player_meta=meta))[0]
        assert e.team_id is None
        assert adapter.n_unresolved_team == 1

    def test_team_id_none_when_no_meta_supplied_at_all(self, adapter, tmp_path):
        row = ["1000", "t", "1164", "Magnus Saugstrup", "Sprint", "1", "1", "1", "1", "Low"]
        path = _write_events_csv(tmp_path, [row])
        e = list(adapter.parse(path))[0]
        assert e.team_id is None
        assert adapter.n_unresolved_team == 1

    def test_two_teams_resolved_independently(self, adapter, tmp_path):
        rows = [
            ["1000", "t", "1164", "Magnus Saugstrup", "Sprint", "1", "1", "1", "1", "Low"],
            ["1001", "t", "2404", "Tristan Kirschner", "Sprint", "1", "1", "1", "1", "Low"],
        ]
        path = _write_events_csv(tmp_path, rows)
        meta = {
            1164: _FakeMeta(group_name="SC Magdeburg"),
            2404: _FakeMeta(group_name="HSG Wetzlar"),
        }
        events = list(adapter.parse(path, player_meta=meta))
        team_by_player = {e.player_id: e.team_id for e in events}
        assert team_by_player == {1164: "SC Magdeburg", 2404: "HSG Wetzlar"}


# ---------------------------------------------------------------------------
# D. Deterministic event IDs
# ---------------------------------------------------------------------------

class TestDeterministicEventIds:

    def test_same_inputs_produce_same_id(self):
        ts = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)
        id1 = make_event_id("3387", "kinexon", ts, 1164, "sprint_event", 0)
        id2 = make_event_id("3387", "kinexon", ts, 1164, "sprint_event", 0)
        assert id1 == id2

    def test_different_player_produces_different_id(self):
        ts = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)
        id1 = make_event_id("3387", "kinexon", ts, 1164, "sprint_event", 0)
        id2 = make_event_id("3387", "kinexon", ts, 1165, "sprint_event", 0)
        assert id1 != id2

    def test_different_occurrence_produces_different_id(self):
        """Tie-breaker for the ~15 duplicate (ts, player, type) keys found in
        session 3387 (e.g. two Impact rows for the same player at the same
        timestamp)."""
        ts = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)
        id1 = make_event_id("3387", "kinexon", ts, 732, "impact_event", 0)
        id2 = make_event_id("3387", "kinexon", ts, 732, "impact_event", 1)
        assert id1 != id2

    def test_reparsing_same_file_yields_identical_ids(self, adapter, tmp_path):
        rows = [
            ["1000", "t", "1", "Player A", "Impact", "1", "1"],
            ["1000", "t", "1", "Player A", "Impact", "2", "2"],  # duplicate key, diff metadata
        ]
        path = _write_events_csv(tmp_path, rows)
        ids_first_pass = [e.event_id for e in adapter.parse(path, match_id="3387")]

        adapter2 = KinexonTacticalEventAdapter()
        ids_second_pass = [e.event_id for e in adapter2.parse(path, match_id="3387")]

        assert ids_first_pass == ids_second_pass
        assert len(set(ids_first_pass)) == 2, "Duplicate keys must still get distinct IDs"

    def test_event_ids_unique_across_real_session(self):
        if not EVENTS_PATH.exists():
            pytest.skip(f"events.csv not found at {EVENTS_PATH}")
        adapter = KinexonTacticalEventAdapter()
        events = list(adapter.parse(EVENTS_PATH, match_id="3387"))
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids)), "All event_ids must be globally unique"


# ---------------------------------------------------------------------------
# E. Ball-entity filtering
# ---------------------------------------------------------------------------

class TestBallEntityFiltering:

    def test_ball_shot_is_filtered(self, adapter, tmp_path):
        row = ["1780837096000", "06/07/2026, 02:58:16 PM", "369", "Ball1 Ball",
               "Shots", "8.05", "54.1"]
        path = _write_events_csv(tmp_path, [row])
        events = list(adapter.parse(path))
        assert events == []
        assert adapter.n_skipped_ball == 1

    def test_real_player_named_ball_like_is_not_filtered(self, adapter, tmp_path):
        """Only exact ' Ball' suffix triggers filtering; do not over-match."""
        row = ["1000", "t", "500", "Bjorn Ballard", "Sprint", "1", "1", "1", "1", "Low"]
        path = _write_events_csv(tmp_path, [row])
        events = list(adapter.parse(path))
        assert len(events) == 1
        assert adapter.n_skipped_ball == 0


# ---------------------------------------------------------------------------
# F. Real-data validation: session 3387
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_adapter_and_events():
    if not EVENTS_PATH.exists():
        pytest.skip(f"events.csv not found at {EVENTS_PATH}")
    adapter = KinexonTacticalEventAdapter()
    events = list(adapter.parse(EVENTS_PATH, match_id="3387"))
    return adapter, events


class TestRealDataValidation:

    def test_parses_without_crashing(self, real_adapter_and_events):
        adapter, events = real_adapter_and_events
        assert len(events) > 0

    def test_no_unknown_event_types_in_real_data(self, real_adapter_and_events):
        adapter, events = real_adapter_and_events
        assert adapter.n_unknown_type == 0, (
            f"Unexpected unknown labels: {adapter.unknown_labels}"
        )

    def test_no_parse_errors_in_real_data(self, real_adapter_and_events):
        adapter, events = real_adapter_and_events
        assert adapter.n_parse_errors == 0

    def test_ball_entities_excluded(self, real_adapter_and_events):
        adapter, events = real_adapter_and_events
        assert adapter.n_skipped_ball > 0, "Session 3387 is known to log Ball1/Ball3 Shots rows"
        names_in_output = {e.player_id for e in events}
        assert 369 not in names_in_output and 371 not in names_in_output

    def test_all_eleven_tier1_types_present(self, real_adapter_and_events):
        adapter, events = real_adapter_and_events
        expected = {
            "possession", "turnover", "pass", "shot", "sprint_event",
            "acceleration_event", "deceleration_event", "change_of_direction",
            "exertion_event", "impact_event", "jump_event",
        }
        assert expected.issubset(set(adapter.counts_by_type.keys()))

    def test_all_event_ids_unique(self, real_adapter_and_events):
        adapter, events = real_adapter_and_events
        ids = [e.event_id for e in events]
        assert len(ids) == len(set(ids))

    def test_every_event_is_a_tactical_event_with_required_fields(self, real_adapter_and_events):
        adapter, events = real_adapter_and_events
        for e in events[:200]:  # spot-check a slice; full set is large
            assert isinstance(e, TacticalEvent)
            assert e.event_id and len(e.event_id) == 64  # sha256 hex digest
            assert e.timestamp.tzinfo is not None
            assert e.match_id == "3387"
            assert e.player_id is not None
            assert e.event_type
            assert e.source == "kinexon"
            assert e.confidence == 1.0

    def test_team_resolution_with_real_statistics_csv(self):
        """End-to-end: load real player_meta and confirm team_ids resolve to
        the two real club names instead of staying None for everyone."""
        stats_path = DATA_DIR / "statistics.csv"
        if not stats_path.exists():
            pytest.skip(f"statistics.csv not found at {stats_path}")
        from ingestion.kinexon_adapter import KinexonAdapter
        kinexon = KinexonAdapter()
        meta = kinexon.load_player_meta(stats_path)

        adapter = KinexonTacticalEventAdapter()
        events = list(adapter.parse(EVENTS_PATH, player_meta=meta, match_id="3387"))

        resolved_teams = {e.team_id for e in events if e.team_id is not None}
        assert resolved_teams, "At least some events must resolve a team_id"
        assert resolved_teams.issubset({"SC Magdeburg", "HSG Wetzlar"})

    def test_summary_report(self, real_adapter_and_events):
        """Not an assertion -- prints the real-data summary requested in the
        TacticalEvent ingestion v1 deliverable (run with -s to see it)."""
        adapter, events = real_adapter_and_events
        stats = adapter.stats()
        print("\n--- TacticalEvent ingestion v1: session 3387 summary ---")
        print(f"Total TacticalEvents parsed: {stats['n_parsed']}")
        print("Counts by type:")
        for t, n in sorted(stats["counts_by_type"].items(), key=lambda kv: -kv[1]):
            print(f"  {t:<22} {n}")
        print(f"Ball entity rows skipped: {stats['n_skipped_ball']}")
        print(f"Unresolved team_id (no meta supplied in this run): {stats['n_unresolved_team']}")
        print(f"Unknown event types: {stats['n_unknown_type']} {stats['unknown_labels']}")
        print(f"Parse errors: {stats['n_parse_errors']}")
        assert stats["n_parsed"] == len(events)
