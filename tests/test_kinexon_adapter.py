"""
test_kinexon_adapter.py

Loads real Kinexon CSV exports, runs the KinexonAdapter, validates output,
and prints a human-readable summary.

Run as pytest:
    pytest tests/test_kinexon_adapter.py -v -s

Run directly:
    python tests/test_kinexon_adapter.py
"""
from __future__ import annotations

import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

# ── path setup for direct execution ──────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest

from config.settings import KinexonConfig
from ingestion.kinexon_adapter import KinexonAdapter, KinexonObservation
from ingestion.pipeline import RawPlayerObservation

# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR       = _REPO_ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH     = DATA_DIR / "statistics.csv"

PREVIEW_COUNT  = 10   # first N observations to print


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def adapter() -> KinexonAdapter:
    return KinexonAdapter(config=KinexonConfig())


@pytest.fixture(scope="module")
def player_meta(adapter: KinexonAdapter) -> dict:
    if not STATS_PATH.exists():
        pytest.skip(f"statistics.csv not found at {STATS_PATH}")
    return adapter.load_player_meta(STATS_PATH)


@pytest.fixture(scope="module")
def observations(adapter: KinexonAdapter, player_meta: dict):
    """Collect ALL observations in memory (needed for statistics over the full stream)."""
    if not POSITIONS_PATH.exists():
        pytest.skip(f"positions.csv not found at {POSITIONS_PATH}")
    return list(
        adapter.stream_positions(
            POSITIONS_PATH,
            meta=player_meta,
            session_id="3387",
            match_id="wetzlar_vs_scm_20260607",
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — run without any data files (synthetic CSV rows)
# ─────────────────────────────────────────────────────────────────────────────

def _write_positions_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Helper: write a minimal positions.csv with real column names."""
    path = tmp_path / "positions.csv"
    cols = [
        "ts in ms", "formatted local time", "sensor id", "mapped id",
        "number", "full name", "league id", "group id", "group name",
        "x in m", "y in m", "z in m", "speed in m/s",
        "direction of movement in deg", "acceleration in m/s2",
        "total distance in m", "heart rate in bpm", "core temperature in celsius",
        "metabolic power in W/kg", "player orientation in deg",
        "player orientation category", "ball possession (id of possessed ball)",
        "acceleration load",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(";".join(cols) + "\n")
        for r in rows:
            fh.write(";".join(str(r.get(c, "")) for c in cols) + "\n")
    return path


_SYNTHETIC_ROWS = [
    # Player 1796 — 3 ticks, moving along X axis
    {"ts in ms": 1780837241000, "mapped id": 1796, "number": 21,
     "full name": "Albin Lagergren", "group name": "SC Magdeburg",
     "x in m": 0.0, "y in m": 0.0, "speed in m/s": 2.5, "acceleration in m/s2": 0.3},
    {"ts in ms": 1780837241050, "mapped id": 1796, "number": 21,
     "full name": "Albin Lagergren", "group name": "SC Magdeburg",
     "x in m": 0.125, "y in m": 0.0, "speed in m/s": 2.6, "acceleration in m/s2": 0.2},
    {"ts in ms": 1780837241100, "mapped id": 1796, "number": 21,
     "full name": "Albin Lagergren", "group name": "SC Magdeburg",
     "x in m": 0.250, "y in m": 0.0, "speed in m/s": 6.0, "acceleration in m/s2": 1.5},
    # Player 2331 — TW, near goal
    {"ts in ms": 1780837241000, "mapped id": 2331, "number": 1,
     "full name": "Nikola Portner", "group name": "SC Magdeburg",
     "x in m": -19.0, "y in m": 0.0, "speed in m/s": 0.5, "acceleration in m/s2": -0.1},
    # Ball entity — must be filtered out
    {"ts in ms": 1780837241000, "mapped id": 369, "number": 0,
     "full name": "Ball1 Ball", "group name": "Ball",
     "x in m": 5.0, "y in m": 2.0, "speed in m/s": 15.0, "acceleration in m/s2": 2.0},
    # Row with outlier speed (33.7 m/s) — must be capped and flagged
    {"ts in ms": 1780837242000, "mapped id": 1796, "number": 21,
     "full name": "Albin Lagergren", "group name": "SC Magdeburg",
     "x in m": 0.5, "y in m": 0.5, "speed in m/s": 33.7, "acceleration in m/s2": 0.0},
    # Row with missing x/y — must flag missing_coordinates
    {"ts in ms": 1780837243000, "mapped id": 1796, "number": 21,
     "full name": "Albin Lagergren", "group name": "SC Magdeburg",
     "x in m": "", "y in m": "", "speed in m/s": 2.0, "acceleration in m/s2": 0.1},
    # Row with missing mapped id — must be skipped entirely
    {"ts in ms": 1780837243000, "mapped id": "", "number": 99,
     "full name": "Unknown", "group name": "SC Magdeburg",
     "x in m": 5.0, "y in m": 1.0, "speed in m/s": 3.0, "acceleration in m/s2": 0.0},
]


class TestSyntheticData:
    """Unit tests using synthetic CSV rows — no real data files needed."""

    @pytest.fixture
    def synth_adapter(self) -> KinexonAdapter:
        return KinexonAdapter(config=KinexonConfig())

    @pytest.fixture
    def synth_observations(self, tmp_path, synth_adapter):
        positions_path = _write_positions_csv(tmp_path, _SYNTHETIC_ROWS)
        return list(synth_adapter.stream_positions(positions_path))

    # ── Filtering ──────────────────────────────────────────────────────────

    def test_ball_entities_excluded(self, synth_observations):
        for obs in synth_observations:
            assert obs.group_name != "Ball", "Ball entity leaked through filter"

    def test_missing_player_id_skipped(self, synth_observations):
        pids = {obs.player_id for obs in synth_observations}
        # The empty mapped_id row should not appear
        assert "" not in pids

    def test_correct_player_ids_present(self, synth_observations):
        pids = {obs.player_id for obs in synth_observations}
        assert 1796 in pids
        assert 2331 in pids

    # ── Coordinate normalisation ───────────────────────────────────────────

    def test_centre_normalises_to_50(self, synth_observations):
        centre = next(o for o in synth_observations
                      if o.player_id == 1796 and o.x_m == 0.0 and o.y_m == 0.0)
        assert abs(centre.x_pitch - 50.0) < 0.01
        assert abs(centre.y_pitch - 50.0) < 0.01

    def test_goalkeeper_left_normalises_correctly(self, synth_observations):
        gk = next(o for o in synth_observations if o.player_id == 2331)
        # x_m = -19.0 → x_pitch = (-19 + 20) / 40 * 100 = 2.5
        assert abs(gk.x_pitch - 2.5) < 0.1

    def test_x_y_pitch_in_range(self, synth_observations):
        for obs in synth_observations:
            assert 0.0 <= obs.x_pitch <= 100.0, f"x_pitch={obs.x_pitch} out of range"
            assert 0.0 <= obs.y_pitch <= 100.0, f"y_pitch={obs.y_pitch} out of range"

    # ── Speed handling ─────────────────────────────────────────────────────

    def test_outlier_speed_capped(self, synth_observations):
        cfg = KinexonConfig()
        for obs in synth_observations:
            if obs.speed_ms is not None:
                assert obs.speed_ms <= cfg.max_speed_ms
            # The 33.7 m/s row should have issues
        capped = [o for o in synth_observations
                  if any("speed_capped" in i for i in o.issues)]
        assert len(capped) == 1, "Expected exactly 1 speed-capped observation"

    # ── Sprint flag ────────────────────────────────────────────────────────

    def test_sprint_flag_set_at_threshold(self, synth_observations):
        cfg = KinexonConfig()
        # Tick with speed=6.0 m/s should be a sprint (≥5.5)
        fast = [o for o in synth_observations
                if o.player_id == 1796 and o.speed_ms is not None and o.speed_ms >= 5.5]
        assert all(o.sprint_flag for o in fast), "Sprint flag not set for speed ≥ threshold"

    def test_no_sprint_below_threshold(self, synth_observations):
        slow = [o for o in synth_observations
                if o.speed_ms is not None and o.speed_ms < 5.5 and not o.issues]
        assert all(not o.sprint_flag for o in slow)

    # ── Distance delta ─────────────────────────────────────────────────────

    def test_first_tick_delta_is_zero(self, synth_observations):
        first_1796 = next(o for o in synth_observations if o.player_id == 1796)
        assert first_1796.distance_delta_m == 0.0

    def test_subsequent_tick_delta_positive(self, synth_observations):
        player_obs = [o for o in synth_observations
                      if o.player_id == 1796 and not o.issues]
        second = player_obs[1]
        assert second.distance_delta_m > 0.0

    def test_delta_matches_euclidean_distance(self, synth_observations):
        import math
        valid_1796 = [o for o in synth_observations
                      if o.player_id == 1796 and "missing_coordinates" not in o.issues]
        if len(valid_1796) >= 2:
            o1, o2 = valid_1796[0], valid_1796[1]
            expected = math.sqrt((o2.x_m - o1.x_m)**2 + (o2.y_m - o1.y_m)**2)
            assert abs(o2.distance_delta_m - expected) < 1e-9

    # ── Validity flags ─────────────────────────────────────────────────────

    def test_missing_coords_flagged_invalid(self, synth_observations):
        missing = [o for o in synth_observations
                   if "missing_coordinates" in o.issues]
        assert len(missing) == 1
        assert not missing[0].valid

    # ── RawPlayerObservation conversion ───────────────────────────────────

    def test_raw_observation_has_correct_source(self, synth_adapter, synth_observations):
        for obs in synth_observations[:3]:
            raw = synth_adapter.to_raw_observation(obs)
            assert raw.source == "kinexon"

    def test_raw_observation_external_id_is_str(self, synth_adapter, synth_observations):
        for obs in synth_observations:
            raw = synth_adapter.to_raw_observation(obs)
            assert isinstance(raw.player_external_id, str)
            assert raw.player_external_id == str(obs.player_id)

    def test_raw_observation_no_gps(self, synth_adapter, synth_observations):
        for obs in synth_observations:
            raw = synth_adapter.to_raw_observation(obs)
            assert raw.latitude is None
            assert raw.longitude is None

    def test_raw_observation_pitch_coords_in_payload(self, synth_adapter, synth_observations):
        for obs in synth_observations:
            raw = synth_adapter.to_raw_observation(obs)
            assert "x_pitch" in raw.raw_payload
            assert "y_pitch" in raw.raw_payload
            assert abs(raw.raw_payload["x_pitch"] - obs.x_pitch) < 1e-9
            assert abs(raw.raw_payload["y_pitch"] - obs.y_pitch) < 1e-9

    def test_valid_raw_observation_passes_is_valid(self, synth_adapter, synth_observations):
        valid_obs = [o for o in synth_observations if o.valid]
        for obs in valid_obs:
            raw = synth_adapter.to_raw_observation(obs)
            assert raw.is_valid(), (
                f"is_valid() failed for valid obs: player_id={obs.player_id} "
                f"speed={obs.speed_ms} hr={obs.heart_rate_bpm}"
            )

    # ── Event dict conversion ──────────────────────────────────────────────

    def test_event_dict_required_keys(self, synth_adapter, synth_observations):
        required = {"player_external_id", "speed_ms", "heart_rate_bpm",
                    "x_pitch", "y_pitch", "distance_delta_m", "sprint_flag"}
        for obs in synth_observations:
            evt = synth_adapter.to_event_dict(obs)
            assert required.issubset(evt.keys())

    def test_event_dict_hr_is_none(self, synth_adapter, synth_observations):
        for obs in synth_observations:
            evt = synth_adapter.to_event_dict(obs)
            assert evt["heart_rate_bpm"] is None, \
                "heart_rate_bpm should be None — wearable absent in synthetic data"


# ─────────────────────────────────────────────────────────────────────────────
# Tests — data loading
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoading:

    def test_positions_file_exists(self):
        assert POSITIONS_PATH.exists(), f"positions.csv missing at {POSITIONS_PATH}"

    def test_stats_file_exists(self):
        assert STATS_PATH.exists(), f"statistics.csv missing at {STATS_PATH}"

    def test_player_meta_loads(self, player_meta: dict):
        assert len(player_meta) > 0, "No player metadata loaded from statistics.csv"

    def test_player_meta_excludes_balls(self, player_meta: dict):
        for pid, pm in player_meta.items():
            assert pm.group_name != "Ball", f"Ball entity {pid} leaked into player meta"

    def test_observations_produced(self, observations: list):
        assert len(observations) > 0, "No observations produced from positions.csv"

    def test_minimum_observation_count(self, observations: list):
        # At 20 Hz with ≥10 players for ≥1 minute: at least 12,000 ticks
        assert len(observations) >= 12_000, (
            f"Too few observations: {len(observations)}. "
            "Expected at least 12,000 for a real match export."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — observation fields
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationFields:

    def test_player_id_is_int(self, observations: list):
        for obs in observations[:100]:
            assert isinstance(obs.player_id, int), \
                f"player_id should be int, got {type(obs.player_id)}: {obs.player_id}"

    def test_no_ball_entities(self, observations: list):
        for obs in observations:
            assert obs.group_name != "Ball", \
                f"Ball entity (player_id={obs.player_id}) should be filtered out"

    def test_timestamps_are_datetime(self, observations: list):
        from datetime import datetime
        for obs in observations[:100]:
            assert isinstance(obs.ts, datetime), \
                f"ts should be datetime, got {type(obs.ts)}"

    def test_x_pitch_range(self, observations: list):
        for obs in observations:
            assert 0.0 <= obs.x_pitch <= 100.0, \
                f"x_pitch={obs.x_pitch} out of [0, 100] for player {obs.player_id}"

    def test_y_pitch_range(self, observations: list):
        for obs in observations:
            assert 0.0 <= obs.y_pitch <= 100.0, \
                f"y_pitch={obs.y_pitch} out of [0, 100] for player {obs.player_id}"

    def test_speed_within_cap(self, observations: list):
        cfg = KinexonConfig()
        for obs in observations:
            if obs.speed_ms is not None:
                assert obs.speed_ms <= cfg.max_speed_ms, \
                    f"speed_ms={obs.speed_ms} exceeds cap {cfg.max_speed_ms}"
                assert obs.speed_ms >= 0.0, \
                    f"speed_ms={obs.speed_ms} is negative for player {obs.player_id}"

    def test_distance_delta_non_negative(self, observations: list):
        for obs in observations:
            assert obs.distance_delta_m >= 0.0, \
                f"distance_delta_m={obs.distance_delta_m} is negative"

    def test_sprint_flag_consistent_with_speed(self, observations: list):
        cfg = KinexonConfig()
        for obs in observations:
            if obs.speed_ms is not None:
                expected = obs.speed_ms >= cfg.sprint_threshold_ms
                assert obs.sprint_flag == expected, (
                    f"sprint_flag mismatch: speed={obs.speed_ms:.2f} "
                    f"threshold={cfg.sprint_threshold_ms} flag={obs.sprint_flag}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — RawPlayerObservation conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestRawObservationConversion:

    def test_converts_without_error(self, adapter: KinexonAdapter, observations: list):
        for obs in observations[:50]:
            raw = adapter.to_raw_observation(obs)
            assert isinstance(raw, RawPlayerObservation)

    def test_player_external_id_matches_player_id(
        self, adapter: KinexonAdapter, observations: list
    ):
        for obs in observations[:50]:
            raw = adapter.to_raw_observation(obs)
            assert raw.player_external_id == str(obs.player_id)

    def test_source_is_kinexon(self, adapter: KinexonAdapter, observations: list):
        raw = adapter.to_raw_observation(observations[0])
        assert raw.source == "kinexon"

    def test_no_gps_coordinates(self, adapter: KinexonAdapter, observations: list):
        for obs in observations[:50]:
            raw = adapter.to_raw_observation(obs)
            assert raw.latitude is None, "latitude should be None for UWB source"
            assert raw.longitude is None, "longitude should be None for UWB source"

    def test_raw_payload_contains_pitch_coords(
        self, adapter: KinexonAdapter, observations: list
    ):
        for obs in observations[:50]:
            raw = adapter.to_raw_observation(obs)
            assert raw.raw_payload is not None
            assert "x_pitch" in raw.raw_payload
            assert "y_pitch" in raw.raw_payload
            assert "distance_delta_m" in raw.raw_payload
            assert "kinexon_player_id" in raw.raw_payload

    def test_raw_observation_is_valid(self, adapter: KinexonAdapter, observations: list):
        valid_obs = [o for o in observations[:200] if o.valid]
        assert len(valid_obs) > 0, "No valid observations found in first 200"
        for obs in valid_obs:
            raw = adapter.to_raw_observation(obs)
            assert raw.is_valid(), (
                f"is_valid() returned False for a KinexonObservation marked valid. "
                f"player_id={obs.player_id} issues={obs.issues}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests — event dict conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestEventDictConversion:

    def test_event_dict_has_required_keys(
        self, adapter: KinexonAdapter, observations: list
    ):
        required = {"player_external_id", "speed_ms", "heart_rate_bpm",
                    "x_pitch", "y_pitch", "distance_delta_m"}
        for obs in observations[:50]:
            evt = adapter.to_event_dict(obs)
            missing = required - evt.keys()
            assert not missing, f"Event dict missing keys: {missing}"

    def test_event_dict_x_y_pitch_range(
        self, adapter: KinexonAdapter, observations: list
    ):
        for obs in observations[:200]:
            evt = adapter.to_event_dict(obs)
            assert 0.0 <= evt["x_pitch"] <= 100.0
            assert 0.0 <= evt["y_pitch"] <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Standalone human-readable output (also runs as part of pytest -s)
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(observations: list[KinexonObservation], adapter: KinexonAdapter) -> None:
    total      = len(observations)
    n_valid    = sum(1 for o in observations if o.valid)
    n_invalid  = total - n_valid
    n_sprint   = sum(1 for o in observations if o.sprint_flag)
    n_no_hr    = sum(1 for o in observations if o.heart_rate_bpm is None)

    players    = Counter(o.player_id for o in observations)
    issues_all = Counter()
    for o in observations:
        for iss in o.issues:
            issues_all[iss] += 1

    # Per-player speed stats
    speed_by_player: dict[int, list[float]] = defaultdict(list)
    for o in observations:
        if o.speed_ms is not None:
            speed_by_player[o.player_id].append(o.speed_ms)

    sep = "-" * 70

    print(f"\n{sep}")
    print("KINEXON ADAPTER - VALIDATION SUMMARY")
    print(sep)
    print(f"  Source file   : {POSITIONS_PATH}")
    print(f"  Total rows    : {total:,}")
    print(f"  Players seen  : {len(players)}")
    print(f"  Valid obs     : {n_valid:,}  ({n_valid/total*100:.1f}%)")
    print(f"  Flagged obs   : {n_invalid:,}  ({n_invalid/total*100:.1f}%)")
    print(f"  Sprint events : {n_sprint:,}  ({n_sprint/total*100:.1f}%)")
    print(f"  HR absent     : {n_no_hr:,}  ({n_no_hr/total*100:.1f}%)")

    if issues_all:
        print(f"\n  Validation issues breakdown:")
        for iss, cnt in issues_all.most_common():
            print(f"    {iss:<45s} {cnt:>8,}")
    else:
        print("\n  No validation issues found.")

    print(f"\n  Players (Kinexon ID -> ticks, max speed):")
    for pid, cnt in sorted(players.items(), key=lambda x: -x[1])[:15]:
        speeds = speed_by_player.get(pid, [])
        max_spd = max(speeds) if speeds else 0.0
        print(f"    {pid:>6}  {cnt:>7,} ticks   top_speed={max_spd:.2f} m/s")

    print(f"\n{sep}")
    print(f"FIRST {PREVIEW_COUNT} OBSERVATIONS")
    print(sep)
    for i, obs in enumerate(observations[:PREVIEW_COUNT]):
        raw = adapter.to_raw_observation(obs)
        evt = adapter.to_event_dict(obs)
        print(
            f"  [{i+1:02d}] pid={obs.player_id} ({obs.group_name[:12]:<12}) "
            f"ts={obs.ts.strftime('%H:%M:%S.%f')[:-3]}  "
            f"x_pitch={obs.x_pitch:6.2f}  y_pitch={obs.y_pitch:6.2f}  "
            f"speed={obs.speed_ms or 0.0:5.2f} m/s  "
            f"accel={obs.acceleration_ms2 or 0.0:+6.2f}  "
            f"delta={obs.distance_delta_m:.4f} m  "
            f"sprint={'Y' if obs.sprint_flag else 'N'}  "
            f"HR={obs.heart_rate_bpm or 'N/A'}  "
            f"valid={'OK' if obs.valid else 'FLAGGED'}"
        )
        if obs.issues:
            print(f"       issues: {', '.join(obs.issues)}")
        print(
            f"       raw_obs: source={raw.source!r}  "
            f"external_id={raw.player_external_id!r}  "
            f"lat={raw.latitude}  lon={raw.longitude}  "
            f"is_valid()={raw.is_valid()}"
        )
        print(
            f"       evt_dict: speed={evt['speed_ms']}  "
            f"hr={evt['heart_rate_bpm']}  "
            f"x_pitch={evt['x_pitch']:.2f}  y_pitch={evt['y_pitch']:.2f}"
        )

    print(sep)
    print("PIPELINE COMPATIBILITY NOTE")
    print(sep)
    print(
        "  SequenceWindowBuilder.build_live_window() marks an event as padding\n"
        "  (is_real=False) when heart_rate_bpm is None.\n"
        f"  {n_no_hr/total*100:.0f}% of events in this export have no HR data.\n"
        "  Anomaly inference requires HR-equipped wearables in future exports."
    )
    print(sep + "\n")


def run_standalone() -> None:
    """Entry point for direct execution: python tests/test_kinexon_adapter.py"""
    if not POSITIONS_PATH.exists():
        print(f"\nERROR: positions.csv not found at:\n  {POSITIONS_PATH}")
        print(
            "\nPlace the Kinexon positions.csv export in that path and re-run.\n"
            "The file is gitignored (data/ in .gitignore) and must be provided manually.\n"
        )
        sys.exit(1)

    cfg     = KinexonConfig()
    adapter = KinexonAdapter(config=cfg)

    print(f"\nLoading player metadata from {STATS_PATH} ...")
    meta = adapter.load_player_meta(STATS_PATH) if STATS_PATH.exists() else {}

    print(f"Streaming positions from {POSITIONS_PATH} ...")
    observations = list(
        adapter.stream_positions(
            POSITIONS_PATH,
            meta=meta,
            session_id="3387",
            match_id="wetzlar_vs_scm_20260607",
        )
    )

    _print_summary(observations, adapter)


# ─────────────────────────────────────────────────────────────────────────────
# Pytest fixture that prints the summary once per session (with -s flag)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def print_summary_once(adapter, observations):
    """Prints the human-readable summary when running under pytest -s."""
    yield   # tests run first
    _print_summary(observations, adapter)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_standalone()
