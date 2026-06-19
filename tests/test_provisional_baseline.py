"""
tests/test_provisional_baseline.py

Validates pilot-mode baseline support added to analysis/baseline.py:

  BaselineBuilder.compute()              -- UNCHANGED (historical, >= 5 sessions)
  BaselineBuilder.compute_provisional()  -- NEW (within-session, pilot mode)
  BaselineBuilder.compute_with_fallback()-- NEW (tries historical, falls back)
  PlayerBaselineProfile.baseline_mode    -- NEW field: "historical" | "provisional"

Covers:
  A. compute() is provably unchanged (still returns None under 5 sessions;
     still returns "historical" by default when sessions are sufficient)
  B. compute_provisional() behavior in isolation (insufficient data -> None,
     sufficient data -> a populated provisional profile)
  C. compute_with_fallback() orchestration (historical takes priority;
     provisional only when historical is unavailable)
  D. Real-player demonstration for Wing / Pivot / Goalkeeper using actual
     Kinexon positions.csv data (session 3387)
  E. AnomalyResult.baseline_mode field (torch-gated, skips cleanly here)

Run:
    pytest tests/test_provisional_baseline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import pytest

from analysis.baseline import BaselineBuilder, PlayerBaselineProfile
from config.settings import CONFIG

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _synthetic_events_df(n_windows=10, window_seconds=120, base_speed=1.5,
                          base_ts=None, ticks_per_window=300) -> pd.DataFrame:
    """
    Builds a synthetic events_df with enough windows/ticks to pass
    compute_provisional()'s MIN_EVENTS_PER_WINDOW and
    min_windows_for_provisional gates. ticks_per_window=300 matches roughly
    15s of 20Hz data per window-fraction, comfortably above the 30-event floor.
    """
    base_ts = base_ts or datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    rows = []
    total_ticks = n_windows * ticks_per_window
    dt_per_tick = (n_windows * window_seconds) / total_ticks
    for i in range(total_ticks):
        rows.append({
            "ts": base_ts + timedelta(seconds=i * dt_per_tick),
            "speed_ms": max(0.0, base_speed + rng.normal(0, 0.3)),
            "x_pitch": 50.0 + rng.normal(0, 5),
            "y_pitch": 50.0 + rng.normal(0, 5),
            "is_sprint": 0,
            # session_id=0 so the UNCHANGED compute()/_compute_fatigue_curve()
            # path (which groups events by session_id) has something to match
            # against when these tests exercise the historical code path.
            # compute_provisional() does not use this column at all.
            "session_id": 0,
        })
    return pd.DataFrame(rows)


def _synthetic_sessions_df(n_sessions=5) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame({
        "session_id": list(range(n_sessions)),
        "started_at": [base + timedelta(days=7 * i) for i in range(n_sessions)],
        "total_distance_m": [8000.0 + 100 * i for i in range(n_sessions)],
        "sprint_count": [20 + i for i in range(n_sessions)],
        "max_speed_ms": [6.5 + 0.1 * i for i in range(n_sessions)],
        "high_speed_distance_m": [500.0 + 10 * i for i in range(n_sessions)],
    })


def _empty_sessions_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "session_id", "started_at", "total_distance_m",
        "sprint_count", "max_speed_ms", "high_speed_distance_m",
    ])


@pytest.fixture()
def builder():
    return BaselineBuilder()


# ---------------------------------------------------------------------------
# A. compute() is unchanged
# ---------------------------------------------------------------------------

class TestComputeUnchanged:

    def test_still_returns_none_below_min_sessions(self, builder):
        sessions = _synthetic_sessions_df(n_sessions=2)  # below default min=5
        events = _synthetic_events_df()
        result = builder.compute(1, "p1", sessions, events)
        assert result is None

    def test_still_returns_none_with_empty_sessions(self, builder):
        result = builder.compute(1, "p1", _empty_sessions_df(), _synthetic_events_df())
        assert result is None

    def test_returns_historical_mode_by_default_when_sufficient(self, builder):
        sessions = _synthetic_sessions_df(n_sessions=5)
        events = _synthetic_events_df()
        result = builder.compute(1, "p1", sessions, events)
        assert result is not None
        assert result.baseline_mode == "historical", (
            "compute() must default to baseline_mode='historical' without "
            "needing to pass the new field explicitly -- proves the dataclass "
            "default preserves old call-site behaviour unchanged"
        )

    def test_historical_values_unaffected_by_new_field(self, builder):
        """Sanity: the actual computed statistics are identical to what
        compute() always produced -- only the new field is additive."""
        sessions = _synthetic_sessions_df(n_sessions=5)
        events = _synthetic_events_df()
        result = builder.compute(1, "p1", sessions, events)
        assert result.distance_mean == pytest.approx(sessions["total_distance_m"].mean())
        assert result.n_sessions == 5


# ---------------------------------------------------------------------------
# B. compute_provisional() in isolation
# ---------------------------------------------------------------------------

class TestComputeProvisional:

    def test_none_on_empty_events(self, builder):
        empty = pd.DataFrame(columns=["ts", "speed_ms"])
        assert builder.compute_provisional(1, "p1", empty) is None

    def test_none_on_missing_required_columns(self, builder):
        df = pd.DataFrame({"x_pitch": [1, 2, 3]})
        assert builder.compute_provisional(1, "p1", df) is None

    def test_none_when_too_few_valid_windows(self, builder):
        """Only 2 windows' worth of data; default min_windows_for_provisional=5."""
        events = _synthetic_events_df(n_windows=2)
        assert builder.compute_provisional(1, "p1", events) is None

    def test_provisional_mode_set_when_sufficient(self, builder):
        events = _synthetic_events_df(n_windows=10)
        result = builder.compute_provisional(1, "p1", events)
        assert result is not None
        assert result.baseline_mode == "provisional"

    def test_provisional_stats_are_sensible(self, builder):
        events = _synthetic_events_df(n_windows=10, base_speed=2.0)
        result = builder.compute_provisional(1, "p1", events)
        assert result.distance_mean > 0
        assert result.distance_std >= 0
        assert np.isfinite(result.top_speed_mean)
        assert result.n_sessions == 1

    def test_fatigue_curve_not_approximated(self, builder):
        """Pilot mode explicitly does not fabricate a fatigue decay curve."""
        events = _synthetic_events_df(n_windows=10)
        result = builder.compute_provisional(1, "p1", events)
        assert result.fatigue_alpha is None
        assert result.fatigue_beta is None
        assert result.fatigue_r_squared is None

    def test_positional_norms_populated(self, builder):
        events = _synthetic_events_df(n_windows=10)
        result = builder.compute_provisional(1, "p1", events)
        assert result.avg_x is not None
        assert result.avg_y is not None
        assert result.position_std_radius is not None

    def test_respects_min_windows_config(self, builder):
        """Raising the config threshold should require more windows."""
        original = CONFIG.baseline.min_windows_for_provisional
        try:
            CONFIG.baseline.min_windows_for_provisional = 20
            events = _synthetic_events_df(n_windows=10)
            assert builder.compute_provisional(1, "p1", events) is None
        finally:
            CONFIG.baseline.min_windows_for_provisional = original


# ---------------------------------------------------------------------------
# C. compute_with_fallback() orchestration
# ---------------------------------------------------------------------------

class TestComputeWithFallback:

    def test_prefers_historical_when_available(self, builder):
        sessions = _synthetic_sessions_df(n_sessions=5)
        events = _synthetic_events_df(n_windows=10)
        result = builder.compute_with_fallback(1, "p1", sessions, events)
        assert result.baseline_mode == "historical"

    def test_historical_result_matches_direct_compute_call(self, builder):
        sessions = _synthetic_sessions_df(n_sessions=5)
        events = _synthetic_events_df(n_windows=10)
        direct = builder.compute(1, "p1", sessions, events)
        via_fallback = builder.compute_with_fallback(1, "p1", sessions, events)
        assert direct.distance_mean == via_fallback.distance_mean
        assert direct.n_sessions == via_fallback.n_sessions

    def test_falls_back_to_provisional_when_insufficient_sessions(self, builder):
        sessions = _synthetic_sessions_df(n_sessions=1)  # pilot scenario
        events = _synthetic_events_df(n_windows=10)
        result = builder.compute_with_fallback(1, "p1", sessions, events)
        assert result is not None
        assert result.baseline_mode == "provisional"

    def test_falls_back_to_provisional_with_zero_sessions(self, builder):
        events = _synthetic_events_df(n_windows=10)
        result = builder.compute_with_fallback(1, "p1", _empty_sessions_df(), events)
        assert result is not None
        assert result.baseline_mode == "provisional"

    def test_returns_none_when_both_paths_exhausted(self, builder):
        events = _synthetic_events_df(n_windows=2)  # too few even for provisional
        result = builder.compute_with_fallback(1, "p1", _empty_sessions_df(), events)
        assert result is None


# ---------------------------------------------------------------------------
# D. Real-player demonstration: Wing / Pivot / Goalkeeper
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_positions() -> pd.DataFrame:
    if not POSITIONS_PATH.exists():
        pytest.skip(f"positions.csv not found at {POSITIONS_PATH}")
    cols = ["ts in ms", "mapped id", "group name", "x in m", "y in m", "speed in m/s"]
    df = pd.read_csv(POSITIONS_PATH, sep=";", usecols=cols, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["group name"] != "Ball"]
    return df


def _build_real_events_df(real_positions: pd.DataFrame, player_id: int) -> pd.DataFrame:
    p = real_positions[real_positions["mapped id"] == player_id].copy()
    p["ts_num"] = pd.to_numeric(p["ts in ms"], errors="coerce")
    p = p.sort_values("ts_num")
    t0 = p["ts_num"].min()
    base_ts = datetime(2026, 6, 7, 15, 0, 0, tzinfo=timezone.utc)
    speed = pd.to_numeric(p["speed in m/s"], errors="coerce").fillna(0).clip(0, CONFIG.kinexon.max_speed_ms)
    x = pd.to_numeric(p["x in m"], errors="coerce")
    y = pd.to_numeric(p["y in m"], errors="coerce")
    return pd.DataFrame({
        "ts": [base_ts + timedelta(seconds=(t - t0) / 1000.0) for t in p["ts_num"]],
        "speed_ms": speed.values,
        "x_pitch": np.clip((x + 20) / 40 * 100, 0, 100).values,
        "y_pitch": np.clip((y + 10) / 20 * 100, 0, 100).values,
        "is_sprint": (speed >= CONFIG.kinexon.sprint_threshold_ms).astype(int).values,
    })


REAL_PLAYERS = {
    2058: "Wing (Lukas Mertens)",
    1164: "Pivot (Magnus Saugstrup)",
    2331: "Goalkeeper (Matej Mandic)",
}


class TestRealPlayerProvisionalDemo:
    """
    Demonstrates pilot-mode baseline construction against the real session
    3387 Kinexon export, for the three representative positions requested:
    Wing, Pivot, Goalkeeper. There are zero historical sessions for any of
    these players in this dataset (only one Kinexon export exists), so
    compute() returns None for all three and compute_with_fallback() must
    produce a provisional baseline for all three.
    """

    @pytest.mark.parametrize("player_id", list(REAL_PLAYERS.keys()))
    def test_compute_returns_none_no_history(self, builder, real_positions, player_id):
        events = _build_real_events_df(real_positions, player_id)
        result = builder.compute(player_id, str(player_id), _empty_sessions_df(), events)
        assert result is None, (
            f"{REAL_PLAYERS[player_id]}: compute() must still return None with "
            "zero historical sessions -- this is the unchanged existing gate"
        )

    @pytest.mark.parametrize("player_id", list(REAL_PLAYERS.keys()))
    def test_provisional_baseline_built_from_real_data(self, builder, real_positions, player_id):
        events = _build_real_events_df(real_positions, player_id)
        result = builder.compute_with_fallback(player_id, str(player_id), _empty_sessions_df(), events)
        label = REAL_PLAYERS[player_id]
        assert result is not None, f"{label}: pilot mode must produce a baseline"
        assert result.baseline_mode == "provisional"
        assert result.distance_mean > 0
        assert result.top_speed_mean > 0
        assert result.avg_x is not None and result.avg_y is not None

    def test_positions_are_workload_differentiated(self, builder, real_positions):
        """
        The provisional baseline must actually distinguish positions using
        real telemetry -- not just produce a non-crashing placeholder.
        Wing and Pivot should show materially higher top speed and sprint
        volume than the Goalkeeper.
        """
        profiles = {}
        for pid in REAL_PLAYERS:
            events = _build_real_events_df(real_positions, pid)
            profiles[pid] = builder.compute_with_fallback(
                pid, str(pid), _empty_sessions_df(), events
            )

        wing, pivot, gk = profiles[2058], profiles[1164], profiles[2331]

        assert wing.top_speed_mean > gk.top_speed_mean, (
            f"Wing top_speed_mean ({wing.top_speed_mean:.2f}) should exceed "
            f"GK ({gk.top_speed_mean:.2f})"
        )
        assert pivot.top_speed_mean > gk.top_speed_mean, (
            f"Pivot top_speed_mean ({pivot.top_speed_mean:.2f}) should exceed "
            f"GK ({gk.top_speed_mean:.2f})"
        )
        assert wing.sprint_count_mean > gk.sprint_count_mean, (
            "Wing should show more sprint activity per window than GK"
        )
        # Goalkeeper's positional spread should be tight relative to a field
        # player covering much more of the pitch.
        assert gk.position_std_radius < wing.position_std_radius or gk.position_std_radius < 15, (
            f"GK position_std_radius ({gk.position_std_radius:.1f}) expected to be "
            f"tight (goal-area movement only)"
        )


# ---------------------------------------------------------------------------
# E. AnomalyResult.baseline_mode field (torch-gated)
# ---------------------------------------------------------------------------

def _import_anomaly_result():
    try:
        import torch  # noqa: F401
        from analysis.anomaly_detection import AnomalyResult
        return AnomalyResult
    except (ImportError, AttributeError) as exc:
        pytest.skip(f"PyTorch not available — cannot import AnomalyResult: {exc}")


class TestAnomalyResultBaselineModeField:

    def test_field_defaults_to_historical(self):
        AnomalyResult = _import_anomaly_result()
        result = AnomalyResult(
            player_id=1, external_id="p1",
            ts=datetime.now(tz=timezone.utc),
            anomaly_score=0.1, is_anomaly=False, confidence=0.5,
        )
        assert result.baseline_mode == "historical", (
            "Existing AnomalyResult(...) call sites that don't pass "
            "baseline_mode explicitly must keep their current meaning"
        )

    def test_field_can_be_set_to_provisional(self):
        AnomalyResult = _import_anomaly_result()
        result = AnomalyResult(
            player_id=1, external_id="p1",
            ts=datetime.now(tz=timezone.utc),
            anomaly_score=0.1, is_anomaly=False, confidence=0.5,
            baseline_mode="provisional",
        )
        assert result.baseline_mode == "provisional"
