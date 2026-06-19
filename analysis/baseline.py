"""
Players Data — IBM CIC Germany
Personal Baseline Modelling
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import zscore

from config.settings import CONFIG, BaselineConfig, FatigueCurveConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass
class SegmentMetrics:
    """Aggregated metrics for a 15-minute match segment."""
    segment_index: int          # 0=0-15min, 1=15-30min, …
    distance_m: float
    sprint_count: int
    high_speed_distance_m: float
    avg_speed_ms: float
    max_speed_ms: float
    avg_hr_bpm: Optional[float]


@dataclass
class PlayerBaselineProfile:
    """Complete baseline profile for one player."""
    player_id: int
    external_id: str
    window_days: int
    computed_at: datetime
    n_sessions: int

    # Per-metric (mean, std) for anomaly scoring
    distance_mean: float
    distance_std: float
    sprint_count_mean: float
    sprint_count_std: float
    top_speed_mean: float
    top_speed_std: float
    high_speed_dist_mean: float
    high_speed_dist_std: float

    # Fatigue decay curve coefficients (exponential fit)
    # y(t) = beta * exp(-alpha * t)  where t = segment index
    fatigue_alpha: Optional[float]
    fatigue_beta: Optional[float]
    fatigue_r_squared: Optional[float]

    # Positional norm
    avg_x: Optional[float]
    avg_y: Optional[float]
    position_std_radius: Optional[float]

    # "historical"  — built by compute() from >= min_sessions_for_baseline
    #                 cross-session aggregates (existing, unchanged behaviour).
    # "provisional" — built by compute_provisional() from this single
    #                 session's own within-session window statistics, used
    #                 only when historical data is insufficient (pilot mode).
    # Defaults to "historical" so every existing call site that builds a
    # PlayerBaselineProfile without naming this field keeps its current,
    # correct meaning without any code change.
    baseline_mode: str = "historical"

    def zscore(self, metric: str, value: float) -> float:
        """Z-score a value against this player's baseline."""
        mean = getattr(self, f"{metric}_mean", None)
        std = getattr(self, f"{metric}_std", None)
        if mean is None or std is None or std == 0:
            return 0.0
        return (value - mean) / std

    def deviation_from_baseline(self, metric: str, value: float) -> dict:
        """Return absolute and relative deviation plus z-score."""
        mean = getattr(self, f"{metric}_mean", 0.0)
        std = getattr(self, f"{metric}_std", 1.0)
        z = self.zscore(metric, value)
        return {
            "value": value,
            "baseline_mean": mean,
            "baseline_std": std,
            "absolute_delta": value - mean,
            "relative_delta_pct": ((value - mean) / mean * 100) if mean != 0 else 0.0,
            "z_score": z,
        }


# ─────────────────────────────────────────────
# Fatigue curve model
# ─────────────────────────────────────────────
def _linear_fatigue(t: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * t + intercept


def fit_fatigue_curve(
    segment_distances: List[float],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit an exponential decay curve to per-segment distances.
    Returns (alpha, beta, r_squared) or (None, None, None) if fitting fails.
    """
    if len(segment_distances) < CONFIG.fatigue.min_segments:
        return None, None, None

    t = np.arange(len(segment_distances), dtype=float)
    y = np.array(segment_distances, dtype=float)

    if np.all(y == 0):
        return None, None, None

    try:
        # Initial guess: alpha=0.1, beta=mean distance
        p0 = [-50.0, float(np.mean(y))]

        bounds = (
            [-10000.0, 0.0],
            [10000.0, y.max() * 2],
        )

        popt, _ = curve_fit(
            _linear_fatigue,
            t,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )

        slope, intercept = popt
        alpha, beta = popt

        # Coefficient of determination R²
        y_pred = _linear_fatigue(t, slope, intercept)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return float(alpha), float(beta), float(r2)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Fatigue curve fitting failed: %s", exc)
        return None, None, None


# ─────────────────────────────────────────────
# Baseline Builder
# ─────────────────────────────────────────────
class BaselineBuilder:
    """
    Computes PlayerBaselineProfile from historical session/event data.
    Operates on DataFrames produced by the DB query layer.
    """

    def __init__(self, config: BaselineConfig = None, fatigue_config: FatigueCurveConfig = None):
        self.cfg = config or CONFIG.baseline
        self.fatigue_cfg = fatigue_config or CONFIG.fatigue

    def compute(
        self,
        player_id: int,
        external_id: str,
        sessions_df: pd.DataFrame,
        events_df: pd.DataFrame,
        window_days: int = 28,
    ) -> Optional[PlayerBaselineProfile]:
        """
        Compute a full baseline profile from historical data.

        Parameters
        ----------
        sessions_df : DataFrame
            Columns: session_id, started_at, total_distance_m, sprint_count,
                     max_speed_ms, high_speed_distance_m
        events_df : DataFrame
            Columns: session_id, ts, x_pitch, y_pitch, speed_ms, is_sprint
        window_days : int
            Rolling window for baseline computation (7 or 28).
        """
        if len(sessions_df) < self.cfg.min_sessions_for_baseline:
            logger.info(
                "Player %s: only %d sessions — need %d for baseline",
                external_id, len(sessions_df), self.cfg.min_sessions_for_baseline
            )
            return None

        # ── Aggregate session-level stats ──
        dist = sessions_df["total_distance_m"].dropna()
        sprints = sessions_df["sprint_count"].dropna()
        top_speed = sessions_df["max_speed_ms"].dropna()
        hi_dist = sessions_df["high_speed_distance_m"].dropna()

        # ── Fatigue curve from event data ──
        alpha, beta, r2 = self._compute_fatigue_curve(events_df, sessions_df)

        # ── Positional norms ──
        avg_x, avg_y, pos_std = self._compute_positional_norms(events_df)

        return PlayerBaselineProfile(
            player_id=player_id,
            external_id=external_id,
            window_days=window_days,
            computed_at=datetime.now(tz=timezone.utc),
            n_sessions=len(sessions_df),
            distance_mean=float(dist.mean()) if len(dist) > 0 else 0.0,
            distance_std=float(dist.std()) if len(dist) > 1 else 1.0,
            sprint_count_mean=float(sprints.mean()) if len(sprints) > 0 else 0.0,
            sprint_count_std=float(sprints.std()) if len(sprints) > 1 else 1.0,
            top_speed_mean=float(top_speed.mean()) if len(top_speed) > 0 else 0.0,
            top_speed_std=float(top_speed.std()) if len(top_speed) > 1 else 1.0,
            high_speed_dist_mean=float(hi_dist.mean()) if len(hi_dist) > 0 else 0.0,
            high_speed_dist_std=float(hi_dist.std()) if len(hi_dist) > 1 else 1.0,
            fatigue_alpha=alpha,
            fatigue_beta=beta,
            fatigue_r_squared=r2,
            avg_x=avg_x,
            avg_y=avg_y,
            position_std_radius=pos_std,
        )

    # ── Pilot mode ──────────────────────────────────────────────────────────

    def compute_with_fallback(
        self,
        player_id: int,
        external_id: str,
        sessions_df: pd.DataFrame,
        events_df: pd.DataFrame,
        window_days: int = 28,
    ) -> Optional[PlayerBaselineProfile]:
        """
        Pilot-mode entry point. Tries the existing historical baseline first
        (compute(), completely unchanged — still requires
        cfg.min_sessions_for_baseline sessions and returns that result
        untouched when available). Falls back to a provisional within-session
        baseline only when historical data is insufficient.

        Existing callers that use compute() directly are entirely unaffected
        by this method's existence — it is purely additive.
        """
        historical = self.compute(player_id, external_id, sessions_df, events_df, window_days)
        if historical is not None:
            return historical
        return self.compute_provisional(player_id, external_id, events_df)

    def compute_provisional(
        self,
        player_id: int,
        external_id: str,
        events_df: pd.DataFrame,
        window_seconds: int = 120,
    ) -> Optional[PlayerBaselineProfile]:
        """
        Build a provisional, within-session baseline when fewer than
        cfg.min_sessions_for_baseline historical sessions exist (pilot mode:
        e.g. a single Kinexon session with no prior match history).

        Splits this player's own current-session telemetry into
        window_seconds-sized buckets (matching the live window size) and
        computes mean/std of per-WINDOW distance, sprint count, top speed,
        and high-speed distance directly from those buckets. This is the
        same granularity _build_xai_feature_vector() actually compares
        against (z_distance, z_sprint_count, etc. are computed per-window) —
        unlike the historical baseline's match-total statistics, so a
        provisional baseline's z-scores are internally consistent even
        though it has no cross-session history yet.

        fatigue_alpha/beta/r_squared are left None: a fatigue decay curve
        needs multiple sessions' worth of segments to fit meaningfully and
        is not approximated here.

        Returns None if there are fewer than cfg.min_windows_for_provisional
        valid windows — i.e. even a within-session estimate isn't reliable
        yet (e.g. a player with only a few minutes of tracked telemetry).
        """
        if events_df.empty or "ts" not in events_df.columns or "speed_ms" not in events_df.columns:
            return None

        events_df = events_df.copy()
        events_df["ts"] = pd.to_datetime(events_df["ts"], utc=True)
        events_df = events_df.sort_values("ts")

        start_ts = events_df["ts"].min()
        events_df["elapsed_s"] = (events_df["ts"] - start_ts).dt.total_seconds()

        sprint_threshold = CONFIG.kinexon.sprint_threshold_ms
        hi_threshold = CONFIG.kinexon.high_intensity_threshold_ms
        speed_cap = CONFIG.kinexon.max_speed_ms

        MIN_EVENTS_PER_WINDOW = 30  # consistent with _compute_fatigue_curve's filter
        max_window = int(events_df["elapsed_s"].max() // window_seconds) + 1

        window_distances: List[float] = []
        window_sprints: List[float] = []
        window_top_speeds: List[float] = []
        window_hi_dists: List[float] = []

        for w in range(max_window):
            w_start = w * window_seconds
            w_end = (w + 1) * window_seconds
            seg = events_df[
                (events_df["elapsed_s"] >= w_start) & (events_df["elapsed_s"] < w_end)
            ]
            if len(seg) < MIN_EVENTS_PER_WINDOW:
                continue

            speeds = seg["speed_ms"].fillna(0).clip(lower=0, upper=speed_cap).values
            if len(seg) < 2:
                continue
            # .dt.total_seconds() is resolution-agnostic (datetime64[us]/[ns]/etc.) —
            # do not convert via .astype("int64") and a hardcoded 1e9 divisor,
            # which silently assumes nanosecond resolution and is wrong on
            # pandas versions/inputs that produce microsecond-resolution dtype.
            dt = seg["ts"].diff().dt.total_seconds().values[1:]
            dt = np.clip(dt, 0, 5)
            dist = float(np.sum(speeds[:-1] * dt))
            if not np.isfinite(dist) or dist < 0:
                continue

            if "is_sprint" in seg.columns:
                sprint_count = float(seg["is_sprint"].fillna(0).astype(bool).sum())
            else:
                sprint_count = float((speeds >= sprint_threshold).sum())

            top_speed = float(speeds.max())
            hi_mask = speeds[:-1] >= hi_threshold
            hi_dist = float(np.sum(speeds[:-1][hi_mask] * dt[hi_mask])) if hi_mask.any() else 0.0

            window_distances.append(dist)
            window_sprints.append(sprint_count)
            window_top_speeds.append(top_speed)
            window_hi_dists.append(hi_dist)

        if len(window_distances) < self.cfg.min_windows_for_provisional:
            logger.info(
                "Player %s: only %d valid within-session windows — need %d "
                "for a provisional baseline",
                external_id, len(window_distances), self.cfg.min_windows_for_provisional,
            )
            return None

        def _mean_std(values: List[float]) -> Tuple[float, float]:
            arr = np.array(values, dtype=float)
            return float(arr.mean()), float(arr.std()) if len(arr) > 1 else 1.0

        d_mean, d_std = _mean_std(window_distances)
        s_mean, s_std = _mean_std(window_sprints)
        t_mean, t_std = _mean_std(window_top_speeds)
        h_mean, h_std = _mean_std(window_hi_dists)

        avg_x, avg_y, pos_std = self._compute_positional_norms(events_df)

        logger.info(
            "Player %s: provisional baseline built from %d within-session "
            "windows (no historical sessions available)",
            external_id, len(window_distances),
        )

        return PlayerBaselineProfile(
            player_id=player_id,
            external_id=external_id,
            window_days=0,  # not applicable -- within-session only
            computed_at=datetime.now(tz=timezone.utc),
            n_sessions=1,
            distance_mean=d_mean, distance_std=d_std,
            sprint_count_mean=s_mean, sprint_count_std=s_std,
            top_speed_mean=t_mean, top_speed_std=t_std,
            high_speed_dist_mean=h_mean, high_speed_dist_std=h_std,
            fatigue_alpha=None, fatigue_beta=None, fatigue_r_squared=None,
            avg_x=avg_x, avg_y=avg_y, position_std_radius=pos_std,
            baseline_mode="provisional",
        )

    def _compute_fatigue_curve(
    self,
    events_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Computes a robust per-player fatigue trend from historical telemetry.

        Method:
        1. Split each session into fixed time windows (e.g. 15 min)
        2. Estimate distance covered in each segment using speed × dt
        3. Fit a linear decay trend per session
        4. Aggregate session trends using median statistics

        Returns
        -------
        (slope, intercept, r_squared)

        slope:
            Negative slope => performance decay over match duration
            Near 0         => stable work rate
            Positive       => unrealistic / noisy telemetry

        intercept:
            Estimated starting segment workload

        r_squared:
            Goodness of fit of the fatigue trend
        """
        if events_df.empty or "ts" not in events_df.columns:
            return None, None, None

        MIN_EVENTS_PER_SEGMENT = 30

        events_df = events_df.copy()
        events_df["ts"] = pd.to_datetime(events_df["ts"], utc=True)

        all_profiles: List[List[float]] = []

        for _, session in sessions_df.iterrows():

            sess_events = events_df[
                events_df["session_id"] == session["session_id"]
            ].copy()

            if sess_events.empty:
                continue

            sess_events = sess_events.sort_values("ts")

            start_ts = sess_events["ts"].min()

            sess_events["elapsed_min"] = (
                (sess_events["ts"] - start_ts).dt.total_seconds() / 60.0
            )

            seg_width = self.fatigue_cfg.window_minutes

            max_seg = int(
                sess_events["elapsed_min"].max() // seg_width
            ) + 1

            segment_distances: List[float] = []

            for seg_idx in range(max_seg):

                seg_start = seg_idx * seg_width
                seg_end   = (seg_idx + 1) * seg_width

                seg_events = sess_events[
                    (sess_events["elapsed_min"] >= seg_start) &
                    (sess_events["elapsed_min"] < seg_end)
                ].copy()

                # Ignore sparse / unreliable telemetry
                if len(seg_events) < MIN_EVENTS_PER_SEGMENT:
                    continue

                if "speed_ms" not in seg_events.columns:
                    continue

                seg_events = seg_events.sort_values("ts")

                times = (
                    pd.to_datetime(seg_events["ts"], utc=True)
                    .astype("int64") / 1e9
                )

                speeds = (
                    seg_events["speed_ms"]
                    .fillna(0)
                    .clip(lower=0, upper=12)
                    .values
                )

                if len(times) < 2:
                    continue

                dt = np.diff(times)

                # Remove corrupted timestamp gaps/spikes
                dt = np.clip(dt, 0, 5)

                # Distance integration
                dist = np.sum(speeds[:-1] * dt)

                # Reject corrupted segments
                if not np.isfinite(dist):
                    continue

                if dist <= 0:
                    continue

                segment_distances.append(float(dist))

            # Need enough valid segments for fatigue estimation
            if len(segment_distances) >= self.fatigue_cfg.min_segments:
                all_profiles.append(segment_distances)

        if not all_profiles:
            return None, None, None

        slopes = []
        intercepts = []
        r2s = []

        for profile in all_profiles:

            slope, intercept, r2 = fit_fatigue_curve(profile)

            if (
                slope is None or
                intercept is None or
                r2 is None
            ):
                continue

            # Reject nonsense fits
            if not np.isfinite(slope):
                continue

            if not np.isfinite(intercept):
                continue

            if not np.isfinite(r2):
                continue

            slopes.append(slope)
            intercepts.append(intercept)
            r2s.append(r2)

        if not slopes:
            return None, None, None

        return (
            float(np.median(slopes)),
            float(np.median(intercepts)),
            float(np.median(r2s)),
        )

    def _compute_positional_norms(
        self, events_df: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute mean position and standard radius from x_pitch, y_pitch columns."""
        if events_df.empty:
            return None, None, None

        pos = events_df[["x_pitch", "y_pitch"]].dropna()
        if len(pos) < 10:
            return None, None, None

        avg_x = float(pos["x_pitch"].mean())
        avg_y = float(pos["y_pitch"].mean())

        # Euclidean distances from mean position
        dists = np.sqrt(
            (pos["x_pitch"] - avg_x) ** 2 + (pos["y_pitch"] - avg_y) ** 2
        )
        std_radius = float(dists.std())

        return avg_x, avg_y, std_radius


# ─────────────────────────────────────────────
# Workload Trend Tracker
# ─────────────────────────────────────────────
class WorkloadTrendTracker:
    """
    Compares current-session load against rolling 7-day and 28-day windows.
    Surfaces overtraining and underprepared states.
    """

    def __init__(self):
        self.cfg = CONFIG.baseline

    def compute_load_ratios(
        self,
        current_distance_m: float,
        sessions_df: pd.DataFrame,
    ) -> dict:
        """
        Returns ACWR (Acute:Chronic Workload Ratio) and related flags.

        Parameters
        ----------
        current_distance_m : float  — Today's session distance
        sessions_df : DataFrame     — Historical sessions with [started_at, total_distance_m]
        """
        now = datetime.now(tz=timezone.utc)
        sessions_df = sessions_df.copy()
        sessions_df["started_at"] = pd.to_datetime(sessions_df["started_at"], utc=True)

        window_7 = sessions_df[
            sessions_df["started_at"] >= now - timedelta(days=self.cfg.short_window_days)
        ]["total_distance_m"].fillna(0)

        window_28 = sessions_df[
            sessions_df["started_at"] >= now - timedelta(days=self.cfg.rolling_window_days)
        ]["total_distance_m"].fillna(0)

        acute_load = float(window_7.mean()) if len(window_7) > 0 else current_distance_m
        chronic_load = float(window_28.mean()) if len(window_28) > 0 else current_distance_m

        acwr = acute_load / chronic_load if chronic_load > 0 else 1.0

        # ACWR flags (based on sports science literature)
        is_overtraining = acwr > 1.5
        is_underprepared = acwr < 0.8

        return {
            "current_distance_m": current_distance_m,
            "acute_avg_7d": acute_load,
            "chronic_avg_28d": chronic_load,
            "acwr": round(acwr, 3),
            "is_overtraining_risk": is_overtraining,
            "is_underprepared": is_underprepared,
            "workload_status": (
                "high_risk" if is_overtraining else
                "low_readiness" if is_underprepared else
                "optimal"
            ),
        }
