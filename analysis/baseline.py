"""
Players Data — IBM CIC Germany
Personal Baseline Modelling

Builds and maintains an individual performance baseline for each player
from their own historical session data.

Metrics tracked against personal baseline:
  - Total distance per 15-min segment
  - Sprint count per 15-min segment
  - High-speed distance
  - Top speed
  - Fatigue decay curve (exponential fit across match segments)
  - Positional norms (mean position, std radius)
  - Rolling 7-day and 28-day workload windows
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
def _exponential_decay(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Exponential decay: y = beta * exp(-alpha * t)."""
    return beta * np.exp(-alpha * t)


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
        p0 = [0.1, float(np.mean(y))]
        bounds = ([0.0, 0.0], [5.0, y.max() * 2])
        popt, _ = curve_fit(_exponential_decay, t, y, p0=p0, bounds=bounds, maxfev=5000)
        alpha, beta = popt

        # Coefficient of determination R²
        y_pred = _exponential_decay(t, alpha, beta)
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

    def _compute_fatigue_curve(
        self,
        events_df: pd.DataFrame,
        sessions_df: pd.DataFrame,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Average fatigue decay curve across all sessions.
        Each session is segmented into 15-min windows; distance per segment is extracted.
        The average distance profile across sessions is fitted with exponential decay.
        """
        if events_df.empty or "ts" not in events_df.columns:
            return None, None, None

        events_df = events_df.copy()
        events_df["ts"] = pd.to_datetime(events_df["ts"])

        all_segment_profiles: List[List[float]] = []

        for _, session in sessions_df.iterrows():
            sess_events = events_df[events_df["session_id"] == session["session_id"]].copy()
            if sess_events.empty:
                continue

            sess_events = sess_events.sort_values("ts")
            start = sess_events["ts"].min()
            sess_events["elapsed_min"] = (
                (sess_events["ts"] - start).dt.total_seconds() / 60.0
            )

            seg_width = self.fatigue_cfg.window_minutes
            max_seg = int(sess_events["elapsed_min"].max() // seg_width) + 1

            segment_distances: List[float] = []
            for seg_idx in range(max_seg):
                seg_start = seg_idx * seg_width
                seg_end = (seg_idx + 1) * seg_width
                seg_events = sess_events[
                    (sess_events["elapsed_min"] >= seg_start) &
                    (sess_events["elapsed_min"] < seg_end)
                ]
                if seg_events.empty or "speed_ms" not in seg_events:
                    segment_distances.append(0.0)
                    continue

                # Approximate distance: sum(speed * dt)
                speeds = seg_events["speed_ms"].fillna(0).values
                dt_s = (seg_width * 60) / max(len(speeds), 1)
                segment_distances.append(float(np.sum(speeds) * dt_s))

            if len(segment_distances) >= self.fatigue_cfg.min_segments:
                all_segment_profiles.append(segment_distances)

        if not all_segment_profiles:
            return None, None, None

        # Align profiles (pad shorter ones) and average
        max_len = max(len(p) for p in all_segment_profiles)
        padded = [p + [np.nan] * (max_len - len(p)) for p in all_segment_profiles]
        avg_profile = np.nanmean(np.array(padded), axis=0).tolist()

        return fit_fatigue_curve(avg_profile)

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
