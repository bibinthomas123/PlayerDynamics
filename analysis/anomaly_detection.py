"""
Players Data — IBM CIC Germany
Pattern Analysis Engine — Anomaly Detection

Implements:
  1. Isolation Forest — per-player anomaly detection against personal baseline
  2. Fatigue curve comparator — live segment vs. personal decay profile
  3. Positional drift scorer — detects tactical zone violations
  4. Feature engineering pipeline — converts raw events into model features

All models are fitted on personal (per-player) historical data, NOT squad averages.
This is the core technical novelty stated in the proposal.
"""
from __future__ import annotations
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from analysis.baseline import PlayerBaselineProfile, WorkloadTrendTracker
from config.settings import CONFIG, IsolationForestConfig, PositionalDriftConfig

# Re-export so orchestrator can import from one place
from explainability.xai_layer import FEATURE_NAMES  # noqa: F401

logger = logging.getLogger(__name__)

MODEL_STORE = Path("/tmp/players_data_models")
MODEL_STORE.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Feature vector definition
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    # Live window metrics
    "window_sprint_count",
    "window_distance_m",
    "window_avg_speed_ms",

    # Deviations from personal baseline (z-scores)
    "z_distance",
    "z_sprint_count",
    "z_top_speed",
    "z_high_speed_dist",

    # Fatigue indicators
    "fatigue_decay_residual",           # actual_dist - predicted_dist from baseline decay curve
    "speed_drop_pct",                   # % drop in avg speed vs session start

    # Positional
    "positional_drift_score",           # Euclidean dist from player's norm position / std_radius

    # Workload
    "acwr",                             # Acute:Chronic Workload Ratio

    # Biometric
    "heart_rate_bpm",
    "hr_recovery_time_s",

    # Coach annotations (encoded)
    "coach_fatigue_severity",           # 0–1, from annotation
    "coach_pre_match_status_encoded",   # 0=good, 0.5=mild concern, 1=concern
]

N_FEATURES = len(FEATURE_NAMES)


# ─────────────────────────────────────────────
# Anomaly result
# ─────────────────────────────────────────────
@dataclass
class AnomalyResult:
    player_id: int
    external_id: str
    ts: datetime
    anomaly_score: float           # Raw Isolation Forest score (higher = more anomalous)
    is_anomaly: bool
    confidence: float              # Normalized 0–1 confidence

    feature_vector: Dict[str, float] = field(default_factory=dict)
    deviations: Dict[str, dict] = field(default_factory=dict)

    # Sub-analysis results
    fatigue_flag: bool = False
    positional_drift_flag: bool = False
    workload_flag: bool = False
    workload_status: str = "optimal"

    recommendation_type: Optional[str] = None
    triggered_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
class FeatureEngineer:
    """
    Converts live window observations + baseline profile into a fixed-length
    feature vector for model inference.
    """

    def __init__(self):
        self.workload_tracker = WorkloadTrendTracker()

    def build_feature_vector(
        self,
        live_window: dict,
        baseline: PlayerBaselineProfile,
        sessions_df: pd.DataFrame,
        segment_index: int = 0,
        coach_annotation: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Parameters
        ----------
        live_window     : Normalized event dict from IngestionNormalizer
        baseline        : Player's pre-computed baseline profile
        sessions_df     : Historical sessions for workload computation
        segment_index   : Current 15-min segment index in the match
        coach_annotation: Latest annotation dict for this player

        Returns
        -------
        (feature_array, feature_dict)
        """
        fv: Dict[str, float] = {}

        # ── Live window metrics ──
        fv["window_sprint_count"] = float(live_window.get("window_sprint_count") or 0)
        fv["window_distance_m"] = float(live_window.get("window_distance_m") or 0)
        fv["window_avg_speed_ms"] = float(live_window.get("window_avg_speed_ms") or 0)

        # ── Z-scores against personal baseline ──
        fv["z_distance"] = baseline.zscore("distance", fv["window_distance_m"])
        fv["z_sprint_count"] = baseline.zscore("sprint_count", fv["window_sprint_count"])
        fv["z_top_speed"] = baseline.zscore(
            "top_speed", float(live_window.get("speed_ms") or 0)
        )
        fv["z_high_speed_dist"] = baseline.zscore("high_speed_dist", fv["window_distance_m"])

        # ── Fatigue decay residual ──
        fv["fatigue_decay_residual"] = self._fatigue_residual(
            baseline, segment_index, fv["window_distance_m"]
        )

        # ── Speed drop ──
        fv["speed_drop_pct"] = self._speed_drop(
            live_window.get("window_avg_speed_ms") or 0,
            baseline.distance_mean / max(baseline.n_sessions, 1)
        )

        # ── Positional drift ──
        fv["positional_drift_score"] = self._positional_drift(
            live_window.get("x_pitch"),
            live_window.get("y_pitch"),
            baseline,
        )

        # ── Workload ──
        if not sessions_df.empty and "total_distance_m" in sessions_df.columns:
            workload = self.workload_tracker.compute_load_ratios(
                fv["window_distance_m"], sessions_df
            )
            fv["acwr"] = workload["acwr"]
        else:
            fv["acwr"] = 1.0

        # ── Biometric ──
        fv["heart_rate_bpm"] = float(live_window.get("heart_rate_bpm") or 0)
        fv["hr_recovery_time_s"] = float(live_window.get("hr_recovery_time_s") or 0)

        # ── Coach annotations ──
        ann = coach_annotation or {}
        fv["coach_fatigue_severity"] = float(ann.get("fatigue_severity", 0.0))
        pre_match = ann.get("pre_match_status", "good")
        fv["coach_pre_match_status_encoded"] = {
            "good": 0.0, "mild": 0.5, "concern": 1.0
        }.get(pre_match, 0.0)

        # Build ordered array
        arr = np.array([fv[name] for name in FEATURE_NAMES], dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0)

        return arr, fv

    def _fatigue_residual(
        self, baseline: PlayerBaselineProfile, segment_index: int, actual_distance: float
    ) -> float:
        """Distance residual: actual - predicted from player's exponential decay curve."""
        if baseline.fatigue_alpha is None or baseline.fatigue_beta is None:
            return 0.0
        predicted = baseline.fatigue_beta * np.exp(-baseline.fatigue_alpha * segment_index)
        return float(actual_distance - predicted)

    def _speed_drop(self, current_speed: float, session_avg_speed: float) -> float:
        """Percentage speed drop from session average."""
        if session_avg_speed <= 0:
            return 0.0
        return (session_avg_speed - current_speed) / session_avg_speed * 100.0

    def _positional_drift(
        self,
        x: Optional[float],
        y: Optional[float],
        baseline: PlayerBaselineProfile,
    ) -> float:
        """
        Euclidean distance from player's baseline mean position,
        normalized by their standard radius. Score > 1.0 = outside norm.
        """
        if x is None or y is None or baseline.avg_x is None:
            return 0.0
        dist = np.sqrt((x - baseline.avg_x) ** 2 + (y - baseline.avg_y) ** 2)
        std_r = baseline.position_std_radius or 1.0
        return float(dist / std_r)


# ─────────────────────────────────────────────
# Per-Player Isolation Forest Model
# ─────────────────────────────────────────────
class PlayerAnomalyModel:
    """
    Isolation Forest trained exclusively on one player's historical feature vectors.
    This is the core HCAI design choice: personal baselines, not squad averages.
    """

    def __init__(self, player_id: int, config: IsolationForestConfig = None):
        self.player_id = player_id
        self.cfg = config or CONFIG.isolation_forest
        self.model: Optional[IsolationForest] = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained = False
        self.model_version = "untrained"
        self._sensitivity_adjustments: Dict[str, float] = {}   # Per-feature sensitivity

    def train(self, historical_features: np.ndarray) -> None:
        """
        Fit Isolation Forest on player's historical feature matrix.

        Parameters
        ----------
        historical_features : ndarray of shape (n_sessions, N_FEATURES)
        """
        if len(historical_features) < 5:
            logger.warning("Player %d: insufficient history for IF training (%d samples)",
                           self.player_id, len(historical_features))
            return

        X = self.scaler.fit_transform(historical_features)

        self.model = IsolationForest(
            contamination=self.cfg.contamination,
            n_estimators=self.cfg.n_estimators,
            max_samples=self.cfg.max_samples,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        self.model.fit(X)
        self.is_trained = True
        self.model_version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}_p{self.player_id}"
        logger.info("Player %d: Isolation Forest trained on %d samples (version: %s)",
                    self.player_id, len(historical_features), self.model_version)

    def predict(self, feature_vector: np.ndarray) -> Tuple[float, bool, float]:
        """
        Score a single observation.

        Returns
        -------
        (raw_score, is_anomaly, confidence)
          raw_score  : IF decision function value (lower = more anomalous)
          is_anomaly : True if score < threshold
          confidence : Normalized 0–1 anomaly confidence
        """
        if not self.is_trained:
            logger.warning("Player %d: model not trained — returning default scores", self.player_id)
            return 0.0, False, 0.0

        fv = feature_vector.reshape(1, -1)
        fv_scaled = self.scaler.transform(fv)

        raw_score = float(self.model.decision_function(fv_scaled)[0])
        pred = int(self.model.predict(fv_scaled)[0])   # -1=anomaly, 1=normal

        is_anomaly = pred == -1

        # Normalize score to [0, 1] confidence
        # decision_function returns negative = anomaly; more negative = higher confidence
        confidence = max(0.0, min(1.0, (-raw_score + 0.5) / 1.0))

        return raw_score, is_anomaly, round(confidence, 4)

    def apply_sensitivity_adjustment(
        self, feature_name: str, reduction: float
    ) -> None:
        """
        Reduce sensitivity for a specific feature based on coach override patterns.
        Called by the recalibration pipeline.
        """
        self._sensitivity_adjustments[feature_name] = max(
            0.0, self._sensitivity_adjustments.get(feature_name, 1.0) - reduction
        )
        logger.info(
            "Player %d: sensitivity for %s reduced to %.2f",
            self.player_id, feature_name,
            self._sensitivity_adjustments[feature_name]
        )

    def save(self) -> Path:
        path = MODEL_STORE / f"player_{self.player_id}_if.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler,
                         "version": self.model_version,
                         "adjustments": self._sensitivity_adjustments}, f)
        return path

    @classmethod
    def load(cls, player_id: int) -> Optional["PlayerAnomalyModel"]:
        path = MODEL_STORE / f"player_{player_id}_if.pkl"
        if not path.exists():
            return None
        obj = cls(player_id)
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.model = data["model"]
        obj.scaler = data["scaler"]
        obj.model_version = data["version"]
        obj._sensitivity_adjustments = data.get("adjustments", {})
        obj.is_trained = obj.model is not None
        return obj


# ─────────────────────────────────────────────
# Positional Drift Analyzer
# ─────────────────────────────────────────────
class PositionalDriftAnalyzer:
    """
    Detects when a player is consistently outside their tactical zone.
    Uses a rolling window of pitch positions and compares against baseline norm.
    """

    def __init__(self, config: PositionalDriftConfig = None):
        self.cfg = config or CONFIG.positional

    def analyze(
        self,
        recent_positions: List[Tuple[float, float]],
        baseline: PlayerBaselineProfile,
    ) -> dict:
        """
        Parameters
        ----------
        recent_positions : List of (x_pitch, y_pitch) in last ~15 minutes
        baseline         : Player baseline with avg_x, avg_y, position_std_radius

        Returns
        -------
        dict with drift_score, is_flagged, fraction_outside_zone
        """
        if not recent_positions or baseline.avg_x is None:
            return {"drift_score": 0.0, "is_flagged": False, "fraction_outside_zone": 0.0}

        std_r = baseline.position_std_radius or self.cfg.zone_radius_meters
        threshold_r = max(std_r * 2.0, self.cfg.zone_radius_meters)

        dists = [
            np.sqrt((x - baseline.avg_x) ** 2 + (y - baseline.avg_y) ** 2)
            for x, y in recent_positions
        ]

        fraction_outside = sum(1 for d in dists if d > threshold_r) / len(dists)
        avg_drift = float(np.mean(dists))
        drift_score = avg_drift / (threshold_r or 1.0)

        is_flagged = fraction_outside >= self.cfg.drift_fraction_threshold

        return {
            "drift_score": round(drift_score, 3),
            "is_flagged": is_flagged,
            "fraction_outside_zone": round(fraction_outside, 3),
            "avg_distance_from_norm_m": round(avg_drift, 2),
            "threshold_radius_m": round(threshold_r, 2),
        }


# ─────────────────────────────────────────────
# Pattern Analysis Engine (Orchestrator)
# ─────────────────────────────────────────────
class PatternAnalysisEngine:
    """
    Top-level analysis orchestrator.
    For each live observation:
      1. Build feature vector
      2. Run Isolation Forest
      3. Run fatigue curve comparator
      4. Run positional drift detector
      5. Combine into AnomalyResult
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.drift_analyzer = PositionalDriftAnalyzer()
        self._models: Dict[int, PlayerAnomalyModel] = {}
        self._baselines: Dict[int, PlayerBaselineProfile] = {}
        self._position_buffers: Dict[int, List[Tuple[float, float]]] = {}

    def register_player(
        self,
        player_id: int,
        baseline: PlayerBaselineProfile,
        model: Optional[PlayerAnomalyModel] = None,
    ) -> None:
        """Register a player's baseline and model before live inference begins."""
        self._baselines[player_id] = baseline
        self._models[player_id] = model or PlayerAnomalyModel(player_id)
        self._position_buffers[player_id] = []

    def train_player_model(
        self,
        player_id: int,
        historical_features: np.ndarray,
    ) -> None:
        """Train (or retrain) the Isolation Forest for a specific player."""
        if player_id not in self._models:
            self._models[player_id] = PlayerAnomalyModel(player_id)
        self._models[player_id].train(historical_features)

    def analyze(
        self,
        player_id: int,
        live_window: dict,
        sessions_df: pd.DataFrame,
        segment_index: int = 0,
        coach_annotation: Optional[dict] = None,
    ) -> Optional[AnomalyResult]:
        """
        Run full pattern analysis for a single live observation.
        Returns AnomalyResult or None if player not registered or model not trained.
        """
        baseline = self._baselines.get(player_id)
        model = self._models.get(player_id)

        if baseline is None:
            logger.debug("Player %d not registered — skipping analysis", player_id)
            return None

        external_id = baseline.external_id

        # Build feature vector
        fv_array, fv_dict = self.feature_engineer.build_feature_vector(
            live_window, baseline, sessions_df, segment_index, coach_annotation
        )

        # Isolation Forest inference
        raw_score, is_anomaly, confidence = (0.0, False, 0.0)
        if model and model.is_trained:
            raw_score, is_anomaly, confidence = model.predict(fv_array)

        # Positional drift
        x, y = live_window.get("x_pitch"), live_window.get("y_pitch")
        if x is not None and y is not None:
            buf = self._position_buffers.setdefault(player_id, [])
            buf.append((x, y))
            # Keep last 15 minutes worth of positions (assume ~1 obs/sec = 900 obs)
            if len(buf) > 900:
                self._position_buffers[player_id] = buf[-900:]

        drift = self.drift_analyzer.analyze(
            self._position_buffers.get(player_id, []), baseline
        )

        # Fatigue flag: significant negative residual
        fatigue_residual = fv_dict.get("fatigue_decay_residual", 0.0)
        fatigue_std = baseline.distance_std or 1.0
        fatigue_flag = fatigue_residual < -fatigue_std * 1.5

        # Workload flag
        acwr = fv_dict.get("acwr", 1.0)
        workload_flag = acwr > 1.5 or acwr < 0.8

        # Derive deviations for XAI layer
        deviations = {
            "distance": baseline.deviation_from_baseline("distance", fv_dict["window_distance_m"]),
            "sprint_count": baseline.deviation_from_baseline("sprint_count", fv_dict["window_sprint_count"]),
        }

        # Determine recommendation type
        recommendation_type = self._determine_recommendation(
            is_anomaly, fatigue_flag, drift["is_flagged"], workload_flag, confidence
        )

        return AnomalyResult(
            player_id=player_id,
            external_id=external_id,
            ts=datetime.fromisoformat(live_window["ts"].isoformat())
            if hasattr(live_window.get("ts"), "isoformat") else datetime.now(tz=timezone.utc),
            anomaly_score=raw_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            feature_vector=fv_dict,
            deviations=deviations,
            fatigue_flag=fatigue_flag,
            positional_drift_flag=drift["is_flagged"],
            workload_flag=workload_flag,
            workload_status=("high_risk" if acwr > 1.5 else "low_readiness" if acwr < 0.8 else "optimal"),
            recommendation_type=recommendation_type,
        )

    def _determine_recommendation(
        self,
        is_anomaly: bool,
        fatigue_flag: bool,
        drift_flag: bool,
        workload_flag: bool,
        confidence: float,
    ) -> Optional[str]:
        """
        Simple priority logic — highest severity first.
        Maps to RecommendationType enum values.
        """
        if is_anomaly and confidence > 0.75:
            if fatigue_flag:
                return "substitution"
            return "fatigue_alert"
        if fatigue_flag:
            return "fatigue_alert"
        if drift_flag:
            return "positional_drift"
        if workload_flag:
            return "workload_warning"
        if is_anomaly:
            return "anomaly_flag"
        return None

    def build_historical_feature_matrix(
        self,
        player_id: int,
        sessions_df: pd.DataFrame,
        events_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Build the historical feature matrix used to train the Isolation Forest.
        One row per training session.
        """
        baseline = self._baselines.get(player_id)
        if baseline is None:
            return np.empty((0, N_FEATURES))

        rows = []
        for _, session in sessions_df.iterrows():
            sess_events = events_df[events_df["session_id"] == session["session_id"]]
            if sess_events.empty:
                continue

            # Construct a "live window" dict from session aggregates
            live_window = {
                "window_sprint_count": session.get("sprint_count", 0),
                "window_distance_m": session.get("total_distance_m", 0),
                "window_avg_speed_ms": session.get("avg_speed_ms") or
                                       sess_events["speed_ms"].mean() if "speed_ms" in sess_events else 0,
                "speed_ms": session.get("max_speed_ms", 0),
                "x_pitch": sess_events["x_pitch"].mean() if "x_pitch" in sess_events else None,
                "y_pitch": sess_events["y_pitch"].mean() if "y_pitch" in sess_events else None,
                "heart_rate_bpm": session.get("avg_heart_rate_bpm", 0),
                "hr_recovery_time_s": None,
                "ts": session.get("started_at", datetime.now(tz=timezone.utc)),
            }

            # Get latest annotation for this session
            ann = {}
            if not annotations_df.empty and "session_id" in annotations_df.columns:
                sess_anns = annotations_df[annotations_df["session_id"] == session["session_id"]]
                if not sess_anns.empty:
                    fatigue_ann = sess_anns[sess_anns["annotation_type"] == "fatigue_flag"]
                    ann = {"fatigue_severity": float(fatigue_ann["severity"].iloc[0])} if not fatigue_ann.empty else {}

            fv_array, _ = self.feature_engineer.build_feature_vector(
                live_window, baseline, sessions_df, coach_annotation=ann
            )
            rows.append(fv_array)

        if not rows:
            return np.empty((0, N_FEATURES))

        return np.vstack(rows)
