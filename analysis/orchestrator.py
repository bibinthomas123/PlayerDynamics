"""
Players Data — IBM CIC Germany
Main Analysis Orchestrator

Entry point for the full analysis pipeline:
  1. Baseline computation (per-player)
  2. Model training (Isolation Forest, per-player)
  3. SHAP explainer setup
  4. Real-time inference loop
  5. Feedback logging
  6. Scheduled recalibration
  7. Fairness auditing

This module wires all components together without a frontend or backend.
It exposes PlayersDataAnalysisPipeline — the production interface.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

from analysis.anomaly_detection import (
    PatternAnalysisEngine, PlayerAnomalyModel, FEATURE_NAMES
)
from analysis.baseline import BaselineBuilder, PlayerBaselineProfile
from explainability.xai_layer import XAILayer, SHAPExplanation
from feedback.recalibration import (
    FeedbackStore, FairnessMonitor, OverrideRecord, RecalibrationPipeline
)
from ingestion.pipeline import IngestionPipeline, RawPlayerObservation
from config.settings import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


# ─────────────────────────────────────────────
# Player registry entry
# ─────────────────────────────────────────────
class PlayerRegistry:
    """Holds all per-player state: metadata, baseline, model, session history."""

    def __init__(self):
        self._players: Dict[int, dict] = {}

    def register(
        self,
        player_id: int,
        external_id: str,
        name: str,
        position: str,
        age: int,
        age_group: str,
        nationality: str = "",
    ) -> None:
        self._players[player_id] = {
            "player_id": player_id,
            "external_id": external_id,
            "name": name,
            "position": position,
            "age": age,
            "age_group": age_group,
            "nationality": nationality,
            "baseline": None,
            "model": None,
            "sessions_df": pd.DataFrame(),
            "events_df": pd.DataFrame(),
            "annotations_df": pd.DataFrame(),
        }

    def get(self, player_id: int) -> Optional[dict]:
        return self._players.get(player_id)

    def get_by_external_id(self, external_id: str) -> Optional[dict]:
        for p in self._players.values():
            if p["external_id"] == external_id:
                return p
        return None

    def all_player_ids(self) -> List[int]:
        return list(self._players.keys())

    def metadata_dataframe(self) -> pd.DataFrame:
        if not self._players:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "player_id": p["player_id"],
                "name": p["name"],
                "position": p["position"],
                "age_group": p["age_group"],
                "nationality": p["nationality"],
            }
            for p in self._players.values()
        ])


# ─────────────────────────────────────────────
# Main Analysis Pipeline
# ─────────────────────────────────────────────
class PlayersDataAnalysisPipeline:
    """
    Production-level analysis pipeline for the Players Data IBM project.

    Usage
    -----
    pipeline = PlayersDataAnalysisPipeline()

    # 1. Register players
    pipeline.register_player(player_id=1, external_id="p001", name="Player 7", ...)

    # 2. Load historical data and train
    pipeline.load_historical_data(player_id=1, sessions_df=..., events_df=..., annotations_df=...)
    pipeline.train_all_models()

    # 3. Start live ingestion
    await pipeline.run_live(enable_ws=True, enable_mqtt=True)

    # 4. Log coach decisions
    pipeline.log_coach_decision(inference_id=1, player_id=1, decision="override", ...)

    # 5. Run recalibration and fairness audit
    pipeline.recalibrate()
    pipeline.run_fairness_audit()
    """

    def __init__(self):
        self.registry = PlayerRegistry()
        self.baseline_builder = BaselineBuilder()
        self.pattern_engine = PatternAnalysisEngine()
        self.xai_layer = XAILayer()
        self.feedback_store = FeedbackStore()
        self.recalibration_pipeline = RecalibrationPipeline()
        self.fairness_monitor = FairnessMonitor()

        self._inference_log: List[dict] = []      # In-memory; DB-persisted in production
        self._on_alert_callback: Optional[Callable] = None
        self._inference_id_counter = 0

    # ──────────────────────────────────────────
    # Registration & Data Loading
    # ──────────────────────────────────────────
    def register_player(
        self,
        player_id: int,
        external_id: str,
        name: str,
        position: str,
        age: int,
        age_group: str = "Senior",
        nationality: str = "",
    ) -> None:
        """Register a player in the system."""
        self.registry.register(
            player_id=player_id,
            external_id=external_id,
            name=name,
            position=position,
            age=age,
            age_group=age_group,
            nationality=nationality,
        )
        logger.info("Registered player: %s (id=%d, pos=%s)", name, player_id, position)

    def load_historical_data(
        self,
        player_id: int,
        sessions_df: pd.DataFrame,
        events_df: pd.DataFrame,
        annotations_df: pd.DataFrame = None,
    ) -> None:
        """
        Load historical data for a player. Required before training.

        Parameters
        ----------
        sessions_df : DataFrame[session_id, started_at, total_distance_m,
                                sprint_count, max_speed_ms, high_speed_distance_m,
                                avg_speed_ms, avg_heart_rate_bpm]
        events_df   : DataFrame[session_id, ts, x_pitch, y_pitch, speed_ms, is_sprint]
        annotations_df : DataFrame[session_id, annotation_type, value, severity]
        """
        player = self.registry.get(player_id)
        if player is None:
            raise ValueError(f"Player {player_id} not registered")

        player["sessions_df"] = sessions_df
        player["events_df"] = events_df
        player["annotations_df"] = annotations_df if annotations_df is not None else pd.DataFrame()

        logger.info(
            "Historical data loaded for player %d: %d sessions, %d events",
            player_id, len(sessions_df), len(events_df)
        )

    # ──────────────────────────────────────────
    # Baseline & Model Training
    # ──────────────────────────────────────────
    def compute_baselines(self, window_days: int = 28) -> Dict[int, PlayerBaselineProfile]:
        """Compute personal baselines for all registered players."""
        baselines = {}
        for pid in self.registry.all_player_ids():
            player = self.registry.get(pid)
            if player["sessions_df"].empty:
                logger.warning("Player %d: no sessions — baseline skipped", pid)
                continue

            baseline = self.baseline_builder.compute(
                player_id=pid,
                external_id=player["external_id"],
                sessions_df=player["sessions_df"],
                events_df=player["events_df"],
                window_days=window_days,
            )

            if baseline:
                player["baseline"] = baseline
                self.pattern_engine.register_player(pid, baseline)
                baselines[pid] = baseline
                logger.info(
                    "Baseline computed for player %d: sessions=%d, "
                    "dist_mean=%.0fm, sprint_mean=%.1f",
                    pid, baseline.n_sessions,
                    baseline.distance_mean, baseline.sprint_count_mean
                )
            else:
                logger.warning("Player %d: baseline computation failed", pid)

        return baselines

    def train_all_models(self) -> None:
        """Train Isolation Forest for every player with a baseline."""
        for pid in self.registry.all_player_ids():
            player = self.registry.get(pid)
            if player["baseline"] is None:
                logger.warning("Player %d: no baseline — model training skipped", pid)
                continue

            # Build historical feature matrix from session data
            feature_matrix = self.pattern_engine.build_historical_feature_matrix(
                player_id=pid,
                sessions_df=player["sessions_df"],
                events_df=player["events_df"],
                annotations_df=player["annotations_df"],
            )

            if len(feature_matrix) == 0:
                logger.warning("Player %d: empty feature matrix — training skipped", pid)
                continue

            model = PlayerAnomalyModel(pid)
            model.train(feature_matrix)
            player["model"] = model
            self.pattern_engine._models[pid] = model

            # Build SHAP explainer
            self.xai_layer.register_explainer(model, feature_matrix)

            # Save model
            save_path = model.save()
            logger.info("Player %d: model trained and saved to %s", pid, save_path)

    # ──────────────────────────────────────────
    # Real-Time Inference
    # ──────────────────────────────────────────
    def process_live_event(
        self,
        normalized_event: dict,
        match_id: Optional[str] = None,
        segment_index: int = 0,
    ) -> Optional[SHAPExplanation]:
        """
        Process one normalized event from the ingestion pipeline.
        Returns a SHAPExplanation if an alert is triggered, None otherwise.
        Max latency target: 200 ms (per proposal spec).
        """
        t_start = time.perf_counter()

        external_id = normalized_event.get("player_external_id")
        if not external_id:
            return None

        player = self.registry.get_by_external_id(external_id)
        if player is None:
            return None

        pid = player["player_id"]

        # Get latest coach annotation for this player
        ann = self._get_latest_annotation(player)

        # Run pattern analysis
        result = self.pattern_engine.analyze(
            player_id=pid,
            live_window=normalized_event,
            sessions_df=player["sessions_df"],
            segment_index=segment_index,
            coach_annotation=ann,
        )

        if result is None or result.recommendation_type is None:
            return None

        # Compute SHAP explanation
        model = player.get("model")
        if model is None:
            logger.debug("Player %d: no trained model — SHAP skipped", pid)
            return None

        explanation = self.xai_layer.explain(result, model, player["name"])

        # Log inference
        self._inference_id_counter += 1
        inference_record = {
            "inference_id": self._inference_id_counter,
            "player_id": pid,
            "session_id": normalized_event.get("session_id"),
            "recommendation_type": result.recommendation_type,
            "confidence": result.confidence,
            "triggered_at": result.triggered_at.isoformat(),
            "shap_values": explanation.shap_values,
            "feature_values": explanation.feature_values,
            "nlg_summary": explanation.nlg_summary,
            "counterfactual": explanation.counterfactual,
            "anomaly_score": result.anomaly_score,
            "model_version": model.model_version,
            "is_anomaly": result.is_anomaly,
        }
        self._inference_log.append(inference_record)

        # Fire callback if registered
        if self._on_alert_callback:
            try:
                self._on_alert_callback(explanation)
            except Exception as exc:
                logger.exception("Alert callback failed: %s", exc)

        t_ms = (time.perf_counter() - t_start) * 1000
        if t_ms > CONFIG.inference.max_latency_ms:
            logger.warning("Inference latency %.1f ms exceeds 200 ms SLA", t_ms)
        else:
            logger.debug("Inference latency: %.1f ms", t_ms)

        return explanation

    def set_alert_callback(self, callback: Callable[[SHAPExplanation], None]) -> None:
        """Register a callback invoked every time an alert is generated."""
        self._on_alert_callback = callback

    # ──────────────────────────────────────────
    # Coach Feedback
    # ──────────────────────────────────────────
    def log_coach_decision(
        self,
        inference_id: int,
        player_id: int,
        decision: str,
        coach_id: str,
        coach_note: Optional[str] = None,
        session_id: Optional[int] = None,
    ) -> None:
        """
        Log a coach's accept/override/defer decision.
        This is the primary human feedback signal for recalibration.
        """
        inference = next(
            (r for r in self._inference_log if r["inference_id"] == inference_id), None
        )
        if inference is None:
            logger.warning("Inference ID %d not found in log", inference_id)
            return

        player = self.registry.get(player_id)
        if player is None:
            logger.warning("Player %d not registered", player_id)
            return

        record = OverrideRecord(
            inference_id=inference_id,
            player_id=player_id,
            player_external_id=player["external_id"],
            session_id=session_id or inference.get("session_id", 0),
            recommendation_type=inference["recommendation_type"],
            decision=decision,
            coach_id=coach_id,
            coach_note=coach_note,
            overridden_at=datetime.now(tz=timezone.utc),
            context_snapshot=inference.get("feature_values", {}),
            position=player.get("position"),
            age_group=player.get("age_group"),
            nationality=player.get("nationality"),
        )

        self.feedback_store.log_override(record)

    # ──────────────────────────────────────────
    # Recalibration & Fairness
    # ──────────────────────────────────────────
    def recalibrate(self, trigger_reason: str = "manual") -> List[dict]:
        """
        Run the recalibration pipeline.
        Returns list of adjustment summaries.
        """
        player_models = {
            pid: self.registry.get(pid)["model"]
            for pid in self.registry.all_player_ids()
            if self.registry.get(pid).get("model") is not None
        }

        results = self.recalibration_pipeline.run(
            feedback_store=self.feedback_store,
            player_models=player_models,
            trigger_reason=trigger_reason,
        )

        summaries = []
        for r in results:
            summaries.append({
                "player_id": r.player_id,
                "recalibrated_at": r.recalibrated_at.isoformat(),
                "trigger": r.trigger_reason,
                "n_overrides": r.n_overrides_analyzed,
                "adjustments": r.adjustments,
                "notes": r.notes,
            })
            logger.info(
                "Recalibration: player=%s reason=%s adjustments=%d",
                r.player_id, r.trigger_reason, len(r.adjustments)
            )

        return summaries

    def run_fairness_audit(self) -> str:
        """Run fairness audit and return a plain-text report."""
        if not self._inference_log:
            return "No inference data available for fairness audit."

        inference_df = pd.DataFrame(self._inference_log)
        inference_df["is_anomaly"] = inference_df.get("is_anomaly", False)

        metadata_df = self.registry.metadata_dataframe()

        audit_results = self.fairness_monitor.audit(inference_df, metadata_df)
        report = self.fairness_monitor.generate_audit_report(audit_results)

        logger.info("Fairness audit complete — %d attributes audited", len(audit_results))
        return report

    # ──────────────────────────────────────────
    # Live Ingestion Integration
    # ──────────────────────────────────────────
    async def run_live(
        self,
        enable_gps: bool = False,
        enable_api: bool = True,
        enable_ws: bool = True,
        enable_mqtt: bool = True,
        pitch_origin: Optional[tuple] = None,
        gps_player_id: Optional[str] = None,
    ) -> None:
        """
        Start the full live ingestion + inference loop.
        Blocks until the pipeline is stopped.
        """
        ingestion = IngestionPipeline(
            on_event=lambda event: self.process_live_event(event),
            pitch_origin=pitch_origin,
        )

        # Schedule weekly recalibration
        recal_task = asyncio.create_task(self._scheduled_recalibration())

        logger.info("Players Data Analysis Pipeline — LIVE MODE STARTED")
        await asyncio.gather(
            ingestion.run(
                enable_gps=enable_gps,
                enable_api=enable_api,
                enable_ws=enable_ws,
                enable_mqtt=enable_mqtt,
                gps_player_id=gps_player_id,
            ),
            recal_task,
            return_exceptions=True,
        )

    async def _scheduled_recalibration(self) -> None:
        """Run recalibration every N days as configured."""
        interval_s = CONFIG.feedback.recalibration_cadence_days * 86400
        while True:
            await asyncio.sleep(interval_s)
            logger.info("Scheduled recalibration triggered")
            self.recalibrate(trigger_reason="weekly_cadence")
            logger.info("Scheduled fairness audit triggered")
            print(self.run_fairness_audit())

    # ──────────────────────────────────────────
    # Inspection / Reporting
    # ──────────────────────────────────────────
    def get_inference_log(self) -> pd.DataFrame:
        """Return all logged inferences as a DataFrame for analysis."""
        return pd.DataFrame(self._inference_log) if self._inference_log else pd.DataFrame()

    def get_override_summary(self) -> dict:
        """Summary statistics for coach overrides."""
        df = self.feedback_store.to_dataframe()
        if df.empty:
            return {"total": 0, "override_rate": 0.0}

        return {
            "total_decisions": len(df),
            "total_overrides": int((df["decision"] == "override").sum()),
            "total_accepts": int((df["decision"] == "accept").sum()),
            "override_rate": round(self.feedback_store.override_rate, 4),
            "by_recommendation_type": df.groupby("recommendation_type")["decision"]
            .value_counts()
            .to_dict(),
        }

    def _get_latest_annotation(self, player: dict) -> Optional[dict]:
        """Get the most recent coach annotation for a player."""
        ann_df = player.get("annotations_df")
        if ann_df is None or ann_df.empty:
            return None

        latest = ann_df.sort_values("annotated_at", ascending=False).iloc[0]
        return {
            "fatigue_severity": float(latest.get("severity", 0.0)),
            "pre_match_status": latest.get("value", "good"),
            "annotation_type": latest.get("annotation_type", ""),
        }
