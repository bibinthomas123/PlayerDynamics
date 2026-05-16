"""
Players Data — IBM CIC Germany
Main Analysis Orchestrator  (v2 — Sequence Models)


"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Tuple
from analysis.telemetry_validity import TelemetryValidityLayer, TelemetryStatus
from utils.alert_manager import AlertLevel
import numpy as np
import pandas as pd
from config.settings import SEQUENCE_FEATURE_NAMES as _SFN
from analysis.anomaly_detection import (
    PatternAnalysisEngine, AnomalyResult
)
from dataclasses import replace
from analysis.match_state import MatchStateManager, MatchState
from utils.reliability.invariants import SystemInvariantGuard
from analysis.baseline import BaselineBuilder, PlayerBaselineProfile
from explainability.xai_layer import XAILayer, SHAPExplanation, FEATURE_NAMES as XAI_FEATURE_NAMES
from feedback.recalibration import (
    FeedbackStore, FairnessMonitor, OverrideRecord, RecalibrationPipeline,
)
from ingestion.pipeline import IngestionPipeline
from config.settings import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)


# ─────────────────────────────────────────────
# Player registry
# ─────────────────────────────────────────────
class PlayerRegistry:
    def __init__(self):
        self._players: Dict[int, dict] = {}

    def register(
        self, player_id: int, external_id: str, name: str,
        position: str, age: int, age_group: str = "Senior", nationality: str = "",
    ) -> None:
        self._players[player_id] = {
            "player_id": player_id, "external_id": external_id,
            "name": name, "position": position, "age": age,
            "age_group": age_group, "nationality": nationality,
            "baseline": None, "model": None,
            "sessions_df": pd.DataFrame(),
            "events_df": pd.DataFrame(),
            "annotations_df": pd.DataFrame(),
        }

    def get(self, player_id: int) -> Optional[dict]:
        return self._players.get(player_id)

    def get_by_external_id(self, eid: str) -> Optional[dict]:
        return next((p for p in self._players.values() if p["external_id"] == eid), None)

    def all_player_ids(self) -> List[int]:
        return list(self._players.keys())

    def metadata_dataframe(self) -> pd.DataFrame:
        if not self._players:
            return pd.DataFrame()
        return pd.DataFrame([
            {"player_id": p["player_id"], "name": p["name"],
             "position": p["position"], "age_group": p["age_group"],
             "nationality": p["nationality"]}
            for p in self._players.values()
        ])


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
class PlayersDataAnalysisPipeline:
    """
    Production-level analysis pipeline.

    Quick start
    ───────────
    pipeline = PlayersDataAnalysisPipeline()
    pipeline.register_player(player_id=7, external_id="p007", ...)
    pipeline.load_historical_data(7, sessions_df, events_df)
    pipeline.compute_baselines()
    pipeline.train_all_models()
    pipeline.set_alert_callback(my_fn)
    pipeline.process_live_event(event_dict)
    """

    def __init__(self, model_type: str = None, replay_mode: bool = False):
        self.model_type = model_type or CONFIG.active_model
        self.replay_mode = replay_mode
        self.registry = PlayerRegistry()
        self.baseline_builder = BaselineBuilder()
        self.pattern_engine = PatternAnalysisEngine()
        self.xai_layer = XAILayer()
        self.feedback_store = FeedbackStore()
        self.recalibration_pipeline = RecalibrationPipeline()
        self.fairness_monitor = FairnessMonitor()
        self.tvl = TelemetryValidityLayer(replay_mode=replay_mode)
        self.guard = SystemInvariantGuard()
        self._last_xai_ts: Dict[int, float] = {}

        # ── Determinism & Recovery Layers ────────────────────────────────────
        from utils.reliability.determinism import MutationJournal, TemporalCausalityGuard
        self.journal = MutationJournal()
        self.causality_guard = TemporalCausalityGuard()

        self._inference_log: List[dict] = []
        self._on_alert_callback: Optional[Callable] = None
        self._inference_id_counter = 0

        # ── Match state ───────────────────────────────────────────────────────
        self._match_state = MatchStateManager()
        self._active_match_id: Optional[str] = None

    def register_player(
        self, player_id: int, external_id: str, name: str,
        position: str, age: int, age_group: str = "Senior", nationality: str = "",
    ) -> None:
        self.registry.register(player_id, external_id, name, position, age, age_group, nationality)
        logger.info("Registered player: %s (id=%d pos=%s)", name, player_id, position)

    def load_historical_data(
        self,
        player_id: int,
        sessions_df: pd.DataFrame,
        events_df: pd.DataFrame,
        annotations_df: pd.DataFrame = None,
    ) -> None:
        player = self.registry.get(player_id)
        if player is None:
            raise ValueError(f"Player {player_id} not registered")
        player["sessions_df"] = sessions_df
        player["events_df"]   = events_df
        player["annotations_df"] = annotations_df if annotations_df is not None else pd.DataFrame()
        logger.info("Data loaded for player %d: %d sessions, %d events",
                    player_id, len(sessions_df), len(events_df))

    # ──────────────────────────────────────────
    # Baseline computation
    # ──────────────────────────────────────────
    def compute_baselines(self, window_days: int = 28) -> Dict[int, PlayerBaselineProfile]:
        baselines = {}
        for pid in self.registry.all_player_ids():
            p = self.registry.get(pid)
            if p["sessions_df"].empty:
                continue
            baseline = self.baseline_builder.compute(
                player_id=pid,
                external_id=p["external_id"],
                sessions_df=p["sessions_df"],
                events_df=p["events_df"],
                window_days=window_days,
            )
            if baseline:
                p["baseline"] = baseline
                self.pattern_engine.register_player(pid, baseline)
                baselines[pid] = baseline
                logger.info(
                    "Baseline: player %d | sessions=%d | dist_mean=%.0fm | "
                    "sprint_mean=%.1f | fatigue_r2=%.3f",
                    pid, baseline.n_sessions,
                    baseline.distance_mean, baseline.sprint_count_mean,
                    baseline.fatigue_r_squared or 0,
                )
        return baselines

    # ──────────────────────────────────────────
    # Model training 
    # ──────────────────────────────────────────
    def train_all_models(self) -> dict:
        """
        Collects sliding-window sequences from all players, then trains ONE
        shared backbone model across all players simultaneously.
        Per-player thresholds are calibrated from each player's held-out slice.
        XAI background is registered per-player from their own windows.
        """

        # ── Phase 1: build sequences for every eligible player ──────────────
        all_windows: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for pid in self.registry.all_player_ids():
            p = self.registry.get(pid)

            if p["baseline"] is None:
                logger.warning("Player %d: no baseline — skipped", pid)
                continue
            if p["events_df"].empty:
                logger.warning("Player %d: no events — skipped", pid)
                continue

            sequences = self.pattern_engine.build_training_sequences(
                events_df=p["events_df"],
                sessions_df=p["sessions_df"],
            )

            if len(sequences) == 0:
                logger.warning("Player %d: 0 sequences built — skipped", pid)
                continue

            all_windows[pid] = sequences
            logger.info("Player %d: %d training sequences built", pid, len(sequences))

        if not all_windows:
            logger.warning("No eligible players — shared model training aborted")
            return {}

        # ── Phase 2: train ONE shared model across all players ───────────────
        result = self.pattern_engine.train_player_model(all_windows)
        logger.info(
            "Shared backbone trained | players=%d | total_windows=%d | version=%s",
            result.get("n_players", 0),
            result.get("n_windows", 0),
            self.pattern_engine._shared_model.model_version
            if self.pattern_engine._shared_model else "n/a",
        )

        # ── Phase 3: per-player XAI background registration ─────────────────
        results: Dict[int, dict] = {}

        for pid, sequences in all_windows.items():
            p = self.registry.get(pid)

            shared_model = self.pattern_engine._shared_model
            if shared_model is None:
                continue

            # Store reference on the player record so process_live_event can reach it
            p["model"] = shared_model

            # ── Build XAI-space background from REAL windows ──────────────────
            # Each background row is derived from an actual training window using
            # the same feature-engineering transforms as inference. This ensures
            # SHAP explanations reflect the true data manifold.
            # Previously used synthetic random samples (bg_rng.normal, bg_rng.uniform)
            # for unmapped features — that creates a fake manifold which can make
            # SHAP attributions directionally misleading.
            _sfn_idx     = {n: i for i, n in enumerate(_SFN)}

            n_bg_samples = min(32, len(sequences))
            results[pid] = {
                    "status": "trained",
                    "n_sequences": len(sequences),
                    "xai_background_rows": n_bg_samples,
                }
            xai_dim      = len(XAI_FEATURE_NAMES)
            bg_xai       = np.zeros((n_bg_samples, xai_dim), dtype=np.float32)
            _c           = {n: i for i, n in enumerate(XAI_FEATURE_NAMES)}
            baseline_p   = p.get("baseline")

            for bi, (w, mask_w,_) in enumerate(sequences[:n_bg_samples]):
                last = w[-1]   # last timestep (N_SEQUENCE_FEATURES,)

                # --- Features derived directly from window data ---------------
                spd = float(last[_sfn_idx["speed_ms"]])
                hr  = float(last[_sfn_idx["heart_rate_bpm"]])
                spr = float(last[_sfn_idx["sprint_flag"]])
                ddt = float(last[_sfn_idx["distance_delta_m"]])

                bg_xai[bi, _c["window_avg_speed_ms"]]  = spd
                bg_xai[bi, _c["heart_rate_bpm"]]       = hr
                bg_xai[bi, _c["window_sprint_count"]]  = spr
                # Window distance: per-tick displacement × number of valid ticks
                n_valid = max(int(mask_w.sum()), 1)
                bg_xai[bi, _c["window_distance_m"]]    = ddt * n_valid

                # HR recovery: fractional HR change × current HR gives bpm-delta display value.
                # hr_recovery_rate is now in [-1, 1] (fractional), not bpm/s.
                # The old * 15.0 multiplier assumed bpm/s units — now incorrect.
                hr_rec_frac = abs(float(last[_sfn_idx["hr_recovery_rate"]]))
                hr_current  = max(float(last[_sfn_idx["heart_rate_bpm"]]), 1.0)
                bg_xai[bi, _c["hr_recovery_time_s"]] = hr_rec_frac * hr_current

                # Missingness ratio as informative feature
                missing_frac = 1.0 - float(mask_w.mean())

                # Baseline z-scores from real baseline (not random)
                if baseline_p is not None:
                    window_dist = ddt * n_valid
                    bg_xai[bi, _c["z_distance"]]       = float(np.clip(
                        baseline_p.zscore("distance", window_dist), -4, 4))
                    bg_xai[bi, _c["z_sprint_count"]]   = float(np.clip(
                        baseline_p.zscore("sprint_count", spr * 10), -4, 4))
                    bg_xai[bi, _c["z_top_speed"]]      = float(np.clip(
                        baseline_p.zscore("top_speed", spd), -4, 4))
                    bg_xai[bi, _c["z_high_speed_dist"]] = float(np.clip(
                        baseline_p.zscore("high_speed_dist", window_dist * 0.28), -4, 4))

                # Fatigue residual: speed vs expected decay at window midpoint
                # Use baseline fatigue curve if available, else 0
                if baseline_p is not None and baseline_p.fatigue_alpha:
                    import math as _math
                    t_mid   = (bi / max(n_bg_samples, 1)) * 90.0  # rough match-minute proxy
                    beta    = baseline_p.fatigue_beta or spd * 1.3
                    alpha   = baseline_p.fatigue_alpha
                    exp_spd = beta * _math.exp(-alpha * t_mid)
                    bg_xai[bi, _c["fatigue_decay_residual"]] = float(
                        np.clip((spd - exp_spd) * n_valid, -500, 500))

                # ACWR default 1.0 for background (normal training load)
                bg_xai[bi, _c["acwr"]] = 1.0

                # Coach features default to 0 (no annotation for training windows)
                # positional_drift_score defaults to 0 (no drift for most windows)

            self.xai_layer.register_explainer_for_player(pid, bg_xai)
            logger.info("Player %d: XAI background registered from real windows (%d rows)",
                        pid, n_bg_samples)

            # ── Store raw sequence background for true SHAP ───────────────────
            # xai_layer._explain_sequence_shap() needs the raw (unnormalised)
            # sequences in (N_bg, T, F) shape. The model normaliser is applied
            # inside _explain_sequence_shap so perturbations happen in the same
            # space the LSTM sees.
            raw_bg_sequences = np.stack(
                [w for w, _ , _ in sequences[:n_bg_samples]], axis=0
            )   # (N_bg, T, F)
            p["sequence_background"] = raw_bg_sequences
            logger.info(
                "Player %d: sequence background stored (%d × %s) for true SHAP",
                pid, len(raw_bg_sequences), raw_bg_sequences.shape[1:],
            )

            
        return {
            "status": "success",
            "shared_model": {
                "n_players": result["n_players"],
                "n_windows": result["n_windows"],
                "model_version": self.pattern_engine.get_model_version(),
            },
            "players": results,
        }
    # ──────────────────────────────────────────
    # Live inference
    # ──────────────────────────────────────────
    def process_live_event(
        self,
        normalized_event: dict,
        segment_index: int = 0,
    ) -> Optional[AnomalyResult]:
        """
        Process one raw normalised event from the ingestion pipeline.
        Returns SHAPExplanation if an alert is triggered, None otherwise.
        <200 ms latency target.
        """
        t0 = time.perf_counter()

        eid = normalized_event.get("player_external_id")
        if not eid:
            return None

        player = self.registry.get_by_external_id(eid)
        if player is None:
            return None

        pid = player["player_id"]

        # ── Determinism Gate: Temporal Causality ───────────────────────────────
        event_ts = normalized_event.get("timestamp")
        if event_ts:
            if not self.causality_guard.validate_sequence(pid, event_ts):
                logger.error("Determinism Error: Out-of-order event for player %d. Rejecting.", pid)
                return None

        # ── Reliability Gate: Telemetry Validity ───────────────────────────────
        from analysis.telemetry_validity import TelemetryStatus
        from utils.reliability.invariants import InvariantSeverity
        validity = self.tvl.validate_event(pid, normalized_event)

        # Invariant: INVALID telemetry must NEVER generate physiological alerts.
        self.guard.check(
            "TELEMETRY_VALIDITY_GATE",
            condition=(validity.status != TelemetryStatus.INVALID),
            severity=InvariantSeverity.WARNING if validity.status == TelemetryStatus.INVALID else InvariantSeverity.INFO,
            message=f"Event rejected or flagged: status={validity.status.name} issues={validity.issues}",
            context={"player_id": pid, "validity": validity}
        )

        if validity.status == TelemetryStatus.INVALID:
            logger.warning("Inference gated: INVALID telemetry for player %d", pid)
            return None

        # Run sequence anomaly analysis
        result = self.pattern_engine.analyze(
            player_id=pid,
            live_event=normalized_event,
            sessions_df=player["sessions_df"],
        )

        if result is None:
            return None
        
       # ── SHAP/XAI Gating ───────────────────────────────────────────────────

        explanation = None

        model = player.get("model")

        if model is not None:

            from utils.reliability.safe_mode import (
                safe_mode,
                SafeModeLevel,
            )

            # Safe Mode suppression
            shap_allowed = safe_mode.is_feature_enabled(
                "shap_explanation",
                SafeModeLevel.LEVEL_1,
            )

            # Only explain sustained operational alerts
            sustained_alert = result.alert_level in (
                AlertLevel.WARNING,
                AlertLevel.CRITICAL,
            )

            # Require persistence before XAI
            sufficient_persistence = result.persistence_windows >= 3

            # Per-player cooldown
            now = time.time()

            last_xai_ts = self._last_xai_ts.get(pid, 0.0)

            # cooldown_ok = (
            #     now - last_xai_ts
            # ) >= 60.0

            cooldown_ok = True

            run_xai = (
                shap_allowed
                and sustained_alert
                and sufficient_persistence
                and cooldown_ok
            )
            # run_xai = True  # FOR TESTING - BYPASS ALL GATES

            # print("\nXAI DEBUG")
            # print("shap_allowed:", shap_allowed)
            # print("sustained_alert:", sustained_alert)
            # print("sufficient_persistence:", sufficient_persistence)
            # print("cooldown_ok:", cooldown_ok)
            # print("run_xai:", run_xai)

            xai_fv = self._build_xai_feature_vector(result)

                # Get/create match state before explanation so context is current
            match_state = self._match_state.get_or_create(
                    player_id=pid,
                    player_name=player["name"],
                    position=player.get("position", ""),
                    match_id=self._active_match_id,
                )

            match_state.record_telemetry(
                speed_ms=xai_fv.get("window_avg_speed_ms", 0.0),
                hr_bpm=xai_fv.get("heart_rate_bpm", 0.0),
                hr_recovery_rate=result.feature_vector.get("hr_recovery_rate", 0.0),
                anomaly_score=result.anomaly_score,
                telemetry_confidence=self._effective_confidence(validity.confidence),
            )

            if run_xai:

    
                # ─────────────────────────────────────────────
                # Stage 1 — base explanation (NO LLM)
                # ─────────────────────────────────────────────

                base_explanation = self.xai_layer.build_base_explanation(
                    player_id=pid,
                    external_id=player["external_id"],
                    player_name=player["name"],
                    model=model,
                    feature_vector=xai_fv,
                    recommendation_type=result.recommendation_type,
                    confidence=result.confidence,
                    workload_status=result.workload_status,
                    anomaly_score=result.anomaly_score,
                    sequence=result.raw_sequence if hasattr(result, "raw_sequence") else None,
                    mask=result.raw_mask if hasattr(result, "raw_mask") else None,
                    sequence_background=player.get("sequence_background"),
                    persistence_windows=result.persistence_windows,
                )

                # logger.info(
                #     "BASE SHAP FEATURES: %s",
                #     list(base_explanation.shap_dict.keys())[:5],
                # )
                semantic_findings = self.xai_layer.build_semantic_findings(
                    shap_dict=base_explanation.shap_values,
                    feature_values=xai_fv,
                    persistence_windows=result.persistence_windows,
                )

                base_explanation = replace(
                    base_explanation,
                    semantic_findings=tuple(
                        f.to_dict() if hasattr(f, "to_dict") else f
                        for f in semantic_findings
                    ),
                )
                # ─────────────────────────────────────────────
                # Stage 2 — update symbolic longitudinal memory
                # ─────────────────────────────────────────────

                elapsed_s = int(xai_fv.get("elapsed_seconds", 0))

                for finding_dict in base_explanation.semantic_findings:
                    match_state.record_finding(
                        finding=finding_dict,
                        elapsed_seconds=elapsed_s,
                    )

                # Record alert AFTER findings so motif/trend/escalation
                # reasoning sees current findings before this alert is stored.
                match_state.record_alert(
                    recommendation_type=result.recommendation_type,
                    confidence=result.confidence,
                    anomaly_score=result.anomaly_score,
                    elapsed_seconds=elapsed_s,
                )

                # ─────────────────────────────────────────────
                # Stage 3 — build CURRENT semantic state
                # ─────────────────────────────────────────────

                semantic_state = match_state.build_semantic_state()

                logger.info(
                        "CURRENT WINDOW STATE | motifs=%s escalation=%s findings=%d",
                        [m["type"] for m in semantic_state.motifs],
                        semantic_state.escalation_level,
                        len(match_state.recent_findings),
                    )

                # ─────────────────────────────────────────────
                # Stage 4 — final narrative generation
                # ─────────────────────────────────────────────

                explanation = self.xai_layer.generate_explanation_from_base(
                    base=base_explanation,
                    match_state=semantic_state,
                )

                self._last_xai_ts[pid] = now

        # Log inference
        self._inference_id_counter += 1
        log_entry = {
            "inference_id":        self._inference_id_counter,
            "player_id":           pid,
            "recommendation_type": result.recommendation_type,
            "confidence":          result.confidence,
            "triggered_at":        result.triggered_at.isoformat(),
            "anomaly_score":       result.anomaly_score,
            "model_type":          result.model_type,
            "model_version":       getattr(model, "model_version", ""),
            "is_anomaly":          result.is_anomaly,
            "feature_values":      result.feature_vector,
            "shap_values":         explanation.shap_values if explanation else {},
            "nlg_summary":         explanation.nlg_summary if explanation else "",
        }
        self._inference_log.append(log_entry)

        if self._on_alert_callback and explanation:
            try:
                self._on_alert_callback(explanation)
            except Exception as exc:
                logger.exception("Alert callback error: %s", exc)

        t_ms = (time.perf_counter() - t0) * 1000
        if t_ms > CONFIG.inference.max_latency_ms:
            logger.warning("Inference latency %.1f ms > 200 ms SLA", t_ms)

        if explanation is not None:
            result.nlg_summary = explanation.nlg_summary
            result.counterfactual = explanation.counterfactual
            result.shap_values = explanation.shap_values
            result.top_contributions = explanation.top_contributions

        return result
    
    def _effective_confidence(
        self,
        validity_confidence: float,
        replay_mode: Optional[bool] = None,
    ) -> float:
        """
        Return the telemetry confidence value that should be passed downstream.

        TVL always reflects live-semantics truthfulness.  In replay mode the
        only degradation is timestamp discontinuity — not a sensor failure —
        so we floor the confidence at 0.8 to allow trajectory accumulation
        and persistence escalation.

        Operational-mode override is intentionally located here (orchestration
        boundary) rather than inside TVL (validity semantics) or MatchState
        (accumulation logic).  TVL semantics remain unmodified.
        """
        is_replay = replay_mode if replay_mode is not None else self.replay_mode
        if is_replay:
            effective = max(validity_confidence, 0.8)
            logger.debug(
                "REPLAY CONF | replay=%s raw=%.3f effective=%.3f",
                is_replay, validity_confidence, effective,
            )
            return effective
        return validity_confidence
        
    def process_window_direct(
        self,
        window_events: list[dict],
        player_id: int,
        replay_mode: Optional[bool] = None,
        nlg_async: bool = False,
    ) -> Optional[AnomalyResult]:

        """
        Score a pre-built accumulator window without touching add_event().
        """

        player = self.registry.get(player_id)

        if player is None:
            return None

        # ─────────────────────────────────────────────
        # TVL validation
        # ─────────────────────────────────────────────
        latest_event = window_events[-1]

        validity = self.tvl.validate_event(
            player_id,
            latest_event,
        )

        if validity.status == TelemetryStatus.INVALID:

            logger.warning(
                "process_window_direct: INVALID telemetry player=%d issues=%s — skipping",
                player_id,
                validity.issues,
            )

            for alert_type in (
                "anomaly",
                "fatigue",
                "drift",
                "workload",
            ):
                self.pattern_engine.alert_manager.process_signal(
                    player_id,
                    alert_type,
                    signal_active=False,
                    confidence=0.0,
                )

            return None

        latest_event = dict(latest_event)

        latest_event["_tvl_confidence"] = (
            self._effective_confidence(
                validity.confidence,
                replay_mode=replay_mode,
            )
        )

        latest_event["_tvl_status"] = validity.status.name

        # ─────────────────────────────────────────────
        # Build sequence
        # ─────────────────────────────────────────────
        seq, mask = (
            self.pattern_engine.window_builder.build_live_window(
                window_events
            )
        )

        result = self.pattern_engine.analyze_window(
            player_id=player_id,
            sequence=seq,
            mask=mask,
            live_event=latest_event,
            sessions_df=player["sessions_df"],
        )

        if result is None:
            return None

        # ─────────────────────────────────────────────
        # XAI
        # ─────────────────────────────────────────────
        explanation = None

        model = player.get("model")

        if model is not None:

            from utils.reliability.safe_mode import (
                safe_mode,
                SafeModeLevel,
            )

            shap_allowed = safe_mode.is_feature_enabled(
                "shap_explanation",
                SafeModeLevel.LEVEL_1,
            )

            active_alert = (
                result.recommendation_type is not None
            )

            now = time.time()

            # IMPORTANT FIX:
            # synchronous mode MUST always run SHAP
            run_xai = (
                shap_allowed
                and active_alert
                and (
                    not nlg_async
                    or True
                )
            )

            if run_xai:
                self._last_xai_ts[player_id] = now

            xai_fv = self._build_xai_feature_vector(
                result
            )

            match_state = self._match_state.get_or_create(
                player_id=player_id,
                player_name=player["name"],
                position=player.get("position", ""),
                match_id=self._active_match_id,
            )

            match_state.record_telemetry(
                speed_ms=xai_fv.get(
                    "window_avg_speed_ms",
                    0.0,
                ),
                hr_bpm=xai_fv.get(
                    "heart_rate_bpm",
                    0.0,
                ),
                hr_recovery_rate=result.feature_vector.get(
                    "hr_recovery_rate",
                    0.0,
                ),
                anomaly_score=result.anomaly_score,
                telemetry_confidence=self._effective_confidence(
                    float(
                        latest_event.get(
                            "_tvl_confidence",
                            1.0,
                        )
                    ),
                    replay_mode=replay_mode,
                ),
            )

            # ─────────────────────────────────────────
            # ALWAYS attach semantic/XAI state
            # THIS FIXES missing_base_explanation
            # ─────────────────────────────────────────
            result.semantic_state = (
                match_state.build_semantic_state()
            )

            result._xai_kwargs = dict(
                player_id=player_id,
                external_id=player["external_id"],
                player_name=player["name"],
                model=model,
                feature_vector=xai_fv,
                recommendation_type=result.recommendation_type,
                confidence=result.confidence,
                workload_status=result.workload_status,
                anomaly_score=result.anomaly_score,
                sequence=result.raw_sequence,
                mask=result.raw_mask,
                sequence_background=player.get(
                    "sequence_background"
                ),
                persistence_windows=result.persistence_windows,
                match_state=result.semantic_state,
                elapsed_s=int(
                    xai_fv.get(
                        "elapsed_seconds",
                        0,
                    )
                ),
            )

            # ─────────────────────────────────────────
            # SHAP + NLG
            # ─────────────────────────────────────────
            if run_xai:

                if nlg_async:

                    # async path
                    result.nlg_summary = (
                        self.xai_layer._template_nlg.generate(
                            recommendation_type=result.recommendation_type,
                            confidence=result.confidence,
                            player_name=player["name"],
                            top_contributions=[],
                            workload_status=result.workload_status,
                            semantic_findings=None,
                        )
                    )

                    result.base_explanation = None

                else:

                    # synchronous SHAP
                    base_explanation = (
                        self.xai_layer.build_base_explanation(
                            player_id=player_id,
                            external_id=player["external_id"],
                            player_name=player["name"],
                            model=model,
                            feature_vector=xai_fv,
                            recommendation_type=result.recommendation_type,
                            confidence=result.confidence,
                            workload_status=result.workload_status,
                            anomaly_score=result.anomaly_score,
                            sequence=result.raw_sequence,
                            mask=result.raw_mask,
                            sequence_background=player.get(
                                "sequence_background"
                            ),
                            persistence_windows=result.persistence_windows,
                        )
                    )

                    semantic_findings = (
                        self.xai_layer.build_semantic_findings(
                            shap_dict=base_explanation.shap_values,
                            feature_values=xai_fv,
                            persistence_windows=result.persistence_windows,
                        )
                    )

                    base_explanation = replace(
                        base_explanation,
                        semantic_findings=tuple(
                            (
                                f.to_dict()
                                if hasattr(f, "to_dict")
                                else f
                            )
                            for f in semantic_findings
                        ),
                    )

                    result.base_explanation = (
                        base_explanation
                    )

                    elapsed_s = int(
                        xai_fv.get(
                            "elapsed_seconds",
                            0,
                        )
                    )

                    for finding_dict in (
                        base_explanation.semantic_findings
                    ):

                        match_state.record_finding(
                            finding=finding_dict,
                            elapsed_seconds=elapsed_s,
                        )

                    match_state.record_alert(
                        recommendation_type=result.recommendation_type,
                        confidence=result.confidence,
                        anomaly_score=result.anomaly_score,
                        elapsed_seconds=elapsed_s,
                    )

                    semantic_state = (
                        match_state.build_semantic_state()
                    )

                    explanation = (
                        self.xai_layer.generate_explanation_from_base(
                            base=base_explanation,
                            match_state=semantic_state,
                        )
                    )

                    if explanation is not None:

                        result.nlg_summary = (
                            explanation.nlg_summary
                        )

                        result.counterfactual = (
                            explanation.counterfactual
                        )

                        result.shap_values = (
                            explanation.shap_values
                        )

                        result.top_contributions = (
                            explanation.top_contributions
                        )

                        result.nlg_engine = getattr(
                            explanation,
                            "nlg_engine",
                            "llm_unknown",
                        )

                    self._last_xai_ts[player_id] = now

        return result

    def _build_xai_feature_vector(self, result: AnomalyResult) -> dict:
        """
        Maps sequence AnomalyResult → 15-feature XAI dict.

        Key corrections vs. prior version:
        • window_distance_m = distance_delta_m * window_steps (not * 8 hardcoded)
        • reconstruction_loss excluded — not in FEATURE_NAMES, was silently dropped
        • z-scores computed from real baseline, not fabricated
        """
        sfv          = result.feature_vector
        window_steps = CONFIG.window.window_steps   # 24 at 5 s/tick

        xai: dict = {}
        xai["window_sprint_count"] = sfv.get("sprint_flag", 0.0)
        # Scale single-tick displacement to full window distance
        xai["window_distance_m"]   = sfv.get("distance_delta_m", 0.0) * window_steps
        xai["window_avg_speed_ms"] = sfv.get("speed_ms", 0.0)
        xai["heart_rate_bpm"]      = sfv.get("heart_rate_bpm", 0.0)
        # hr_recovery_rate is now a fractional HR change in [-1, 1]
        # Convert to a display-friendly bpm delta: |fraction| x current HR.
        # The old abs(hr_recovery_rate) * 15.0 assumed bpm/s and produced junk values.
        _hr_rec_frac = sfv.get("hr_recovery_rate", 0.0)
        _hr_bpm      = sfv.get("heart_rate_bpm", 150.0)
        xai["hr_recovery_time_s"] = sfv.get("hr_recovery_time_s",
                                             abs(_hr_rec_frac) * max(_hr_bpm, 1.0))
        xai["acwr"]                    = sfv.get("acwr", 1.0)
        xai["positional_drift_score"]  = sfv.get("drift_score", 0.0)
        xai["fatigue_decay_residual"]  = sfv.get("fatigue_decay_residual", 0.0)
        xai["speed_drop_pct"]          = sfv.get("speed_drop_pct", 0.0)
        xai["coach_fatigue_severity"]  = sfv.get("coach_fatigue_severity", 0.0)
        xai["coach_pre_match_status_encoded"] = sfv.get("coach_pre_match_status_encoded", 0.0)

        player   = self.registry.get_by_external_id(result.external_id)
        baseline = player["baseline"] if player else None
        if baseline is not None:
            speed       = xai["window_avg_speed_ms"]
            window_dist = xai["window_distance_m"]
            xai["z_distance"]        = baseline.zscore("distance",       window_dist)
            xai["z_sprint_count"]    = baseline.zscore("sprint_count",   xai["window_sprint_count"] * 10)
            xai["z_top_speed"]       = baseline.zscore("top_speed",      speed)
            xai["z_high_speed_dist"] = baseline.zscore("high_speed_dist", window_dist * 0.28)
        else:
            xai["z_distance"] = xai["z_sprint_count"] = xai["z_top_speed"] = xai["z_high_speed_dist"] = 0.0

        from explainability.xai_layer import FEATURE_NAMES as _XAI_FN
        for fname in _XAI_FN:
            xai.setdefault(fname, 0.0)

        return xai

    def set_alert_callback(self, cb: Callable) -> None:
        self._on_alert_callback = cb

    # ──────────────────────────────────────────
    # Match lifecycle  (Fix 2)
    # ──────────────────────────────────────────
    def start_match(self, match_id: str) -> None:
        """Call at kickoff. Ties all subsequent state to this match_id."""
        self._active_match_id = match_id
        self._match_state.start_match(match_id)
        logger.info("Match started: %s", match_id)

    def end_match(self) -> None:
        """Call at full time. Frees per-match memory and resets match identity."""
        if self._active_match_id:
            self._match_state.end_match(self._active_match_id)
            logger.info("Match ended: %s", self._active_match_id)
            self._active_match_id = None

    # ──────────────────────────────────────────
    # Coach feedback
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
        inference = next(
            (r for r in self._inference_log if r["inference_id"] == inference_id), None
        )
        if not inference:
            logger.warning("Inference ID %d not found", inference_id)
            return

        player = self.registry.get(player_id)
        if not player:
            return

        record = OverrideRecord(
            inference_id=inference_id,
            player_id=player_id,
            player_external_id=player["external_id"],
            session_id=session_id or 0,
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
        logger.info("Coach decision logged: player=%d decision=%s coach=%s",
                    player_id, decision, coach_id)

    # ──────────────────────────────────────────
    # Recalibration & fairness
    # ──────────────────────────────────────────
    def recalibrate(self, trigger_reason: str = "manual") -> List[dict]:
        player_models = {
            pid: self.registry.get(pid)["model"]
            for pid in self.registry.all_player_ids()
            if self.registry.get(pid).get("model") is not None
        }
        results = self.recalibration_pipeline.run(
            self.feedback_store, player_models, trigger_reason
        )
        return [
            {"player_id": r.player_id, "recalibrated_at": r.recalibrated_at.isoformat(),
             "trigger": r.trigger_reason, "adjustments": r.adjustments, "notes": r.notes}
            for r in results
        ]

    def run_fairness_audit(self) -> str:
        if not self._inference_log:
            return "No inference data available for fairness audit."
        inference_df = pd.DataFrame(self._inference_log)
        metadata_df  = self.registry.metadata_dataframe()
        audit_results = self.fairness_monitor.audit(inference_df, metadata_df)
        return self.fairness_monitor.generate_audit_report(audit_results)

    # ──────────────────────────────────────────
    # Live ingestion
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
        from analysis.live_window_accumulator import LiveWindowAccumulator

        match_id = datetime.now(timezone.utc).strftime(
            "live_match_%Y%m%d_%H%M%S"
        )
        self.start_match(match_id)

        # Non-overlapping window accumulator — same parameters as cmd_serve so
        # the two serving paths are architecturally identical.
        _accumulator = LiveWindowAccumulator(
            window_size=CONFIG.window.window_steps,
            stride=CONFIG.window.window_steps,
        )

        def _on_event(ev: dict) -> None:
            ext_id = ev.get("player_external_id", "")
            player = self.registry.get_by_external_id(ext_id)
            if player is None:
                return

            player_id = player["player_id"]
            window = _accumulator.push(player_id=ext_id, event=ev)

            # Session/temporal reset — mirror cmd_serve reset logic.
            if _accumulator.consume_reset_flag(ext_id):
                try:
                    self.pattern_engine.reset_ema_state(player_id)
                    self.pattern_engine.alert_manager.clear_player(player_id)
                except Exception as exc:
                    logger.warning("run_live session reset failed for %s: %s", ext_id, exc)

            if window is None:
                return  # Still accumulating

            try:
                result = self.process_window_direct(
                    window_events=window,
                    player_id=player_id,
                )
            except Exception as exc:
                logger.warning("run_live inference error for %s: %s", ext_id, exc)
                return

            if result is not None and self._on_alert_callback:
                try:
                    self._on_alert_callback(result)
                except Exception as exc:
                    logger.exception("run_live alert callback error: %s", exc)

        ingestion = IngestionPipeline(
            on_event=_on_event,
            pitch_origin=pitch_origin,
        )
        recal_task = asyncio.create_task(self._scheduled_recalibration())
        logger.info("PlayersDataAnalysisPipeline LIVE — model=%s", self.model_type.upper())
        try:
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
        finally:
            self.end_match()

            
    async def _scheduled_recalibration(self) -> None:
        interval_s = CONFIG.feedback.recalibration_cadence_days * 86400
        while True:
            await asyncio.sleep(interval_s)
            logger.info("Scheduled recalibration triggered")
            self.recalibrate("weekly_cadence")
            # print(self.run_fairness_audit())

    # ──────────────────────────────────────────
    # Inspection helpers
    # ──────────────────────────────────────────
    def get_inference_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._inference_log) if self._inference_log else pd.DataFrame()

    def get_override_summary(self) -> dict:
        df = self.feedback_store.to_dataframe()
        if df.empty:
            return {"total": 0, "override_rate": 0.0}
        return {
            "total_decisions": len(df),
            "total_overrides": int((df["decision"] == "override").sum()),
            "override_rate":   round(self.feedback_store.override_rate, 4),
        }