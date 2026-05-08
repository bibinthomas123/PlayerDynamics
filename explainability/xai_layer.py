"""
Players Data — IBM CIC Germany
XAI / Explainability Layer  (shap-compat version)

Wraps shap_compat so the full explanation pipeline works whether or not
the `shap` library is installed.  All public interfaces are identical to
the architecture spec in the proposal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
from explainability.shap_compat import compute_shap_values, build_kmeans_background, SHAP_AVAILABLE
from config.settings import CONFIG, SHAPConfig

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Feature registry
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "window_sprint_count",
    "window_distance_m",
    "window_avg_speed_ms",
    "z_distance",
    "z_sprint_count",
    "z_top_speed",
    "z_high_speed_dist",
    "fatigue_decay_residual",
    "speed_drop_pct",
    "positional_drift_score",
    "acwr",
    "heart_rate_bpm",
    "hr_recovery_time_s",
    "coach_fatigue_severity",
    "coach_pre_match_status_encoded",
]

FEATURE_LABELS: Dict[str, str] = {
    "window_sprint_count":            "Sprint count (last 30 s window)",
    "window_distance_m":              "Distance covered (last 30 s window)",
    "window_avg_speed_ms":            "Average speed (last 30 s window)",
    "z_distance":                     "Distance deviation from personal baseline",
    "z_sprint_count":                 "Sprint count deviation from personal baseline",
    "z_top_speed":                    "Top speed deviation from personal baseline",
    "z_high_speed_dist":              "High-speed distance deviation from baseline",
    "fatigue_decay_residual":         "Fatigue decay residual vs. personal curve",
    "speed_drop_pct":                 "Speed drop vs. session start (%)",
    "positional_drift_score":         "Positional drift from tactical zone",
    "acwr":                           "Acute:Chronic Workload Ratio (7d/28d)",
    "heart_rate_bpm":                 "Heart rate (bpm)",
    "hr_recovery_time_s":             "HR recovery time (s)",
    "coach_fatigue_severity":         "Coach fatigue annotation",
    "coach_pre_match_status_encoded": "Coach pre-match status",
}


def _format_value(name: str, value: float) -> str:
    if "z_" in name:
        d = "above" if value > 0 else "below"
        return f"{abs(value):.1f} SD {d} personal baseline"
    if name == "window_sprint_count":    return f"{int(value)} sprints"
    if name == "window_distance_m":      return f"{value:.0f} m"
    if name == "window_avg_speed_ms":    return f"{value:.1f} m/s"
    if name == "speed_drop_pct":         return f"{value:.1f}% speed drop"
    if name == "positional_drift_score": return f"{value:.2f}x norm radius"
    if name == "acwr":                   return f"ACWR = {value:.2f}"
    if name == "heart_rate_bpm":         return f"{int(value)} bpm"
    if name == "fatigue_decay_residual":
        d = "above" if value >= 0 else "below"
        return f"{abs(value):.0f} m {d} decay curve"
    return f"{value:.2f}"


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
@dataclass
class FeatureContribution:
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str
    human_label: str
    formatted_value: str


@dataclass
class SHAPExplanation:
    player_id: int
    external_id: str
    recommendation_type: str
    confidence: float
    computed_at: datetime
    base_value: float
    shap_values: Dict[str, float]
    feature_values: Dict[str, float]
    top_contributions: List[FeatureContribution]
    nlg_summary: str
    counterfactual: str
    waterfall_data: List[dict]
    shap_method: str = field(default_factory=lambda: "kernel" if SHAP_AVAILABLE else "magnitude_proxy")

    def to_dict(self) -> dict:
        return {
            "player_id":           self.player_id,
            "external_id":         self.external_id,
            "recommendation_type": self.recommendation_type,
            "confidence":          self.confidence,
            "computed_at":         self.computed_at.isoformat(),
            "base_value":          self.base_value,
            "shap_method":         self.shap_method,
            "shap_values":         self.shap_values,
            "feature_values":      self.feature_values,
            "top_contributions": [
                {
                    "feature":         c.feature_name,
                    "value":           c.feature_value,
                    "shap":            c.shap_value,
                    "direction":       c.direction,
                    "label":           c.human_label,
                    "formatted_value": c.formatted_value,
                }
                for c in self.top_contributions
            ],
            "nlg_summary":   self.nlg_summary,
            "counterfactual": self.counterfactual,
            "waterfall_data": self.waterfall_data,
        }


# ─────────────────────────────────────────────
# Counterfactual generator
# ─────────────────────────────────────────────
class CounterfactualGenerator:
    def generate(self, shap_dict: Dict[str, float], feature_values: Dict[str, float]) -> str:
        if not shap_dict:
            return "Insufficient data for counterfactual generation."
        top = max(shap_dict, key=lambda k: shap_dict[k])
        val = feature_values.get(top, 0.0)
        label = FEATURE_LABELS.get(top, top)

        if "z_" in top:
            return (
                f"If {label} were within 1.0 standard deviation of the personal baseline "
                f"(currently {val:.1f} SD), this flag would not trigger."
            )
        if top == "fatigue_decay_residual":
            return (
                f"If distance output matched the player's personal fatigue decay curve "
                f"(current residual: {val:.0f} m), this flag would not trigger."
            )
        if top == "positional_drift_score":
            return (
                f"If the player were within their normal tactical zone "
                f"(current drift: {val:.2f}x, threshold: 1.0x), this flag would not trigger."
            )
        if top == "window_sprint_count":
            target = max(0, val + 2)
            return (
                f"If sprint count were >= {int(target)} "
                f"(currently {int(val)}), this flag would not trigger."
            )
        return (
            f"If {label} were closer to the personal baseline "
            f"(current value: {val:.2f}), this flag would likely not trigger."
        )


# ─────────────────────────────────────────────
# Template NLG engine
# ─────────────────────────────────────────────
class TemplateNLGEngine:
    def generate(
        self,
        recommendation_type: str,
        confidence: float,
        player_name: str,
        top_contributions: List[FeatureContribution],
        workload_status: str = "optimal",
    ) -> str:
        conf_pct = int(confidence * 100)
        labels = {
            "substitution":     f"Consider substituting {player_name}",
            "fatigue_alert":    f"Fatigue alert for {player_name}",
            "positional_drift": f"Positional drift detected for {player_name}",
            "workload_warning": f"Workload warning for {player_name}",
        }
        action = labels.get(recommendation_type, f"Performance anomaly — {player_name}")
        summary = f"{action} (confidence: {conf_pct}%). "

        top_pos = [c for c in top_contributions[:3] if c.shap_value > 0]
        if top_pos:
            parts = [f"{c.human_label} ({c.formatted_value})" for c in top_pos]
            summary += "Primary factors: " + "; ".join(parts) + ". "

        if workload_status == "high_risk":
            summary += "Acute workload significantly exceeds chronic baseline — elevated injury risk. "
        elif workload_status == "low_readiness":
            summary += "Recent load is below chronic baseline — reduced physical readiness. "

        summary += "Analysis is based on this player's own historical data, not squad averages."
        return summary.strip()


# ─────────────────────────────────────────────
# Per-player background cache
# ─────────────────────────────────────────────
class _ExplainerCache:
    def __init__(self):
        self._backgrounds: Dict[int, np.ndarray] = {}

    def register(self, player_id: int, data: np.ndarray, n_bg: int = 50) -> None:
        self._backgrounds[player_id] = build_kmeans_background(data, k=n_bg)
        logger.info(
            "SHAP background registered for player %d (%s, %d samples)",
            player_id,
            "KernelExplainer" if SHAP_AVAILABLE else "magnitude_proxy",
            len(self._backgrounds[player_id]),
        )

    def get(self, player_id: int) -> Optional[np.ndarray]:
        return self._backgrounds.get(player_id)


# ─────────────────────────────────────────────
# XAI Layer
# ─────────────────────────────────────────────
class XAILayer:
    """
    Top-level XAI orchestrator.
    Takes an AnomalyResult + player model -> SHAPExplanation.
    Works with or without the `shap` library.
    """

    def __init__(self):
        self.cfg: SHAPConfig = CONFIG.shap
        self._cache = _ExplainerCache()
        self._cf_gen = CounterfactualGenerator()
        self._nlg = TemplateNLGEngine()

    def register_explainer(self, model, background_data: np.ndarray) -> None:
        """Register background data for a player. Call once after model training."""
        self._cache.register(model.player_id, background_data, n_bg=self.cfg.n_background_samples)

    def explain(self, result, model, player_name: str) -> SHAPExplanation:
        """Produce a SHAPExplanation for one AnomalyResult."""
        fv_array = np.array(
            [result.feature_vector.get(n, 0.0) for n in FEATURE_NAMES],
            dtype=np.float32,
        )

        background = self._cache.get(model.player_id)

        def predict_fn(X: np.ndarray) -> np.ndarray:
            if not model.is_trained:
                return np.zeros(len(X))
            scores = model.model.decision_function(model.scaler.transform(X))
            return np.clip(-scores + 0.5, 0.0, 1.0)

        shap_array, base_value = compute_shap_values(
            predict_fn=predict_fn,
            feature_vector=fv_array,
            background_data=background,
            n_background=self.cfg.n_background_samples,
        )

        shap_dict = {n: float(shap_array[i]) for i, n in enumerate(FEATURE_NAMES)}
        base_value = base_value or 0.0

        contributions = sorted(
            [
                FeatureContribution(
                    feature_name=n,
                    feature_value=result.feature_vector.get(n, 0.0),
                    shap_value=v,
                    direction="positive" if v >= 0 else "negative",
                    human_label=FEATURE_LABELS.get(n, n),
                    formatted_value=_format_value(n, result.feature_vector.get(n, 0.0)),
                )
                for n, v in shap_dict.items()
            ],
            key=lambda c: abs(c.shap_value),
            reverse=True,
        )

        rec_type = result.recommendation_type or "anomaly_flag"
        counterfactual = self._cf_gen.generate(shap_dict, result.feature_vector)
        nlg_summary = self._nlg.generate(
            recommendation_type=rec_type,
            confidence=result.confidence,
            player_name=player_name,
            top_contributions=contributions[:self.cfg.max_display_features],
            workload_status=result.workload_status,
        )
        waterfall = self._build_waterfall(shap_dict, base_value, result.confidence)

        return SHAPExplanation(
            player_id=result.player_id,
            external_id=result.external_id,
            recommendation_type=rec_type,
            confidence=result.confidence,
            computed_at=datetime.now(tz=timezone.utc),
            base_value=base_value,
            shap_values=shap_dict,
            feature_values=result.feature_vector,
            top_contributions=contributions[:self.cfg.max_display_features],
            nlg_summary=nlg_summary,
            counterfactual=counterfactual,
            waterfall_data=waterfall,
        )

    def _build_waterfall(
        self, shap_dict: Dict[str, float], base_value: float, final_value: float
    ) -> List[dict]:
        top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top = top[:self.cfg.max_display_features]
        wf = [{"name": "Base value", "value": base_value, "cumulative": base_value}]
        cum = base_value
        for name, sv in top:
            cum += sv
            wf.append({
                "name":       FEATURE_LABELS.get(name, name),
                "value":      sv,
                "cumulative": round(cum, 4),
                "direction":  "positive" if sv >= 0 else "negative",
            })
        wf.append({"name": "Model output", "value": final_value, "cumulative": round(final_value, 4)})
        return wf
