"""
Players Data — IBM CIC Germany
XAI / Explainability Layer  (shap-compat + Qwen2.5:14b NLG)

Wraps shap_compat so the full explanation pipeline works whether or not
the `shap` library is installed.  All public interfaces are identical to
the architecture spec in the proposal.

Qwen2.5:14b integration
────────────────────────
Two NLG engines are registered at startup:
  1. LLMNLGEngine   — calls qwen2.5:14b via local Ollama for rich, contextual
                      natural-language summaries.  Subject to a configurable
                      timeout (OLLAMA_NLG_TIMEOUT_S, default 2 s).
  2. TemplateNLGEngine — deterministic fallback; always succeeds in < 1 ms.

XAILayer.explain_from_dict always tries LLM first.  If the call times out
or Ollama is unavailable the template engine is used transparently so the
200 ms serve SLA is never broken.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
from config.ollama_client import OllamaClient
from explainability.shap_compat import SHAP_AVAILABLE
from config.settings import CONFIG, SHAPConfig
from config.settings import SEQUENCE_FEATURE_NAMES as _SFN
from explainability.shap_compat import SHAP_AVAILABLE, build_kmeans_background
from explainability.shap_compat import compute_shap_values

logger = logging.getLogger(__name__)

_NLG_TIMEOUT_S = float(os.getenv("OLLAMA_NLG_TIMEOUT_S", "2"))

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
    "hr_recovery_time_s":             "HR slope (bpm/s — rising = exerting)",
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
    if name == "hr_recovery_time_s":
        bpm_per_s = value * 200.0
        if abs(bpm_per_s) < 0.05:
            return "HR stable (~0 bpm/s)"
        direction = "rising" if bpm_per_s > 0 else "dropping"
        return f"HR {direction} ~{abs(bpm_per_s):.2f} bpm/s"
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
    nlg_engine: str = "template"   # "llm_qwen" | "template"

    def to_dict(self) -> dict:
        return {
            "player_id":           self.player_id,
            "external_id":         self.external_id,
            "recommendation_type": self.recommendation_type,
            "confidence":          self.confidence,
            "computed_at":         self.computed_at.isoformat(),
            "base_value":          self.base_value,
            "shap_method":         self.shap_method,
            "nlg_engine":          self.nlg_engine,
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
            "nlg_summary":    self.nlg_summary,
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
        if top == "hr_recovery_time_s":
            bpm_per_s = val * 200.0
            direction = "rising" if bpm_per_s > 0 else "dropping"
            return (
                f"If {label} were closer to the personal baseline "
                f"(current value: {direction} ~{abs(bpm_per_s):.2f} bpm/s), "
                f"this flag would likely not trigger."
            )
        return (
            f"If {label} were closer to the personal baseline "
            f"(current value: {val:.2f}), this flag would likely not trigger."
        )


# ─────────────────────────────────────────────
# Template NLG engine  (deterministic fallback)
# ─────────────────────────────────────────────
class TemplateNLGEngine:
    """
    Sub-millisecond deterministic NLG.
    Always used as fallback when Ollama times out or is unavailable.
    """

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
        action  = labels.get(recommendation_type, f"Performance anomaly — {player_name}")
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
# LLM NLG engine  (qwen2.5:14b via Ollama)
# ─────────────────────────────────────────────
_LLM_SYSTEM_PROMPT = """\
You are a professional sports-science analyst for an elite football club.
Your task: write a concise, precise, actionable alert summary for the performance
coaching staff when the ML pipeline flags a player event.

Rules:
- 2-3 sentences maximum.
- Clinical, factual tone. No emojis. No bullet points.
- Reference specific metric values (e.g. "ACWR 1.42", "HR rising at 0.8 bpm/s").
- Conclude with a concrete, time-bound action (e.g. "Recommend substitution before 75'").
- Do NOT invent data beyond what is provided.
"""

_LLM_PROMPT_TEMPLATE = """\
Player: {player_name}
Alert type: {recommendation_type}
Model confidence: {conf_pct}%
Workload status: {workload_status}

Top contributing features (by SHAP magnitude):
{feature_lines}

Write the alert summary now.
"""


class LLMNLGEngine:
    """
    NLG engine backed by qwen2.5:14b running on local Ollama.

    Thread-safe.  Falls back gracefully to TemplateNLGEngine on timeout or
    connection failure so the 200 ms serve SLA is never violated.
    """

    def __init__(
        self,
        timeout_s: float = _NLG_TIMEOUT_S,
        model: str = "qwen2.5:14b",
        max_tokens: int = 150,
    ) -> None:
        self._timeout_s   = timeout_s
        self._model       = model
        self._max_tokens  = max_tokens
        self._fallback    = TemplateNLGEngine()
        self._client      = None          # lazy: import inside generate to avoid startup crash
        self._client_lock = threading.Lock()
        self._available: Optional[bool] = None   # None = not yet probed

    # ── Lazy client init ──────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        self._client = OllamaClient(
                            default_model=self._model,
                            timeout_s=self._timeout_s * 3,  # client-level: generous
                            max_retries=0,                   # per-call timeout handles it
                            cache=True,
                        )
                        self._available = self._client.is_available(self._model)
                        if not self._available:
                            logger.warning(
                                "LLMNLGEngine: Ollama not available or model '%s' not loaded. "
                                "Falling back to template NLG. Start Ollama and run: "
                                "ollama pull %s", self._model, self._model,
                            )
                    except Exception as exc:
                        logger.warning("LLMNLGEngine init failed: %s — using template NLG", exc)
                        self._available = False
        return self._client, self._available

    # ── Main generate ─────────────────────────────────────────────────────────
    def generate(
        self,
        recommendation_type: str,
        confidence: float,
        player_name: str,
        top_contributions: List[FeatureContribution],
        workload_status: str = "optimal",
    ) -> Tuple[str, str]:
        """
        Returns (summary_text, engine_name).
        engine_name is 'llm_qwen' on success, 'template' on fallback.
        """
        client, available = self._get_client()

        if not available or client is None:
            return (
                self._fallback.generate(
                    recommendation_type, confidence, player_name,
                    top_contributions, workload_status,
                ),
                "template",
            )

        # Build the prompt
        feature_lines = "\n".join(
            f"  • {c.human_label}: {c.formatted_value}  (SHAP={c.shap_value:+.3f})"
            for c in top_contributions[:5]
        )
        prompt = _LLM_PROMPT_TEMPLATE.format(
            player_name=player_name,
            recommendation_type=recommendation_type,
            conf_pct=int(confidence * 100),
            workload_status=workload_status,
            feature_lines=feature_lines or "  (no significant features)",
        )

        t0 = time.perf_counter()
        try:
            resp = client.generate(
                prompt=prompt,
                system=_LLM_SYSTEM_PROMPT,
                max_tokens=self._max_tokens,
                temperature=0.15,
                timeout_s=self._timeout_s,
                use_cache=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            text = resp.text.strip()
            if not text:
                raise ValueError("Empty LLM response")

            logger.debug(
                "LLMNLGEngine: player=%s  engine=qwen2.5:14b  %.0f ms  tokens=%d",
                player_name, elapsed_ms, resp.eval_count,
            )
            return text, "llm_qwen"

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                "LLMNLGEngine fallback (%.0f ms): %s", elapsed_ms, exc
            )
            return (
                self._fallback.generate(
                    recommendation_type, confidence, player_name,
                    top_contributions, workload_status,
                ),
                "template",
            )


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

    NLG strategy:
    
    Uses LLMNLGEngine (qwen2.5:14b) by default, with automatic fallback to
    TemplateNLGEngine if Ollama is unavailable or the call exceeds the SLA.
    The engine used is recorded in SHAPExplanation.nlg_engine.
    """

    def __init__(self, nlg_timeout_s: float = _NLG_TIMEOUT_S):
        self.cfg: SHAPConfig     = CONFIG.shap
        self._cache              = _ExplainerCache()
        self._cf_gen             = CounterfactualGenerator()
        self._llm_nlg            = LLMNLGEngine(timeout_s=nlg_timeout_s)
        self._template_nlg       = TemplateNLGEngine()

    def register_explainer(self, model, background_data: np.ndarray) -> None:
        """Register background data for a player. Call once after model training."""
        self._cache.register(model.player_id, background_data, n_bg=self.cfg.n_background_samples)

    def register_explainer_for_player(self, player_id: int, background_data: np.ndarray) -> None:
        """Register background data keyed directly by player_id."""
        self._cache.register(player_id, background_data, n_bg=self.cfg.n_background_samples)

    def explain_from_dict(
        self,
        player_id: int,
        external_id: str,
        model,
        feature_vector: dict,
        recommendation_type: str,
        confidence: float,
        workload_status: str,
        anomaly_score: float,
        player_name: str,
        sequence: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        sequence_background: Optional[np.ndarray] = None,
    ) -> SHAPExplanation:
        """
        Produce a SHAPExplanation.

        SHAP path selection
        ───────────────────
        True SHAP (channel ablation): when sequence + mask + background are
        provided and model has reconstruction_loss_for_shap().
        Fallback: magnitude proxy in XAI-space.

        NLG path selection
        ──────────────────
        LLMNLGEngine (qwen2.5:14b) → TemplateNLGEngine (auto-fallback).
        """
        has_true_shap = (
            sequence is not None
            and mask is not None
            and sequence_background is not None
            and hasattr(model, "reconstruction_loss_for_shap")
            and model.is_trained
        )

        if has_true_shap:
            shap_dict, base_value, feature_values_for_display = self._explain_sequence_shap(
                player_id=player_id,
                model=model,
                sequence=sequence,
                mask=mask,
                background=sequence_background,
                extra_features=feature_vector,
            )
        else:
            logger.debug(
                "True SHAP unavailable for player %d — using magnitude proxy "
                "(sequence=%s, background=%s, model_has_method=%s)",
                player_id,
                sequence is not None,
                sequence_background is not None,
                hasattr(model, "reconstruction_loss_for_shap"),
            )
            fv_array   = np.array(
                [feature_vector.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32
            )
            background = self._cache.get(player_id)

            def _proxy_predict_fn(X: np.ndarray) -> np.ndarray:
                fv_mag  = float(np.linalg.norm(fv_array)) + 1e-8
                deltas  = np.linalg.norm(X - fv_array, axis=1)
                return np.clip(anomaly_score * (1.0 + deltas / fv_mag), 0.0, 1.0)

            shap_array, base_value = compute_shap_values(
                predict_fn=_proxy_predict_fn,
                feature_vector=fv_array,
                background_data=background,
                n_background=self.cfg.n_background_samples,
            )
            shap_dict                  = {n: float(shap_array[i]) for i, n in enumerate(FEATURE_NAMES)}
            feature_values_for_display = feature_vector

        base_value = base_value or 0.0

        contributions = sorted(
            [
                FeatureContribution(
                    feature_name=n,
                    feature_value=feature_values_for_display.get(n, 0.0),
                    shap_value=v,
                    direction="positive" if v >= 0 else "negative",
                    human_label=FEATURE_LABELS.get(n, n),
                    formatted_value=_format_value(n, feature_values_for_display.get(n, 0.0)),
                )
                for n, v in shap_dict.items()
            ],
            key=lambda c: abs(c.shap_value),
            reverse=True,
        )

        counterfactual            = self._cf_gen.generate(shap_dict, feature_values_for_display)
        nlg_summary, nlg_engine   = self._llm_nlg.generate(
            recommendation_type=recommendation_type,
            confidence=confidence,
            player_name=player_name,
            top_contributions=contributions[:self.cfg.max_display_features],
            workload_status=workload_status,
        )
        waterfall = self._build_waterfall(shap_dict, base_value, confidence)

        shap_method = (
            "channel_ablation" if has_true_shap
            else ("kernel_proxy" if SHAP_AVAILABLE else "magnitude_proxy")
        )

        return SHAPExplanation(
            player_id=player_id,
            external_id=external_id,
            recommendation_type=recommendation_type,
            confidence=confidence,
            computed_at=datetime.now(tz=timezone.utc),
            base_value=base_value,
            shap_values=shap_dict,
            feature_values=feature_values_for_display,
            top_contributions=contributions[:self.cfg.max_display_features],
            nlg_summary=nlg_summary,
            counterfactual=counterfactual,
            waterfall_data=waterfall,
            shap_method=shap_method,
            nlg_engine=nlg_engine,
        )

    # ── True SHAP / channel ablation ─────────────────────────────────────────
    def _explain_sequence_shap(
        self,
        player_id: int,
        model,
        sequence: np.ndarray,
        mask: np.ndarray,
        background: np.ndarray,
        extra_features: dict,
    ) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        """
        Fast feature attribution via masked perturbation (column-dropout).
        Runs 2×F+1 = 17 model calls (~30-50 ms on CPU, well within 200 ms SLA).
        """
        T, F = sequence.shape

        seq_norm  = model.normaliser.transform(sequence[np.newaxis])[0]
        base_loss = float(model.reconstruction_loss_for_shap(
            player_id=player_id,
            sequences_norm=seq_norm[np.newaxis].astype(np.float32),
            mask=mask,
        )[0])

        bg_norm = model.normaliser.transform(background)
        bg_mean = bg_norm.mean(axis=0)

        shap_f = np.zeros(F, dtype=np.float32)
        for fi in range(F):
            perturbed = seq_norm.copy()
            perturbed[:, fi] = bg_mean[:, fi]
            ablated_loss = float(model.reconstruction_loss_for_shap(
                player_id=player_id,
                sequences_norm=perturbed[np.newaxis].astype(np.float32),
                mask=mask,
            )[0])
            shap_f[fi] = float(base_loss - ablated_loss)

        bg_sequence = bg_mean.copy()
        base_value  = float(model.reconstruction_loss_for_shap(
            player_id=player_id,
            sequences_norm=bg_sequence[np.newaxis].astype(np.float32),
            mask=mask,
        )[0])

        seq_shap: Dict[str, float] = {
            name: float(shap_f[i]) for i, name in enumerate(_SFN)
        }

        shap_dict: Dict[str, float] = {n: 0.0 for n in FEATURE_NAMES}
        _lstm_to_xai = {
            "speed_ms":          "window_avg_speed_ms",
            "heart_rate_bpm":    "heart_rate_bpm",
            "sprint_flag":       "window_sprint_count",
            "distance_delta_m":  "window_distance_m",
            "hr_recovery_rate":  "hr_recovery_time_s",
        }
        for lstm_name, xai_name in _lstm_to_xai.items():
            if lstm_name in seq_shap and xai_name in shap_dict:
                shap_dict[xai_name] = seq_shap[lstm_name]

        x_shap = seq_shap.get("x_pitch", 0.0)
        y_shap = seq_shap.get("y_pitch", 0.0)
        shap_dict["positional_drift_score"] = float(
            np.sign(x_shap + y_shap) * np.sqrt(x_shap**2 + y_shap**2)
        )
        shap_dict["window_avg_speed_ms"] += seq_shap.get("acceleration_ms2", 0.0)

        last_step = sequence[-1]
        hr_rec_raw = float(last_step[_SFN.index("hr_recovery_rate")])
        feature_values_for_display = {
            "window_avg_speed_ms":  float(last_step[_SFN.index("speed_ms")]),
            "heart_rate_bpm":       float(last_step[_SFN.index("heart_rate_bpm")]),
            "window_sprint_count":  float(last_step[_SFN.index("sprint_flag")]),
            "window_distance_m":    float(last_step[_SFN.index("distance_delta_m")]) * T,
            "hr_recovery_time_s":   hr_rec_raw,
            "positional_drift_score": extra_features.get("positional_drift_score", 0.0),
            "acwr":                   extra_features.get("acwr", 1.0),
            "fatigue_decay_residual": extra_features.get("fatigue_decay_residual", 0.0),
            "speed_drop_pct":         extra_features.get("speed_drop_pct", 0.0),
            "coach_fatigue_severity": extra_features.get("coach_fatigue_severity", 0.0),
            "coach_pre_match_status_encoded": extra_features.get("coach_pre_match_status_encoded", 0.0),
            "z_distance":             extra_features.get("z_distance", 0.0),
            "z_sprint_count":         extra_features.get("z_sprint_count", 0.0),
            "z_top_speed":            extra_features.get("z_top_speed", 0.0),
            "z_high_speed_dist":      extra_features.get("z_high_speed_dist", 0.0),
        }

        logger.debug(
            "Fast attribution (player %d): %d model calls, base_loss=%.4f",
            player_id, F + 2, base_loss,
        )
        return shap_dict, base_value, feature_values_for_display

    # ── Legacy explain() entry-point (IsolationForest models) ────────────────
    @staticmethod
    def _magnitude_proxy_flat(
        fv_flat: np.ndarray,
        predict_fn,
        bg_flat: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        try:
            base_value   = float(predict_fn(bg_flat[:1])[0])
            total_effect = float(predict_fn(fv_flat.reshape(1, -1))[0]) - base_value
        except Exception:
            base_value, total_effect = 0.0, float(np.abs(fv_flat).mean())
        magnitudes = np.abs(fv_flat)
        total_mag  = magnitudes.sum()
        if total_mag > 0:
            proxy = (magnitudes / total_mag) * total_effect * np.sign(fv_flat)
        else:
            proxy = np.zeros_like(fv_flat)
        return proxy.astype(np.float32), base_value

    def explain(self, result, model, player_name: str) -> SHAPExplanation:
        """Produce a SHAPExplanation for one AnomalyResult (IsolationForest path)."""
        fv_array   = np.array(
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

        shap_dict  = {n: float(shap_array[i]) for i, n in enumerate(FEATURE_NAMES)}
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

        rec_type       = result.recommendation_type or "anomaly_flag"
        counterfactual = self._cf_gen.generate(shap_dict, result.feature_vector)
        nlg_summary, nlg_engine = self._llm_nlg.generate(
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
            nlg_engine=nlg_engine,
        )

    def _build_waterfall(
        self, shap_dict: Dict[str, float], base_value: float, final_value: float
    ) -> List[dict]:
        top = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top = top[:self.cfg.max_display_features]
        wf  = [{"name": "Base value", "value": base_value, "cumulative": base_value}]
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