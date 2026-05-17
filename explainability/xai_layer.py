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

Semantic Layer integration
────────────────────────────────
SHAP values are routed through SemanticInterpreter OUTSIDE this layer — in the
orchestrator — before reaching the LLM.  This file does not call
SemanticInterpreter.interpret() from any public path.  The interpreter converts
raw attributions into symbolic SemanticFinding objects in the orchestrator, which
injects them into BaseExplanation.semantic_findings before passing to
generate_explanation_from_base().  The LLM receives these findings and acts as
narrator/communicator only — physiological reasoning lives in the symbolic layer,
not in the prompt.

Architecture:
    build_base_explanation()          raw SHAP + attribution only
    ↓
    orchestrator: build_semantic_findings()   symbolic reasoning only
    ↓
    generate_explanation_from_base()  LLM narration only
    ↓
    SHAPExplanation.nlg_summary


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
from explainability.semantics_layer import (
    SemanticFinding,
    SemanticInterpreter,
    build_semantic_prompt_block,
)
from analysis.episodic_context import CompressedTemporalContext

logger = logging.getLogger(__name__)

_NLG_TIMEOUT_S = float(os.getenv("OLLAMA_NLG_TIMEOUT_S", "60.0"))

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

FEATURE_SEMANTICS = {
    "window_avg_speed_ms":
        "movement intensity",

    "window_distance_m":
        "ground coverage",

    "window_sprint_count":
        "explosive sprint activity",

    "heart_rate_bpm":
        "cardiovascular strain",

    "hr_recovery_time_s":
        "cardiovascular recovery dynamics",

    "positional_drift_score":
        "tactical positioning stability",

    "fatigue_decay_residual":
        "fatigue adaptation consistency",

    "acwr":
        "workload balance",
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
    uncertainty: float = 0.0
    shap_method: str = field(default_factory=lambda: "kernel" if SHAP_AVAILABLE else "magnitude_proxy")
    nlg_engine: str = "template"   # "llm_qwen" | "template"
    semantic_findings: List[dict] = field(default_factory=list)  # serialized SemanticFinding dicts

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
            "nlg_summary":         self.nlg_summary,
            "counterfactual":      self.counterfactual,
            "waterfall_data":      self.waterfall_data,
            "semantic_findings":   self.semantic_findings,
        }


@dataclass(frozen=True)
class BaseExplanation:
    """
    Immutable SHAP result from the realtime path.
    Contains everything except the NLG summary.
    Safe to pass across threads — holds no model references or tensors.
    """

    player_id:           int
    external_id:         str
    player_name:         str
    recommendation_type: str
    confidence:          float
    workload_status:     str
    computed_at:         datetime
    # SHAP core outputs
    base_value:          float
    shap_values:         Dict[str, float]
    feature_values:      Dict[str, float]
    # Structured explainability artifacts
    top_contributions:   Tuple[FeatureContribution, ...]
    counterfactual:      str
    waterfall_data:      Tuple[dict, ...]
    # Model uncertainty / anomaly metadata
    uncertainty:         float
    anomaly_score:       float
    shap_method:         str
    # Symbolic reasoning outputs
    # Filled AFTER build_semantic_findings() in orchestrator.
    semantic_findings:   Tuple[dict, ...] = ()
    compressed_context: Optional[CompressedTemporalContext] = None

# ─────────────────────────────────────────────
# Counterfactual generator
# ─────────────────────────────────────────────
class CounterfactualGenerator:

    def generate(self, shap_dict: Dict[str, float], feature_values: Dict[str, float]) -> str:
        if not shap_dict:
            return "Insufficient data for counterfactual generation."
        positive_items = {
            k: v for k, v in shap_dict.items()
            if v > 0
        }

        if not positive_items:
            return "No strong anomaly-driving feature identified."

        top = max(positive_items, key=lambda k: positive_items[k])
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
    Accepts semantic_findings for richer output when available.
    """

    def generate(
        self,
        recommendation_type: str,
        confidence: float,
        player_name: str,
        top_contributions: List[FeatureContribution],
        workload_status: str = "optimal",
        semantic_findings: Optional[List[SemanticFinding]] = None,
    ) -> str:
        conf_pct = int(confidence * 100)

        # If semantic findings are available, use them for a cleaner template summary
        if semantic_findings:
            labels = {
                "substitution":     f"Consider substituting {player_name}",
                "fatigue_alert":    f"Fatigue alert for {player_name}",
                "positional_drift": f"Positional drift detected for {player_name}",
                "workload_warning": f"Workload warning for {player_name}",
            }
            action = labels.get(recommendation_type, f"Performance anomaly — {player_name}")
            summary = f"{action} (confidence: {conf_pct}%). "

            # Lead with highest severity finding
            top_finding = semantic_findings[0]
            summary += top_finding.summary + " "

            if len(semantic_findings) > 1:
                others = ", ".join(
                    f.finding_type.replace("_", " ")
                    for f in semantic_findings[1:3]
                )
                summary += f"Additional signals: {others}. "

            if workload_status == "high_risk":
                summary += "Acute workload significantly exceeds chronic baseline — elevated injury risk. "
            elif workload_status == "low_readiness":
                summary += "Recent load is below chronic baseline — reduced physical readiness. "

            summary += "Analysis is based on this player's own historical data, not squad averages."
            return summary.strip()

        # Legacy path: raw SHAP contributions
        labels = {
            "substitution":     f"Consider substituting {player_name}",
            "fatigue_alert":    f"Fatigue alert for {player_name}",
            "positional_drift": f"Positional drift detected for {player_name}",
            "workload_warning": f"Workload warning for {player_name}",
        }
        action  = labels.get(recommendation_type, f"Performance anomaly — {player_name}")
        summary = f"{action} (confidence: {conf_pct}%). "

        # SHAP sign = contribution to anomaly score, not physiological direction.
        # Positive SHAP = raises anomaly likelihood; negative = lowers it.
        # Do NOT label these "stabilizing" or "risk factors" — that conflates
        # model-internal attribution with clinical interpretation.
        anomaly_increasing = [c for c in top_contributions[:5] if c.shap_value > 0]
        anomaly_decreasing = [c for c in top_contributions[:5] if c.shap_value < 0]

        if anomaly_increasing:
            parts = [f"{c.human_label} ({c.formatted_value})" for c in anomaly_increasing[:3]]
            summary += "Primary anomaly contributors: " + "; ".join(parts) + ". "

        if anomaly_decreasing:
            parts = [f"{c.human_label} ({c.formatted_value})" for c in anomaly_decreasing[:2]]
            summary += "Anomaly-dampening signals: " + "; ".join(parts) + ". "
        if workload_status == "high_risk":
            summary += "Acute workload significantly exceeds chronic baseline — elevated injury risk. "
        elif workload_status == "low_readiness":
            summary += "Recent load is below chronic baseline — reduced physical readiness. "

        summary += "Analysis is based on this player's own historical data, not squad averages."
        return summary.strip()


# ─────────────────────────────────────────────
# LLM NLG engine  (qwen2.5:14b via Ollama)
# ─────────────────────────────────────────────
_LLM_SYSTEM_PROMPT = """
You are a sports science communication engine.

Your role:
- Verbalize pre-computed semantic findings and session trajectory context for coaching staff.
- Do NOT perform physiological reasoning — the symbolic engine has already done this.
- Do NOT speculate about injuries, tactics, or causes not stated in the findings.
- Do NOT reference SHAP values, z-scores, anomaly scores, or internal metrics by name.
- Translate findings and trajectory patterns into clear, direct operational language.
- If a finding is marked CRITICAL or HIGH severity, lead with it.
- If the session context describes a motif (e.g. sprint-collapse pattern, repeated overload),
  communicate it as a confirmed behavioral pattern — not speculation.
- If a trend is described as worsening, communicate urgency without hedging.
- Maximum 3 sentences. Address the coaching staff directly.
"""

_LLM_PROMPT_TEMPLATE = """\
Player: {player_name}
Alert type: {recommendation_type}
Model confidence: {conf_pct}%
Workload status: {workload_status}

{semantic_block}
"""


def _build_context_block(match_context: str) -> str:
    """
    Wrap the semantic summary in a clearly labelled prompt section.
    The label tells the LLM this is pre-reasoned trajectory data,
    not raw telemetry it should interpret itself.
    """
    if not match_context:
        return ""
    return f"\n\nSession trajectory context (pre-computed by symbolic engine):\n{match_context}"


def format_match_state_prompt(state) -> str:
    """
    Convert a SemanticMatchState into a prose prompt block for the LLM.

    This is the ONLY place where structured symbolic state is converted to
    natural language. The LLM receives this as pre-reasoned trajectory context
    — it narrates, it does not re-reason.

    Parameters
    ----------
    state : SemanticMatchState
        Structured state returned by MatchState.build_semantic_state().

    Returns
    -------
    str
        Formatted prose block ready for LLM prompt injection.
        Returns empty string if state carries no meaningful content.
    """
    if state is None:
        return ""

    if not state.motifs and not state.trends and not state.persistent_findings:
        return ""

    lines = [f"Session trajectory — escalation level: {state.escalation_level.upper()}"]

    # ── Motif block ───────────────────────────────────────────────────────────
    if state.motifs:
        lines.append("  Behavioral patterns detected:")
        for m in state.motifs:
            conf_pct = int(m.get("confidence", 0.5) * 100)
            lines.append(f"    • {m['description']} (confidence: {conf_pct}%)")

    # ── Trend block ───────────────────────────────────────────────────────────
    trend_lines = []
    trend_labels = {
        "anomaly":  "Anomaly trajectory",
        "recovery": "Recovery efficiency",
        "workload": "Locomotor output",
    }
    for key, label in trend_labels.items():
        trend = state.trends.get(key, {})
        direction  = trend.get("direction", "")
        volatility = trend.get("volatility", "low")
        if not direction or direction in ("stable", "insufficient data"):
            continue
        vol_note = f", volatility: {volatility}" if volatility != "low" else ""
        trend_lines.append(f"    {label}: {direction}{vol_note}")

    if trend_lines:
        lines.append("  Progression analysis:")
        lines.extend(trend_lines)


    # ── Coupled physiological states ─────────────────────────────────────

    if state.coupled_states:
        lines.append("\nCoupled physiological states:")

        for cs in state.coupled_states:
            lines.append(
                f"  - {cs.state_type} "
                f"(severity={cs.severity}, "
                f"confidence={cs.confidence:.2f})"
            )

            if cs.supporting_trends:
                lines.append(
                    f"    Supporting trends: {', '.join(cs.supporting_trends)}"
                )

            lines.append(
                f"    {cs.description}"
            )

    # ── Persistent high/critical findings ─────────────────────────────────────
    if state.persistent_findings:
        lines.append("  Persistent high-severity findings:")
        seen: set = set()
        for f in state.persistent_findings[-5:]:  # cap at 5 most recent
            ftype = f.get("type", "unknown")
            if ftype in seen:
                continue
            seen.add(ftype)
            minute = f.get("minute")
            min_note = f" (min ~{minute})" if minute is not None else ""
            lines.append(
                f"    • {ftype.replace('_', ' ')} [{f.get('severity', 'high')}]{min_note}"
            )

    return "\n".join(lines)


def format_episodic_context(compressed_context) -> str:
    """
    Convert a CompressedTemporalContext into a prompt block for the LLM.
 
    This supplements format_match_state_prompt() when episodic context is
    available. Both can coexist in the same prompt:
 
        semantic_block   = build_semantic_prompt_block(semantic_findings)
        state_block      = format_match_state_prompt(semantic_state)
        episodic_block   = format_episodic_context(compressed_context)
 
        full_prompt = semantic_block + "\\n\\n" + state_block + "\\n\\n" + episodic_block
 
    The episodic block communicates match evolution — what happened earlier,
    how it evolved, and which prior episodes are relevant to now. The LLM
    receives this as pre-computed symbolic context; it does NOT re-reason.
 
    Returns empty string if context is None or empty.
    """
    if compressed_context is None or compressed_context.is_empty():
        return ""
    try:
        block = compressed_context.to_prompt_block()

        logger.warning(
        "TO_PROMPT_BLOCK OUTPUT:\n%s",
        compressed_context.to_prompt_block(),
    )
        

        if not block:
            return ""
        return (
            "\nMatch history context "
            "(pre-computed by symbolic engine — narrate only, do not re-reason):\n"
            + block
        )
    except Exception:
        return ""
 
 
_NLG_CIRCUIT_BREAKER_S = float(os.getenv("NLG_CIRCUIT_BREAKER_SECONDS", "30"))


class LLMNLGEngine:
    """
    NLG engine backed by qwen2.5:14b running on local Ollama.

    Thread-safe.  Falls back gracefully to TemplateNLGEngine on timeout or
    connection failure so the 200 ms serve SLA is never violated.

    When semantic_findings are provided, the prompt is built from symbolic
    findings (SemanticFinding objects) rather than raw SHAP feature lines.
    The LLM acts as narrator/communicator only — reasoning lives in semantic_layer.py.

    Circuit breaker
    ───────────────
    After a timeout the engine enters degraded mode for NLG_CIRCUIT_BREAKER_SECONDS
    (default 30 s).  During that window all NLG calls are served immediately by
    TemplateNLGEngine without touching Ollama.  This prevents cancellation storms:
    the pattern where every timed-out request triggers a new Ollama runner startup,
    partial graph allocation, cancellation, and teardown.

    Model warm state
    ────────────────
    Call warmup() once during startup (before the serve loop) to force Ollama to
    fully load the model into memory.  Without this, the first N requests race
    model loading against the NLG timeout and are reliably cancelled (HTTP 499 /
    context canceled), which is operationally different from Ollama being unavailable.
    """

    def __init__(
        self,
        timeout_s: float = _NLG_TIMEOUT_S,
        model: str = "qwen2.5:14b",
        max_tokens: int = 150,
        circuit_breaker_s: float = _NLG_CIRCUIT_BREAKER_S,
    ) -> None:
        self._timeout_s          = timeout_s
        self._model              = model
        self._max_tokens         = max_tokens
        self._circuit_breaker_s  = circuit_breaker_s
        self._fallback           = TemplateNLGEngine()
        self._client             = None          # lazy: import inside generate to avoid startup crash
        self._client_lock        = threading.Lock()
        self._available: Optional[bool] = None   # None = not yet probed
        self._last_probe_ts: float      = 0.0

        # Circuit breaker state
        self._circuit_open: bool  = False        # True = degraded mode, skip Ollama
        self._circuit_open_until: float = 0.0   # monotonic timestamp

    # ── Lazy client init ──────────────────────────────────────────────────────
    _REPROBE_INTERVAL_S = 30.0  # re-check Ollama every 30 s after a failure

    def _get_client(self):
        """
        Lazy singleton client getter.

        Availability is checked ONCE only.
        Realtime inference must never health-check Ollama repeatedly.
        """
        with self._client_lock:

            # Build client once
            if self._client is None:
                try:
                    self._client = OllamaClient(
                        default_model=self._model,
                        timeout_s=self._timeout_s,
                        max_retries=0,  # Realtime systems should not retry
                        cache=True,
                    )

                except Exception as exc:
                    logger.warning(
                        "LLMNLGEngine init failed: %s — using template NLG",
                        exc,
                    )
                    self._available = False
                    return None, False

            # Probe ONCE only
            if self._available is None:

                self._available = self._client.is_available(self._model)

                if not self._available:
                    logger.warning(
                        "LLMNLGEngine: Ollama unreachable or model '%s' not found. "
                        "Using template fallback. "
                        "Hint: run `ollama pull %s` and ensure Ollama is running.",
                        self._model, self._model,
                    )

                else:
                    logger.info(
                        "LLMNLGEngine: Ollama available, model '%s' registered.",
                        self._model,
                    )

            return self._client, self._available

    def warmup(self) -> bool:
        """
        Pre-load the model into Ollama memory before the serve loop starts.

        Delegates to OllamaClient.warmup().  Call once synchronously during
        startup.  If warmup fails the engine continues normally with template
        fallback — it is never a hard error.
        """
        client, available = self._get_client()
        if not available or client is None:
            return False
        return client.warmup(model=self._model, timeout_s=30.0)

    # ── Circuit breaker helpers ───────────────────────────────────────────────

    def _circuit_is_open(self) -> bool:
        if not self._circuit_open:
            return False
        if time.monotonic() >= self._circuit_open_until:
            self._circuit_open = False
            logger.info(
                "LLMNLGEngine: circuit breaker reset — resuming Ollama NLG calls"
            )
            return False
        return True

    def _trip_circuit(self) -> None:
        self._circuit_open       = True
        self._circuit_open_until = time.monotonic() + self._circuit_breaker_s
        logger.warning(
            "LLMNLGEngine: circuit breaker tripped — "
            "switching to template NLG for %.0f s to prevent cancellation storm",
            self._circuit_breaker_s,
        )

        # ── Main generate ─────────────────────────────────────────────────────────

    def generate(
        self,
        recommendation_type: str,
        confidence: float,
        player_name: str,
        top_contributions,           # List[FeatureContribution]
        workload_status: str = "optimal",
        match_context: str = "",
        semantic_findings: Optional[List[SemanticFinding]] = None,
        compressed_context: Optional[CompressedTemporalContext] = None
    ):
            
            """
            Returns (summary_text, engine_name, uncertainty).

            If semantic_findings is provided (non-empty list), the LLM prompt is
            built from symbolic findings via build_semantic_prompt_block().
            Otherwise falls back to the legacy raw-SHAP feature_lines format for
            backward compatibility.

            NLG is always opportunistic: if Ollama is unavailable, the circuit
            breaker is open, or the call exceeds the SLA timeout, TemplateNLGEngine
            is used transparently.  The calling inference path is never blocked.
            """
            def _template_response(reason: str) -> Tuple[str, str, float]:
                logger.debug("LLMNLGEngine: using template (%s)", reason)
                return (
                    self._fallback.generate(
                        recommendation_type, confidence, player_name,
                        top_contributions, workload_status, semantic_findings,
                    ),
                    "template",
                    0.0,
                )

            # Fast path: circuit open or Ollama known unavailable
            if self._circuit_is_open():
                return _template_response("circuit_open")

            # Hard gate: no symbolic findings means telemetry is degraded.
            # The LLM must NOT attempt to narrate degraded-window raw SHAP —
            # it produces directionally wrong clinical inferences.
            # Template engine handles this case correctly and sub-millisecond.
            if not semantic_findings:
                return _template_response("telemetry_degraded")

            client, available = self._get_client()

            if not available or client is None:
                return _template_response("ollama_unavailable")

            # ── Build prompt block ────────────────────────────────────────────
            # semantic_findings is guaranteed non-empty here: the degraded-telemetry
            # gate above returns to template before reaching this point.
            prompt_parts: list[str] = [build_semantic_prompt_block(semantic_findings)]

            semantic_block = "\n\n".join(prompt_parts)

            context_block = _build_context_block(match_context)

            # ── Episodic context block ────────────────────────────────────────────────
            episodic_block = ""
            if compressed_context and not compressed_context.is_empty():
                try:
                    episodic_block = format_episodic_context(compressed_context)
                except Exception as e:
                    logger.exception(
                        "Failed to format episodic context: %s",
                        e,
                    )
                    episodic_block = ""
            logger.warning(
                    "EPISODIC BLOCK:\n%s",
                    episodic_block,
                )
            prompt = (
                _LLM_PROMPT_TEMPLATE.format(
                    player_name=player_name,
                    recommendation_type=recommendation_type,
                    conf_pct=int(confidence * 100),
                    workload_status=workload_status,
                    semantic_block=semantic_block,
                )
                + context_block
                + episodic_block
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
                logger.warning(
                    "LLM PROMPT:\n%s",
                    prompt,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000
                text = resp.text.strip()
                if not text:
                    raise ValueError("Empty LLM response")

                logger.debug(
                    "LLMNLGEngine: player=%s  engine=qwen2.5:14b  %.0f ms  tokens=%d",
                    player_name, elapsed_ms, resp.eval_count,
                )
                # Using prompt evaluation count as a proxy for uncertainty/complexity
                uncertainty = float(resp.prompt_eval_count) / 1000.0
                return text, "llm_qwen", uncertainty

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                # Distinguish timeout (SLA tradeoff) from genuine unavailability.
                # Both result in template fallback, but they signal different things
                # operationally: timeout means the model IS running but we chose not
                # to wait; unavailable means the infrastructure is down.
                exc_str = str(exc)
                if "timeout" in exc_str.lower() or "timed out" in exc_str.lower():
                    logger.debug(
                        "LLMNLGEngine: NLG timeout after %.0f ms (SLA tradeoff, not an error) "
                        "— template fallback activated",
                        elapsed_ms,
                    )
                    self._trip_circuit()
                else:
                    logger.warning(
                        "LLMNLGEngine: NLG unavailable (%.0f ms): %s — template fallback",
                        elapsed_ms, exc,
                    )
                return _template_response(f"exception_{type(exc).__name__}")


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

    NLG is always asynchronous / opportunistic with respect to inference:
    the inference path (model inference → attribution → alert decision) never
    blocks on NLG generation.  If Ollama is slow or the circuit breaker is open,
    TemplateNLGEngine is used sub-millisecond.

    Startup: call warmup_nlg() once before the serve loop to pre-load the model
    into Ollama memory.  Without this, the first N requests race model loading
    against the NLG timeout and are reliably cancelled (HTTP 499 / context canceled).

    Semantic strategy (v2 — decoupled):
    Semantic interpretation is NOT performed inside this class on any direct
    explain path.  The orchestrator calls build_semantic_findings() after
    build_base_explanation() and injects findings into BaseExplanation before
    calling generate_explanation_from_base().  This cleanly separates:
        build_base_explanation()    → SHAP extraction only
        build_semantic_findings()   → symbolic reasoning only
        generate_explanation_from_base() → LLM narration only
    """

    def __init__(self, nlg_timeout_s: float = _NLG_TIMEOUT_S):
        self.cfg: SHAPConfig     = CONFIG.shap
        self._cache              = _ExplainerCache()
        self._cf_gen             = CounterfactualGenerator()
        self._llm_nlg            = LLMNLGEngine(timeout_s=nlg_timeout_s)
        self._template_nlg       = TemplateNLGEngine()
        self._semantic_interp    = SemanticInterpreter()

    def warmup_nlg(self) -> bool:
        """
        Pre-load the Ollama model before the serve loop starts.

        Call once synchronously during startup.  Returns True if the model
        loaded successfully.  Failure is non-fatal — template NLG continues.
        """
        return self._llm_nlg.warmup()

    def register_explainer(self, model, background_data: np.ndarray) -> None:
        """Register background data for a player. Call once after model training."""
        self._cache.register(model.player_id, background_data, n_bg=self.cfg.n_background_samples)

    def register_explainer_for_player(self, player_id: int, background_data: np.ndarray) -> None:
        """Register background data keyed directly by player_id."""
        self._cache.register(player_id, background_data, n_bg=self.cfg.n_background_samples)

    def build_semantic_findings(
        self,
        shap_dict: Dict[str, float],
        feature_values: Dict[str, float],
        persistence_windows: int = 0,
    ) -> List[SemanticFinding]:
        """
        Convert SHAP attributions into symbolic findings.

        Pure symbolic reasoning stage.
        No LLM.
        No formatting.
        """
        return self._semantic_interp.interpret(
            shap_values=shap_dict,
            feature_values=feature_values,
            persistence_windows=persistence_windows,
        )

    def generate_explanation_from_base(
        self,
        base: "BaseExplanation",
        match_state=None,
        compressed_context: Optional[CompressedTemporalContext] = None
    ) -> "SHAPExplanation":
        """
        Final narrative generation stage.
 
        Match state and compressed episodic context are both injected AFTER
        symbolic findings have updated longitudinal memory.
 
        compressed_context, when provided, is a CompressedTemporalContext built
        by MatchState.build_compressed_context(). It is appended to the LLM
        prompt as a pre-reasoned match history block. The LLM narrates it;
        it does not re-reason from it.
        """
        return self.generate_nlg(
            base=base,
            match_context=match_state,
            compressed_context=compressed_context
        )


    def build_base_explanation(
        self,
        player_id: int,
        external_id: str,
        player_name: str,
        model,
        feature_vector: dict,
        recommendation_type: str,
        confidence: float,
        workload_status: str,
        anomaly_score: float,
        sequence: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        sequence_background: Optional[np.ndarray] = None,
        persistence_windows: int = 0,
    ) -> "BaseExplanation":
        """
        Realtime path: SHAP extraction + attribution packaging + counterfactual + waterfall.
        No LLM call. No semantic interpretation.
        Returns immutable BaseExplanation (semantic_findings=[]) safe to enqueue to the NLG worker.
        Orchestrator injects findings via build_semantic_findings() before calling generate_explanation_from_base().
        Target: <100 ms.
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
            fv_array   = np.array(
                [feature_vector.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32
            )
            background = self._cache.get(player_id)

            def _proxy_predict_fn(X: np.ndarray) -> np.ndarray:
                fv_mag = float(np.linalg.norm(fv_array)) + 1e-8
                deltas = np.linalg.norm(X - fv_array, axis=1)
                return np.clip(anomaly_score * (1.0 + deltas / fv_mag), 0.0, 1.0)

            shap_array, base_value = compute_shap_values(
                predict_fn=_proxy_predict_fn,
                feature_vector=fv_array,
                background_data=background,
                n_background=self.cfg.n_background_samples,
            )
            shap_dict                  = {n: float(shap_array[i]) for i, n in enumerate(FEATURE_NAMES)}
            feature_values_for_display = feature_vector

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

        counterfactual = self._cf_gen.generate(shap_dict, feature_values_for_display)
        waterfall      = self._build_waterfall(shap_dict, base_value or 0.0, confidence)
        shap_method    = (
            "temporal_feature_ablation" if has_true_shap
            else ("kernel_proxy" if SHAP_AVAILABLE else "magnitude_proxy")
        )

        return BaseExplanation(
            player_id=player_id,
            external_id=external_id,
            player_name=player_name,
            recommendation_type=recommendation_type,
            confidence=confidence,
            workload_status=workload_status,
            computed_at=datetime.now(tz=timezone.utc),
            base_value=base_value or 0.0,
            shap_values=shap_dict,
            feature_values=feature_values_for_display,
            top_contributions=tuple(contributions[:self.cfg.max_display_features]),
            counterfactual=counterfactual,
            waterfall_data=tuple(waterfall),
            uncertainty=0.0,
            anomaly_score=anomaly_score,
            shap_method=shap_method,
        )

    def generate_nlg(
        self,
        base: "BaseExplanation",
        match_context=None,          # Optional[SemanticMatchState]
        compressed_context=None,     # Optional[CompressedTemporalContext]  ← NEW
    ) -> SHAPExplanation:
        """
        Async path: LLM narrative generation only.
        Receives immutable BaseExplanation, returns full SHAPExplanation.
        No SHAP recomputation. No model access. No forward passes.
        Deserializes semantic_findings from BaseExplanation for the NLG call.
        """
        from explainability.semantics_layer import SemanticFinding as _SF
 
        # Convert SemanticMatchState → prompt string. None → empty string.
        try:
            match_context_str = format_match_state_prompt(match_context) if match_context is not None else ""
        except Exception:
            match_context_str = ""
 
        # Reconstruct SemanticFinding objects from serialized dicts stored in BaseExplanation
        semantic_findings_objs = []
        for fd in base.semantic_findings:
            try:
                semantic_findings_objs.append(
                    _SF(
                        finding_type=fd["finding_type"],
                        severity=fd["severity"],
                        confidence=fd["confidence"],
                        summary=fd["summary"],
                        supporting_features=fd["supporting_features"],
                        evidence=fd["evidence"],
                        shap_evidence=fd.get("shap_evidence", {}),
                        persistence_windows=fd.get("persistence_windows", 0),
                        trend=fd.get("trend", "stable"),
                        domain=fd.get("domain", ""),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to deserialize SemanticFinding: %s", exc)
 
        nlg_summary, nlg_engine, uncertainty = self._llm_nlg.generate(
            recommendation_type=base.recommendation_type,
            confidence=base.confidence,
            player_name=base.player_name,
            top_contributions=list(base.top_contributions),
            workload_status=base.workload_status,
            match_context=match_context_str,
            semantic_findings=semantic_findings_objs or None,
            compressed_context=compressed_context,   # ← NEW
        )
 
        return SHAPExplanation(
            player_id=base.player_id,
            external_id=base.external_id,
            recommendation_type=base.recommendation_type,
            confidence=base.confidence,
            computed_at=base.computed_at,
            base_value=base.base_value,
            shap_values=base.shap_values,
            feature_values=base.feature_values,
            top_contributions=list(base.top_contributions),
            nlg_summary=nlg_summary,
            counterfactual=base.counterfactual,
            waterfall_data=list(base.waterfall_data),
            uncertainty=uncertainty,
            shap_method=base.shap_method,
            nlg_engine=nlg_engine,
            semantic_findings=list(base.semantic_findings),
        )

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
        match_context=None,   # Optional[SemanticMatchState]
        persistence_windows: int = 0,
    ) -> SHAPExplanation:
        """
        Produce a SHAPExplanation.

        SHAP path selection
        ───────────────────
        True SHAP (channel ablation): when sequence + mask + background are
        provided and model has reconstruction_loss_for_shap().
        Fallback: magnitude proxy in XAI-space.

        Semantic path
        ─────────────
        After SHAP values are computed, SemanticInterpreter runs and produces
        List[SemanticFinding]. These are forwarded to LLMNLGEngine so the LLM
        acts as narrator rather than physiological reasoner.

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

        # Semantic interpretation is NOT performed here.
        # Findings are injected externally by the orchestrator via build_semantic_findings().
        semantic_findings = []

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

        counterfactual = self._cf_gen.generate(shap_dict, feature_values_for_display)

        # ── NLG: pass semantic findings to LLM ───────────────────────────────
        # Convert SemanticMatchState → prompt string. None → empty string.
        try:
            match_context_str = format_match_state_prompt(match_context) if match_context is not None else ""
        except Exception:
            match_context_str = ""

        nlg_summary, nlg_engine, uncertainty = self._llm_nlg.generate(
            recommendation_type=recommendation_type,
            confidence=confidence,
            player_name=player_name,
            top_contributions=contributions[:self.cfg.max_display_features],
            workload_status=workload_status,
            match_context=match_context_str,
            semantic_findings=semantic_findings if semantic_findings else None,
        )

        waterfall = self._build_waterfall(shap_dict, base_value, confidence)

        shap_method = (
            "temporal_feature_ablation" if has_true_shap
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
            uncertainty=uncertainty,
            shap_method=shap_method,
            nlg_engine=nlg_engine,
            semantic_findings=[f.to_dict() for f in semantic_findings],
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
        bg_norm   = model.normaliser.transform(background)
        bg_mean   = bg_norm.mean(axis=0)   # (T, F)

        # ── Base loss (unperturbed) ────────────────────────────────────────────
        base_loss = float(model.reconstruction_loss_for_shap(
            player_id=player_id,
            sequences_norm=seq_norm[np.newaxis].astype(np.float32),
            mask=mask,
        )[0])

        # ── Batch all F perturbations into a SINGLE model call ────────────────
        # Old: F individual forward passes (17 round-trips total)
        # New: stack all perturbed sequences → (F, T, F_feat) → 1 batch call
        # Reduces SHAP latency by ~70 % on CPU.
        perturbed_batch = np.stack(
            [
                np.where(
                    np.eye(F, dtype=bool)[fi][np.newaxis, :],   # (1, F) bool mask
                    bg_mean,                                      # replace channel fi
                    seq_norm,                                     # keep all others
                )
                for fi in range(F)
            ],
            axis=0,
        ).astype(np.float32)   # (F, T, F_feat)

        ablated_losses = model.reconstruction_loss_for_shap(
            player_id=player_id,
            sequences_norm=perturbed_batch,
            mask=mask,
        )   # (F,) array

        shap_f = (ablated_losses - base_loss).astype(np.float32)

        # ── Background (full-ablation) base value ─────────────────────────────
        base_value = float(model.reconstruction_loss_for_shap(
            player_id=player_id,
            sequences_norm=bg_mean[np.newaxis].astype(np.float32),
            mask=mask,
        )[0])

        seq_shap: Dict[str, float] = {
            name: float(shap_f[i]) for i, name in enumerate(_SFN)
        }

        shap_dict: Dict[str, float] = {n: 0.0 for n in FEATURE_NAMES}

        _lstm_to_xai = {
            _SFN[0]: "window_avg_speed_ms",
            _SFN[2]: "heart_rate_bpm",
            _SFN[3]: "window_sprint_count",
            _SFN[6]: "window_distance_m",
            _SFN[7]: "hr_recovery_time_s",
        }

        for lstm_name, xai_name in _lstm_to_xai.items():
            if lstm_name in seq_shap and xai_name in shap_dict:
                shap_dict[xai_name] = seq_shap[lstm_name]

        x_shap = seq_shap.get("x_pitch", 0.0)
        y_shap = seq_shap.get("y_pitch", 0.0)

        # Default positional attribution from latent spatial channels
        positional_effect = float(
            np.sign(x_shap + y_shap) * np.sqrt(x_shap**2 + y_shap**2)
        )

        # If the actual alert came from drift logic,
        # inject the real tactical drift contribution.
        if extra_features.get("positional_drift_score", 0.0) > 0:
            positional_effect = float(
                extra_features.get("positional_drift_score", 0.0)
            )

        shap_dict["positional_drift_score"] = positional_effect

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
            "Batched attribution (player %d): 3 model calls (base + batch-F + bg), base_loss=%.4f",
            player_id, base_loss,
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

        # Semantic interpretation is NOT performed here.
        # Findings are injected externally by the orchestrator via build_semantic_findings().
        semantic_findings = []

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
        nlg_summary, nlg_engine, uncertainty = self._llm_nlg.generate(
            recommendation_type=rec_type,
            confidence=result.confidence,
            player_name=player_name,
            top_contributions=contributions[:self.cfg.max_display_features],
            workload_status=result.workload_status,
            semantic_findings=semantic_findings if semantic_findings else None,
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
            uncertainty=uncertainty,
            nlg_engine=nlg_engine,
            semantic_findings=[f.to_dict() for f in semantic_findings],
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