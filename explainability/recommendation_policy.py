"""
Players Data — IBM CIC Germany
Recommendation Policy Layer

Sits between SemanticInterpreter findings and the final recommendation_type
emitted on each alert.

Architecture position:
    SemanticInterpreter  → List[SemanticFinding]
    MatchState / CompressedTemporalContext → episodic signals
    ↓
    RecommendationPolicyEngine      
    ↓
    recommendation_type  (str)        → XAILayer / alert payload

Design contract
───────────────
• Pure symbolic decision logic — no LLM, no ML models, no I/O.
• Reads SemanticFinding objects + CompressedTemporalContext envelope.
• Returns exactly one recommendation_type string.
• Deterministic: same inputs → same output every call.
• The LLM narrates; this engine decides.  These responsibilities never merge.

Recommendation vocabulary
──────────────────────────
  substitute            — immediate substitution warranted
  recovery_intervention — active recovery protocol required
  tactical_adjustment   — positional / tactical correction needed
  workload_restriction  — training/match load reduction required
  performance_monitor   — flag + monitor; no immediate action yet
  anomaly_flag          — generic fallback; no specific policy matched

Priority ordering (first matching rule wins):
  1. substitute
  2. recovery_intervention
  3. workload_restriction
  4. tactical_adjustment
  5. performance_monitor
  6. anomaly_flag  (default)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Policy thresholds — tune here, never in rule bodies
# ─────────────────────────────────────────────────────────────────────────────

# Substitution
_SUB_MIN_CROSS_MATCH_RECURRENCE = 2   # closed episodes in prior matches
_SUB_MIN_PERSISTENCE_WINDOWS    = 4   # consecutive windows in current episode
_SUB_ESCALATION_LEVELS          = {"high", "critical"}

# Recovery intervention
_REC_MIN_PERSISTENCE_WINDOWS    = 3
_REC_FINDING_TYPES              = {"recovery_degradation", "cardiovascular_overload",
                                   "elevated_cardiovascular_response"}
_REC_MIN_SEVERITY               = {"high", "critical"}

# Workload restriction
_WL_FINDING_TYPES               = {"fatigue_accumulation"}
_WL_ACWR_HIGH_RISK              = 1.30   # mirrors semantics_layer THRESHOLDS
_WL_MIN_PERSISTENCE_WINDOWS     = 2

# Tactical adjustment
_TAC_FINDING_TYPES              = {"tactical_instability"}

# Performance monitor (locomotor suppression still active but not yet critical)
_MON_FINDING_TYPES = {"locomotor_suppression", "locomotor_overload"}
_MON_TREND_WORSENING            = "worsening"


# ─────────────────────────────────────────────────────────────────────────────
# Input container — keeps the public API stable regardless of internal changes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyInput:
    """
    All signals the policy engine needs in one flat container.

    The caller (orchestrator / process_window_direct) is responsible for
    populating this from the objects it already has; the policy engine itself
    never imports MatchState, CompressedTemporalContext, or SemanticFinding
    directly, keeping coupling minimal.
    """

    # ── Episodic context (from CompressedTemporalContext.to_prompt_block / fields) ──
    cross_match_recurrence: int = 0      # closed episodes matching current pattern in prior matches
    in_match_recurrence:    int = 0      # recurring pattern count within current match
    persistence_windows:    int = 0      # consecutive windows the current episode has lasted
    current_escalation:     str = "low"  # "low" | "medium" | "high" | "critical"
    trend:                  str = "stable"  # "stable" | "worsening" | "recovering" | "transitioning"

    # ── Current-window semantic findings ─────────────────────────────────────
    # List of dicts, each matching SemanticFinding.to_dict() schema:
    # {"finding_type": str, "severity": str, "confidence": float, ...}
    semantic_findings: List[dict] = None  # type: ignore[assignment]

    # ── Feature values (raw, for ACWR workload gate) ──────────────────────────
    feature_values: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.semantic_findings is None:
            self.semantic_findings = []
        if self.feature_values is None:
            self.feature_values = {}


# ─────────────────────────────────────────────────────────────────────────────
# Policy engine
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationPolicyEngine:
    """
    Converts symbolic episodic signals + current-window semantic findings
    into a single actionable recommendation_type string.

    Usage
    ─────
        engine = RecommendationPolicyEngine()
        rec_type = engine.determine(policy_input)

    The engine is stateless — instantiate once, call determine() per window.
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def determine(self, ctx: PolicyInput) -> str:
        """
        Evaluate rules in priority order and return the first matching type.
        Falls through to "anomaly_flag" if no rule fires.

        Parameters
        ----------
        ctx : PolicyInput
            All signals needed for policy decisions, pre-extracted by the caller.

        Returns
        -------
        str
            One of: substitute | recovery_intervention | workload_restriction |
                    tactical_adjustment | performance_monitor | anomaly_flag
        """
        finding_types = {f.get("finding_type", "") for f in ctx.semantic_findings}
        severities    = {f.get("severity", "low")   for f in ctx.semantic_findings}

        logger.warning(
            "POLICY DEBUG | finding_types=%s trend=%s persistence=%d "
            "cross_match=%d escalation=%s",
            finding_types,
            ctx.trend,
            ctx.persistence_windows,
            ctx.cross_match_recurrence,
            ctx.current_escalation,
        )

        # ── Rule 1: Substitute ────────────────────────────────────────────────
        # Recurrent cross-match pattern + sustained persistence + high/critical escalation.
        # This is the strongest signal: a pattern that has re-emerged across multiple
        # matches and is currently sustained and escalating.
        if (
            ctx.cross_match_recurrence >= _SUB_MIN_CROSS_MATCH_RECURRENCE
            and ctx.persistence_windows >= _SUB_MIN_PERSISTENCE_WINDOWS
            and ctx.current_escalation  in _SUB_ESCALATION_LEVELS
        ):
            logger.debug(
                "Policy: substitute | cross_match_recurrence=%d persistence=%d escalation=%s",
                ctx.cross_match_recurrence,
                ctx.persistence_windows,
                ctx.current_escalation,
            )
            return "substitute"

        # ── Rule 2: Recovery intervention ─────────────────────────────────────
        # Cardiovascular or recovery degradation finding, sustained, high/critical severity.
        # The player is not recovering adequately — medical/physio staff attention required.
        if (
            finding_types & _REC_FINDING_TYPES
            and severities & _REC_MIN_SEVERITY
            and ctx.persistence_windows >= _REC_MIN_PERSISTENCE_WINDOWS
        ):
            logger.debug(
                "Policy: recovery_intervention | findings=%s persistence=%d",
                finding_types & _REC_FINDING_TYPES,
                ctx.persistence_windows,
            )
            return "recovery_intervention"

        # ── Rule 3: Workload restriction ──────────────────────────────────────
        # Fatigue accumulation finding, OR ACWR in high-risk zone for ≥2 windows.
        # Training/match load must be reduced before the next session.
        acwr = float(ctx.feature_values.get("acwr", 0.0))
        if (
            finding_types & _WL_FINDING_TYPES
            and ctx.persistence_windows >= _WL_MIN_PERSISTENCE_WINDOWS
        ) or (
            acwr >= _WL_ACWR_HIGH_RISK
            and ctx.persistence_windows >= _WL_MIN_PERSISTENCE_WINDOWS
        ):
            logger.debug(
                "Policy: workload_restriction | findings=%s acwr=%.2f persistence=%d",
                finding_types & _WL_FINDING_TYPES,
                acwr,
                ctx.persistence_windows,
            )
            return "workload_restriction"

        # ── Rule 4: Tactical adjustment ───────────────────────────────────────
        # Tactical instability finding present, any severity.
        # No physiological risk signal — purely positional/tactical correction.
        if finding_types & _TAC_FINDING_TYPES:
            logger.debug("Policy: tactical_adjustment | findings=%s", finding_types & _TAC_FINDING_TYPES)
            return "tactical_adjustment"

        # ── Rule 5: Performance monitor ───────────────────────────────────────
        # Locomotor suppression active and worsening but not yet escalated.
        # Flag for the coaching staff to watch; no substitution warranted yet.
        if (
            finding_types & _MON_FINDING_TYPES
            and ctx.trend == _MON_TREND_WORSENING
        ):
            logger.debug("Policy: performance_monitor | findings=%s trend=%s", finding_types, ctx.trend)
            return "performance_monitor"

        # ── Default ───────────────────────────────────────────────────────────
        return "anomaly_flag"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build PolicyInput from the objects already present in the orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_policy_input(
    semantic_findings: List[dict],
    compressed_context,          # Optional[CompressedTemporalContext]
    feature_values: Optional[dict] = None,
) -> PolicyInput:
    """
    Convenience constructor that extracts the fields the policy engine needs
    from a CompressedTemporalContext object (may be None for the first window).

    Call this in the orchestrator immediately after build_semantic_findings()
    returns and before build_base_explanation() is called.

    Parameters
    ----------
    semantic_findings   : list of SemanticFinding.to_dict() dicts from the current window
    compressed_context  : CompressedTemporalContext (or None if not yet available)
    feature_values      : raw feature dict for ACWR gate (optional)

    Returns
    -------
    PolicyInput
    """
    ctx = PolicyInput(
        semantic_findings=semantic_findings or [],
        feature_values=feature_values or {},
    )

    if compressed_context is None:
        return ctx

    # Pull fields directly from the dataclass — no string parsing.
    ctx.cross_match_recurrence = getattr(compressed_context, "cross_match_recurrence", 0)
    ctx.persistence_windows    = getattr(compressed_context, "persistence_windows", 0)
    ctx.current_escalation     = getattr(compressed_context, "current_escalation", "low") or "low"

    # trend: CompressedTemporalContext.trend_summaries is a dict {axis: direction}.
    # Use a fixed priority ordering so the most actionable direction is always
    # selected, regardless of dict insertion order.
    # "worsening" > "recovering" > "increasing" > "declining" > "stable"
    _TREND_PRIORITY = ["worsening", "recovering", "increasing", "declining", "stable"]
    trend_summaries = getattr(compressed_context, "trend_summaries", {}) or {}
    _trend_vals = set(trend_summaries.values()) - {"insufficient data"}
    ctx.trend = next(
        (t for t in _TREND_PRIORITY if t in _trend_vals),
        "stable",
    )

    # in_match_recurrence: derived from recurring_patterns matching active_pattern.
    # Replicate the to_prompt_block() calculation without string parsing.
    active_pattern    = getattr(compressed_context, "active_pattern", None)
    recurring         = getattr(compressed_context, "recurring_patterns", []) or []
    ctx.in_match_recurrence = sum(
        1 for pat in recurring
        if active_pattern and active_pattern in pat
    )

    # cross_match_recurrence: computed inside to_prompt_block() from historical_episodes.
    # Re-derive here so the policy engine has a numeric value without touching the prompt.
    if ctx.cross_match_recurrence == 0 and active_pattern:
        target_readable = {active_pattern}
        target_raw      = {active_pattern.replace(" ", "_")}
        target_set      = target_readable | target_raw
        historical      = getattr(compressed_context, "historical_episodes", []) or []
        ctx.cross_match_recurrence = sum(
            1 for ep in historical
            if set(ep.get("dominant_findings", [])) & target_set
        )

    return ctx