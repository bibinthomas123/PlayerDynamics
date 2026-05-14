"""
Players Data — IBM CIC Germany
Semantic Interpretation Layer  (v1)

Sits between SHAP attribution and LLM narrative generation.

Architecture position:
    Temporal SHAP attribution
    ↓
    SemanticInterpreter          ← this module
        ├─ cardiovascular reasoning
        ├─ locomotor reasoning
        ├─ workload / ACWR reasoning
        ├─ tactical reasoning
        └─ persistence reasoning
    ↓
    List[SemanticFinding]
    ↓
    LLM narrative generation  (xai_layer.py — LLMNLGEngine)

Design principles
─────────────────
• Thresholds and ontology are centralized here, never scattered in if-blocks elsewhere.
• The LLM receives SemanticFinding objects — not raw SHAP values.
  It narrates; it does not reason physiologically.
• Five finding types for v1. Extend via SEMANTIC_RULES (see below).
• All public interfaces are typed and dataclass-based so callers can serialize freely.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Semantic ontology — feature groupings into physiological / tactical domains
# ─────────────────────────────────────────────────────────────────────────────

SEMANTIC_DOMAINS: Dict[str, Dict] = {
    "cardiovascular_load": {
        "description": "Heart rate and cardiac recovery dynamics",
        "features": [
            "heart_rate_bpm",
            "hr_recovery_time_s",
        ],
    },
    "locomotor_load": {
        "description": "Sprint activity, distance covered, and movement speed",
        "features": [
            "window_distance_m",
            "window_avg_speed_ms",
            "window_sprint_count",
            "z_distance",
            "z_sprint_count",
            "z_top_speed",
            "z_high_speed_dist",
        ],
    },
    "workload_balance": {
        "description": "Acute-to-Chronic Workload Ratio and fatigue accumulation",
        "features": [
            "acwr",
            "fatigue_decay_residual",
            "speed_drop_pct",
        ],
    },
    "tactical_positioning": {
        "description": "Spatial displacement from assigned tactical zone",
        "features": [
            "positional_drift_score",
            "z_distance",       # deviation from baseline territory coverage
        ],
    },
    "coach_context": {
        "description": "Coach-annotated fatigue severity and pre-match status",
        "features": [
            "coach_fatigue_severity",
            "coach_pre_match_status_encoded",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#.    Centralized thresholds
#     All rule comparisons draw from here. Never use bare literals in rules.
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS: Dict[str, float] = {
    # Cardiovascular
    "hr_high":                175.0,   # bpm — elevated exertion zone
    "hr_critical":            185.0,   # bpm — near-maximal sustained effort
    "hr_recovery_negative":  -0.05,    # fractional — negative = dropping HR (recovery)
    "hr_recovery_flat":       0.02,    # fractional — near-zero = HR not recovering

    # Locomotor
    "z_score_high":           1.5,     # SD — individual deviation flagged
    "z_score_very_high":      2.5,     # SD — strong individual deviation
    "sprint_count_low":       1.0,     # sprints per window — possible suppression
    "speed_ms_low":           2.5,     # m/s — walking / very slow pace

    # Workload
    "acwr_high_risk":         1.30,    # ACWR — injury risk zone
    "acwr_low_readiness":     0.80,    # ACWR — underloaded / low readiness
    "fatigue_residual_high":  80.0,    # m — above personal decay curve
    "speed_drop_significant": 15.0,    # % — speed has dropped significantly vs session start

    # Tactical
    "drift_elevated":         1.2,     # × norm radius — above positional norm
    "drift_high":             1.8,     # × norm radius — clear zone violation

    # SHAP attribution relevance cutoff
    "shap_relevant":          0.05,    # |SHAP| threshold to count a feature as driving
    "shap_strong":            0.15,    # |SHAP| threshold for a strongly driving feature

    # Persistence
    "persistence_confirmed":  3,       # windows — sustained = confirmed finding
    "persistence_severe":     6,       # windows — worsening pattern
}


# ─────────────────────────────────────────────────────────────────────────────
#  SemanticFinding dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticFinding:
    """
    One symbolic finding produced by SemanticInterpreter.
    The LLM receives a list of these instead of raw SHAP values.
    It verbalizes the findings; physiological reasoning lives here.
    """

    finding_type: str
    """
    One of:
      cardiovascular_overload
      locomotor_overload
      recovery_degradation
      tactical_instability
      fatigue_accumulation
    """

    severity: str
    """'low' | 'moderate' | 'high' | 'critical'"""

    confidence: float
    """0.0–1.0  — derived from SHAP magnitudes and threshold margins"""

    summary: str
    """One-sentence human-readable summary. The LLM may rephrase but not contradict this."""

    supporting_features: List[str]
    """Feature names from FEATURE_NAMES that back this finding."""

    evidence: Dict[str, float]
    """Observed values for supporting features: {feature_name: observed_value}."""

    shap_evidence: Dict[str, float] = field(default_factory=dict)
    """SHAP attributions for supporting features (subset of shap_values dict)."""

    persistence_windows: int = 0
    """Number of consecutive windows this condition has been active."""

    trend: str = "stable"
    """'stable' | 'worsening' | 'improving'"""

    domain: str = ""
    """Semantic domain from SEMANTIC_DOMAINS this finding belongs to."""

    def to_dict(self) -> dict:
        return {
            "finding_type":       self.finding_type,
            "severity":           self.severity,
            "confidence":         round(self.confidence, 3),
            "summary":            self.summary,
            "supporting_features": self.supporting_features,
            "evidence":           {k: round(v, 4) for k, v in self.evidence.items()},
            "shap_evidence":      {k: round(v, 4) for k, v in self.shap_evidence.items()},
            "persistence_windows": self.persistence_windows,
            "trend":              self.trend,
            "domain":             self.domain,
        }

    def __str__(self) -> str:
        return (
            f"[{self.finding_type} | {self.severity} | conf={self.confidence:.2f} | "
            f"persist={self.persistence_windows}w | trend={self.trend}] {self.summary}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#   SemanticInterpreter — the rule engine
# ─────────────────────────────────────────────────────────────────────────────

class SemanticInterpreter:
    """
    Converts SHAP attribution dicts + observed feature values into symbolic findings.

    Usage
    ─────
    interpreter = SemanticInterpreter()
    findings = interpreter.interpret(
        shap_values=shap_dict,
        feature_values=fv_dict,
        persistence_windows=result.persistence_windows,
    )

    Returns up to five SemanticFinding objects (v1 finding types).
    Empty list = no meaningful semantic signal detected.
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None) -> None:
        self._t = {**THRESHOLDS, **(thresholds or {})}

    # ── Public interface ──────────────────────────────────────────────────────

    def interpret(
        self,
        shap_values: Dict[str, float],
        feature_values: Dict[str, float],
        persistence_windows: int = 0,
    ) -> List[SemanticFinding]:
        """
        Run all interpretation rules and return a deduplicated, severity-sorted
        list of SemanticFinding objects.

        Parameters
        ----------
        shap_values        : {feature_name: shap_float} from XAILayer
        feature_values     : {feature_name: observed_float} from _build_xai_feature_vector
        persistence_windows: how many consecutive windows this alert has been active
        """
        findings: List[SemanticFinding] = []

        fv = feature_values   # short alias for readability
        sv = shap_values

        # Run each rule — each appends at most one finding
        finding = self._rule_cardiovascular_overload(fv, sv, persistence_windows)
        if finding:
            findings.append(finding)

        finding = self._rule_locomotor_overload(fv, sv, persistence_windows)
        if finding:
            findings.append(finding)

        finding = self._rule_recovery_degradation(fv, sv, persistence_windows)
        if finding:
            findings.append(finding)

        finding = self._rule_tactical_instability(fv, sv, persistence_windows)
        if finding:
            findings.append(finding)

        finding = self._rule_fatigue_accumulation(fv, sv, persistence_windows)
        if finding:
            findings.append(finding)

        # Sort: critical → high → moderate → low; then by confidence descending
        _sev_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
        findings.sort(key=lambda f: (_sev_order.get(f.severity, 9), -f.confidence))

        if findings:
            logger.debug(
                "SemanticInterpreter: %d finding(s) — %s",
                len(findings),
                [f.finding_type for f in findings],
            )

        return findings

    # ── Helper utilities ──────────────────────────────────────────────────────

    def _shap_supports(self, feature: str, shap_values: Dict[str, float]) -> bool:
        """True if |SHAP| for feature exceeds the relevance cutoff."""
        return abs(shap_values.get(feature, 0.0)) >= self._t["shap_relevant"]

    def _shap_strongly_supports(self, feature: str, shap_values: Dict[str, float]) -> bool:
        return abs(shap_values.get(feature, 0.0)) >= self._t["shap_strong"]

    def _persistence_trend(self, persistence_windows: int) -> str:
        if persistence_windows >= self._t["persistence_severe"]:
            return "worsening"
        if persistence_windows >= self._t["persistence_confirmed"]:
            return "stable"
        return "stable"

    def _confidence_from_shap(
        self,
        features: List[str],
        shap_values: Dict[str, float],
        base: float = 0.70,
    ) -> float:
        """
        Estimate confidence by how strongly SHAP endorses the supporting features.
        Each strongly-attributed feature adds up to 0.1; capped at 0.98.
        """
        bonus = sum(
            min(abs(shap_values.get(f, 0.0)) / self._t["shap_strong"], 1.0) * 0.10
            for f in features
        )
        return min(base + bonus, 0.98)

    # ── Rule: cardiovascular overload ─────────────────────────────────────────

    def _rule_cardiovascular_overload(
        self,
        fv: Dict[str, float],
        sv: Dict[str, float],
        persistence_windows: int,
    ) -> Optional[SemanticFinding]:
        """
        Fires when: sustained high HR AND recovery dynamics are impaired.

        Requires:
          • heart_rate_bpm  > hr_high threshold
          • hr_recovery_time_s (fractional rate) > hr_recovery_flat
            (positive = HR still rising or not dropping — recovery impaired)
          • at least one of the two features has meaningful SHAP attribution
        """
        hr    = fv.get("heart_rate_bpm", 0.0)
        rec   = fv.get("hr_recovery_time_s", 0.0)

        hr_elevated = hr >= self._t["hr_high"]
        recovery_impaired = rec >= self._t["hr_recovery_flat"]  # HR not dropping

        if not (hr_elevated and recovery_impaired):
            return None
        if not (self._shap_supports("heart_rate_bpm", sv) or
                self._shap_supports("hr_recovery_time_s", sv)):
            return None

        # Severity escalation
        if hr >= self._t["hr_critical"] and persistence_windows >= self._t["persistence_confirmed"]:
            severity = "critical"
        elif hr >= self._t["hr_critical"]:
            severity = "high"
        else:
            severity = "moderate"

        confidence = self._confidence_from_shap(
            ["heart_rate_bpm", "hr_recovery_time_s"], sv, base=0.75
        )

        trend = self._persistence_trend(persistence_windows)
        if persistence_windows >= self._t["persistence_confirmed"] and rec > self._t["hr_recovery_flat"]:
            trend = "worsening"

        return SemanticFinding(
            finding_type="cardiovascular_overload",
            severity=severity,
            confidence=confidence,
            summary=(
                f"Sustained cardiovascular strain detected: HR {int(hr)} bpm with impaired "
                f"recovery dynamics (rate: {rec:+.3f})."
            ),
            supporting_features=["heart_rate_bpm", "hr_recovery_time_s"],
            evidence={"heart_rate_bpm": hr, "hr_recovery_time_s": rec},
            shap_evidence={
                k: sv.get(k, 0.0)
                for k in ["heart_rate_bpm", "hr_recovery_time_s"]
            },
            persistence_windows=persistence_windows,
            trend=trend,
            domain="cardiovascular_load",
        )

    # ── Rule: locomotor overload ───────────────────────────────────────────────

    def _rule_locomotor_overload(
        self,
        fv: Dict[str, float],
        sv: Dict[str, float],
        persistence_windows: int,
    ) -> Optional[SemanticFinding]:
        """
        Fires when locomotor output is significantly above the player's personal baseline.

        Requires ≥ 2 of:
          • z_distance > z_score_high
          • z_sprint_count > z_score_high
          • z_top_speed > z_score_high
          • z_high_speed_dist > z_score_high
        Plus SHAP attribution supporting at least one locomotor feature.
        """
        z_dist  = fv.get("z_distance", 0.0)
        z_spr   = fv.get("z_sprint_count", 0.0)
        z_spd   = fv.get("z_top_speed", 0.0)
        z_hsd   = fv.get("z_high_speed_dist", 0.0)

        threshold = self._t["z_score_high"]
        flagged = [
            ("z_distance",        z_dist >= threshold),
            ("z_sprint_count",    z_spr  >= threshold),
            ("z_top_speed",       z_spd  >= threshold),
            ("z_high_speed_dist", z_hsd  >= threshold),
        ]
        active_features = [name for name, triggered in flagged if triggered]

        if len(active_features) < 2:
            return None

        locomotor_features = [
            "window_distance_m", "window_avg_speed_ms", "window_sprint_count",
            "z_distance", "z_sprint_count", "z_top_speed", "z_high_speed_dist",
        ]
        if not any(self._shap_supports(f, sv) for f in locomotor_features):
            return None

        # Use maximum z-score to determine severity
        max_z = max(z_dist, z_spr, z_spd, z_hsd)
        if max_z >= self._t["z_score_very_high"]:
            severity = "high"
        else:
            severity = "moderate"

        confidence = self._confidence_from_shap(active_features, sv, base=0.72)

        supporting = active_features + [
            f for f in ["window_distance_m", "window_avg_speed_ms", "window_sprint_count"]
            if self._shap_supports(f, sv)
        ]
        supporting = list(dict.fromkeys(supporting))   # deduplicate, preserve order

        return SemanticFinding(
            finding_type="locomotor_overload",
            severity=severity,
            confidence=confidence,
            summary=(
                f"Locomotor output significantly exceeds personal baseline "
                f"({len(active_features)} of 4 deviation indicators active, "
                f"max z={max_z:.1f} SD)."
            ),
            supporting_features=supporting,
            evidence={
                "z_distance":        z_dist,
                "z_sprint_count":    z_spr,
                "z_top_speed":       z_spd,
                "z_high_speed_dist": z_hsd,
                "window_distance_m": fv.get("window_distance_m", 0.0),
                "window_sprint_count": fv.get("window_sprint_count", 0.0),
            },
            shap_evidence={f: sv.get(f, 0.0) for f in supporting},
            persistence_windows=persistence_windows,
            trend=self._persistence_trend(persistence_windows),
            domain="locomotor_load",
        )

    # ── Rule: recovery degradation ────────────────────────────────────────────

    def _rule_recovery_degradation(
        self,
        fv: Dict[str, float],
        sv: Dict[str, float],
        persistence_windows: int,
    ) -> Optional[SemanticFinding]:
        """
        Fires when multiple recovery markers suggest the player is not recovering
        between high-intensity efforts.

        Requires ≥ 2 of:
          • hr_recovery_time_s > hr_recovery_flat (HR not dropping after effort)
          • speed_drop_pct >= speed_drop_significant
          • fatigue_decay_residual >= fatigue_residual_high (above decay curve)
          • acwr >= acwr_high_risk
        """
        rec   = fv.get("hr_recovery_time_s", 0.0)
        drop  = fv.get("speed_drop_pct", 0.0)
        fat   = fv.get("fatigue_decay_residual", 0.0)
        acwr  = fv.get("acwr", 1.0)

        conditions = [
            ("hr_recovery_time_s",    rec  >= self._t["hr_recovery_flat"]),
            ("speed_drop_pct",        drop >= self._t["speed_drop_significant"]),
            ("fatigue_decay_residual", fat >= self._t["fatigue_residual_high"]),
            ("acwr",                  acwr >= self._t["acwr_high_risk"]),
        ]
        active = [name for name, ok in conditions if ok]

        if len(active) < 2:
            return None

        if not any(self._shap_supports(f, sv) for f in active):
            return None

        severity = "high" if len(active) >= 3 else "moderate"

        confidence = self._confidence_from_shap(active, sv, base=0.68)

        return SemanticFinding(
            finding_type="recovery_degradation",
            severity=severity,
            confidence=confidence,
            summary=(
                f"Recovery capacity appears compromised: {len(active)} markers active "
                f"(HR recovery, speed decline, fatigue curve, workload ratio)."
            ),
            supporting_features=active,
            evidence={
                "hr_recovery_time_s":    rec,
                "speed_drop_pct":        drop,
                "fatigue_decay_residual": fat,
                "acwr":                  acwr,
            },
            shap_evidence={f: sv.get(f, 0.0) for f in active},
            persistence_windows=persistence_windows,
            trend=(
                "worsening"
                if persistence_windows >= self._t["persistence_confirmed"] and len(active) >= 3
                else self._persistence_trend(persistence_windows)
            ),
            domain="workload_balance",
        )

    # ── Rule: tactical instability ────────────────────────────────────────────

    def _rule_tactical_instability(
        self,
        fv: Dict[str, float],
        sv: Dict[str, float],
        persistence_windows: int,
    ) -> Optional[SemanticFinding]:
        """
        Fires when the player has drifted meaningfully from their assigned tactical zone.

        Requires:
          • positional_drift_score >= drift_elevated
          • SHAP attribution on positional_drift_score
        """
        drift = fv.get("positional_drift_score", 0.0)

        if drift < self._t["drift_elevated"]:
            return None
        if not self._shap_supports("positional_drift_score", sv):
            return None

        severity = "high" if drift >= self._t["drift_high"] else "low"

        confidence = self._confidence_from_shap(
            ["positional_drift_score"], sv, base=0.65
        )

        supporting = ["positional_drift_score"]
        if self._shap_supports("z_distance", sv):
            supporting.append("z_distance")

        return SemanticFinding(
            finding_type="tactical_instability",
            severity=severity,
            confidence=confidence,
            summary=(
                f"Player is operating {drift:.1f}× outside their normal positional zone "
                f"({'sustained' if persistence_windows >= 3 else 'transient'} displacement)."
            ),
            supporting_features=supporting,
            evidence={
                "positional_drift_score": drift,
                "z_distance":             fv.get("z_distance", 0.0),
            },
            shap_evidence={f: sv.get(f, 0.0) for f in supporting},
            persistence_windows=persistence_windows,
            trend=self._persistence_trend(persistence_windows),
            domain="tactical_positioning",
        )

    # ── Rule: fatigue accumulation ────────────────────────────────────────────

    def _rule_fatigue_accumulation(
        self,
        fv: Dict[str, float],
        sv: Dict[str, float],
        persistence_windows: int,
    ) -> Optional[SemanticFinding]:
        """
        Fires when cumulative fatigue signals are combining across domains:
        locomotor output is dropping AND workload is above healthy range.

        Requires:
          • speed_drop_pct >= speed_drop_significant  (output declining)
          • acwr >= acwr_high_risk OR fatigue_decay_residual >= fatigue_residual_high
          • SHAP attribution on at least one fatigue-domain feature
        """
        drop  = fv.get("speed_drop_pct", 0.0)
        acwr  = fv.get("acwr", 1.0)
        fat   = fv.get("fatigue_decay_residual", 0.0)
        spr   = fv.get("window_sprint_count", 0.0)
        spd   = fv.get("window_avg_speed_ms", 0.0)

        output_declining = drop >= self._t["speed_drop_significant"]
        load_elevated = (
            acwr >= self._t["acwr_high_risk"] or
            fat  >= self._t["fatigue_residual_high"]
        )

        if not (output_declining and load_elevated):
            return None

        fatigue_features = [
            "speed_drop_pct", "acwr", "fatigue_decay_residual",
            "window_avg_speed_ms", "window_sprint_count",
        ]
        if not any(self._shap_supports(f, sv) for f in fatigue_features):
            return None

        # Severity: persistence makes this serious
        if persistence_windows >= self._t["persistence_confirmed"]:
            severity = "high"
        else:
            severity = "moderate"

        # Also check for locomotor suppression (low sprint despite expectation)
        locomotor_suppressed = (
            spr <= self._t["sprint_count_low"] and
            spd <= self._t["speed_ms_low"]
        )
        summary_tail = (
            " Locomotor output is suppressed."
            if locomotor_suppressed else ""
        )

        active = [
            f for f in fatigue_features
            if self._shap_supports(f, sv) or fv.get(f, 0.0) > 0.0
        ]

        confidence = self._confidence_from_shap(active, sv, base=0.70)

        return SemanticFinding(
            finding_type="fatigue_accumulation",
            severity=severity,
            confidence=confidence,
            summary=(
                f"Fatigue accumulation pattern detected: speed output has dropped "
                f"{drop:.0f}% with elevated workload (ACWR={acwr:.2f}).{summary_tail}"
            ),
            supporting_features=active,
            evidence={
                "speed_drop_pct":         drop,
                "acwr":                   acwr,
                "fatigue_decay_residual": fat,
                "window_avg_speed_ms":    spd,
                "window_sprint_count":    spr,
            },
            shap_evidence={f: sv.get(f, 0.0) for f in active},
            persistence_windows=persistence_windows,
            trend=(
                "worsening"
                if persistence_windows >= self._t["persistence_confirmed"]
                else "stable"
            ),
            domain="workload_balance",
        )


# ─────────────────────────────────────────────────────────────────────────────
#     LLM prompt builder
#     Converts List[SemanticFinding] → structured prompt block for the NLG engine.
#     Import this in xai_layer.py and replace the raw SHAP feature_lines block.
# ─────────────────────────────────────────────────────────────────────────────

def build_semantic_prompt_block(findings: List[SemanticFinding]) -> str:
    """
    Format SemanticFinding objects into the prompt section consumed by LLMNLGEngine.

    The LLM is given symbolic findings — not raw SHAP values — so it acts as
    a narrator and communicator, not as a physiological reasoning engine.

    Returns a string ready to embed directly in the LLM prompt template.
    """
    if not findings:
        return "No significant semantic findings detected in this window."

    lines = ["Semantic findings (pre-interpreted by symbolic engine):"]
    for i, f in enumerate(findings, 1):
        lines.append(
            f"  {i}. [{f.severity.upper()} | confidence={f.confidence:.0%}] "
            f"{f.finding_type.replace('_', ' ').title()}"
        )
        lines.append(f"     {f.summary}")
        if f.trend != "stable":
            lines.append(f"     Trend: {f.trend} over {f.persistence_windows} windows.")
        if f.evidence:
            evidence_str = ", ".join(
                f"{k}={v:.2f}" for k, v in list(f.evidence.items())[:4]
            )
            lines.append(f"     Evidence: {evidence_str}")

    lines.append("")
    lines.append(
        "Generate a concise operational sports report. "
        "Do not add physiological reasoning beyond what is stated above. "
        "Address the coaching staff directly. Maximum 3 sentences."
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#     Convenience: extract semantic findings from a SHAPExplanation-like dict
#     Allows the orchestrator to call semantic interpretation with minimal change.
# ─────────────────────────────────────────────────────────────────────────────

def interpret_explanation(
    shap_values: Dict[str, float],
    feature_values: Dict[str, float],
    persistence_windows: int = 0,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[SemanticFinding]:
    """
    Thin convenience wrapper.  Drop-in callable from orchestrator or xai_layer.

    Parameters
    ----------
    shap_values        : from SHAPExplanation.shap_values
    feature_values     : from SHAPExplanation.feature_values
    persistence_windows: from AnomalyResult.persistence_windows
    thresholds         : optional override dict (merged into THRESHOLDS)

    Returns
    -------
    List[SemanticFinding] sorted by severity then confidence.
    """
    interpreter = SemanticInterpreter(thresholds=thresholds)
    return interpreter.interpret(
        shap_values=shap_values,
        feature_values=feature_values,
        persistence_windows=persistence_windows,
    )