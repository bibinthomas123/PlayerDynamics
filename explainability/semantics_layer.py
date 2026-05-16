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
    Orchestrator                 ← calls match_state.record_finding() per finding
    ↓
    MatchState (longitudinal memory)
        ├─ motif detection
        ├─ trend reasoning
        └─ build_semantic_summary()
    ↓
    LLM narrative generation  (xai_layer.py — LLMNLGEngine)

Design principles
─────────────────
• Thresholds and ontology are centralized here, never scattered in if-blocks elsewhere.
• The LLM receives SemanticFinding objects — not raw SHAP values.
  It narrates; it does not reason physiologically.
• Five finding types for v1. Extend via the rule methods below.
• All public interfaces are typed and dataclass-based so callers can serialize freely.

Layer responsibility boundary
──────────────────────────────
  semantic_layer  : interprets CURRENT window state → List[SemanticFinding]
  match_state     : interprets LONGITUDINAL evolution → motifs, trends, summary
  orchestrator    : wires findings from semantic_layer into match_state
  LLM             : communicates pre-reasoned output, never reasons itself

DO NOT call match_state from inside this module.
The orchestrator owns the wiring between these two layers.
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
      cardiovascular_overload          — sustained high HR with impaired recovery (Path A: active exertion)
      elevated_cardiovascular_response — anomalous HR recovery pattern at low activity (Path B: resting)
      locomotor_overload
      recovery_degradation
      tactical_instability
      fatigue_accumulation
    """

    severity: str
    """'low' | 'medium' | 'high' | 'critical'"""

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
        list of SemanticFinding objects for the CURRENT window.

        Parameters
        ----------
        shap_values        : {feature_name: shap_float} from XAILayer
        feature_values     : {feature_name: observed_float} from _build_xai_feature_vector
        persistence_windows: how many consecutive windows this alert has been active

        Returns
        -------
        List[SemanticFinding] sorted by severity then confidence.

        Handoff
        -------
        The caller (orchestrator) is responsible for calling
        match_state.record_finding() on each returned finding.
        This module must NOT access MatchState directly — it only
        interprets the current window. Longitudinal reasoning lives
        in match_state.py (motifs, trends, build_semantic_summary).
        """
        findings: List[SemanticFinding] = []

        fv = feature_values   # short alias for readability
        sv = shap_values

        # ── Global window quality gate ────────────────────────────────────────
        # Assess telemetry coherence ONCE before any rule runs.
        # Rules are suppressed for incoherent feature combinations so that a
        # stale accumulator or sensor dropout does not produce findings across
        # multiple domains simultaneously.
        quality = self._assess_window_quality(fv)
        if quality["degraded"]:
            logger.warning(
                "SemanticInterpreter: window quality degraded — "
                "suppressing all semantic findings. Reasons: %s",
                "; ".join(quality["reasons"]),
            )
            return []

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

        finding = self._rule_locomotor_suppression(fv, sv, persistence_windows)
        if finding:
            findings.append(finding)

        # Sort: critical → high → medium → low; then by confidence descending
        _sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
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

    def _assess_window_quality(self, fv: Dict[str, float]) -> Dict:
        """
        Assess telemetry coherence for the entire window before any rule runs.

        Returns a dict:
            {"degraded": bool, "reasons": List[str]}

        Rules
        ─────
        A window is degraded if any of the following hold:

        1. Speed/distance incoherence
           speed == 0 (or near-zero) but distance >> plausible maximum.
           A 30-second window at threshold speed covers ~90 m; more implies
           a stale accumulator from a different telemetry epoch.

        2. HR/speed incoherence
           HR is reported as 0 bpm while speed is > 0.
           Zero HR is biologically impossible during movement — sensor dropout.

        3. Negative distance
           Physical distance cannot be negative; indicates a corrupt window.

        4. HR physiologically implausible
           HR below resting floor (< 30 bpm) or above physiological maximum
           (> 220 bpm) while speed is non-zero.

        These checks are intentionally conservative — only clear physical
        impossibilities are flagged. Borderline values pass through to the
        individual rules, which apply domain-specific thresholds.
        """
        reasons: List[str] = []

        speed    = fv.get("window_avg_speed_ms", 0.0)
        distance = fv.get("window_distance_m", 0.0)
        hr       = fv.get("heart_rate_bpm", 0.0)

        # 1. Speed/distance incoherence
        _max_plausible_distance = (self._t["speed_ms_low"] + 0.5) * 30  # ~90 m
        if speed <= 0.1 and distance > _max_plausible_distance:
            reasons.append(
                f"speed={speed:.2f} m/s but distance={distance:.0f} m "
                f"(max plausible={_max_plausible_distance:.0f} m — stale accumulator suspected)"
            )

        # 2. HR dropout during movement
        if hr == 0.0 and speed > 0.5:
            reasons.append(
                f"HR=0 bpm while speed={speed:.1f} m/s — HR sensor dropout during movement"
            )

        # 3. Negative distance
        if distance < 0.0:
            reasons.append(f"distance={distance:.1f} m — negative distance is physically impossible")

        # 4. Implausible HR during movement
        if speed > 0.5 and hr > 0.0 and (hr < 30.0 or hr > 220.0):
            reasons.append(
                f"HR={hr:.0f} bpm during movement — outside physiological range [30, 220]"
            )

        return {"degraded": bool(reasons), "reasons": reasons}

    def _persistence_trend(self, persistence_windows: int) -> str:
        """
        Return the correct trend label for a finding based solely on persistence count.

        IMPORTANT: This reflects how long a condition has been present, NOT whether
        the underlying signal is numerically deteriorating. Use 'persistent' rather
        than 'worsening' here. Numerical slope-based worsening is determined by
        match_state.py (_compute_trend_signal) from the longitudinal deques.

        'worsening' should only be set by callers that have confirmed a deteriorating
        slope from match_state — not inferred from persistence count alone.
        """
        if persistence_windows >= self._t["persistence_severe"]:
            return "persistent"      # ≥6 windows: chronic, not necessarily worsening
        if persistence_windows >= self._t["persistence_confirmed"]:
            return "persistent"      # ≥3 windows: confirmed recurrence
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

        Two paths:
          Path A — High-intensity overload:
            • heart_rate_bpm  > hr_high threshold (175 bpm)
            • hr_recovery_time_s > hr_recovery_flat (HR not dropping)
            • SHAP attribution on either feature

          Path B — Resting non-recovery (low HR but recovery still impaired):
            • hr_recovery_time_s > hr_recovery_flat
            • window_avg_speed_ms near zero (player has stopped / is walking)
            • SHAP on hr_recovery_time_s is strongly driving the anomaly
            This catches the pattern where a player stops moving but HR
            recovery is anomalous relative to their personal baseline — a
            situation where hr_high (175 bpm) would never fire but the
            recovery signal is the dominant SHAP driver.
        """
        hr    = fv.get("heart_rate_bpm", 0.0)
        rec   = fv.get("hr_recovery_time_s", 0.0)
        speed = fv.get("window_avg_speed_ms", 0.0)

        recovery_impaired = rec >= self._t["hr_recovery_flat"]

        # Path A: classic high-intensity overload
        path_a = (
            hr >= self._t["hr_high"]
            and recovery_impaired
            and (
                self._shap_supports("heart_rate_bpm", sv)
                or self._shap_supports("hr_recovery_time_s", sv)
            )
        )

        # Path B: anomalous non-recovery at low activity
        # hr_recovery_time_s SHAP must be POSITIVE (anomaly-increasing) and strong.
        # _shap_strongly_supports uses abs() — intentionally, for other rules — but
        # Path B requires sign-aware gating: a negative SHAP means this feature is
        # SUPPRESSING the anomaly score, which is the opposite of what Path B requires.
        # The autoencoder may assign negative SHAP to hr_recovery_time_s when the model
        # has learned chronic abnormality — that is NOT an HR recovery anomaly signal.
        path_b = (
            recovery_impaired
            and speed <= self._t["speed_ms_low"]
            and sv.get("hr_recovery_time_s", 0.0) >= self._t["shap_strong"]
        )

        if not (path_a or path_b):
            return None

        # Severity: path_a escalates with persistence; path_b is medium by default
        if path_a:
            if hr >= self._t["hr_critical"] and persistence_windows >= self._t["persistence_confirmed"]:
                severity = "critical"
            elif hr >= self._t["hr_critical"]:
                severity = "high"
            else:
                severity = "medium"
        else:
            # Path B: non-recovery at rest — unusual but not critical without persistence
            severity = "high" if persistence_windows >= self._t["persistence_confirmed"] else "medium"

        confidence = self._confidence_from_shap(
            ["heart_rate_bpm", "hr_recovery_time_s"], sv, base=0.75
        )

        trend = self._persistence_trend(persistence_windows)
        # Do not upgrade to 'worsening' purely from persistence count.
        # Slope-confirmed deterioration comes from match_state longitudinal analysis.
        # Persistence with continued impairment is 'persistent', not proven 'worsening'.

        summary = (
            f"Sustained cardiovascular strain detected: HR {int(hr)} bpm with impaired "
            f"recovery dynamics (rate: {rec:+.3f})."
            if path_a
            else
            f"Elevated HR persistence observed during low movement: {int(hr)} bpm with "
            f"slow recovery dynamics (rate: {rec:+.3f}) at near-zero movement ({speed:.1f} m/s). "
            f"Anomalous relative to this player's personal baseline — not necessarily effort-driven."
        )

        return SemanticFinding(
            # Path A = active exertion overload; Path B = anomalous recovery pattern.
            # Distinct types prevent downstream components from treating both identically.
            finding_type="cardiovascular_overload" if path_a else "elevated_cardiovascular_response",
            severity=severity,
            confidence=confidence,
            summary=summary,
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
            severity = "medium"

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

        severity = "high" if len(active) >= 3 else "medium"

        confidence = self._confidence_from_shap(active, sv, base=0.68)

        # Map feature names to short human labels for the summary line.
        # Only active features are listed — candidates that did NOT cross threshold
        # are excluded. Previously all four candidate names appeared in the summary
        # string regardless of which ones were actually active, causing the LLM to
        # narrate all four while the count said only 2 or 3 were flagged.
        _feature_labels = {
            "hr_recovery_time_s":    "HR recovery",
            "speed_drop_pct":        "speed decline",
            "fatigue_decay_residual": "fatigue curve",
            "acwr":                  "workload ratio",
        }
        active_labels = [_feature_labels[f] for f in active if f in _feature_labels]

        return SemanticFinding(
            finding_type="recovery_degradation",
            severity=severity,
            confidence=confidence,
            summary=(
                f"Recovery-related markers exceeded configured thresholds {len(active)} of 4 markers active "
                f"({', '.join(active_labels)})."
            ),
            supporting_features=active,
            evidence={f: fv.get(f, 0.0) for f in active},
            shap_evidence={f: sv.get(f, 0.0) for f in active},
            persistence_windows=persistence_windows,
            trend=self._persistence_trend(persistence_windows),
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
            severity = "medium"

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
            trend=self._persistence_trend(persistence_windows),
            domain="workload_balance",
        )

    # ── Rule: locomotor suppression ───────────────────────────────────────────

    def _rule_locomotor_suppression(
        self,
        fv: Dict[str, float],
        sv: Dict[str, float],
        persistence_windows: int,
    ) -> Optional[SemanticFinding]:
        """
        Fires when a player's movement output has effectively ceased and SHAP
        attributes the anomaly primarily to speed/distance suppression.

        This is distinct from locomotor_overload (which fires on high z-scores).
        Suppression fires on the opposite pattern: near-zero movement that is
        anomalous relative to the player's personal baseline — the model flags
        it because the player *should* be moving but isn't.

        Requires:
          • window_avg_speed_ms <= speed_ms_low (walking pace or stopped)
          • SHAP on window_avg_speed_ms or window_distance_m is anomaly-driving
            (positive SHAP = driving the anomaly flag)
          • The combined positive SHAP from locomotor features exceeds shap_strong
            to ensure the movement suppression is the primary driver, not incidental

        Does NOT require z-scores — the player may be new or have limited
        baseline history. The SHAP signal alone is sufficient when strong.
        """
        speed    = fv.get("window_avg_speed_ms", 0.0)
        distance = fv.get("window_distance_m", 0.0)

        # Must be genuinely stopped / walking
        if speed > self._t["speed_ms_low"]:
            return None

        # SHAP must be driving the anomaly via the speed/distance channel
        speed_shap    = sv.get("window_avg_speed_ms", 0.0)
        distance_shap = sv.get("window_distance_m", 0.0)

        # Positive SHAP = this feature is pushing the anomaly score up
        if speed_shap <= 0.0 and distance_shap <= 0.0:
            return None

        combined_locomotor_shap = max(speed_shap, 0.0) + max(distance_shap, 0.0)
        if combined_locomotor_shap < self._t["shap_strong"]:
            return None

        severity = (
            "high"
            if persistence_windows >= self._t["persistence_confirmed"]
            else "medium"
        )

        confidence = self._confidence_from_shap(
            ["window_avg_speed_ms", "window_distance_m"], sv, base=0.72
        )

        supporting = ["window_avg_speed_ms", "window_distance_m"]
        # Include positional drift if it's also driving the anomaly
        if sv.get("positional_drift_score", 0.0) > self._t["shap_relevant"]:
            supporting.append("positional_drift_score")

        return SemanticFinding(
            finding_type="locomotor_suppression",
            severity=severity,
            confidence=confidence,
            summary=(
                f"Movement output has effectively ceased: speed {speed:.1f} m/s "
                f"(distance {distance:.0f} m this window). "
                f"This suppression is the primary anomaly driver relative to "
                f"this player's personal baseline."
            ),
            supporting_features=supporting,
            evidence={
                "window_avg_speed_ms": speed,
                "window_distance_m":   distance,
                "window_sprint_count": fv.get("window_sprint_count", 0.0),
            },
            shap_evidence={f: sv.get(f, 0.0) for f in supporting},
            persistence_windows=persistence_windows,
            trend=self._persistence_trend(persistence_windows),
            domain="locomotor_load",
        )


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
        if f.trend not in ("stable", ""):
            if f.trend == "worsening":
                # 'worsening' is only set when match_state confirms a deteriorating slope.
                lines.append(
                    f"     Trend: numerically deteriorating over {f.persistence_windows} windows "
                    f"(slope-confirmed — signal is getting worse, not merely recurring)."
                )
            elif f.trend == "persistent":
                lines.append(
                    f"     Persistence: condition has been active for {f.persistence_windows} "
                    f"consecutive windows (recurring, trajectory not yet slope-confirmed)."
                )
            else:
                lines.append(f"     Trend: {f.trend} over {f.persistence_windows} windows.")
        if f.evidence:
            evidence_str = ", ".join(
                f"{k}={v:.2f}" for k, v in list(f.evidence.items())[:4]
            )
            lines.append(f"     Evidence: {evidence_str}")

    lines.append("")
    lines.append(
            "Render ONLY the findings explicitly stated above.\n"
            "Do NOT infer physiology, fatigue, injury risk, recovery quality, "
            "or coaching recommendations.\n"
            "Do NOT escalate severity beyond the provided labels.\n"
            "Use ONLY these sections:\n"
            "OBSERVED CONDITION:\n"
            "PERSISTENCE:\n"
            "MATCH CONTEXT:\n"
            "Maximum 3 short sentences total."
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