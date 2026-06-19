"""
CoachInsightEngine — PlayerDynamics

First coach-facing intelligence layer. Converts TeamStateTrend
(analysis/team_state_trend.py) deltas into deterministic, human-readable
OBSERVATIONS about a team's recent attack/load/possession behaviour.

Explicitly out of scope for this module:
    Recommendations of any kind (substitutions, tactics, lineup changes),
    LLM logic, ML, Frontend.

Insights are observations only. "Attack activity is rising" is in scope;
"substitute the right wing" is not.

Determinism
------------
Every insight is the result of comparing existing TeamStateTrend fields
against fixed thresholds (CoachInsightConfig in config/settings.py). No ML,
no learned models. The same ordered TeamStateTrend sequence always produces
the same CoachInsight sequence.

Why insight thresholds differ from trend thresholds
-------------------------------------------------------
TeamStateTrendConfig's thresholds already filter window-to-window noise
down to a binary "increasing" / "stable" / "decreasing" label. But not
every "increasing" trend is significant enough to surface to a coach as a
standalone observation -- across a 60-minute match, roughly half of all
60s-window trends already carry a directional label (see
TEAMSTATE_TREND_IMPLEMENTATION.md's distribution table). CoachInsightConfig
sets a SECOND, stricter threshold per metric so only the most notable swings
(calibrated to roughly the 75th percentile of |delta| in real session 3387
data) become an insight. A trend can be labelled "increasing" without
producing any CoachInsight at all -- that is the expected, intended behaviour.

v1 insight categories
------------------------
Attack:
    attack_activity_rising      -- attack_trend=="increasing" AND attack_activity_delta >= threshold
    attack_activity_falling     -- attack_trend=="decreasing" AND attack_activity_delta <= -threshold
Load:
    workload_spike               -- load_trend=="increasing"   AND physical_load_delta   >= threshold
    workload_drop                 -- load_trend=="decreasing"   AND physical_load_delta   <= -threshold
Possession:
    possession_pressure_increasing -- possession_pressure_delta >= threshold
    possession_pressure_decreasing -- possession_pressure_delta <= -threshold
    (TeamStateTrend carries no possession_pressure trend label, so direction
     here is determined directly from the signed delta against the threshold.)
Composite (both component conditions must independently hold for the SAME
TeamStateTrend row -- composites are emitted IN ADDITION TO, not instead of,
the singular insights that compose them):
    high_attack_high_load  -- attack_activity_rising-condition AND workload_spike-condition
    low_attack_high_load    -- attack_activity_falling-condition AND workload_spike-condition
                               (physical effort rising while attacking output falls)

Every insight carries, in .metadata:
    source_metrics    -- which TeamStateTrend field(s) triggered it
    values             -- the actual delta value(s)
    thresholds_crossed -- the SIGNED threshold actually crossed (e.g. -19.0
                          for a "falling"/"drop"/"decreasing" category, not +19.0)
    window_seconds      -- the originating TeamStateTrend's window length
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config.settings import CONFIG, CoachInsightConfig
from analysis.team_state_trend import TeamStateTrend

logger = logging.getLogger(__name__)

_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class CoachInsight:
    """
    A single deterministic, coach-facing observation derived from one
    TeamStateTrend row. Never a recommendation.
    """
    timestamp: datetime
    team_id: Optional[str]
    severity: str        # "low" | "medium" | "high"
    category: str
    message: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _severity_and_confidence(ratio: float, config: CoachInsightConfig) -> Tuple[str, float]:
    if ratio >= config.severity_high_ratio:
        severity = "high"
    elif ratio >= config.severity_medium_ratio:
        severity = "medium"
    else:
        severity = "low"
    confidence = min(1.0, round(config.confidence_base + config.confidence_slope * (ratio - 1.0), 3))
    return severity, confidence


def _max_severity(a: str, b: str) -> str:
    return a if _SEVERITY_ORDER[a] >= _SEVERITY_ORDER[b] else b


class CoachInsightEngine:
    """
    Usage
    -----
        engine = CoachInsightEngine()
        insights = engine.generate(trends)   # trends: Iterable[TeamStateTrend]
    """

    def __init__(self, config: Optional[CoachInsightConfig] = None) -> None:
        self.config = config or CONFIG.coach_insight

    def generate(self, trends: Iterable[TeamStateTrend]) -> List[CoachInsight]:
        ordered = sorted(
            trends,
            key=lambda t: (t.team_id is None, str(t.team_id), t.window_seconds, t.timestamp),
        )
        insights: List[CoachInsight] = []
        for t in ordered:
            insights.extend(self._evaluate(t))
        return insights

    # ─────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────

    def _evaluate(self, t: TeamStateTrend) -> List[CoachInsight]:
        cfg = self.config
        out: List[CoachInsight] = []

        attack_up = (
            t.attack_trend == "increasing"
            and t.attack_activity_delta >= cfg.attack_activity_insight_threshold
        )
        attack_down = (
            t.attack_trend == "decreasing"
            and t.attack_activity_delta <= -cfg.attack_activity_insight_threshold
        )
        load_up = (
            t.load_trend == "increasing"
            and t.physical_load_delta >= cfg.physical_load_insight_threshold
        )
        load_down = (
            t.load_trend == "decreasing"
            and t.physical_load_delta <= -cfg.physical_load_insight_threshold
        )
        pp_up = t.possession_pressure_delta >= cfg.possession_pressure_insight_threshold
        pp_down = t.possession_pressure_delta <= -cfg.possession_pressure_insight_threshold

        attack_sev = attack_conf = None
        if attack_up or attack_down:
            ratio = abs(t.attack_activity_delta) / cfg.attack_activity_insight_threshold
            attack_sev, attack_conf = _severity_and_confidence(ratio, cfg)

        load_sev = load_conf = None
        if load_up or load_down:
            ratio = abs(t.physical_load_delta) / cfg.physical_load_insight_threshold
            load_sev, load_conf = _severity_and_confidence(ratio, cfg)

        if attack_up:
            out.append(self._single(
                t, "attack_activity_rising", "attack_activity_delta",
                t.attack_activity_delta, cfg.attack_activity_insight_threshold,
                attack_sev, attack_conf,
                f"Attack activity rising for {t.team_id}: "
                f"{t.attack_activity_delta:+.1f} events/min "
                f"(threshold {cfg.attack_activity_insight_threshold:.1f}).",
            ))
        if attack_down:
            out.append(self._single(
                t, "attack_activity_falling", "attack_activity_delta",
                t.attack_activity_delta, -cfg.attack_activity_insight_threshold,
                attack_sev, attack_conf,
                f"Attack activity falling for {t.team_id}: "
                f"{t.attack_activity_delta:+.1f} events/min "
                f"(threshold -{cfg.attack_activity_insight_threshold:.1f}).",
            ))
        if load_up:
            out.append(self._single(
                t, "workload_spike", "physical_load_delta",
                t.physical_load_delta, cfg.physical_load_insight_threshold,
                load_sev, load_conf,
                f"Workload spike for {t.team_id}: physical load "
                f"{t.physical_load_delta:+.1f} events/min "
                f"(threshold {cfg.physical_load_insight_threshold:.1f}).",
            ))
        if load_down:
            out.append(self._single(
                t, "workload_drop", "physical_load_delta",
                t.physical_load_delta, -cfg.physical_load_insight_threshold,
                load_sev, load_conf,
                f"Workload drop for {t.team_id}: physical load "
                f"{t.physical_load_delta:+.1f} events/min "
                f"(threshold -{cfg.physical_load_insight_threshold:.1f}).",
            ))
        if pp_up:
            ratio = abs(t.possession_pressure_delta) / cfg.possession_pressure_insight_threshold
            sev, conf = _severity_and_confidence(ratio, cfg)
            out.append(self._single(
                t, "possession_pressure_increasing", "possession_pressure_delta",
                t.possession_pressure_delta, cfg.possession_pressure_insight_threshold,
                sev, conf,
                f"Possession pressure increasing for {t.team_id}: "
                f"{t.possession_pressure_delta:+.2f} "
                f"(threshold {cfg.possession_pressure_insight_threshold:.2f}) -- "
                f"team losing the ball more often relative to clean possessions.",
            ))
        if pp_down:
            ratio = abs(t.possession_pressure_delta) / cfg.possession_pressure_insight_threshold
            sev, conf = _severity_and_confidence(ratio, cfg)
            out.append(self._single(
                t, "possession_pressure_decreasing", "possession_pressure_delta",
                t.possession_pressure_delta, -cfg.possession_pressure_insight_threshold,
                sev, conf,
                f"Possession pressure decreasing for {t.team_id}: "
                f"{t.possession_pressure_delta:+.2f} "
                f"(threshold -{cfg.possession_pressure_insight_threshold:.2f}) -- "
                f"team retaining the ball more securely.",
            ))

        if attack_up and load_up:
            out.append(self._composite(
                t, "high_attack_high_load",
                ("attack_activity_delta", "physical_load_delta"),
                (t.attack_activity_delta, t.physical_load_delta),
                (cfg.attack_activity_insight_threshold, cfg.physical_load_insight_threshold),
                _max_severity(attack_sev, load_sev), min(attack_conf, load_conf),
                f"{t.team_id} is pushing hard on both fronts: attack activity "
                f"{t.attack_activity_delta:+.1f}/min and physical load "
                f"{t.physical_load_delta:+.1f}/min both rising together.",
            ))
        if attack_down and load_up:
            out.append(self._composite(
                t, "low_attack_high_load",
                ("attack_activity_delta", "physical_load_delta"),
                (t.attack_activity_delta, t.physical_load_delta),
                (-cfg.attack_activity_insight_threshold, cfg.physical_load_insight_threshold),
                _max_severity(attack_sev, load_sev), min(attack_conf, load_conf),
                f"{t.team_id} shows rising physical load ({t.physical_load_delta:+.1f}/min) "
                f"without a matching rise in attack activity "
                f"({t.attack_activity_delta:+.1f}/min) -- effort not translating "
                f"into attacking output.",
            ))

        return out

    @staticmethod
    def _single(
        t: TeamStateTrend,
        category: str,
        metric: str,
        value: float,
        threshold_crossed: float,
        severity: str,
        confidence: float,
        message: str,
    ) -> CoachInsight:
        return CoachInsight(
            timestamp=t.timestamp,
            team_id=t.team_id,
            severity=severity,
            category=category,
            message=message,
            confidence=confidence,
            metadata={
                "source_metrics": [metric],
                "values": {metric: value},
                "thresholds_crossed": {metric: threshold_crossed},
                "window_seconds": t.window_seconds,
            },
        )

    @staticmethod
    def _composite(
        t: TeamStateTrend,
        category: str,
        metrics: Tuple[str, str],
        values: Tuple[float, float],
        thresholds_crossed: Tuple[float, float],
        severity: str,
        confidence: float,
        message: str,
    ) -> CoachInsight:
        return CoachInsight(
            timestamp=t.timestamp,
            team_id=t.team_id,
            severity=severity,
            category=category,
            message=message,
            confidence=confidence,
            metadata={
                "source_metrics": list(metrics),
                "values": dict(zip(metrics, values)),
                "thresholds_crossed": dict(zip(metrics, thresholds_crossed)),
                "window_seconds": t.window_seconds,
            },
        )
