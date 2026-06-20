"""
CoachSituation — PlayerDynamics

The first match-context layer. Aggregates Possession, TeamState,
TeamStateTrend, and CoachInsight signals into a single higher-level
tactical state per (team_id, window_seconds, timestamp) evaluation point.

Pipeline position:

    TacticalEvent -> Possession -> TeamState -> TeamStateTrend -> CoachInsight
                                                                          |
                                                                          v
                                                                  CoachSituation

Explicitly out of scope for this module:
    Recommendations of any kind, Frontend, Redis contract changes.
    Situations are observations about match CHARACTER, never directives.

Determinism
------------
Every situation_type is the result of a fixed-priority rule list over
already-deterministic inputs (TeamStateTrend labels/deltas, the CURRENT
TeamState's absolute attack_activity level, CoachInsight categories, and
simple aggregates over recent Possession objects). No ML, no learned
models. The same inputs always produce the same CoachSituation sequence.

Why TeamState (the absolute level), not just TeamStateTrend (the delta)
---------------------------------------------------------------------------
TeamStateTrend's attack_trend=="stable" says nothing about whether that
stable level is high or low -- "stable at 25 events/min" and "stable at
1 event/min" are both "stable". Several situation types need the absolute
level too (e.g. SUSTAINED_PRESSURE requires genuinely high tempo, not just
non-declining tempo from near zero), so this engine also looks up the
matching TeamState snapshot (same team_id/window_seconds/timestamp key as
the trend) and reads its attack_activity field directly. If no matching
TeamState is supplied, any rule that depends on the absolute level simply
does not fire for that point (fails closed, never guesses).

Possession aggregation per evaluation point
-----------------------------------------------
For a given (team_id, window_seconds, timestamp), "recent possessions" are
every Possession for that team whose end_timestamp falls in
(timestamp - window_seconds, timestamp] -- the same window the trend/state
themselves describe. From these we compute:
    recent_possession_count
    recent_turnover_rate      = turnovers / max(count, 1)
    recent_high_quality_count = count(possession_quality >= quality_high_threshold)
    recent_mean_quality        = mean(possession_quality), 0.0 if none

Only one situation_type per evaluation point
------------------------------------------------
Unlike CoachInsight (which emits every category that independently fires),
a CoachSituation is meant to describe THE tactical character of a moment --
so rules are evaluated in a fixed priority order and the FIRST match wins.
If no rule matches, no CoachSituation is produced for that point (silence
is the correct, expected outcome for an unremarkable period).

Situation taxonomy (10 types) and their signals
-----------------------------------------------------
 1. ATTACKING_SURGE_WITH_RISK -- high_attack_high_load insight present AND recent turnovers are heavy
 2. HIGH_TEMPO_ATTACK          -- attack rising + already-high absolute tempo + high-quality recent possessions + load rising
 3. INEFFICIENT_HIGH_EFFORT    -- low_attack_high_load insight present (workload up, attack down)
 4. POSSESSION_INSTABILITY     -- turnover-heavy recent possessions + worsening possession pressure
 5. SUSTAINED_PRESSURE         -- multiple high-quality recent possessions + attack stable/rising at a high level + possession pressure improving
 6. FATIGUE_ONSET              -- fatigue burden rising, load not falling, attack not rising
 7. DEFENSIVE_RECOVERY_PHASE   -- attack stable/falling at a low level + load falling + possession pressure improving
 8. RECOVERY_CONSOLIDATION     -- fatigue burden falling, load stable/falling, possession pressure not worsening
 9. EFFICIENT_TRANSITION       -- attack rising while load is falling (more output, less physical cost)
10. CONTROLLED_TEMPO           -- everything stable (attack/load/fatigue trends all "stable", possession pressure barely moving)

Every type's relevant CoachInsight categories (used to populate
source_insights -- only those ACTUALLY present at this point are included,
never invented) are listed in _RELEVANT_INSIGHTS below.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from config.settings import CONFIG, CoachSituationConfig
from analysis.coach_insight import CoachInsight
from analysis.possession import Possession
from analysis.team_state import TeamState
from analysis.team_state_trend import TeamStateTrend

logger = logging.getLogger(__name__)

_RELEVANT_INSIGHTS: Dict[str, Set[str]] = {
    "ATTACKING_SURGE_WITH_RISK": {"high_attack_high_load", "attack_activity_rising", "workload_spike"},
    "HIGH_TEMPO_ATTACK": {"attack_activity_rising", "workload_spike"},
    "INEFFICIENT_HIGH_EFFORT": {"low_attack_high_load", "attack_activity_falling", "workload_spike"},
    "POSSESSION_INSTABILITY": {"possession_pressure_increasing"},
    "SUSTAINED_PRESSURE": {"possession_pressure_decreasing", "attack_activity_rising"},
    "FATIGUE_ONSET": {"workload_spike"},
    "DEFENSIVE_RECOVERY_PHASE": {"possession_pressure_decreasing", "attack_activity_falling", "workload_drop"},
    "RECOVERY_CONSOLIDATION": {"workload_drop", "possession_pressure_decreasing"},
    "EFFICIENT_TRANSITION": {"attack_activity_rising", "workload_drop"},
    "CONTROLLED_TEMPO": set(),
}


@dataclass
class CoachSituation:
    """A single higher-level tactical state for one team at one point in time."""
    timestamp: datetime
    team_id: Optional[str]
    situation_type: str
    severity: str          # "low" | "medium" | "high"
    confidence: float
    source_insights: List[str] = field(default_factory=list)
    source_metrics: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


def _possession_aggregates(possessions: List[Possession], config: CoachSituationConfig) -> Dict[str, Any]:
    if not possessions:
        return {
            "recent_possession_count": 0,
            "recent_turnover_rate": 0.0,
            "recent_high_quality_count": 0,
            "recent_mean_quality": 0.0,
        }
    n = len(possessions)
    n_turnovers = sum(1 for p in possessions if p.outcome == "turnover")
    n_high_quality = sum(1 for p in possessions if p.possession_quality >= config.quality_high_threshold)
    mean_quality = sum(p.possession_quality for p in possessions) / n
    return {
        "recent_possession_count": n,
        "recent_turnover_rate": round(n_turnovers / n, 4),
        "recent_high_quality_count": n_high_quality,
        "recent_mean_quality": round(mean_quality, 4),
    }


def _severity_and_confidence(
    source_insights: List[str], insights_by_category: Dict[str, CoachInsight], config: CoachSituationConfig
) -> Tuple[str, float]:
    n = len(source_insights)
    severity = "high" if n >= 2 else ("medium" if n == 1 else "low")
    if source_insights:
        confidence = min(insights_by_category[c].confidence for c in source_insights)
    else:
        confidence = config.trend_only_confidence
    return severity, confidence


def _relevant_insights_present(situation_type: str, insight_categories: Set[str]) -> List[str]:
    return sorted(_RELEVANT_INSIGHTS[situation_type] & insight_categories)


class CoachSituationEngine:
    """
    Usage
    -----
        engine = CoachSituationEngine()
        situations = engine.generate(
            possessions=possessions,     # Iterable[Possession]
            team_states=team_states,     # Iterable[TeamState]
            trends=trends,                # Iterable[TeamStateTrend]
            insights=insights,            # Iterable[CoachInsight]
        )
    """

    def __init__(self, config: Optional[CoachSituationConfig] = None) -> None:
        self.config = config or CONFIG.coach_situation

    def generate(
        self,
        possessions: Iterable[Possession],
        team_states: Iterable[TeamState],
        trends: Iterable[TeamStateTrend],
        insights: Iterable[CoachInsight],
    ) -> List[CoachSituation]:
        possessions_by_team: Dict[Optional[str], List[Possession]] = {}
        for p in possessions:
            possessions_by_team.setdefault(p.team_id, []).append(p)
        for plist in possessions_by_team.values():
            plist.sort(key=lambda p: p.end_timestamp)

        state_by_key: Dict[Tuple[Optional[str], int, datetime], TeamState] = {
            (s.team_id, s.window_seconds, s.timestamp): s for s in team_states
        }

        insights_by_key: Dict[Tuple[Optional[str], int, datetime], List[CoachInsight]] = {}
        for i in insights:
            key = (i.team_id, i.metadata.get("window_seconds"), i.timestamp)
            insights_by_key.setdefault(key, []).append(i)

        ordered_trends = sorted(
            trends, key=lambda t: (t.team_id is None, str(t.team_id), t.window_seconds, t.timestamp)
        )

        situations: List[CoachSituation] = []
        for trend in ordered_trends:
            key = (trend.team_id, trend.window_seconds, trend.timestamp)
            state = state_by_key.get(key)
            team_insights = insights_by_key.get(key, [])
            insights_by_category = {i.category: i for i in team_insights}
            insight_categories = set(insights_by_category.keys())

            window_start = trend.timestamp - timedelta(seconds=trend.window_seconds)
            recent = [
                p for p in possessions_by_team.get(trend.team_id, [])
                if window_start < p.end_timestamp <= trend.timestamp
            ]
            agg = _possession_aggregates(recent, self.config)

            situation = self._classify(trend, state, insight_categories, agg)
            if situation is None:
                continue

            situation_type, raw_source_metrics = situation
            source_insights = _relevant_insights_present(situation_type, insight_categories)
            severity, confidence = _severity_and_confidence(source_insights, insights_by_category, self.config)
            source_metrics = {
                "window_seconds": trend.window_seconds,
                "attack_trend": trend.attack_trend,
                "load_trend": trend.load_trend,
                "fatigue_trend": trend.fatigue_trend,
                "possession_pressure_delta": trend.possession_pressure_delta,
                **agg,
                **raw_source_metrics,
            }
            situations.append(CoachSituation(
                timestamp=trend.timestamp,
                team_id=trend.team_id,
                situation_type=situation_type,
                severity=severity,
                confidence=confidence,
                source_insights=source_insights,
                source_metrics=source_metrics,
                explanation=_explain(situation_type, trend, agg),
            ))

        return situations

    # ─────────────────────────────────────────────────────────────────────
    # Internal: fixed-priority classification
    # ─────────────────────────────────────────────────────────────────────

    def _classify(
        self,
        t: TeamStateTrend,
        state: Optional[TeamState],
        insight_categories: Set[str],
        agg: Dict[str, Any],
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        cfg = self.config

        if "high_attack_high_load" in insight_categories and agg["recent_possession_count"] > 0 \
                and agg["recent_turnover_rate"] >= cfg.turnover_heavy_threshold:
            return "ATTACKING_SURGE_WITH_RISK", {}

        if (
            t.attack_trend == "increasing"
            and state is not None and state.attack_activity >= cfg.attack_activity_floor
            and agg["recent_possession_count"] > 0 and agg["recent_mean_quality"] >= cfg.quality_high_threshold
            and t.load_trend == "increasing"
        ):
            return "HIGH_TEMPO_ATTACK", {"attack_activity": state.attack_activity}

        if "low_attack_high_load" in insight_categories:
            return "INEFFICIENT_HIGH_EFFORT", {}

        if (
            agg["recent_possession_count"] > 0
            and agg["recent_turnover_rate"] >= cfg.turnover_heavy_threshold
            and t.possession_pressure_delta >= cfg.possession_pressure_worsening_threshold
        ):
            return "POSSESSION_INSTABILITY", {}

        if (
            agg["recent_high_quality_count"] >= cfg.min_high_quality_possessions
            and t.attack_trend in ("stable", "increasing")
            and state is not None and state.attack_activity >= cfg.attack_activity_floor
            and t.possession_pressure_delta <= -cfg.possession_pressure_improving_threshold
        ):
            return "SUSTAINED_PRESSURE", {"attack_activity": state.attack_activity}

        if (
            t.fatigue_trend == "increasing"
            and t.load_trend in ("increasing", "stable")
            and t.attack_trend != "increasing"
        ):
            return "FATIGUE_ONSET", {}

        if (
            t.attack_trend in ("stable", "decreasing")
            and state is not None and state.attack_activity <= cfg.attack_activity_low_ceiling
            and t.load_trend == "decreasing"
            and t.possession_pressure_delta <= -cfg.possession_pressure_improving_threshold
        ):
            return "DEFENSIVE_RECOVERY_PHASE", {"attack_activity": state.attack_activity}

        if (
            t.fatigue_trend == "decreasing"
            and t.load_trend in ("stable", "decreasing")
            and t.possession_pressure_delta <= 0.0
        ):
            return "RECOVERY_CONSOLIDATION", {}

        if t.attack_trend == "increasing" and t.load_trend == "decreasing":
            return "EFFICIENT_TRANSITION", {}

        if (
            t.attack_trend == "stable" and t.load_trend == "stable" and t.fatigue_trend == "stable"
            and abs(t.possession_pressure_delta) < cfg.possession_pressure_stable_threshold
        ):
            return "CONTROLLED_TEMPO", {}

        return None


_EXPLANATIONS = {
    "ATTACKING_SURGE_WITH_RISK": (
        "{team}: pushing hard offensively and physically, but {turnover_pct:.0f}% of recent "
        "possessions ended in a turnover -- aggressive tempo carrying real risk."
    ),
    "HIGH_TEMPO_ATTACK": (
        "{team}: attack tempo rising from an already-high base, paired with rising physical "
        "output and high-quality recent possessions (mean quality {mean_quality:.2f})."
    ),
    "INEFFICIENT_HIGH_EFFORT": (
        "{team}: physical effort rising without a matching rise in attacking output."
    ),
    "POSSESSION_INSTABILITY": (
        "{team}: {turnover_pct:.0f}% of recent possessions ended in a turnover, and possession "
        "pressure is still worsening."
    ),
    "SUSTAINED_PRESSURE": (
        "{team}: {n_high_quality} high-quality possessions recently, attack tempo holding at a "
        "high level, and possession pressure improving."
    ),
    "FATIGUE_ONSET": (
        "{team}: fatigue burden rising while attacking output is not increasing to match."
    ),
    "DEFENSIVE_RECOVERY_PHASE": (
        "{team}: low attack tempo, easing physical load, and improving possession pressure -- "
        "a defensive consolidation period."
    ),
    "RECOVERY_CONSOLIDATION": (
        "{team}: fatigue burden easing with physical load steady or falling, and possession "
        "control not deteriorating."
    ),
    "EFFICIENT_TRANSITION": (
        "{team}: attack tempo rising while physical load is falling -- more attacking output "
        "for less physical cost."
    ),
    "CONTROLLED_TEMPO": (
        "{team}: attack, load, and fatigue trends are all stable, with possession pressure barely moving."
    ),
}


def _explain(situation_type: str, t: TeamStateTrend, agg: Dict[str, Any]) -> str:
    template = _EXPLANATIONS[situation_type]
    return template.format(
        team=t.team_id,
        turnover_pct=agg["recent_turnover_rate"] * 100.0,
        mean_quality=agg["recent_mean_quality"],
        n_high_quality=agg["recent_high_quality_count"],
    )
