"""
Players Data — IBM CIC Germany
Feedback Loop, Recalibration Pipeline & Fairness Monitor

Implements:
  1. Override logging — every coach decision is stored with full context
  2. Weekly recalibration — adjusts model thresholds from accumulated overrides
  3. Per-player sensitivity adjustment — reduces feature sensitivity when repeatedly overridden
  4. Bias audit — detects systematic over/under-flagging by position, age group, etc.

This is the mechanism that makes the system genuinely human-centered:
it learns the coaching philosophy, not just the statistics.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import CONFIG, FeedbackConfig, FairnessConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Override Record
# ─────────────────────────────────────────────
@dataclass
class OverrideRecord:
    """A single coach override event."""
    inference_id: int
    player_id: int
    player_external_id: str
    session_id: int
    recommendation_type: str       # "substitution", "fatigue_alert", etc.
    decision: str                  # "accept" | "override" | "defer"
    coach_id: str
    coach_note: Optional[str]
    overridden_at: datetime
    context_snapshot: dict         # Feature values at decision time
    position: Optional[str] = None
    age_group: Optional[str] = None
    nationality: Optional[str] = None


# ─────────────────────────────────────────────
# Feedback Store (in-memory, DB-backed in production)
# ─────────────────────────────────────────────
class FeedbackStore:
    """
    In-memory store for override records during a session.
    In production: backed by the override_logs PostgreSQL table.
    """

    def __init__(self):
        self._records: List[OverrideRecord] = []

    def log_override(self, record: OverrideRecord) -> None:
        """Log a coach override. This is the primary learning signal."""
        self._records.append(record)
        logger.info(
            "Override logged: player=%s type=%s decision=%s coach=%s",
            record.player_external_id,
            record.recommendation_type,
            record.decision,
            record.coach_id,
        )

    def get_recent(
        self,
        since: Optional[datetime] = None,
        player_id: Optional[int] = None,
        recommendation_type: Optional[str] = None,
    ) -> List[OverrideRecord]:
        """Filter override records by recency, player, or recommendation type."""
        records = self._records

        if since:
            records = [r for r in records if r.overridden_at >= since]
        if player_id is not None:
            records = [r for r in records if r.player_id == player_id]
        if recommendation_type:
            records = [r for r in records if r.recommendation_type == recommendation_type]

        return records

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        return pd.DataFrame([vars(r) for r in self._records])

    @property
    def total_overrides(self) -> int:
        return sum(1 for r in self._records if r.decision == "override")

    @property
    def override_rate(self) -> float:
        total = len(self._records)
        return self.total_overrides / total if total > 0 else 0.0


# ─────────────────────────────────────────────
# Recalibration Engine
# ─────────────────────────────────────────────
@dataclass
class RecalibrationResult:
    """Output of one recalibration run."""
    player_id: Optional[int]          # None = squad-wide
    recalibrated_at: datetime
    trigger_reason: str
    n_overrides_analyzed: int
    adjustments: Dict[str, dict]       # {feature: {old_threshold, new_threshold}}
    model_version_before: str
    model_version_after: str
    notes: str


class RecalibrationPipeline:
    """
    Analyzes accumulated override patterns and adjusts model behavior.

    Two types of adjustment:
      1. Squad-wide threshold tuning  — shifts contamination parameter
      2. Per-player sensitivity reduction — dampens specific features per player
    """

    def __init__(self, config: FeedbackConfig = None):
        self.cfg = config or CONFIG.feedback

    def run(
        self,
        feedback_store: FeedbackStore,
        player_models: Dict[int, object],   # Dict[player_id, PlayerAnomalyModel]
        trigger_reason: str = "weekly_cadence",
    ) -> List[RecalibrationResult]:
        """
        Run full recalibration.
        Returns list of RecalibrationResult (one per player + one squad-wide).
        """
        since = datetime.now(tz=timezone.utc) - timedelta(
            days=self.cfg.recalibration_cadence_days
        )
        recent = feedback_store.get_recent(since=since)

        if len(recent) < self.cfg.min_overrides_for_recalibration:
            logger.info(
                "Recalibration skipped — only %d overrides (need %d)",
                len(recent), self.cfg.min_overrides_for_recalibration
            )
            return []

        results = []

        # ── Squad-wide: if override rate is high, increase contamination (more permissive)
        squad_result = self._squad_recalibration(recent, player_models, trigger_reason)
        if squad_result:
            results.append(squad_result)

        # ── Per-player: reduce sensitivity for repeatedly overridden features
        player_ids = {r.player_id for r in recent if r.decision == "override"}
        for player_id in player_ids:
            player_result = self._player_recalibration(
                player_id, recent, player_models, trigger_reason
            )
            if player_result:
                results.append(player_result)

        return results

    def _squad_recalibration(
        self,
        recent_overrides: List[OverrideRecord],
        player_models: Dict[int, object],
        trigger_reason: str,
    ) -> Optional[RecalibrationResult]:
        """Adjust squad-wide contamination parameter based on overall override rate."""
        total = len(recent_overrides)
        n_overrides = sum(1 for r in recent_overrides if r.decision == "override")
        override_rate = n_overrides / total if total > 0 else 0.0

        adjustments: Dict[str, dict] = {}

        if override_rate > 0.4:
            # Too many overrides — model is too aggressive, increase threshold (less sensitive)
            delta = self.cfg.threshold_adjustment_step
            new_contamination = min(0.2, CONFIG.isolation_forest.contamination + delta)
            adjustments["contamination"] = {
                "old": CONFIG.isolation_forest.contamination,
                "new": new_contamination,
                "reason": f"Override rate {override_rate:.1%} > 40% — reducing sensitivity",
            }
            CONFIG.isolation_forest.contamination = new_contamination

        elif override_rate < 0.05 and override_rate > 0:
            # Very few overrides — model may be under-detecting
            delta = self.cfg.threshold_adjustment_step
            new_contamination = max(0.01, CONFIG.isolation_forest.contamination - delta)
            adjustments["contamination"] = {
                "old": CONFIG.isolation_forest.contamination,
                "new": new_contamination,
                "reason": f"Override rate {override_rate:.1%} < 5% — increasing sensitivity",
            }
            CONFIG.isolation_forest.contamination = new_contamination

        if not adjustments:
            return None

        ts = datetime.now(tz=timezone.utc)
        return RecalibrationResult(
            player_id=None,
            recalibrated_at=ts,
            trigger_reason=trigger_reason,
            n_overrides_analyzed=total,
            adjustments=adjustments,
            model_version_before="squad_v_prev",
            model_version_after=f"squad_v_{ts.strftime('%Y%m%d')}",
            notes=f"Squad-wide override rate: {override_rate:.1%}",
        )

    def _player_recalibration(
        self,
        player_id: int,
        recent_overrides: List[OverrideRecord],
        player_models: Dict[int, object],
        trigger_reason: str,
    ) -> Optional[RecalibrationResult]:
        """Reduce sensitivity for features repeatedly overridden for a specific player."""
        player_overrides = [
            r for r in recent_overrides
            if r.player_id == player_id and r.decision == "override"
        ]

        if len(player_overrides) < 3:
            return None

        model = player_models.get(player_id)
        if model is None:
            return None

        # Identify features that were at extreme values during overridden predictions
        feature_override_counts: Dict[str, int] = defaultdict(int)

        for override in player_overrides:
            snapshot = override.context_snapshot or {}
            for feature, value in snapshot.items():
                if abs(value) > 2.0:    # Feature was extreme when coach overrode
                    feature_override_counts[feature] += 1

        adjustments: Dict[str, dict] = {}

        for feature, count in feature_override_counts.items():
            if count >= 3:
                old_sensitivity = model._sensitivity_adjustments.get(feature, 1.0)
                model.apply_sensitivity_adjustment(feature, self.cfg.per_player_sensitivity_decay)
                new_sensitivity = model._sensitivity_adjustments.get(feature, 1.0)
                adjustments[feature] = {
                    "old_sensitivity": old_sensitivity,
                    "new_sensitivity": new_sensitivity,
                    "override_count": count,
                }

        if not adjustments:
            return None

        ts = datetime.now(tz=timezone.utc)
        return RecalibrationResult(
            player_id=player_id,
            recalibrated_at=ts,
            trigger_reason=trigger_reason,
            n_overrides_analyzed=len(player_overrides),
            adjustments=adjustments,
            model_version_before=getattr(model, "model_version", "unknown"),
            model_version_after=f"{getattr(model, 'model_version', 'v0')}_recal",
            notes=f"Per-player sensitivity adjustment for {len(adjustments)} features",
        )


# ─────────────────────────────────────────────
# Fairness Monitor
# ─────────────────────────────────────────────
@dataclass
class FairnessAuditResult:
    """Output of one fairness audit run."""
    audited_at: datetime
    attribute: str
    group_results: List[dict]        # Per-group flag rates and disparity
    is_biased: bool
    biased_groups: List[str]
    action_recommended: str
    squad_avg_flag_rate: float


class FairnessMonitor:
    """
    Detects systematic over/under-representation in flagged insights
    across positions, age groups, or other protected attributes.

    Alert if any group's flag rate differs from squad average by > threshold.
    """

    def __init__(self, config: FairnessConfig = None):
        self.cfg = config or CONFIG.fairness

    def audit(
        self,
        inference_df: pd.DataFrame,
        player_metadata_df: pd.DataFrame,
    ) -> List[FairnessAuditResult]:
        """
        Run fairness audit across all protected attributes.

        Parameters
        ----------
        inference_df : DataFrame with columns [player_id, is_anomaly, triggered_at]
        player_metadata_df : DataFrame with [player_id, position, age_group, nationality]
        """
        if inference_df.empty or player_metadata_df.empty:
            logger.warning("Fairness audit: empty data — skipping")
            return []

        merged = inference_df.merge(player_metadata_df, on="player_id", how="left")
        squad_flag_rate = float(merged["is_anomaly"].mean()) if "is_anomaly" in merged else 0.0

        results = []
        for attr in self.cfg.protected_attributes:
            if attr not in merged.columns:
                continue
            result = self._audit_attribute(merged, attr, squad_flag_rate)
            if result:
                results.append(result)

        return results

    def _audit_attribute(
        self,
        df: pd.DataFrame,
        attribute: str,
        squad_avg_flag_rate: float,
    ) -> Optional[FairnessAuditResult]:
        """Audit flag rate disparity for one attribute."""
        group_stats = (
            df.groupby(attribute)["is_anomaly"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "flag_rate", "count": "n"})
            .reset_index()
        )

        group_results = []
        biased_groups = []

        for _, row in group_stats.iterrows():
            group_label = str(row[attribute])
            flag_rate = float(row["flag_rate"])
            disparity = flag_rate - squad_avg_flag_rate
            is_biased = abs(disparity) > self.cfg.flag_rate_disparity_threshold

            group_results.append({
                "group": group_label,
                "flag_rate": round(flag_rate, 4),
                "squad_avg": round(squad_avg_flag_rate, 4),
                "disparity": round(disparity, 4),
                "n_observations": int(row["n"]),
                "is_biased": is_biased,
            })

            if is_biased:
                biased_groups.append(group_label)

        is_biased = len(biased_groups) > 0

        if is_biased:
            biased_str = ", ".join(biased_groups)
            action = (
                f"Review model thresholds for: {biased_str}. "
                f"Flag rate disparity exceeds {self.cfg.flag_rate_disparity_threshold:.0%} threshold. "
                f"Consider per-group sensitivity adjustment."
            )
            logger.warning(
                "FAIRNESS ALERT: %s — biased groups: %s", attribute, biased_str
            )
        else:
            action = "No bias detected — flag rates are within acceptable disparity bounds."

        return FairnessAuditResult(
            audited_at=datetime.now(tz=timezone.utc),
            attribute=attribute,
            group_results=group_results,
            is_biased=is_biased,
            biased_groups=biased_groups,
            action_recommended=action,
            squad_avg_flag_rate=round(squad_avg_flag_rate, 4),
        )

    def generate_audit_report(self, results: List[FairnessAuditResult]) -> str:
        """Generate a plain-text fairness audit report for human review."""
        lines = [
            "=" * 60,
            "PLAYERS DATA — FAIRNESS AUDIT REPORT",
            f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 60,
        ]

        if not results:
            lines.append("No audit results available.")
            return "\n".join(lines)

        for r in results:
            lines.append(f"\nAttribute: {r.attribute.upper()}")
            lines.append(f"Squad average flag rate: {r.squad_avg_flag_rate:.1%}")
            lines.append(f"Status: {'⚠ BIAS DETECTED' if r.is_biased else '✓ FAIR'}")

            for g in r.group_results:
                bias_marker = " ⚠" if g["is_biased"] else ""
                lines.append(
                    f"  {g['group']:20s} flag_rate={g['flag_rate']:.1%}  "
                    f"disparity={g['disparity']:+.1%}  n={g['n_observations']}{bias_marker}"
                )

            if r.is_biased:
                lines.append(f"Action: {r.action_recommended}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
