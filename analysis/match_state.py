"""
Per-player match state manager.
Structured semantic memory for LLM prompt enrichment.
Keyed by (player_id, match_id) — no cross-match state leakage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchState:
    """
    Compressed per-player statistics for the current match.
    Stores counters and aggregates — never raw event timelines.
    LLMs reason better over semantic summaries than raw sequences.
    """
    player_id:   int
    player_name: str
    position:    str
    match_id:    str

    # Per-type alert counters
    fatigue_alert_count:  int = 0
    workload_alert_count: int = 0
    anomaly_alert_count:  int = 0

    # Persistence tracking
    consecutive_alerts:    int = 0
    last_alert_type:       str = ""
    last_alert_ts:         Optional[datetime] = None
    first_alert_elapsed_s: Optional[int] = None

    # Online anomaly score stats (incremental mean — no list accumulation)
    _anomaly_score_sum:   float = 0.0
    _anomaly_score_count: int   = 0
    peak_anomaly_score:   float = 0.0

    # Telemetry aggregates
    peak_hr_bpm:  float = 0.0
    sprint_count: int   = 0

    # Thread-safety lock (Fix 3)
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def record_alert(
        self,
        recommendation_type: str,
        confidence: float,
        anomaly_score: float,
        elapsed_seconds: int,
    ) -> None:
        with self._lock:
            if "fatigue" in recommendation_type:
                self.fatigue_alert_count += 1
            elif "workload" in recommendation_type:
                self.workload_alert_count += 1
            else:
                self.anomaly_alert_count += 1

            self.consecutive_alerts = (
                self.consecutive_alerts + 1
                if self.last_alert_type == recommendation_type
                else 1
            )
            self.last_alert_type       = recommendation_type
            self.last_alert_ts         = datetime.now(tz=timezone.utc)
            self._anomaly_score_sum   += anomaly_score
            self._anomaly_score_count += 1
            self.peak_anomaly_score    = max(self.peak_anomaly_score, anomaly_score)

            if self.first_alert_elapsed_s is None:
                self.first_alert_elapsed_s = elapsed_seconds

    def record_telemetry(self, speed_ms: float, hr_bpm: float) -> None:
        with self._lock:
            self.peak_hr_bpm = max(self.peak_hr_bpm, hr_bpm)
            if speed_ms >= 7.0:
                self.sprint_count += 1

    @property
    def mean_anomaly_score(self) -> float:
        with self._lock:
            if self._anomaly_score_count == 0:
                return 0.0
            return self._anomaly_score_sum / self._anomaly_score_count

    def build_llm_context(self) -> str:
        """
        Compressed semantic summary injected into the LLM prompt.
        Never sends raw timelines — just statistics and trends.
        """
        with self._lock:
            total_alerts = (
                self.fatigue_alert_count
                + self.workload_alert_count
                + self.anomaly_alert_count
            )
            if total_alerts == 0:
                return "No prior alerts this match."

            lines = [
                f"Match context for {self.player_name} ({self.position}):",
                (
                    f"  Total alerts: {total_alerts}"
                    f"  (fatigue={self.fatigue_alert_count},"
                    f" workload={self.workload_alert_count},"
                    f" anomaly={self.anomaly_alert_count})"
                ),
                f"  Consecutive same-type alerts: {self.consecutive_alerts}",
                (
                    f"  Mean anomaly score: {self._anomaly_score_sum / self._anomaly_score_count:.2f}"
                    f"  Peak: {self.peak_anomaly_score:.2f}"
                ),
                (
                    f"  Peak HR: {self.peak_hr_bpm:.0f} bpm"
                    f"  Sprint count: {self.sprint_count}"
                ),
            ]
            if self.first_alert_elapsed_s is not None:
                lines.append(
                    f"  First alert at ~{self.first_alert_elapsed_s // 60} min"
                )
            return "\n".join(lines)


class MatchStateManager:
    """
    Registry keyed by (player_id, match_id).
    Explicit lifecycle via start_match() / end_match().
    Never keyed by player_id alone — that causes cross-match leakage.
    """

    def __init__(self) -> None:
        self._states: Dict[Tuple[int, str], MatchState] = {}
        self._active_match_id: Optional[str] = None
        self._lock = Lock()

    def start_match(self, match_id: str) -> None:
        """Call at kickoff. Drops all state from prior matches."""
        logger.info("MatchStateManager: starting match %s", match_id)
        with self._lock:
            stale = [k for k in self._states if k[1] != match_id]

            for k in stale:
                del self._states[k]

            self._active_match_id = match_id

    def end_match(self, match_id: str) -> None:
        """Call at full time. Frees all per-match memory."""
        logger.info("MatchStateManager: ending match %s", match_id)
        with self._lock:
            keys = [k for k in self._states if k[1] == match_id]

            for k in keys:
                del self._states[k]

            if self._active_match_id == match_id:
                self._active_match_id = None

    def get_or_create(
        self,
        player_id: int,
        player_name: str,
        position: str,
        match_id: Optional[str] = None,
    ) -> MatchState:
        
        with self._lock:
            mid = match_id or self._active_match_id or "default"
            key = (player_id, mid)
            if key not in self._states:
                self._states[key] = MatchState(
                    player_id=player_id,
                    player_name=player_name,
                    position=position,
                    match_id=mid,
                )
            return self._states[key]