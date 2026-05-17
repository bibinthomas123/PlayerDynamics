"""
Per-player match state manager.
Structured semantic memory for LLM prompt enrichment.
Keyed by (player_id, match_id) — no cross-match state leakage.

State Evolution Memory
────────────────────────────
Upgraded from alert counter storage to longitudinal trajectory memory.

Four memory layers:
  Layer 1 — Trend memory        rolling deques for HR, speed, anomaly score, recovery
  Layer 2 — Motif detection     repeated-pattern recognition over recent findings
  Layer 3 — Progression analysis slope-based trend reasoning (worsening / stable / recovering)
  Layer 4 — Semantic summary    human-readable state evolution narrative for the LLM

Architecture rule:
  The symbolic layer detects motifs and trends.
  The LLM only communicates them.
  These responsibilities must never be merged.

Runtime Persistence
────────────────────────────
MatchState exposes to_dict() / from_dict() for full serialization.
MatchStateManager exposes checkpoint() / restore_if_active() / mark_dirty() /
end_match_and_wipe() to persist and resume live match intelligence across
process restarts.

Checkpoint lifecycle:
  start_match(id)    → manager.restore_if_active(id)   # resume if same match
  per window         → manager.mark_dirty()             # flag state as changed
  on alert / N wins  → manager.checkpoint_if_due()      # throttled flush
  end_match(id)      → manager.end_match_and_wipe(id)  # final flush + wipe

Throttle policy (all tuneable via constructor):
  checkpoint_interval_s   — minimum seconds between periodic checkpoints (default 30)
  checkpoint_every_n      — also checkpoint every N windows regardless of time (default 10)
  checkpoint_on_alert     — always checkpoint when an alert window is recorded (default True)

CheckpointStore abstraction
────────────────────────────
Storage is fully decoupled from orchestration logic via a Protocol:

    class CheckpointStore(Protocol):
        def save(key: str, payload: dict) -> None: ...
        def load(key: str) -> dict | None: ...
        def delete(key: str) -> None: ...

Concrete implementations:
  JsonFileCheckpointStore   — atomic JSON file (default, good for debugging)
  (future) RedisCheckpointStore, S3CheckpointStore — same interface, no orchestrator changes

Schema version is stamped on every checkpoint. A version mismatch causes the
checkpoint to be discarded and deleted rather than silently loaded as corrupt state.
A checkpoint older than MAX_CHECKPOINT_AGE_HOURS is treated as stale and
discarded — protects against orphaned files from a previous match day after an
unclean shutdown.

Match identity
────────────────────────────
Checkpoint metadata includes match_id + match_start_ts + schema_version so
old runtime motifs from a prior session can never contaminate a new match even
if match_ids were reused or the process crashed mid-match.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock, RLock
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import numpy as np

from analysis.episodic_context import (
      PlayerEpisode,
      TacticalEpisode,
      TemporalContextCompressor,
      CompressedTemporalContext,
  )
from config.redis_client import EpisodeStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Persistence constants
# ─────────────────────────────────────────────
_CHECKPOINT_SCHEMA_VERSION  = 2    # bump when MatchState fields change
MAX_CHECKPOINT_AGE_HOURS    = 6    # checkpoints older than this are treated as stale


# ─────────────────────────────────────────────
# CheckpointStore protocol
# ─────────────────────────────────────────────

@runtime_checkable
class CheckpointStore(Protocol):
    """
    Storage backend for match state checkpoints.

    The key is a logical identifier (e.g. ``"runtime_match_state"``).
    The payload is a plain JSON-safe dict — serialization is the
    responsibility of the store, not the manager.

    Implementations must be thread-safe.  Concrete stores:
      JsonFileCheckpointStore — atomic JSON file (default)
      RedisCheckpointStore    — future, same interface
      S3CheckpointStore       — future, same interface
    """

    def save(self, key: str, payload: dict) -> None:
        """Persist payload under key.  Must be atomic (no partial writes)."""
        ...

    def load(self, key: str) -> Optional[dict]:
        """Return payload for key, or None if not found / unreadable."""
        ...

    def delete(self, key: str) -> None:
        """Delete the stored payload.  Silent if key does not exist."""
        ...


class JsonFileCheckpointStore:
    """
    Atomic JSON file backend.

    Writes via <path>.tmp + os.replace() so a crash mid-write always leaves
    a valid previous checkpoint on disk.  Thread-safe via an internal RLock.

    Parameters
    ----------
    base_dir : directory that holds checkpoint files (created if absent)
    """

    def __init__(self, base_dir: str = "models") -> None:
        self._base_dir = base_dir
        self._lock = RLock()

    def _path(self, key: str) -> str:
        return os.path.join(self._base_dir, f"{key}.json")

    def save(self, key: str, payload: dict) -> None:
        path = self._path(key)
        tmp  = path + ".tmp"
        with self._lock:
            try:
                os.makedirs(self._base_dir, exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, path)
                logger.debug("CheckpointStore.save: %s (%d bytes)", path, os.path.getsize(path))
            except Exception:
                logger.exception("CheckpointStore.save: failed for key=%r", key)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def load(self, key: str) -> Optional[dict]:
        path = self._path(key)
        with self._lock:
            if not os.path.exists(path):
                return None
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                logger.exception("CheckpointStore.load: could not read %r — discarding", path)
                self.delete(key)
                return None

    def delete(self, key: str) -> None:
        path = self._path(key)
        with self._lock:
            try:
                os.unlink(path)
                logger.debug("CheckpointStore.delete: removed %s", path)
            except FileNotFoundError:
                pass
            except Exception:
                logger.exception("CheckpointStore.delete: failed for %s", path)


_CHECKPOINT_KEY = "runtime_match_state"

# Minimum number of consecutive windows an episode must span before it is
# written to EpisodeStore.  Single-window flickers (persistence_duration=1)
# carry too little signal and bloat the episode index.
_MIN_EPISODE_PERSIST_WINDOWS: int = 2

logger.info("_MIN_EPISODE_PERSIST_WINDOWS = %d", _MIN_EPISODE_PERSIST_WINDOWS)
# ─────────────────────────────────────────────
# Trend slope thresholds
# ─────────────────────────────────────────────
_ANOMALY_SLOPE_WORSENING   =  0.03   # rising anomaly score per window
_ANOMALY_SLOPE_RECOVERING  = -0.03

_RECOVERY_SLOPE_WORSENING  =  0.005  # hr_recovery_rate rising = impaired recovery
_RECOVERY_SLOPE_IMPROVING  = -0.005

_SPEED_SLOPE_DECLINING     = -0.05   # m/s per window
_SPEED_SLOPE_INCREASING    =  0.05

_MIN_TREND_SAMPLES = 4               # minimum deque length before slope is meaningful

# ─────────────────────────────────────────────
# Metric-specific volatility thresholds
# ─────────────────────────────────────────────
# A std of 0.15 means very different things for anomaly score vs speed vs recovery slope.
# These are calibrated per-signal so volatility labels are semantically meaningful.
#
# anomaly score   — [0, 1] range;   0.05 = medium spread, 0.12 = wide spread
# recovery slope  — fractional HR;  0.01 = noticeable instability, 0.03 = high instability
# speed (m/s)     — typical 2–7 m/s; 0.15 = medium, 0.35 = sprinting irregularity
_VOLATILITY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "anomaly":  {"medium": 0.05, "high": 0.12},
    "recovery": {"medium": 0.01, "high": 0.03},
    "workload": {"medium": 0.15, "high": 0.35},
}

_CURVATURE_THRESHOLDS = {
    "anomaly": 0.01,
    "recovery": 0.003,
    "workload": 0.02,
}

@dataclass
class TrendSignal:
    """
    Raw numerical output of signal processing.
    Deliberately free of semantic labels — only physics.

    slope     : first-order rate of change per window
    curvature : second-order acceleration (positive = accelerating degradation)
    volatility: standard deviation of the signal
    """
    slope:      float
    curvature:  float
    volatility: float


@dataclass
class TrendInterpretation:
    """
    Semantic labeling derived from a TrendSignal.
    Kept separate so thresholds and labels can be recalibrated without
    touching signal extraction, and so each layer is independently testable.

    direction    : "worsening" | "recovering" | "stable" | "declining" |
                   "increasing" | "insufficient data"
    acceleration : "accelerating" | "steady"
    volatility   : "high" | "medium" | "low"
    """
    direction:    str
    acceleration: str
    volatility:   str


@dataclass
class CoupledState:
    """
    First-class symbolic entity representing a detected multi-system
    physiological interaction.

    Replaces loose dict coupling so the type system enforces completeness
    and the layer can be evolved (e.g. adding temporal span) without silent
    schema drift.

    state_type        : machine-readable identifier
    severity          : "low" | "medium" | "high" | "critical"
    confidence        : belief score in [0, 1]  — interpretable symbolic prior,
                        not statistical confidence
    supporting_trends : which trend axes contributed (e.g. ["anomaly", "recovery"])
    description       : human-readable explanation for the LLM
    """
    state_type:        str
    severity:          str
    confidence: float
    supporting_trends: List[str]
    description:       str

    def to_dict(self) -> dict:
        return {
            "type":              self.state_type,
            "severity":          self.severity,
            "confidence": self.confidence,
            "supporting_trends": self.supporting_trends,
            "description":       self.description,
        }


@dataclass
class TrendState:
    signal: Optional[TrendSignal]
    interpretation: TrendInterpretation

    def to_dict(self) -> dict:
        base = {
            "direction":    self.interpretation.direction,
            "volatility":   self.interpretation.volatility,
            "acceleration": self.interpretation.acceleration,
        }

        if self.signal is not None:
            base["slope"] = round(self.signal.slope, 4)
            base["curvature"] = round(self.signal.curvature, 4)

        return base

@dataclass
class SemanticMatchState:
    """
    Structured machine-readable session state.
    Replaces the string-only build_semantic_summary() output.
    Passed to xai_layer.format_match_state_prompt() for LLM formatting.
    """
    motifs:              List[dict]
    trends:              Dict[str, TrendState]
    persistent_findings: List[dict]
    escalation_level:    str
    coupled_states:      List[CoupledState]
    risk_breakdown:      Dict[str, float]


@dataclass
class MatchState:
    """
    Per-player session intelligence memory for the current match.

    Stores rolling trajectories of physiological signals and symbolic findings
    to support longitudinal reasoning across windows — not just current-state
    snapshot counters.

    build_semantic_state() exposes structured symbolic session state.
    Formatting into LLM-ready narrative is delegated to xai_layer.
    """
    player_id:   int
    player_name: str
    position:    str
    match_id:    str

    # ── Per-type alert counters ───────────────────────────────────────────────
    fatigue_alert_count:  int = 0
    workload_alert_count: int = 0
    anomaly_alert_count:  int = 0

    # ── Persistence tracking ──────────────────────────────────────────────────
    consecutive_alerts:    int = 0
    last_alert_type:       str = ""
    last_alert_ts:         Optional[datetime] = None
    first_alert_elapsed_s: Optional[int] = None

    # ── Online anomaly score stats (incremental mean — no list accumulation) ──
    _anomaly_score_sum:   float = 0.0
    _anomaly_score_count: int   = 0
    peak_anomaly_score:   float = 0.0

    # ── Telemetry aggregates ──────────────────────────────────────────────────
    peak_hr_bpm:  float = 0.0
    sprint_count: int   = 0

    # ── Layer 1: Rolling trajectory deques ───────────────────────────────────
    # Shape over time is more informative than any single value.
    # "HR steadily rising over 6 windows" >> "HR high"
    recent_hr:             deque = field(default_factory=lambda: deque(maxlen=50))
    recent_speed:          deque = field(default_factory=lambda: deque(maxlen=50))
    recent_anomaly_scores: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_recovery:       deque = field(default_factory=lambda: deque(maxlen=50))

    # ── Layer 2: Symbolic finding history ────────────────────────────────────
    # Each entry: {"type", "severity", "minute", "confidence", "trend", "domain"}
    # Expanded schema supports confidence propagation and temporal motif reasoning.
    recent_findings: deque = field(default_factory=lambda: deque(maxlen=50))

    # ── Layer 2b: Finding transition memory ──────────────────────────────────
    # Counts how often finding type B immediately follows finding type A.
    # Key: (from_type, to_type) — Value: occurrence count.
    # Enables empirical causal chain discovery:
    #   P(recovery_degradation | locomotor_overload) = count(A→B) / count(A)
    # Currently write-only; motif engine will read this in future versions.
    transition_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    # ── Layer 5: Episodic temporal memory ────────────────────────────────────
#   # Compressed episode abstractions replace raw history injection to the LLM.
#   # The compressor reads recent_findings + deques and produces episodes.
#   # Episodes are stable once closed (end_minute is set); only the last
#   # (ongoing) episode is mutated on each window.
    episodes:             List["PlayerEpisode"] = field(default_factory=list)
    trend_summaries:      Dict[str, str] = field(default_factory=dict)
    recent_context:       str = ""
    intervention_history: List[str] = field(default_factory=list)
    _episode_counter:     int = field(default=0, repr=False)
    # Tracks which episode IDs have already been written to EpisodeStore.
    # Prevents re-persisting the same closed episode on every window refresh.
    _persisted_episode_ids: set = field(default_factory=set, repr=False, compare=False)

    # ── Thread-safety lock ────────────────────────────────────────────────────
    _lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    # ── SemanticMatchState cache ───────────────────────────────────────────────
    # Invalidated on every record_finding() / record_telemetry() / record_alert()
    # so build_semantic_state() is only ever computed once per window, not 3+ times.
    _cached_semantic_state: Optional["SemanticMatchState"] = field(
        default=None, repr=False, compare=False
    )

    _episode_store = EpisodeStore()


    # ─────────────────────────────────────────────
    # Existing interface — extended
    # ─────────────────────────────────────────────

    def record_alert(
            self,
            recommendation_type: str,
            confidence: float,
            anomaly_score: float,
            elapsed_seconds: int,
            episode_id: Optional[int] = None,
        ) -> None:
        with self._lock:
            if "fatigue" in recommendation_type:
                self.fatigue_alert_count += 1
            elif "workload" in recommendation_type:
                self.workload_alert_count += 1
            else:
                self.anomaly_alert_count += 1

            if (episode_id is not None and episode_id == self.last_episode_id):
                return

            self.consecutive_alerts = (
                self.consecutive_alerts + 1
                if self.last_alert_type == recommendation_type
                else 1
            )

            self.last_episode_id = episode_id
            self.last_alert_type       = recommendation_type
            self.last_alert_ts         = datetime.now(tz=timezone.utc)

            if self.first_alert_elapsed_s is None:
                self.first_alert_elapsed_s = elapsed_seconds

            # Invalidate cached semantic state — counters changed
            self._cached_semantic_state = None

    def record_telemetry(
        self,
        speed_ms: float,
        hr_bpm: float,
        hr_recovery_rate: float = 0.0,
        anomaly_score: float = 0.0,
        telemetry_confidence: float = 1.0,
        enforce_confidence_gate: bool = True,   # ← add this
    ) -> None:
        """
        Extended to feed all four rolling trajectory deques.

        hr_recovery_rate : fractional HR change per window (from XAI feature vector).
                           Positive  = HR still rising  (recovery impaired).
                           Negative  = HR dropping      (recovering well).
        anomaly_score    : current window anomaly score.
                   All windows are appended so temporal trends remain
                   uniformly sampled.
        """
        with self._lock:
            if enforce_confidence_gate or telemetry_confidence >= 0.8:
                # Only update trajectory memory when telemetry is trustworthy.
                # DEGRADED telemetry (confidence < 0.8) must not reinforce motifs,
                # inflate anomaly trajectories, or distort trend slopes.
                if telemetry_confidence >= 0.8:
                    self.peak_hr_bpm = max(self.peak_hr_bpm, hr_bpm)
                    if speed_ms >= 7.0:
                        self.sprint_count += 1

                    # Layer 1: rolling physiological trajectories
                    self.recent_hr.append(hr_bpm)
                    self.recent_speed.append(speed_ms)
                    self.recent_recovery.append(hr_recovery_rate)

                    self._anomaly_score_sum   += anomaly_score
                    self._anomaly_score_count += 1
                    self.peak_anomaly_score    = max(self.peak_anomaly_score, anomaly_score)

                    self.recent_anomaly_scores.append(anomaly_score)
                else:
                    logger.debug(
                        "record_telemetry: skipped trajectory update "
                        "(telemetry_confidence=%.2f < 0.8)",
                        telemetry_confidence,
                    )

            # Invalidate cached semantic state — deques changed
            self._cached_semantic_state = None
            
    # ─────────────────────────────────────────────
    # Layer 2: Finding memory — producer interface
    # ─────────────────────────────────────────────

    def record_finding(
        self,
        finding: dict,
        elapsed_seconds: Optional[int] = None,
    ) -> None:
        """
        Record one serialized SemanticFinding dict into the findings deque.

        Called from the orchestrator after each XAI explanation so findings
        accumulate across windows. The symbolic motif engine reads this deque.

        The orchestrator calls this — semantic_layer.py and xai_layer.py
        must NOT call this directly. Layer separation must be maintained.

        Parameters
        ----------
        finding         : SemanticFinding.to_dict() output
        elapsed_seconds : match clock at time of finding (used to compute minute label)
        """
        minute = (elapsed_seconds // 60) if elapsed_seconds is not None else None
        entry = {
            "type":       finding.get("finding_type", "unknown"),
            "severity":   finding.get("severity", "low"),
            "minute":     minute,
            "confidence": finding.get("confidence", 0.5),
            "trend":      finding.get("trend", "stable"),
            "domain":     finding.get("domain", ""),
            "state":      finding.get("state", "active"),
        }
        with self._lock:
            # Record transition from the previous finding to this one
            if self.recent_findings:
                prev = self.recent_findings[-1]

                prev_min = prev.get("minute")
                curr_min = entry.get("minute")

                temporal_local = (
                    prev_min is not None
                    and curr_min is not None
                    and 0 <= (curr_min - prev_min) <= 3
                )

                if (
                    temporal_local
                    and entry.get("confidence", 0.0) >= 0.55
                    and prev.get("confidence", 0.0) >= 0.55
                ):
                    prev_type = prev["type"]
                    key = (prev_type, entry["type"])
                    self.transition_counts[key] = (
                        self.transition_counts.get(key, 0) + 1
                    )
            previous = None

            for old in reversed(self.recent_findings):
                if old["type"] == entry["type"]:
                    previous = old
                    break

            entry["state"] = self._infer_finding_state(entry, previous)
            self.recent_findings.append(entry)
            # Invalidate cached semantic state — findings changed
            self._cached_semantic_state = None

    # ─────────────────────────────────────────────
    # Layer 2b: Transition memory — query interface
    # ─────────────────────────────────────────────

    def transition_probability(self, from_type: str, to_type: str) -> float:
        """
        Empirical P(to_type | from_type) from observed finding transitions.

        Returns 0.0 when from_type has never been observed as a predecessor.
        This will feed into motif confidence scoring once enough transitions
        accumulate across matches (requires persistence layer — not yet wired).

        Example:
            p = state.transition_probability("locomotor_overload", "recovery_degradation")
            # 0.67 → "2 out of 3 locomotor overloads led to recovery degradation"
        """
        with self._lock:
            from_count = sum(
                v for (a, _), v in self.transition_counts.items() if a == from_type
            )
            if from_count == 0:
                return 0.0
            to_count = self.transition_counts.get((from_type, to_type), 0)
            smoothed = (to_count + 1) / (from_count + 2) # laplace smoothing
            return round(smoothed,3)

    def top_transitions(self, n: int = 5) -> List[dict]:
        """
        Return the n most frequent finding transitions observed this match,
        ordered by count descending.

        Useful for: dashboards, analytics, future RL reward shaping.

        Example output:
            [{"from": "locomotor_overload", "to": "recovery_degradation", "count": 4, "p": 0.8}, ...]
        """
        with self._lock:
            counts = dict(self.transition_counts)

        if not counts:
            return []

        # Compute from-totals for conditional probabilities
        from_totals: Dict[str, int] = {}
        for (a, _), v in counts.items():
            from_totals[a] = from_totals.get(a, 0) + v

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {
                "from":  a,
                "to":    b,
                "count": v,
                "p":     round(v / from_totals[a], 3),
            }
            for (a, b), v in ranked[:n]
        ]

    # ─────────────────────────────────────────────
    # Layer 2: Motif detection
    # ─────────────────────────────────────────────

    @staticmethod
    def _contains_progression(seq: List[str], pattern: List[str], max_window: int = 10) -> Optional[Tuple[int, int]]:
        """
        Check whether `pattern` appears as an ordered subsequence within any
        contiguous `max_window`-length slice of `seq`.

        Allows intervening noise events:
            A -> X -> B -> C  still matches pattern [A, B, C]

        Returns (start_idx, end_idx) of the earliest matching window, or None.

        max_window limits how far apart the pattern elements can be spread —
        a 10-finding window prevents spuriously matching across half a match.
        """
        n = len(seq)
        p = len(pattern)
        if n < p:
            return None

        for start in range(n - p + 1):
            end = min(start + max_window, n)
            window = seq[start:end]
            # Greedy subsequence scan
            pi = 0
            match_end = start
            for si, item in enumerate(window):
                if item == pattern[pi]:
                    pi += 1
                    match_end = start + si
                    if pi == p:
                        return (start, match_end + 1)
        return None

    def detect_motifs(
        self,
        findings_snapshot: List[dict],
    ) -> List[dict]:
        """
        Scan recent_findings for repeated behavioral patterns.

        Motif detection is purely symbolic. The LLM communicates discovered
        motifs; it must not discover them itself.

        MUST be called with self._lock already held (called from
        build_semantic_state which holds the lock for the snapshot read,
        then releases before computing trends).

        Returns a list of structured motif dicts:
          {"type", "severity", "confidence", "description"}
        """
        if not findings_snapshot:
            return []
        
        findings_list = findings_snapshot
        types          = [f["type"] for f in findings_list]
        severities     = [f["severity"] for f in findings_list]
        motifs: List[dict] = []
        already_flagged: set = set()

        # ── Temporal sequence: sprint-collapse progression ────────────────────
        # Ordered chain: locomotor_overload → cardiovascular_overload → recovery_degradation
        # Uses subsequence detection so intervening noise findings don't break the motif.
        _SPRINT_COLLAPSE_PATTERN = [
            "locomotor_overload",
            "cardiovascular_overload",
            "recovery_degradation",
        ]
        match_span = self._contains_progression(types, _SPRINT_COLLAPSE_PATTERN, max_window=10)
        if match_span is not None:
            start_i, end_i = match_span
            evidence = findings_list[start_i:end_i]
            motifs.append({
                "type":        "sprint_collapse_progression",
                "severity":    "high",
                "confidence":  self._motif_confidence(evidence),
                "description": (
                    "Sprint-collapse progression: locomotor overload → "
                    "cardiovascular overload → recovery degradation detected in sequence."
                ),
            })
            already_flagged.update(_SPRINT_COLLAPSE_PATTERN)

        # ── Motif 1: Repeated cardiovascular overload ─────────────────────────
        if "cardiovascular_overload" not in already_flagged:
            cv_findings = [f for f in findings_list if f["type"] == "cardiovascular_overload"]
            if len(cv_findings) >= 3:
                motifs.append({
                    "type":          "persistent_cardiovascular_overload",
                    "severity":      "high",
                    "confidence":    self._motif_confidence(cv_findings),
                    "evidence_count": len(cv_findings),
                    "description": (
                        "Repeated cardiovascular overload episodes detected — "
                        "pattern suggests sustained cardiac stress, not isolated spikes."
                    ),
                })
                already_flagged.add("cardiovascular_overload")

        # ── Motif 2: Sprint-collapse motif (unordered fallback) ───────────────
        if (
            "sprint_collapse_progression" not in {m["type"] for m in motifs}
            and types.count("locomotor_overload") >= 2
            and "cardiovascular_overload" in types
            and "recovery_degradation" in types
        ):
            evidence = [f for f in findings_list if f["type"] in (
                "locomotor_overload", "cardiovascular_overload", "recovery_degradation"
            )]
            motifs.append({
                "type":        "sprint_collapse_motif",
                "severity":    "high",
                "confidence":  self._motif_confidence(evidence),
                "description": (
                    "Sprint-collapse motif: repeated high-intensity bursts followed by "
                    "cardiovascular overload and degraded recovery."
                ),
            })

        # ── Motif 3: Tactical drift fatigue ───────────────────────────────────
        if "fatigue_accumulation" in types and "tactical_instability" in types:
            evidence = [
                f for f in findings_list
                if f["type"] in ("fatigue_accumulation", "tactical_instability")
            ]
            motifs.append({
                "type":        "fatigue_tactical_drift",
                "severity":    "medium",
                "confidence":  self._motif_confidence(evidence),
                "description": (
                    "Fatigue-associated tactical instability emerging — "
                    "positional discipline declining under accumulated fatigue load."
                ),
            })

        # ── Motif 4: Persistent overload (any single finding type ≥3 times) ───
        for ftype in set(types):
            if ftype in already_flagged:
                continue
            ftype_findings = [f for f in findings_list if f["type"] == ftype]
            if len(ftype_findings) >= 3:
                motifs.append({
                    "type":          f"persistent_{ftype}",
                    "severity":      "medium",
                    "confidence":    self._motif_confidence(ftype_findings),
                    "evidence_count": len(ftype_findings),
                    "description": (
                        f"Persistent {ftype.replace('_', ' ')} pattern: "
                        f"same condition recurred {len(ftype_findings)}x — state worsening, not isolated."
                    ),
                })
                already_flagged.add(ftype)

        # ── Motif 5: Acute severity escalation ───────────────────────────────
        recent3_sev = severities[-3:] if len(severities) >= 3 else []
        if len(recent3_sev) == 3 and all(s in ("high", "critical") for s in recent3_sev):
            evidence = findings_list[-3:]
            motifs.append({
                "type":        "acute_severity_escalation",
                "severity":    "critical",
                "confidence":  self._motif_confidence(evidence),
                "description": (
                    "Acute deterioration trajectory: last three consecutive findings "
                    "all rated high or critical severity."
                ),
            })

        return motifs

    def _motif_confidence(self, findings: list) -> float:
        """
        Compute evidence strength for a detected motif.

        The score combines:
        1. Confidence of supporting findings
        2. Recurrence density
        3. Temporal compactness
        4. Temporal recency weighting

        Older findings contribute less than recent ones via exponential decay,
        preventing stale match states from dominating motif strength.

        Returns
        -------
        float
            Evidence strength in [0.0, 1.0]
        """
        if not findings:
            return 0.5

        import math

        # ─────────────────────────────────────────────
        # A. Base finding strengths
        # ─────────────────────────────────────────────
        strengths = [
            f.get("confidence", 0.5)
            for f in findings
        ]

        # ─────────────────────────────────────────────
        # B. Temporal weighting (recency decay)
        # More recent findings contribute more strongly.
        #
        # weight = exp(-λ * age_minutes)
        #
        # λ = 0.15:
        #   ~0.86 after 1 min
        #   ~0.47 after 5 min
        #   ~0.22 after 10 min
        # ─────────────────────────────────────────────
        minutes = [
            f.get("minute")
            for f in findings
            if f.get("minute") is not None
        ]

        if minutes:
            current_min = max(minutes)

            weights = []
            for f in findings:
                minute = f.get("minute", current_min)
                age = max(0, current_min - minute)

                weight = math.exp(-0.15 * age)
                weights.append(weight)

            weighted_strength = float(
                np.average(strengths, weights=weights)
            )

            # Temporal compactness:
            # smaller time span = stronger motif
            window_span = max(minutes) - min(minutes)

            compactness_score = max(
                0.2,
                1.0 - (window_span / 12.0)
            )

        else:
            # No timing metadata available
            weighted_strength = float(np.mean(strengths))
            compactness_score = 0.6

        # ─────────────────────────────────────────────
        # C. Recurrence density
        # Log-scaled so recurrence grows sublinearly
        #
        # 1 finding  → 0.30
        # 3 findings → 0.60
        # 9 findings → 1.00
        # ─────────────────────────────────────────────
        recurrence_score = min(
            1.0,
            math.log(len(findings) + 1) / math.log(10)
        )

        # ─────────────────────────────────────────────
        # D. Final synthesis
        # Weighted symbolic prior — NOT probability.
        # ─────────────────────────────────────────────
        confidence = (
            0.50 * weighted_strength
            + 0.30 * recurrence_score
            + 0.20 * compactness_score
        )

        return round(
            float(np.clip(confidence, 0.0, 1.0)),
            3,
        )
    # ─────────────────────────────────────────────
    # Layer 3: Progression analysis
    # ─────────────────────────────────────────────

    @staticmethod
    def _slope(values: list) -> Optional[float]:
        """
        Least-squares linear slope over a value sequence.
        Returns None when fewer than _MIN_TREND_SAMPLES values are available.
        """
        if len(values) < _MIN_TREND_SAMPLES:
            return None
        arr = np.array(values, dtype=float)
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])

    @staticmethod
    def _volatility(values: list) -> float:
        """Standard deviation of a value sequence — measures signal instability."""
        if len(values) < 2:
            return 0.0
        return float(np.std(values))
    
    @staticmethod
    def _curvature(values: list) -> float:
        if len(values) < 6:
            return 0.0

        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 2)

        return float(coeffs[0])

    # ─────────────────────────────────────────────
    # Layer 3a: Signal extraction  (pure math, no semantics)
    # ─────────────────────────────────────────────

    @staticmethod
    def _compute_trend_signal(values: list) -> Optional["TrendSignal"]:
        """
        Extract slope, curvature, and volatility from a value sequence.

        Returns None when the sequence is too short for meaningful slope
        estimation (< _MIN_TREND_SAMPLES).  Curvature requires ≥ 6 points;
        falls back to 0.0 otherwise.  Volatility is always computable.

        This method must contain NO semantic labels.  It is the boundary
        between signal processing and symbolic reasoning.
        """
        if len(values) < _MIN_TREND_SAMPLES:
            return None

        arr = np.array(values, dtype=float)
        slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])

        curvature: float
        if len(values) >= 6:
            coeffs   = np.polyfit(np.arange(len(arr)), arr, 2)
            curvature = float(coeffs[0])
        else:
            curvature = 0.0

        volatility = float(np.std(arr)) if len(arr) >= 2 else 0.0

        return TrendSignal(slope=slope, curvature=curvature, volatility=volatility)

    # ─────────────────────────────────────────────
    # Layer 3b: Semantic interpretation  (labels, no math)
    # ─────────────────────────────────────────────

    @staticmethod
    def _interpret_trend(
        metric:            str,
        signal:            Optional["TrendSignal"],
        worsening_thresh:  float,
        recovering_thresh: float,
        flip:              bool = False,
    ) -> "TrendInterpretation":
        """
        Map a TrendSignal to human-readable semantic labels.

        This method must contain NO numerical computation beyond threshold
        comparisons.  All arithmetic belongs in _compute_trend_signal.

        flip=True inverts direction semantics for speed/workload signals,
        where a declining slope means the player is slowing down (bad),
        and an increasing slope means acceleration (potentially good).
        """
        if signal is None:
            return TrendInterpretation(
                direction="insufficient data",
                acceleration="steady",
                volatility="low",
            )

        # Primary direction from first derivative
        if not flip:
            if signal.slope > worsening_thresh:
                direction = "worsening"
            elif signal.slope < recovering_thresh:
                direction = "recovering"
            else:
                direction = "stable"
        else:
            if signal.slope < recovering_thresh:
                direction = "declining"
            elif signal.slope > worsening_thresh:
                direction = "increasing"
            else:
                direction = "stable"

        # Secondary acceleration from second derivative
        curvature_threshold = _CURVATURE_THRESHOLDS.get(metric, 0.01)

    
        if signal.curvature > curvature_threshold:
            acceleration = "accelerating_deterioration"

        elif signal.curvature < -curvature_threshold:
            acceleration = "accelerating_recovery"

        else:
            acceleration = "steady"

        return TrendInterpretation(
            direction=direction,
            acceleration=acceleration,
            volatility=MatchState._volatility_label(metric, signal.volatility),
        )

    @staticmethod
    def _volatility_label(metric: str, std: float) -> str:
        """
        Classify volatility relative to the signal's own scale.

        Uses per-metric thresholds from _VOLATILITY_THRESHOLDS so that
        a std of 0.15 on anomaly score (wide spread) and 0.15 on speed
        (medium spread) are not treated identically.

        Falls back to a safe "low" label for unknown metrics.
        """
        thresholds = _VOLATILITY_THRESHOLDS.get(metric, {"medium": 0.10, "high": 0.25})
        if std >= thresholds["high"]:
            return "high"
        if std >= thresholds["medium"]:
            return "medium"
        return "low"

    # def anomaly_trend(self) -> dict:
    #     """
    #     Direction + volatility of anomaly score evolution over recent windows.
    #     Returns {"direction": str, "volatility": str}
    #     """
    #     with self._lock:
    #         vals = list(self.recent_anomaly_scores)
    #     slope = self._slope(vals)
    #     vol   = self._volatility(vals)
    #     if slope is None:
    #         return {"direction": "insufficient data", "volatility": "low"}
    #     if slope > _ANOMALY_SLOPE_WORSENING:
    #         direction = "worsening"
    #     elif slope < _ANOMALY_SLOPE_RECOVERING:
    #         direction = "recovering"
    #     else:
    #         direction = "stable"
    #     return {"direction": direction, "volatility": self._volatility_label("anomaly", vol)}

    # def recovery_trend(self) -> dict:
    #     """
    #     Direction + volatility of HR recovery rate across recent telemetry windows.
    #     Returns {"direction": str, "volatility": str}
    #     """
    #     with self._lock:
    #         vals = list(self.recent_recovery)
    #     slope = self._slope(vals)
    #     vol   = self._volatility(vals)
    #     if slope is None:
    #         return {"direction": "insufficient data", "volatility": "low"}
    #     if slope > _RECOVERY_SLOPE_WORSENING:
    #         direction = "worsening"
    #     elif slope < _RECOVERY_SLOPE_IMPROVING:
    #         direction = "improving"
    #     else:
    #         direction = "stable"
    #     return {"direction": direction, "volatility": self._volatility_label("recovery", vol)}

    # def workload_trend(self) -> dict:
    #     """
    #     Direction + volatility of movement speed across recent telemetry windows.
    #     Returns {"direction": str, "volatility": str}
    #     """
    #     with self._lock:
    #         vals = list(self.recent_speed)
    #     slope = self._slope(vals)
    #     vol   = self._volatility(vals)
    #     if slope is None:
    #         return {"direction": "insufficient data", "volatility": "low"}
    #     if slope < _SPEED_SLOPE_DECLINING:
    #         direction = "declining"
    #     elif slope > _SPEED_SLOPE_INCREASING:
    #         direction = "increasing"
    #     else:
    #         direction = "stable"
    #     return {"direction": direction, "volatility": self._volatility_label("workload", vol)}

    # ─────────────────────────────────────────────
    # Layer 4: Semantic session summary (LLM interface)
    # ─────────────────────────────────────────────

    def build_semantic_state(self) -> SemanticMatchState:
        """
        Build a structured SemanticMatchState for the current player.

        Returns machine-readable symbolic state — NOT a prose string.
        Formatting into prose for the LLM is the responsibility of
        xai_layer.format_match_state_prompt().

        Replaces build_semantic_summary() (which collapsed structure too early).

        Thread-safe. All mutable state is snapshotted under a single lock
        acquisition; all computation runs outside the lock to minimise
        contention between the inference thread and checkpoint/NLG threads.

        Result is cached on self._cached_semantic_state and invalidated by
        record_finding() / record_telemetry() / record_alert(), so only one
        full computation runs per window instead of the previous 3+.
        """
        # ── Fast path: return cached result if state hasn't changed ──────────
        with self._lock:
            if self._cached_semantic_state is not None:
                return self._cached_semantic_state

            # ── Single snapshot under lock — release before all computation ──
            total_alerts = (
                self.fatigue_alert_count
                + self.workload_alert_count
                + self.anomaly_alert_count
            )
            n_findings = len(self.recent_findings)

            if total_alerts == 0 and n_findings == 0:
                empty = SemanticMatchState(
                    motifs=[],
                    trends={},
                    coupled_states=[],
                    persistent_findings=[],
                    escalation_level="none",
                    risk_breakdown={
                        "motif_risk": 0.0,
                        "coupled_risk": 0.0,
                        "persistence_risk": 0.0,
                        "instability_risk": 0.0,
                        "transition_risk": 0.0,
                        "total_risk": 0.0,
                    },
                )
                self._cached_semantic_state = empty
                return empty

            # Snapshot all mutable collections in one lock pass
            findings_snapshot = list(self.recent_findings)
            anomaly_snap      = list(self.recent_anomaly_scores)
            recovery_snap     = list(self.recent_recovery)
            speed_snap        = list(self.recent_speed)
            peak_score        = self.peak_anomaly_score
            transition_snap   = dict(self.transition_counts)

        # ── All computation outside the lock ─────────────────────────────────

        # Motif detection (pure function of findings_snapshot)
        motifs = self.detect_motifs(findings_snapshot)

        # Persistent findings
        finding_counts: Dict[str, int] = {}
        for f in findings_snapshot:
            if f.get("severity") not in ("high", "critical"):
                continue
            ftype = f.get("type")
            finding_counts[ftype] = finding_counts.get(ftype, 0) + 1

        persistent_map: Dict[str, dict] = {}
        for f in findings_snapshot:
            ftype = f.get("type")
            if (
                f.get("severity") in ("high", "critical")
                and finding_counts.get(ftype, 0) >= 3
            ):
                persistent_map[ftype] = f

        persistent = list(persistent_map.values())

        if False:  # dead block — keeps linter happy about removed second lock
            with self._lock:
                anomaly_snap  = list(self.recent_anomaly_scores)
                recovery_snap = list(self.recent_recovery)
                speed_snap    = list(self.recent_speed)

        anomaly_signal = self._compute_trend_signal(anomaly_snap)
        recovery_signal = self._compute_trend_signal(recovery_snap)
        workload_signal = self._compute_trend_signal(speed_snap)


        anomaly_interp  = self._interpret_trend("anomaly",  anomaly_signal,  _ANOMALY_SLOPE_WORSENING,  _ANOMALY_SLOPE_RECOVERING)
        recovery_interp = self._interpret_trend("recovery", recovery_signal, _RECOVERY_SLOPE_WORSENING, _RECOVERY_SLOPE_IMPROVING)
        workload_interp = self._interpret_trend("workload", workload_signal, _SPEED_SLOPE_INCREASING,   _SPEED_SLOPE_DECLINING,    flip=True)

        # Build the outward-facing trends dict that downstream consumers expect.
        # Carries both the raw signal numbers and the semantic labels so callers
        # can choose how deeply to inspect each layer.

        # def _as_trend_dict(signal: Optional[TrendSignal], interp: TrendInterpretation) -> dict:
        #     base = {
        #         "direction":    interp.direction,
        #         "volatility":   interp.volatility,
        #         "acceleration": interp.acceleration,
        #     }
        #     if signal is not None:
        #         base["slope"]     = round(signal.slope,     4)
        #         base["curvature"] = round(signal.curvature, 4)
        #     return base

        trends = {
            "anomaly": TrendState(
                signal=anomaly_signal,
                interpretation=anomaly_interp,
            ),

            "recovery": TrendState(
                signal=recovery_signal,
                interpretation=recovery_interp,
            ),

            "workload": TrendState(
                signal=workload_signal,
                interpretation=workload_interp,
            ),
            }

        # ── Coupled physiological reasoning ────────────────────────────────
        # Each condition is a first-class CoupledState, not a loose dict.
        # Transition memory boosts confidence when the same physiological
        # pathway has been empirically observed in prior findings this match.

        coupled_states: List[CoupledState] = []

        workload_dir = trends["workload"].interpretation.direction
        recovery_dir = trends["recovery"].interpretation.direction
        anomaly_dir  = trends["anomaly"].interpretation.direction

        # ── Helper: apply transition-memory boost ─────────────────────────────
        def _with_transition_boost(base_confidence: float, *pairs: Tuple[str, str]) -> float:
            """
            Boost confidence by up to 0.15 per observed transition pair.
            Uses the transition snapshot captured before the lock was released
            so this runs safely outside the lock without touching self state.
            """
            def _tp(a: str, b: str) -> float:
                count_ab = transition_snap.get((a, b), 0)
                count_a  = sum(v for (fa, _), v in transition_snap.items() if fa == a)
                if count_a == 0:
                    return 0.0
                return round(count_ab / count_a, 3)

            probs = [_tp(a, b) for a, b in pairs]
            boost = 0.15 * max(probs, default=0.0)
            return round(float(np.clip(base_confidence + boost, 0.0, 1.0)), 3)

        # High workload + worsening recovery
        if workload_dir == "declining" and recovery_dir == "worsening":
            base = self._coupled_confidence(trends, ["workload", "recovery"])
            conf = _with_transition_boost(
                base,
                ("locomotor_overload", "recovery_degradation"),
                ("cardiovascular_overload", "recovery_degradation"),
            )
            coupled_states.append(CoupledState(
                state_type="fatigue_accumulation_under_load",
                severity="high",
                confidence=conf,
                supporting_trends=["workload", "recovery"],
                description=(
                    "Sustained workload decline combined with worsening "
                    "recovery dynamics suggests accumulating fatigue."
                ),
            ))

        # General instability across physiology + anomaly system
        if anomaly_dir == "worsening" and recovery_dir == "worsening":
            base = self._coupled_confidence(trends, ["anomaly", "recovery"])
            conf = _with_transition_boost(
                base,
                ("cardiovascular_overload", "recovery_degradation"),
                ("locomotor_overload", "cardiovascular_overload"),
            )
            coupled_states.append(CoupledState(
                state_type="systemic_instability",
                severity="critical",
                confidence=conf,
                supporting_trends=["anomaly", "recovery"],
                description=(
                    "Concurrent anomaly escalation and recovery degradation "
                    "indicate systemic instability."
                ),
            ))

        # High volatility + worsening recovery
        if trends["recovery"].interpretation.volatility == "high" and recovery_dir == "worsening":
            base = self._coupled_confidence(trends, ["recovery"])
            conf = _with_transition_boost(
                base,
                ("recovery_degradation", "cardiovascular_overload"),
            )
            coupled_states.append(CoupledState(
                state_type="unstable_recovery_response",
                severity="medium",
                confidence=conf,
                supporting_trends=["recovery"],
                description="Recovery trajectory is both degrading and unstable.",
            ))

        # Accelerating anomaly growth
        if (
                    anomaly_dir == "worsening"
                    and trends["anomaly"].interpretation.acceleration
                    == "accelerating_deterioration"
                ):
            base = self._coupled_confidence(trends, ["anomaly"])
            conf = _with_transition_boost(
                base,
                ("locomotor_overload", "cardiovascular_overload"),
            )
            coupled_states.append(CoupledState(
                state_type="accelerating_instability",
                severity="high",
                confidence=conf,
                supporting_trends=["anomaly"],
                description="Anomaly progression is accelerating over recent windows.",
            ))

        # ── Escalation level — weighted risk accumulation ─────────────────────
        # Each evidence layer contributes a bounded risk component.
        # Final score maps to an escalation tier without nested if-chains.
        # Weights are intentional symbolic priors; calibrate from outcomes later.

        # Component 1: motif risk (named high-risk motifs score 4; others 1)
        _MOTIF_RISK_WEIGHTS = {
            "sprint_collapse_progression":        4,
            "persistent_cardiovascular_overload": 3,
            "fatigue_tactical_drift":             3,
            "acute_severity_escalation":          4,
        }
        raw_motif_score = sum(
            _MOTIF_RISK_WEIGHTS.get(m.get("type"), 1) for m in motifs
        )
        motif_risk = min(1.0, raw_motif_score / 8.0)

        # Component 2: coupled-state risk (severity-weighted)
        _COUPLED_SEVERITY_WEIGHT = {"critical": 1.0, "high": 0.6, "medium": 0.3, "low": 0.1}
        coupled_risk = min(1.0, sum(
            _COUPLED_SEVERITY_WEIGHT.get(c.severity, 0.1)
            for c in coupled_states
        ) / 2.0)

        # Component 3: persistence risk
        persistence_risk = min(1.0, len(persistent) / 5.0)

        # Component 4: trend instability risk
        worsening_trends = sum(
            1 for t in trends.values()
            if t.interpretation.direction in ("worsening", "declining")
        )
        instability_risk = min(1.0, worsening_trends / 3.0)

        # Component 5: transition-evidence risk
        # If the most dangerous chain (locomotor → cardiovascular → recovery)
        # has high empirical probability, escalate risk accordingly.
        # Uses transition_snap captured before the lock was released.
        def _tp_snap(a: str, b: str) -> float:
            count_ab = transition_snap.get((a, b), 0)
            count_a  = sum(v for (fa, _), v in transition_snap.items() if fa == a)
            return round(count_ab / count_a, 3) if count_a else 0.0

        p_loco_to_cv     = _tp_snap("locomotor_overload",    "cardiovascular_overload")
        p_cv_to_recovery = _tp_snap("cardiovascular_overload", "recovery_degradation")
        transition_risk  = min(1.0, (p_loco_to_cv + p_cv_to_recovery) / 2.0)

        risk_score = (
            0.25 * motif_risk
            + 0.25 * coupled_risk
            + 0.20 * persistence_risk
            + 0.15 * instability_risk
            + 0.15 * transition_risk
        )

        critical_evidence = (
            any(m.get("severity") == "critical" for m in motifs)
            or any(c.severity == "critical" for c in coupled_states)
            or any(f.get("severity") == "critical" for f in persistent)
        )

        high_evidence = (
            sum(
                1 for f in persistent
                if f.get("severity") in ("high", "critical")
            ) >= 2
        )

        if critical_evidence:
            escalation_level = "critical"

        elif high_evidence or risk_score >= 0.45:
            escalation_level = "high"

        elif risk_score >= 0.20:
            escalation_level = "elevated"

        else:
            escalation_level = "normal"

        result = SemanticMatchState(
            motifs=motifs,
            trends=trends,
            persistent_findings=list(persistent),
            escalation_level=escalation_level,
            coupled_states=coupled_states,
            risk_breakdown={
                "motif_risk":        round(motif_risk, 3),
                "coupled_risk":      round(coupled_risk, 3),
                "persistence_risk":  round(persistence_risk, 3),
                "instability_risk":  round(instability_risk, 3),
                "transition_risk":   round(transition_risk, 3),
                "total_risk":        round(risk_score, 3),
            },
        )

        # Store in cache — guarded write so concurrent callers see the same object
        with self._lock:
            if self._cached_semantic_state is None:
                self._cached_semantic_state = result

        return result


    def _coupled_confidence(
            self,
            trends: dict,
            involved: list[str],
        ) -> float:
        """
        Compute confidence for a coupled physiological state.

        Confidence depends on:
        1. Directional agreement
        2. Slope magnitude
        3. Curvature magnitude
        4. Volatility penalty

        The goal is to estimate how strongly the involved systems
        are jointly deteriorating — not merely whether labels match.
        """

        involved_trends = [
            trends[name]
            for name in involved
            if name in trends
        ]

        if not involved_trends:
            return 0.5

        # ── Directional agreement ─────────────────────────────
        directional_strength = (
            sum(
                1
                for t in involved_trends
                if t.interpretation.direction in (
                    "worsening",
                    "declining",
                )
            )
            / len(involved_trends)
        )

        # ── Magnitude of degradation ──────────────────────────
        slope_strength = float(np.mean([
            abs(t.signal.slope if t.signal else 0.0)
            for t in involved_trends
        ]))

        # ── Acceleration / curvature intensity ───────────────
        curve_strength = float(np.mean([
            abs(t.signal.curvature if t.signal else 0.0)
            for t in involved_trends
        ]))

        # Normalize to avoid runaway values
        slope_strength = min(slope_strength * 5.0, 1.0)
        curve_strength = min(curve_strength * 20.0, 1.0)

        # ── Volatility penalty ────────────────────────────────
        vols = [
            t.interpretation.volatility
            for t in involved_trends
        ]

        volatility_penalty = (
            0.15 if "high" in vols else
            0.05 if "medium" in vols else
            0.0
        )

        # ── Final synthesis ───────────────────────────────────
        conf = (
            0.45
            + 0.30 * directional_strength
            + 0.15 * slope_strength
            + 0.10 * curve_strength
            - volatility_penalty
        )

        return round(float(np.clip(conf, 0.0, 1.0)), 3)
    # ─────────────────────────────────────────────
    # Legacy interface — preserved for backward compat
    # ─────────────────────────────────────────────

    @property
    def mean_anomaly_score(self) -> float:
        with self._lock:
            if self._anomaly_score_count == 0:
                return 0.0
            return self._anomaly_score_sum / self._anomaly_score_count
        

    def build_semantic_summary(self) -> "SemanticMatchState":
        """
        Deprecated — replaced by build_semantic_state().
        Forwarded so any un-updated callers continue to work.
        """
        logger.warning(
            "build_semantic_summary() is deprecated. "
            "Switch callers to build_semantic_state(). Forwarding."
        )
        return self.build_semantic_state()

    def build_llm_context(self) -> str:
        """
        Deprecated — replaced by build_semantic_state() + xai_layer.format_match_state_prompt().
        Forwarded so any un-updated callers continue to work.
        """
        logger.warning(
            "build_llm_context() is deprecated. "
            "Switch callers to build_semantic_state(). Forwarding."
        )
        # Import here to avoid circular dependency at module load
        from explainability.xai_layer import format_match_state_prompt  # noqa: F401
        return format_match_state_prompt(self.build_semantic_state())

    # ─────────────────────────────────────────────
    # Persistence — serialization
    # ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialize this MatchState to a plain JSON-safe dict.

        Covers every mutable field. The _lock is excluded (not serializable,
        not needed on restore). deque → list; transition_counts tuple-keys →
        "from|to" strings (JSON requires string keys); last_alert_ts →
        ISO-8601 string.
        """
        ts = self.last_alert_ts
        return {
            "_schema_version": _CHECKPOINT_SCHEMA_VERSION,
            # Identity
            "player_id":   self.player_id,
            "player_name": self.player_name,
            "position":    self.position,
            "match_id":    self.match_id,
            # Alert counters
            "fatigue_alert_count":  self.fatigue_alert_count,
            "workload_alert_count": self.workload_alert_count,
            "anomaly_alert_count":  self.anomaly_alert_count,
            # Persistence tracking
            "consecutive_alerts":    self.consecutive_alerts,
            "last_alert_type":       self.last_alert_type,
            "last_alert_ts":         ts.isoformat() if ts is not None else None,
            "first_alert_elapsed_s": self.first_alert_elapsed_s,
            "last_episode_id":       getattr(self, "last_episode_id", None),
            # Online anomaly stats
            "_anomaly_score_sum":   self._anomaly_score_sum,
            "_anomaly_score_count": self._anomaly_score_count,
            "peak_anomaly_score":   self.peak_anomaly_score,
            # Telemetry aggregates
            "peak_hr_bpm":  self.peak_hr_bpm,
            "sprint_count": self.sprint_count,
            # Layer 1 — rolling deques
            "recent_hr":             list(self.recent_hr),
            "recent_speed":          list(self.recent_speed),
            "recent_anomaly_scores": list(self.recent_anomaly_scores),
            "recent_recovery":       list(self.recent_recovery),
            # Layer 2 — finding history + transition memory
            "recent_findings":   list(self.recent_findings),
            "transition_counts": {
                f"{a}|{b}": v for (a, b), v in self.transition_counts.items()
            },
            "episodes":             self._episodes_to_list(),
            "trend_summaries":      self.trend_summaries,
            "recent_context":       self.recent_context,
            "intervention_history": self.intervention_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MatchState":
        """
        Reconstruct a MatchState from a serialized dict.

        Raises ValueError on unknown schema version so callers can discard
        cleanly rather than silently loading corrupt state.
        """
        version = data.get("_schema_version", 0)
        if version != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unrecognised checkpoint schema version {version!r} "
                f"(expected {_CHECKPOINT_SCHEMA_VERSION})"
            )

        def _load_deque(lst: list, maxlen: int = 50) -> deque:
            d: deque = deque(maxlen=maxlen)
            d.extend(lst)
            return d

        ts_str = data.get("last_alert_ts")
        last_alert_ts = (
            datetime.fromisoformat(ts_str) if ts_str is not None else None
        )

        state = cls(
            player_id=data["player_id"],
            player_name=data["player_name"],
            position=data["position"],
            match_id=data["match_id"],
        )

        # Alert counters
        state.fatigue_alert_count  = data.get("fatigue_alert_count", 0)
        state.workload_alert_count = data.get("workload_alert_count", 0)
        state.anomaly_alert_count  = data.get("anomaly_alert_count", 0)

        # Persistence tracking
        state.consecutive_alerts    = data.get("consecutive_alerts", 0)
        state.last_alert_type       = data.get("last_alert_type", "")
        state.last_alert_ts         = last_alert_ts
        state.first_alert_elapsed_s = data.get("first_alert_elapsed_s")

        if hasattr(state, "last_episode_id"):
            state.last_episode_id = data.get("last_episode_id")

        # Online anomaly stats
        state._anomaly_score_sum   = data.get("_anomaly_score_sum", 0.0)
        state._anomaly_score_count = data.get("_anomaly_score_count", 0)
        state.peak_anomaly_score   = data.get("peak_anomaly_score", 0.0)

        # Telemetry aggregates
        state.peak_hr_bpm  = data.get("peak_hr_bpm", 0.0)
        state.sprint_count = data.get("sprint_count", 0)

        # Layer 1 — rolling deques (maxlen matches dataclass defaults)
        state.recent_hr             = _load_deque(data.get("recent_hr", []))
        state.recent_speed          = _load_deque(data.get("recent_speed", []))
        state.recent_anomaly_scores = _load_deque(data.get("recent_anomaly_scores", []))
        state.recent_recovery       = _load_deque(data.get("recent_recovery", []))

        # Layer 2
        state.recent_findings = _load_deque(data.get("recent_findings", []))
        state.transition_counts = {
            (a, b): v
            for raw_key, v in data.get("transition_counts", {}).items()
            for a, b in [raw_key.split("|", 1)]
        }
        state.episodes             = MatchState._episodes_from_list(data.get("episodes", []))
        state.trend_summaries      = data.get("trend_summaries", {})
        state.recent_context       = data.get("recent_context", "")
        state.intervention_history = data.get("intervention_history", [])

        return state

    def _infer_finding_state(self, current: Dict, previous: Optional[Dict]) -> str:
        """
        Infer lifecycle state of a finding from temporal evolution.
        """
        if previous is None:
            return "active"

        curr_conf = current.get("confidence", 0.5)
        prev_conf = previous.get("confidence", 0.5)

        curr_sev = current.get("severity", "low")
        prev_sev = previous.get("severity", "low")

        sev_order = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "critical": 3,
        }

        if sev_order[curr_sev] > sev_order[prev_sev]:
            return "escalating"

        if curr_conf > prev_conf + 0.10:
            return "escalating"

        if curr_conf < prev_conf - 0.10:
            return "resolving"

        if curr_sev == prev_sev and abs(curr_conf - prev_conf) < 0.05:
            return "stabilizing"

        return "active"
    
    def log_intervention(self, description: str) -> None:
        """
        Record a coach or system intervention.
 
        Examples:
            "substitution requested min~72"
            "load reduction instruction issued min~55"
            "player repositioned min~38"
 
        Thread-safe. The intervention is stored in intervention_history
        and also appended to the ongoing episode (if any).
        """
        from analysis.episodic_context import PlayerEpisode as _PE
        with self._lock:
            self.intervention_history.append(description)
            # Attach to the current ongoing episode
            for ep in reversed(self.episodes):
                if isinstance(ep, _PE) and ep.is_ongoing():
                    ep.interventions.append(description)
                    break
 
    def refresh_episodes(
        self,
        current_minute: Optional[int] = None,
        current_escalation: str = "normal",
        semantic_state: Optional["SemanticMatchState"] = None,
    ) -> None:
        """
        Re-segment the finding history into episodes and rebuild the
        compressed temporal summaries.
 
        Call once per window AFTER record_finding() and record_telemetry()
        have been called, so the episode list reflects the latest state.
 
        Parameters
        ----------
        current_minute      : current match minute.
        current_escalation  : escalation level string (deprecated param kept for
                              compatibility — ignored when semantic_state is provided).
        semantic_state      : pre-computed SemanticMatchState from build_semantic_state().
                              Pass this in to avoid a redundant recomputation; if None,
                              build_semantic_state() is called internally.
 
        Thread-safe: takes a consistent snapshot under the lock, then
        runs the compressor outside the lock.
        """
        from analysis.episodic_context import (
            TemporalContextCompressor,
            PlayerEpisode as _PE,
        )
 
        with self._lock:
            findings_snap   = list(self.recent_findings)
            anomaly_snap    = list(self.recent_anomaly_scores)
            interventions   = list(self.intervention_history)
            existing_eps    = [
                ep for ep in self.episodes
                if isinstance(ep, _PE)
            ]
 
        compressor = TemporalContextCompressor()
 
        # Rebuild episodes from the current finding snapshot.
        updated_episodes = compressor.build_episodes_from_findings(
            findings_snapshot=findings_snap,
            anomaly_scores_snapshot=anomaly_snap,
            current_minute=current_minute,
            existing_episodes=existing_eps,
        )
 
        # Use the caller-supplied state if available (avoids 2nd polyfit pass).
        # Fall back to building it when called standalone.
        if semantic_state is None:
            semantic_state = self.build_semantic_state()

        trend_sums: Dict[str, str] = {}
        for axis, ts in semantic_state.trends.items():
            if hasattr(ts, "interpretation"):
                trend_sums[axis] = ts.interpretation.direction
            elif isinstance(ts, dict):
                trend_sums[axis] = ts.get("direction", "stable")
 
        # Build recent context string (symbol-only, no raw telemetry)
        recent_ctx = compressor.build_recent_context(
            findings_snapshot=findings_snap,
            current_minute=current_minute,
        )
 
        with self._lock:
            prev_episodes = list(self.episodes)
            self.episodes             = updated_episodes
            self.trend_summaries      = trend_sums
            self.recent_context       = recent_ctx

        # Persist newly closed episodes to EpisodeStore.
        # A closed episode is one where is_ongoing() is False and it was not
        # already in prev_episodes (i.e. it just transitioned closed this window).
        self._persist_newly_closed_episodes(
            prev_episodes=prev_episodes,
            updated_episodes=updated_episodes,
        )

    def _persist_newly_closed_episodes(
        self,
        prev_episodes: list,
        updated_episodes: list,
    ) -> None:
        """
        Persist closed episodes to EpisodeStore exactly once each.

        Uses self._persisted_episode_ids as a write-once guard: an episode is
        written the first time it appears closed (is_ongoing() == False) and
        never written again.  This replaces the previous prev_open_ids logic
        which was broken because build_episodes_from_findings() rebuilds the
        list from scratch each window, so prev_episodes never carried live
        is_ongoing() state — prev_open_ids was always empty and nothing wrote.
        """
        try:
            from config.redis_client import EpisodeStore
        except ImportError:
            logger.warning(
            "REDIS IS NOT IMPORTED FROM REDIS_CLIENT")
            return  # Redis not available in test/offline environment

        from analysis.episodic_context import PlayerEpisode as _PE
        import time as _time

        store = EpisodeStore()
        for ep in updated_episodes:
            if not isinstance(ep, _PE):
                continue
            if ep.is_ongoing():
                continue  # still active; will persist when it closes

            episode_id = f"{self.match_id}_{ep.episode_index}"
            if episode_id in self._persisted_episode_ids:
                continue  # already written, skip

            # Skip single-window flickers — they carry no meaningful signal
            # and would bloat the episode index with noise.
            if ep.persistence_duration < _MIN_EPISODE_PERSIST_WINDOWS:
                logger.debug(
                    "EpisodeStore: skipping short ep=%s player=%d "
                    "(persistence=%d < %d)",
                    episode_id, self.player_id,
                    ep.persistence_duration, _MIN_EPISODE_PERSIST_WINDOWS,
                )
                continue

            # Use end_minute as score when available (including minute 0 = match start).
            # Do NOT use truthiness check — end_minute=0 is valid and must not fall
            # back to wall-clock time, which would corrupt ZRANGE ordering by mixing
            # match-minute scores (~0-90) with unix timestamps (~1.7 billion).
            score = (
                float(ep.end_minute)
                if ep.end_minute is not None
                else _time.time()
            )

            store.persist_episode_with_meta(   # was: persist_episode
                player_id=self.player_id,
                episode_id=episode_id,
                episode_dict=ep.to_dict(),
                score=score,
            )
            self._persisted_episode_ids.add(episode_id)
            logger.debug(
                "EpisodeStore: persisted ep=%s player=%d score=%.1f",
                episode_id, self.player_id, score,
            )

    def build_compressed_context(
        self,
        current_finding_types: Optional[List[str]] = None,
        current_minute: Optional[int] = None,
        semantic_state: Optional["SemanticMatchState"] = None,
    ) -> "CompressedTemporalContext":
        """
        Build a CompressedTemporalContext from all episodic memory.
 
        Call after refresh_episodes() to get the LLM-ready compressed context.
 
        Parameters
        ----------
        current_finding_types : finding types from the current anomaly window.
        current_minute        : current match minute.
        semantic_state        : pre-computed SemanticMatchState from build_semantic_state().
                                Pass this in to avoid a redundant recomputation (avoids
                                the 3rd polyfit pass per window); if None, build_semantic_state()
                                is called internally.
 
        Returns
        -------
        CompressedTemporalContext
            Token-efficient, LLM-ready. Zero raw telemetry.
        """
        from analysis.episodic_context import (
            TemporalContextCompressor,
            PlayerEpisode as _PE,
        )
 
        with self._lock:
            episodes      = [ep for ep in self.episodes if isinstance(ep, _PE)]
            trend_sums    = dict(self.trend_summaries)
            findings_snap = list(self.recent_findings)
            interventions = list(self.intervention_history)
            recent_ctx    = self.recent_context
 
        # Use caller-supplied state to avoid redundant polyfit.
        if semantic_state is None:
            semantic_state = self.build_semantic_state()
        escalation = semantic_state.escalation_level

        # ── Retrieve cross-match historical episodes ───────────────────────────
        historical_episodes: List[dict] = []

        if self._episode_store is not None:
            try:
                current_trend = "stable"

                if semantic_state is not None:
                    anomaly_trend = semantic_state.trends.get("anomaly")

                    # TrendState object/dataclass, not dict
                    current_trend = getattr(
                        anomaly_trend,
                        "direction",
                        "stable",
                    )

                historical_episodes = self._episode_store.retrieve_relevant(
                    player_id=self.player_id,
                    current_findings=current_finding_types or [],
                    current_trend=current_trend,
                    top_k=3,
                    candidate_pool=20,
                )

                logger.debug(
                    "build_compressed_context: player=%d retrieved=%d historical episodes",
                    self.player_id,
                    len(historical_episodes),
                )

            except Exception as exc:
                logger.warning(
                    "build_compressed_context: episode retrieval failed player=%d: %s",
                    self.player_id,
                    exc,
                )

        compressor = TemporalContextCompressor()

        ctx = compressor.compress(
            episodes=episodes,
            trend_summaries=trend_sums,
            recent_findings=findings_snap,
            intervention_history=interventions,
            current_minute=current_minute,
            current_escalation=escalation,
            current_finding_types=current_finding_types,
            historical_episodes=historical_episodes,
        )

        # Retrieve cross-match historical episodes from EpisodeStore and
        # attach them to the context BEFORE prompt construction.
        # The symbolic layer (EpisodeStore.retrieve_relevant) does the
        # reasoning; the LLM only receives the conclusions.
        current_trend = next(
            (v for v in trend_sums.values() if v not in ("stable", "insufficient data")),
            "stable",
        )
        # try:
        #     from config.redis_client import EpisodeStore as _ES
        #     hist = _ES().retrieve_relevant(
        #         player_id=self.player_id,
        #         current_findings=current_finding_types or [],
        #         current_trend=current_trend,
        #         top_k=3,
        #     )
        #     from dataclasses import replace as _replace
        #     ctx = _replace(ctx, historical_episodes=hist)
        # except Exception as exc:
        #     logger.warning(
        #         "EpisodeStore retrieval failed for player=%d (degrading to current-match context only): %s",
        #         self.player_id, exc,
        #     )

        return ctx
 
    # ─────────────────────────────────────────────
    # Serialization additions (extend to_dict / from_dict)
    # ─────────────────────────────────────────────
 
    def _episodes_to_list(self) -> list:
        """Serialize episodes for checkpointing."""
        from analysis.episodic_context import PlayerEpisode as _PE
        result = []
        for ep in self.episodes:
            if isinstance(ep, _PE):
                result.append(ep.to_dict())
        return result
 
    @staticmethod
    def _episodes_from_list(raw: list) -> list:
        """Deserialize episodes from checkpoint."""
        from analysis.episodic_context import PlayerEpisode as _PE
        result = []
        for d in raw:
            try:
                result.append(_PE.from_dict(d))
            except Exception as exc:
                logger.warning("Could not deserialize PlayerEpisode: %s", exc)
        return result
    

class MatchStateManager:
    """
    Registry keyed by (player_id, match_id).
    Explicit lifecycle via start_match() / end_match_and_wipe().
    Never keyed by player_id alone — that causes cross-match leakage.

    Persistence
    ───────────
    Checkpointing is throttled to avoid serializing the full state on every
    window.  Three triggers are supported, all configurable:

    * Periodic: flush if at least ``checkpoint_interval_s`` seconds have
      elapsed since the last successful write.
    * Window-count: flush every ``checkpoint_every_n`` windows regardless of
      elapsed time (guards against very-low-frequency matches).
    * Alert: flush immediately whenever ``mark_dirty(is_alert=True)`` is
      called, regardless of throttle state.

    All mutations and checkpoint operations share a single RLock so
    concurrent inference and async NLG paths cannot produce a partially
    updated snapshot.

    Storage is delegated to a ``CheckpointStore`` instance so the backend
    (filesystem, Redis, S3) can be swapped without touching this class or
    the orchestrator.
    """

    def __init__(
        self,
        store: Optional[CheckpointStore] = None,
        checkpoint_interval_s: float = 30.0,
        checkpoint_every_n:    int   = 10,
        checkpoint_on_alert:   bool  = True,
    ) -> None:
        self._states: Dict[Tuple[int, str], MatchState] = {}
        self._active_match_id:  Optional[str]      = None
        self._match_start_ts:   Optional[datetime]  = None  # stamped in checkpoint metadata
        self._lock = RLock()   # RLock: checkpoint() may be called while already holding lock

        # Storage backend — defaults to JSON files in models/
        self._store: CheckpointStore = store or JsonFileCheckpointStore()

        # Throttle state
        self._checkpoint_interval_s = checkpoint_interval_s
        self._checkpoint_every_n    = checkpoint_every_n
        self._checkpoint_on_alert   = checkpoint_on_alert
        self._last_checkpoint_ts:  float = 0.0
        self._window_count:        int   = 0   # incremented by mark_dirty()
        self._dirty:               bool  = False

    # ─────────────────────────────────────────────
    # Match lifecycle
    # ─────────────────────────────────────────────

    def start_match(self, match_id: str) -> None:
        """Call at kickoff. Drops all state from prior matches."""
        logger.info("MatchStateManager: starting match %s", match_id)
        with self._lock:
            stale = [k for k in self._states if k[1] != match_id]
            for k in stale:
                del self._states[k]
            self._active_match_id = match_id
            self._match_start_ts  = datetime.now(tz=timezone.utc)
            self._dirty           = False
            self._window_count    = 0
            self._last_checkpoint_ts = 0.0

    def end_match(self, match_id: str) -> None:
        """
        Free per-match memory.

        Prefer end_match_and_wipe() in production so the checkpoint file is
        also removed and cannot contaminate the next match.
        """
        logger.info("MatchStateManager: ending match %s", match_id)
        with self._lock:
            keys = [k for k in self._states if k[1] == match_id]
            for k in keys:
                del self._states[k]
            if self._active_match_id == match_id:
                self._active_match_id = None
                self._match_start_ts  = None
                self._dirty           = False
                self._window_count    = 0

    def end_match_and_wipe(self, match_id: str) -> None:
        """
        Flush a final checkpoint, free all per-match memory, then delete the
        checkpoint so stale runtime state cannot leak into the next match.

        Always use this instead of end_match() in production.
        """
        with self._lock:
            if self._dirty:
                self._do_checkpoint()   # final flush before wipe
        self.end_match(match_id)
        self._store.delete(_CHECKPOINT_KEY)
        logger.info("end_match_and_wipe: checkpoint wiped for match %s", match_id)

    # ─────────────────────────────────────────────
    # State registry
    # ─────────────────────────────────────────────

    def get_or_create(
        self,
        player_id: int,
        player_name: str,
        position: str,
        match_id: Optional[str] = None,
    ) -> MatchState:
        with self._lock:
            if not match_id:
                raise ValueError("match_id is required")
            key = (player_id, match_id)
            if key not in self._states:
                self._states[key] = MatchState(
                    player_id=player_id,
                    player_name=player_name,
                    position=position,
                    match_id=match_id,
                )
            return self._states[key]

    # ─────────────────────────────────────────────
    # Dirty flagging — call from orchestrator
    # ─────────────────────────────────────────────

    def mark_dirty(self, is_alert: bool = False) -> None:
        """
        Signal that state has changed since the last checkpoint.

        Call once per processed window.  The orchestrator decides whether
        this window triggered an alert so the store can flush immediately
        when ``checkpoint_on_alert`` is set.

        Parameters
        ----------
        is_alert : True when the current window produced an alert —
                   triggers an immediate checkpoint regardless of throttle.
                   Must NOT be set to True just to force a write; use
                   force_checkpoint() for that.
        """
        with self._lock:
            self._dirty = True
            self._window_count += 1
            flush_now = (
                (is_alert and self._checkpoint_on_alert)
                or (self._window_count % self._checkpoint_every_n == 0)
            )
        if flush_now:
            self.checkpoint_if_due(force=True)
        else:
            self.checkpoint_if_due()

    def force_checkpoint(self, reason: str = "explicit") -> None:
        """
        Unconditionally flush the current state to the checkpoint store.

        Use this whenever you need a guaranteed write without coupling to
        alert semantics or window-count cadence — e.g. at match start to
        verify store connectivity, or after a significant lifecycle event.

        Unlike mark_dirty(is_alert=True) this method:
          • does NOT increment the window counter
          • does NOT affect the alert-triggered persistence path
          • does NOT alter checkpoint cadence for subsequent windows
          • marks state dirty so the write is never skipped if state is clean

        Parameters
        ----------
        reason : free-form label written to the debug log so forced writes
                 are always identifiable in production logs.
        """
        with self._lock:
            self._dirty = True   # ensure _do_checkpoint won't no-op
            logger.debug("force_checkpoint: reason=%r", reason)
            self._do_checkpoint()

    def checkpoint_if_due(self, force: bool = False) -> None:
        """
        Write a checkpoint if state is dirty and the throttle interval has
        elapsed (or ``force=True``).

        Safe to call from any thread.  No-op when state is clean.
        """
        with self._lock:
            if not self._dirty:
                return
            now = time.monotonic()
            if not force and (now - self._last_checkpoint_ts) < self._checkpoint_interval_s:
                return
            self._do_checkpoint()

    # ─────────────────────────────────────────────
    # Checkpoint write (lock must already be held)
    # ─────────────────────────────────────────────

    def _do_checkpoint(self) -> None:
        """
        Serialize and persist the current match state.

        Must be called with self._lock already held.  Takes a consistent
        snapshot of all per-player states under the lock before serializing,
        so concurrent mutations cannot produce a partial write.

        Checkpoint envelope adds match_start_ts and pipeline_version to the
        existing match_id + saved_at fields so restoration can detect
        same-match_id / different-session collisions.
        """
        match_id = self._active_match_id
        if match_id is None:
            return

        # Take consistent snapshot under the lock
        snapshot = {
            key: state.to_dict()
            for key, state in self._states.items()
            if key[1] == match_id
        }

        payload = {
            "match_id":       match_id,
            "match_start_ts": (
                self._match_start_ts.isoformat()
                if self._match_start_ts is not None else None
            ),
            "saved_at":       datetime.now(tz=timezone.utc).isoformat(),
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "players": {
                str(pid): state_dict
                for (pid, _mid), state_dict in snapshot.items()
            },
        }

        self._store.save(_CHECKPOINT_KEY, payload)
        self._last_checkpoint_ts = time.monotonic()
        self._dirty = False
        logger.debug(
            "_do_checkpoint: %d player(s) persisted for match %s",
            len(payload["players"]), match_id,
        )

    # ─────────────────────────────────────────────
    # Restore on boot / crash-recovery
    # ─────────────────────────────────────────────

    def restore_if_active(
        self,
        match_id: str,
        max_age_hours: float = MAX_CHECKPOINT_AGE_HOURS,
    ) -> int:
        """
        Load a previous checkpoint if it belongs to match_id.

        Call immediately after start_match() on boot or crash-recovery.
        Returns the number of player states restored (0 if nothing loaded).

        Discard conditions — store entry is deleted, manager starts clean:

        * No stored checkpoint found
        * ``match_id`` does not match the checkpoint's stored match_id
        * ``match_start_ts`` differs — same match_id, different session
          (crash then restart with reused id, or replayed telemetry)
        * Checkpoint older than ``max_age_hours``
        * Schema version mismatch (fields changed between deployments)
        """
        payload = self._store.load(_CHECKPOINT_KEY)
        if payload is None:
            logger.debug("restore: no checkpoint found")
            return 0

        # ── match_id guard ────────────────────────────────────────────────────
        saved_match_id = payload.get("match_id")
        if saved_match_id != match_id:
            logger.info(
                "restore: checkpoint match_id=%r != active=%r — discarding",
                saved_match_id, match_id,
            )
            self._store.delete(_CHECKPOINT_KEY)
            return 0

        # ── match_start_ts guard (same match_id, different session) ───────────
        saved_start_str  = payload.get("match_start_ts")
        current_start_ts = self._match_start_ts
        if saved_start_str is not None and current_start_ts is not None:
            try:
                saved_start = datetime.fromisoformat(saved_start_str)
                # Allow a 5-second grace window for clock imprecision on restart
                delta_s = abs((current_start_ts - saved_start).total_seconds())
                if delta_s > 5:
                    logger.info(
                        "restore: match_start_ts mismatch (delta=%.1fs) — "
                        "same match_id, different session — discarding",
                        delta_s,
                    )
                    self._store.delete(_CHECKPOINT_KEY)
                    return 0
            except Exception:
                logger.warning(
                    "restore: could not parse match_start_ts %r — skipping session guard",
                    saved_start_str,
                )

        # ── age guard ─────────────────────────────────────────────────────────
        saved_at_str = payload.get("saved_at")
        if saved_at_str is not None:
            try:
                saved_at  = datetime.fromisoformat(saved_at_str)
                age_hours = (
                    datetime.now(tz=timezone.utc) - saved_at
                ).total_seconds() / 3600.0
                if age_hours > max_age_hours:
                    logger.warning(
                        "restore: checkpoint age %.1fh > limit %.1fh — discarding",
                        age_hours, max_age_hours,
                    )
                    self._store.delete(_CHECKPOINT_KEY)
                    return 0
            except Exception:
                logger.warning(
                    "restore: could not parse saved_at %r — skipping age check",
                    saved_at_str,
                )

        # ── schema version guard ──────────────────────────────────────────────
        saved_schema = payload.get("schema_version", 0)
        if saved_schema != _CHECKPOINT_SCHEMA_VERSION:
            logger.warning(
                "restore: schema_version=%r != expected=%r — discarding",
                saved_schema, _CHECKPOINT_SCHEMA_VERSION,
            )
            self._store.delete(_CHECKPOINT_KEY)
            return 0

        # ── deserialize ───────────────────────────────────────────────────────
        players_raw: dict = payload.get("players", {})
        restored = 0

        with self._lock:
            for pid_str, state_data in players_raw.items():
                try:
                    state = MatchState.from_dict(state_data)
                    key = (state.player_id, match_id)
                    self._states[key] = state
                    restored += 1
                    logger.info(
                        "restore: player=%d match=%s findings=%d hr_deque=%d",
                        state.player_id, match_id,
                        len(state.recent_findings),
                        len(state.recent_hr),
                    )
                except ValueError as exc:
                    logger.warning("restore: skipping player %s — %s", pid_str, exc)
                except Exception:
                    logger.exception("restore: failed to deserialize player %s", pid_str)

        logger.info(
            "restore: %d/%d player states loaded for match %s",
            restored, len(players_raw), match_id,
        )
        return restored