"""
Players Data — IBM CIC Germany
Episodic Temporal Context System

Compressed episodic memory for LLM temporal reasoning.

Architecture
────────────────────────────────────────────────────────────────
This module replaces raw telemetry history injection with a structured
episodic compression layer. The LLM receives compressed trajectory
narratives, not raw windows.

Two episode abstractions:

  TacticalEpisode   — team/positional events (motif bursts, tactical drift,
                      escalation plateaus, transitions between dominant states)
  PlayerEpisode     — per-player physiological episodes (anomaly bursts,
                      recovery degradations, workload spikes)

Each episode captures:
  start_ts / end_ts       — match-minute timestamps
  dominant_findings       — symbolic finding types that dominated
  trend_direction         — worsening / recovering / stable / transitioning
  interventions           — any intervention signals logged during the episode
  response / outcome      — how the player responded (resolved / escalated /
                            persisted / unknown)
  persistence_duration    — number of windows the episode spanned

TemporalContextCompressor
────────────────────────────────────────────────────────────────
Consumes MatchState's rolling memory (findings, deques, motifs) and produces:

  trajectory_narrative    — a single compressed prose block describing the
                            match arc from kickoff to now
  trend_summaries         — per-axis direction + acceleration labels
  recent_context          — compressed last-N-minutes state
  intervention_history    — list of logged interventions with outcomes

Compression rules:
  - Summarise each episode in ≤ 2 sentences
  - Collapse consecutive same-finding windows into one episode
  - Detect worsening / improving / stable trends symbolically
  - Detect recurring patterns (same motif ≥ 2 episodes)
  - Detect transitions (state A → state B)
  - Keep total output under MAX_CONTEXT_TOKENS (token-efficient)

The compressed output is injected into the LLM prompt via xai_layer.py.
The LLM reasons over this compressed state, never raw windows.

Architecture rule:
  Symbolic detection here.
  LLM narrates the compressed output.
  These responsibilities must never be merged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Token budget
# ─────────────────────────────────────────────
MAX_CONTEXT_TOKENS        = 600   # hard cap on compressed context chars (≈ tokens)
MAX_EPISODES_IN_CONTEXT   = 6     # most recent episodes surfaced to LLM
MAX_RELEVANT_EPISODES     = 3     # prior episodes relevant to current anomaly
EPISODE_MERGE_WINDOW_MIN  = 2     # consecutive windows within this many minutes
                                   # with the same dominant finding are merged

# ── Episode quality gates — tunable independently ─────────────────────────
#
# These four constants control episode suppression.  They were designed to
# be tuned separately: changing one does not require changing the others.
#
# WARNING: tightening all four simultaneously creates state inertia —
# the engine becomes reluctant to acknowledge real rapid state changes
# (acute overload spikes, recovery collapses, tactical transitions).
# Always instrument suppression counts before tightening.
#
# MIN_EPISODE_WINDOWS
#   A non-last group must span this many windows before it is promoted to a
#   closed episode.  Single-window closures are usually noise; 2-window
#   closures still capture acute spikes.
#   BYPASS: any group with severity >= "high" is always promoted regardless.
#
# MIN_EPISODE_CONFIDENCE
#   Peak confidence required to close a low-severity episode.
#   Applied only when severity < "high" — high/critical episodes bypass this.
#
# EPISODE_COOLDOWN_WINDOWS
#   Minimum episode indices between closing and re-opening the SAME finding type.
#   Prevents ep11/ep14/ep17 fragmentation of a sustained locomotor pattern.
#   BYPASS: severity >= "critical" always opens immediately (acute re-escalation).
#   Set to 0 to disable cooldown entirely.
#
# EPISODE_MERGE_WINDOW_MIN (above)
#   Same-type findings within this many minutes are merged, not split.
#   Widening this too much suppresses legitimate rapid type transitions.
#   Keep at 2; only widen if the match clock is known to be unreliable.

MIN_EPISODE_WINDOWS      = 5     # min persistence before non-last closure (acute spike bypass for high+)
MIN_EPISODE_CONFIDENCE   = 0.55  # min peak confidence for low/medium severity closure
EPISODE_COOLDOWN_WINDOWS = 2     # min episode-index gap before same type re-opens (0 = disabled)


# ─────────────────────────────────────────────
# Episode data structures
# ─────────────────────────────────────────────

@dataclass
class PlayerEpisode:
    """
    A compressed representation of a physiological episode for one player.

    Spans one or more consecutive windows where a coherent finding cluster
    dominated. Captures what happened, how it evolved, and how it resolved —
    NOT the raw telemetry that produced it.
    """
    episode_index:       int                    # monotonic counter within match
    start_minute:        Optional[int]          # match minute the episode began
    end_minute:          Optional[int]          # match minute the episode ended (None = ongoing)
    dominant_findings:   List[str]              # primary finding types during episode
    trend_direction:     str                    # "worsening" | "recovering" | "stable" | "transitioning"
    severity:            str                    # peak severity during episode
    interventions:       List[str]              # coach actions or load-reduction signals
    response:            str                    # "resolved" | "escalated" | "persisted" | "unknown"
    persistence_duration: int                   # windows the episode spanned
    peak_anomaly_score:  float                  # max anomaly score during episode
    peak_confidence:     float                  # max finding confidence during episode
    notes:               str = ""              # optional free-form compressor note

    def is_ongoing(self) -> bool:
        return self.end_minute is None

    def duration_minutes(self) -> Optional[int]:
        if self.start_minute is None or self.end_minute is None:
            return None
        return max(0, self.end_minute - self.start_minute)

    def compressed_summary(self) -> str:
        """
        One-line episode summary for the LLM context block.
        ≤ 120 chars. Human-readable, coach-facing language.
        """
        start = f"min~{self.start_minute}" if self.start_minute is not None else "early"
        end   = f"→min~{self.end_minute}" if self.end_minute is not None else "→ongoing"
        findings = ", ".join(
            f.replace("_", " ") for f in self.dominant_findings[:3]
        )
        response_note = (
            f" [{self.response}]"
            if self.response not in ("unknown", "")
            else ""
        )
        return (
            f"Ep{self.episode_index} ({start}{end}): "
            f"{self.severity.upper()} — {findings}{response_note}"
        )

    def to_dict(self) -> dict:
        return {
            "episode_index":       self.episode_index,
            "start_minute":        self.start_minute,
            "end_minute":          self.end_minute,
            "dominant_findings":   self.dominant_findings,
            "trend_direction":     self.trend_direction,
            "severity":            self.severity,
            "interventions":       self.interventions,
            "response":            self.response,
            "persistence_duration": self.persistence_duration,
            "peak_anomaly_score":  round(self.peak_anomaly_score, 3),
            "peak_confidence":     round(self.peak_confidence, 3),
            "notes":               self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlayerEpisode":
        return cls(
            episode_index=d.get("episode_index", 0),
            start_minute=d.get("start_minute"),
            end_minute=d.get("end_minute"),
            dominant_findings=d.get("dominant_findings", []),
            trend_direction=d.get("trend_direction", "stable"),
            severity=d.get("severity", "low"),
            interventions=d.get("interventions", []),
            response=d.get("response", "unknown"),
            persistence_duration=d.get("persistence_duration", 1),
            peak_anomaly_score=d.get("peak_anomaly_score", 0.0),
            peak_confidence=d.get("peak_confidence", 0.0),
            notes=d.get("notes", ""),
        )


@dataclass
class TacticalEpisode:
    """
    A compressed representation of a tactical / team-level episode.

    Tactical episodes describe positional, motif, or escalation-level events
    that span multiple windows, capturing team or role-level patterns rather
    than individual physiology.

    Intentionally lightweight — the focus is on what changed tactically
    and how the team/player responded.
    """
    episode_index:      int
    start_minute:       Optional[int]
    end_minute:         Optional[int]
    episode_type:       str              # "motif_burst" | "escalation_plateau" |
                                         # "tactical_drift" | "state_transition"
    dominant_findings:  List[str]
    trend_direction:    str
    escalation_level:   str              # from SemanticMatchState
    interventions:      List[str]
    response:           str
    notes:              str = ""

    def compressed_summary(self) -> str:
        start = f"min~{self.start_minute}" if self.start_minute is not None else "early"
        end   = f"→min~{self.end_minute}"  if self.end_minute is not None else "→ongoing"
        return (
            f"TacEp{self.episode_index} ({start}{end}): "
            f"{self.episode_type.replace('_', ' ')} "
            f"[{self.escalation_level}] {self.trend_direction}"
        )

    def to_dict(self) -> dict:
        return {
            "episode_index":    self.episode_index,
            "start_minute":     self.start_minute,
            "end_minute":       self.end_minute,
            "episode_type":     self.episode_type,
            "dominant_findings": self.dominant_findings,
            "trend_direction":  self.trend_direction,
            "escalation_level": self.escalation_level,
            "interventions":    self.interventions,
            "response":         self.response,
            "notes":            self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TacticalEpisode":
        return cls(
            episode_index=d.get("episode_index", 0),
            start_minute=d.get("start_minute"),
            end_minute=d.get("end_minute"),
            episode_type=d.get("episode_type", "unknown"),
            dominant_findings=d.get("dominant_findings", []),
            trend_direction=d.get("trend_direction", "stable"),
            escalation_level=d.get("escalation_level", "normal"),
            interventions=d.get("interventions", []),
            response=d.get("response", "unknown"),
            notes=d.get("notes", ""),
        )


# ─────────────────────────────────────────────
# Compressed context output
# ─────────────────────────────────────────────

@dataclass
class CompressedTemporalContext:
    """
    Token-efficient temporal context ready for LLM injection.

    Built by TemporalContextCompressor. Consumed by xai_layer.format_episodic_context().
    Never contains raw telemetry values.
    """
    trajectory_narrative:   str                  # full match arc narrative (≤ 300 chars)
    trend_summaries:        Dict[str, str]        # {"anomaly": "worsening", ...}
    recent_context:         str                  # last ~5 minutes compressed (≤ 150 chars)
    intervention_history:   List[str]            # interventions + outcomes
    recurring_patterns:     List[str]            # motifs that recurred across episodes
    state_transitions:      List[str]            # detected A → B state transitions
    relevant_prior_episodes: List[str]           # compressed summaries of relevant past episodes
    current_escalation:     str                  # current escalation level
    episode_count:          int                  # total episodes this match
    # Persistence of the current ongoing episode (window count).
    # Set by the compressor so to_prompt_block() never parses strings to recover it.
    # LOCKED SEMANTICS: persistence_windows == total_risk_windows whenever
    # trajectory data is available (i.e. after the first window of a match).
    # The only exception is the very first window before ActiveRiskTrajectory
    # has been updated — in that case both fields are 0.
    # Do NOT use this field for pattern-specific persistence; use
    # pattern_persistence_windows for that.  This field exists for backward
    # compatibility with existing consumers; new code should read
    # total_risk_windows directly.  Will be removed in a future version.
    persistence_windows:    int = 0
    # ── Dual persistence counters (from ActiveRiskTrajectory) ─────────────────
    # pattern_persistence_windows: consecutive windows the *exact* current pattern
    #   has been active.  Resets when semantic subtype changes.
    # total_risk_windows: cumulative alert windows since last clean reset.
    #   Never resets on subtype transitions.  Used for escalation.
    # Both are 0 when no trajectory data is available (first window).
    pattern_persistence_windows: int = 0
    total_risk_windows:          int = 0
    # Cross-match historical episodes retrieved from EpisodeStore (optional).
    # Scored and filtered before compression so only salient episodes reach the prompt.
    historical_episodes:    List[dict] = field(default_factory=list)
    active_pattern:         Optional[str] = None 

    def is_empty(self) -> bool:
        return (
            self.episode_count == 0
            and not self.trend_summaries
            and not self.recent_context
            and not self.recurring_patterns
            and not self.state_transitions
            and not self.relevant_prior_episodes
        )

    def to_prompt_block(self) -> str:
        """
        Render as a compact JSON envelope for LLM injection (~40-80 tokens).

        The LLM narrates these conclusions; it does not re-reason from raw history.

        Fields
        ------
        active_pattern          : dominant finding type string e.g. "locomotor suppression"
        trend                   : first non-stable trend direction, or "stable"
        persistence_windows     : consecutive windows the pattern has lasted (from compressor)
        in_match_recurrence     : closed episodes this match with the same dominant finding
        cross_match_recurrence  : closed episodes in prior matches (from EpisodeStore)
        last_similar_outcome    : response from most recent similar prior-match episode
        current_escalation      : escalation level string
        recent_transitions      : up to 2 detected state transitions (absent if none)
        last_intervention       : most recent intervention logged (absent if none)
        """
        import json as _json
        import re as _re

        if self.is_empty():
            return ""

        # ── Active pattern ────────────────────────────────────────────────────
        # Use the canonical field set by compress().  Never parse episode summary
        # strings — that path was fragile and produced None on first occurrences.
        #
        # Fallback chain (only reached when compress() didn't set the field,
        # e.g. when deserialising an older checkpoint that predates this field):
        #   1. recurring_patterns  → "locomotor suppression (×6 episodes)"
        #   2. relevant_prior_episodes → parse the summary string as last resort
        active_pattern = self.active_pattern

        if not active_pattern and self.recurring_patterns:
            m = _re.match(r"^(.+?)\s+\(", self.recurring_patterns[0])
            active_pattern = m.group(1).strip() if m else self.recurring_patterns[0]

        if not active_pattern and self.relevant_prior_episodes:
            # Last resort: parse "Ep7 (min~0→min~0): HIGH — locomotor suppression [persisted]"
            # The em-dash (—) or ASCII dash separates severity from finding label.
            m = _re.search(r"[—\u2014]\s*(.+?)(?:\s*\[|$)", self.relevant_prior_episodes[0])
            if m:
                active_pattern = m.group(1).strip()

        # ── Persistence ───────────────────────────────────────────────────────
        # Authoritative value written by compress() — never re-derive from strings.
        persistence_w = self.persistence_windows

        # ── In-match recurrence ───────────────────────────────────────────────
        in_match_recurrence = sum(
            1 for pat in self.recurring_patterns
            if active_pattern and active_pattern in pat
        )

        # ── Cross-match recurrence (EpisodeStore) ─────────────────────────────
        # Match against both the human-readable form ("locomotor suppression") and
        # the raw underscore form ("locomotor_suppression") so naming differences
        # between the compressor and EpisodeStore payloads don't silently zero this.
        target_readable  = {active_pattern} if active_pattern else set()
        target_raw       = {active_pattern.replace(" ", "_")} if active_pattern else set()
        target_set       = target_readable | target_raw

        cross_match_recurrence = sum(
            1 for ep in self.historical_episodes
            if set(ep.get("dominant_findings", [])) & target_set
        )
        last_similar_outcome = None
        for ep in reversed(self.historical_episodes):
            if set(ep.get("dominant_findings", [])) & target_set:
                resp = ep.get("response", "unknown")
                if resp not in ("unknown", ""):
                    last_similar_outcome = resp
                break

        # ── Trend ─────────────────────────────────────────────────────────────
        trend = next(
            (v for v in self.trend_summaries.values()
             if v not in ("stable", "insufficient data")),
            "stable",
        )

        # ── Envelope ──────────────────────────────────────────────────────────
        envelope = {
            "active_pattern":              active_pattern,
            "trend":                       trend,
            # pattern_persistence_windows: how long the *current subtype* has been active.
            # total_risk_windows: how long the player has been in ANY degraded state.
            # The LLM should use both: "Recovery degradation for 2 windows within
            # an 8-window sustained degradation trajectory."
            "pattern_persistence_windows": self.pattern_persistence_windows,
            "total_risk_windows":          self.total_risk_windows,
            # Kept for backward compatibility with existing prompt consumers.
            # Equals total_risk_windows when trajectory data is available.
            "persistence_windows":         persistence_w,
            "in_match_recurrence":         in_match_recurrence,
            "cross_match_recurrence":      cross_match_recurrence,
            "last_similar_outcome":        last_similar_outcome,
            "current_escalation":          self.current_escalation,
        }

        if self.state_transitions:
            envelope["recent_transitions"] = self.state_transitions[:2]

        if self.intervention_history:
            envelope["last_intervention"] = self.intervention_history[-1]

        return _json.dumps(envelope, separators=(",", ":"))
# ─────────────────────────────────────────────
# Temporal context compressor
# ─────────────────────────────────────────────

class TemporalContextCompressor:
    """
    Converts MatchState's rolling symbolic memory into CompressedTemporalContext.

    Consumes:
      - match_state.recent_findings   (deque of symbolic finding dicts)
      - match_state.episodes          (list of PlayerEpisode)
      - match_state.trend_summaries   (dict built from trend analysis)
      - match_state.recent_context    (last-N-minutes summary)
      - match_state.intervention_history

    Produces:
      - CompressedTemporalContext

    No raw telemetry values enter the output. The LLM reasons over
    symbolic summaries, not sensor data.
    """
    @staticmethod
    def _merge_adjacent_same_type(
        episodes: List[PlayerEpisode],
        merge_gap_minutes: int = 3,
    ) -> List[PlayerEpisode]:
        """
        Post-process: merge consecutive closed episodes that share a dominant finding
        and are separated by <= merge_gap_minutes.

        This is the last line of defence against ep11/ep14/ep17 fragmentation when
        the cooldown gate alone doesn't catch it (e.g. burst of distinct intermediate
        types between same-type groups that reset the cooldown counter).
        """
        if len(episodes) < 2:
            return episodes

        merged: List[PlayerEpisode] = [episodes[0]]
        for ep in episodes[1:]:
            prev = merged[-1]
            same_type = (
                prev.dominant_findings
                and ep.dominant_findings
                and prev.dominant_findings[0] == ep.dominant_findings[0]
            )
            # Only merge closed→closed; ongoing is handled by extend-existing logic
            both_closed = not prev.is_ongoing() and not ep.is_ongoing()
            # Gap check: None minutes → always allow merge (clock absent)
            gap_ok = (
                prev.end_minute is None
                or ep.start_minute is None
                or (ep.start_minute - prev.end_minute) <= merge_gap_minutes
            )
            if same_type and both_closed and gap_ok:
                # Absorb ep into prev
                prev.end_minute         = ep.end_minute
                prev.persistence_duration += ep.persistence_duration
                prev.peak_anomaly_score = max(prev.peak_anomaly_score, ep.peak_anomaly_score)
                prev.peak_confidence    = max(prev.peak_confidence,    ep.peak_confidence)
                if ep.trend_direction == "worsening":
                    prev.trend_direction = "worsening"
                if ep.response in ("escalated", "persisted"):
                    prev.response = ep.response
                _SEV = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                if _SEV.get(ep.severity, 0) > _SEV.get(prev.severity, 0):
                    prev.severity = ep.severity
                logger.debug(
                    "Episode merge: Ep%d absorbed Ep%d (type=%s)",
                    prev.episode_index, ep.episode_index, prev.dominant_findings[0],
                )
            else:
                merged.append(ep)
        return merged

    # ── Episode builder ────────────────────────────────────────────────────────

    def build_episodes_from_findings(
        self,
        findings_snapshot: List[dict],
        anomaly_scores_snapshot: List[float],
        current_minute: Optional[int] = None,
        existing_episodes: Optional[List[PlayerEpisode]] = None,
    ) -> List[PlayerEpisode]:
        """
        Segment the finding history into discrete episodes.

        Strategy
        ────────
        - Group consecutive findings where the dominant type stays the same
          within EPISODE_MERGE_WINDOW_MIN minutes.
        - A change of dominant type triggers a new episode.
        - Episodes that share the same dominant type as an existing ongoing
          episode are merged with it (persistence extension), not duplicated.

        Quality gates (applied independently — each has severity bypasses)
        ──────────────────────────────────────────────────────────────────
        1. MIN_EPISODE_WINDOWS  — non-last groups must span ≥ 2 windows.
           BYPASS: severity >= "high" always closes regardless (acute spikes).

        2. MIN_EPISODE_CONFIDENCE — peak confidence gate for low/medium severity.
           BYPASS: severity >= "high" always closes regardless.

        3. EPISODE_COOLDOWN_WINDOWS — prevents same-type fragmentation.
           BYPASS: severity == "critical" always re-opens immediately.
           BYPASS: first-ever episode of this type (no prior close recorded).

        4. Clock propagation — start/end minutes taken from finding["minute"]
           (real match clock). Never defaulted to 0; None when absent.

        Returns the updated full episode list (existing + new).
        """
        _SEV = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        if not findings_snapshot:
            return existing_episodes or []

        logger.debug(
            "Episode builder: findings=%d existing_episodes=%d",
            len(findings_snapshot),
            len(existing_episodes or []),
        )

        episodes: List[PlayerEpisode] = list(existing_episodes or [])
        next_idx = max((ep.episode_index for ep in episodes), default=0) + 1

        # ── Instrumentation counters ──────────────────────────────────────────
        _stats: Dict[str, int] = {
            "merged":                 0,
            "suppressed_min_windows": 0,
            "suppressed_confidence":  0,
            "suppressed_cooldown":    0,
        }

        # ── Group findings into runs by dominant type ─────────────────────────
        groups: List[List[dict]] = []
        current_group: List[dict] = []
        current_dominant: Optional[str] = None

        for f in findings_snapshot:
            ftype    = f.get("type", "unknown")
            fmin     = f.get("minute")
            prev_min = current_group[-1].get("minute") if current_group else None

            time_gap_ok = (
                prev_min is None
                or fmin is None
                or abs(fmin - prev_min) <= EPISODE_MERGE_WINDOW_MIN
            )
            if ftype == current_dominant and time_gap_ok:
                current_group.append(f)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [f]
                current_dominant = ftype

        if current_group:
            groups.append(current_group)

        # ── Cooldown tracker: last closed episode_index per finding type ───────
        last_closed_idx: Dict[str, int] = {}
        for ep in episodes:
            if not ep.is_ongoing() and ep.dominant_findings:
                last_closed_idx[ep.dominant_findings[0]] = ep.episode_index

        # ── Convert groups → episodes ──────────────────────────────────────────
        for group in groups:
            dominant = group[0].get("type", "unknown")
            is_last  = (group is groups[-1])

            # Real clock: first/last non-None minute in the group
            minutes_in_group = [
                f.get("minute") for f in group if f.get("minute") is not None
            ]
            start_m = minutes_in_group[0]  if minutes_in_group else None
            end_m   = minutes_in_group[-1] if minutes_in_group else None

            severity = max(
                (f.get("severity", "low") for f in group),
                key=lambda s: _SEV.get(s, 0),
            )
            sev_int     = _SEV.get(severity, 0)
            is_high_sev = sev_int >= 2   # "high" or "critical"

            confidences = [f.get("confidence", 0.5) for f in group]
            peak_conf   = max(confidences) if confidences else 0.5

            n_scores   = len(anomaly_scores_snapshot)
            peak_score = (
                max(anomaly_scores_snapshot[-len(group):])
                if n_scores >= len(group) else 0.0
            )

            trend_dir = self._infer_episode_trend(group)
            response  = self._infer_episode_response(group, findings_snapshot)

            # ── Try to extend an existing ongoing episode of the same type ────
            matched = False
            for ep in reversed(episodes):
                if (
                    ep.dominant_findings
                    and ep.dominant_findings[0] == dominant
                    and ep.is_ongoing()
                ):
                    ep.persistence_duration = len(group)
                    ep.peak_anomaly_score   = max(ep.peak_anomaly_score, peak_score)
                    ep.peak_confidence      = max(ep.peak_confidence, peak_conf)
                    ep.trend_direction      = trend_dir
                    ep.response             = response
                    ep.severity             = severity
                    if start_m is not None and ep.start_minute is None:
                        ep.start_minute = start_m

                    if not is_last:
                        # Gate 1: min-windows (bypass for high/critical)
                        if not is_high_sev and ep.persistence_duration < MIN_EPISODE_WINDOWS:
                            _stats["suppressed_min_windows"] += 1
                            logger.debug(
                                "Episode gate [min_windows]: type=%s windows=%d sev=%s — "
                                "not closing (below threshold, not high severity)",
                                dominant, ep.persistence_duration, severity,
                            )
                        # Gate 2: confidence (bypass for high/critical)
                        elif not is_high_sev and ep.peak_confidence < MIN_EPISODE_CONFIDENCE:
                            _stats["suppressed_confidence"] += 1
                            logger.debug(
                                "Episode gate [confidence]: type=%s conf=%.2f sev=%s — "
                                "not closing (below threshold, not high severity)",
                                dominant, ep.peak_confidence, severity,
                            )
                        else:
                            ep.end_minute = end_m
                            last_closed_idx[dominant] = ep.episode_index

                    _stats["merged"] += 1
                    matched = True
                    break

            if not matched:
                # Gate 3: cooldown — prevent same-type fragmentation.
                # BYPASS 1: critical severity always re-opens immediately.
                # BYPASS 2: first-ever episode of this type — no prior close
                #           recorded so cooldown does not apply yet.
                if EPISODE_COOLDOWN_WINDOWS > 0 and not is_last and sev_int < 3:
                    last_close  = last_closed_idx.get(dominant, -999)
                    since_close = next_idx - last_close
                    if last_close != -999 and since_close < EPISODE_COOLDOWN_WINDOWS:
                        _stats["suppressed_cooldown"] += 1
                        logger.info(
                            "Episode gate [cooldown]: type=%s since_close=%d "
                            "threshold=%d sev=%s — suppressed",
                            dominant, since_close, EPISODE_COOLDOWN_WINDOWS, severity,
                        )
                        continue

                # Gate 1+2 on new episodes: non-last, non-high-severity short groups
                if not is_last and not is_high_sev:
                    if len(group) < MIN_EPISODE_WINDOWS:
                        _stats["suppressed_min_windows"] += 1
                        logger.debug(
                            "Episode gate [min_windows]: type=%s windows=%d — "
                            "new episode suppressed",
                            dominant, len(group),
                        )
                        continue
                    if peak_conf < MIN_EPISODE_CONFIDENCE:
                        _stats["suppressed_confidence"] += 1
                        logger.debug(
                            "Episode gate [confidence]: type=%s conf=%.2f — "
                            "new episode suppressed",
                            dominant, peak_conf,
                        )
                        continue

                ep = PlayerEpisode(
                    episode_index=next_idx,
                    start_minute=start_m,
                    end_minute=end_m if not is_last else None,
                    dominant_findings=[dominant],
                    trend_direction=trend_dir,
                    severity=severity,
                    interventions=[],
                    response=response,
                    persistence_duration=len(group),
                    peak_anomaly_score=peak_score,
                    peak_confidence=peak_conf,
                )
                episodes.append(ep)
                if not is_last:
                    last_closed_idx[dominant] = next_idx
                next_idx += 1

        if any(_stats.values()):
            logger.info(
                "Episode suppression | merged=%d min_windows=%d confidence=%d "
                "cooldown=%d total_episodes=%d",
                _stats["merged"],
                _stats["suppressed_min_windows"],
                _stats["suppressed_confidence"],
                _stats["suppressed_cooldown"],
                len(episodes),
            )

        episodes = self._merge_adjacent_same_type(
            episodes, merge_gap_minutes=EPISODE_MERGE_WINDOW_MIN + 1
        )
        return episodes
    # ── Trend detection helpers ────────────────────────────────────────────────

    @staticmethod
    def _infer_episode_trend(group: List[dict]) -> str:
        """
        Detect worsening / recovering / stable / transitioning from a finding group.
        Uses severity and confidence trajectories — no raw telemetry.
        """
        if len(group) < 2:
            return "stable"

        sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        sevs  = [sev_order.get(f.get("severity", "low"), 0) for f in group]
        confs = [f.get("confidence", 0.5) for f in group]

        states = [f.get("state", "active") for f in group]
        if "escalating" in states:
            return "worsening"
        if "resolving" in states:
            return "recovering"

        # Slope heuristic on severity
        first_half = sevs[: len(sevs) // 2]
        second_half = sevs[len(sevs) // 2 :]
        if second_half and first_half:
            if sum(second_half) / len(second_half) > sum(first_half) / len(first_half):
                return "worsening"
            if sum(second_half) / len(second_half) < sum(first_half) / len(first_half):
                return "recovering"

        # Confidence slope
        first_conf = confs[: len(confs) // 2]
        second_conf = confs[len(confs) // 2 :]
        if second_conf and first_conf:
            if sum(second_conf) / len(second_conf) > sum(first_conf) / len(first_conf) + 0.05:
                return "worsening"

        return "stable"

    @staticmethod
    def _infer_episode_response(group: List[dict], full_history: List[dict]) -> str:
        """
        Infer whether the episode resolved, escalated, persisted, or is unknown.

        Looks at findings that appear *after* the last member of this group in
        full_history to determine the outcome.  For the ongoing (last) group,
        uses the state label on the final finding rather than returning "unknown".
        """
        if not group:
            return "unknown"

        dominant = group[-1].get("type", "unknown")
        group_ids = {id(f) for f in group}

        # Locate the position of the last group member in the full history list
        # so we can inspect everything that comes after it.
        last_pos = -1
        for i, f in enumerate(full_history):
            if id(f) in group_ids:
                last_pos = i
        after = full_history[last_pos + 1:] if last_pos >= 0 else []

        same_type_later = [f for f in after if f.get("type") == dominant]

        if same_type_later:
            last_state = same_type_later[-1].get("state", "active")
            if last_state == "resolving":
                return "resolved"
            if last_state == "escalating":
                return "escalated"
            # Same finding type recurs after the group — still ongoing
            return "persisted"

        if after:
            # Different finding types follow — this type has been superseded
            last_state = group[-1].get("state", "active")
            if last_state == "escalating":
                return "escalated"
            return "resolved"

        # This is the trailing group (ongoing episode) — no later findings exist.
        # Classify from the current state label rather than defaulting to "unknown".
        last_state = group[-1].get("state", "active")
        if last_state == "resolving":
            return "resolved"
        if last_state == "escalating":
            return "escalated"
        if last_state in ("stabilizing", "active"):
            # Finding is still present without a resolution signal — persisting
            return "persisted"
        return "unknown"

    # ── Pattern / transition detection ────────────────────────────────────────

    def detect_recurring_patterns(
        self, episodes: List[PlayerEpisode]
    ) -> List[str]:
        """
        Identify finding types that recur across ≥ 2 distinct episodes.
        Returns human-readable strings.
        """
        counts: Dict[str, int] = {}
        for ep in episodes:
            for ftype in set(ep.dominant_findings):
                counts[ftype] = counts.get(ftype, 0) + 1

        recurring = sorted(
            [(ft, c) for ft, c in counts.items() if c >= 2],
            key=lambda x: -x[1],
        )
        return [
            f"{ft.replace('_', ' ')} (×{c} episodes)"
            for ft, c in recurring[:4]
        ]

    def detect_state_transitions(
        self, episodes: List[PlayerEpisode]
    ) -> List[str]:
        """
        Detect A → B transitions between consecutive episode dominant states.
        Returns human-readable transition strings.
        """
        transitions: List[str] = []
        seen: Dict[Tuple[str, str], int] = {}

        for i in range(1, len(episodes)):
            prev_type = (
                episodes[i - 1].dominant_findings[0]
                if episodes[i - 1].dominant_findings
                else None
            )
            curr_type = (
                episodes[i].dominant_findings[0]
                if episodes[i].dominant_findings
                else None
            )
            if prev_type and curr_type and prev_type != curr_type:
                key = (prev_type, curr_type)
                seen[key] = seen.get(key, 0) + 1

        for (a, b), count in sorted(seen.items(), key=lambda x: -x[1]):
            label = (
                f"{a.replace('_', ' ')} → {b.replace('_', ' ')}"
                + (f" (×{count})" if count > 1 else "")
            )
            transitions.append(label)

        return transitions[:4]

    # ── Trajectory narrative ───────────────────────────────────────────────────

    def build_trajectory_narrative(
        self,
        episodes: List[PlayerEpisode],
        current_escalation: str,
    ) -> str:
        """
        Produce a single compressed match-arc narrative sentence.
        ≤ 200 chars. Focuses on direction and key turning points.
        """
        if not episodes:
            return "No significant episodes detected yet."

        n = len(episodes)
        first = episodes[0]
        last  = episodes[-1]

        # Arc direction
        worsening = sum(1 for ep in episodes if ep.trend_direction == "worsening")
        recovering = sum(1 for ep in episodes if ep.trend_direction == "recovering")

        if worsening > recovering:
            arc = "progressive deterioration"
        elif recovering > worsening:
            arc = "improving trajectory"
        else:
            arc = "fluctuating state"

        start_note = (
            f"from min~{first.start_minute}" if first.start_minute is not None else "from early match"
        )
        end_note = (
            f"to min~{last.end_minute}" if last.end_minute is not None else "ongoing"
        )

        first_finding = (
            first.dominant_findings[0].replace("_", " ")
            if first.dominant_findings
            else "anomaly"
        )
        last_finding = (
            last.dominant_findings[0].replace("_", " ")
            if last.dominant_findings
            else "anomaly"
        )

        if first_finding == last_finding:
            return (
                f"{n} episode(s) of {first_finding} — "
                f"{arc} {start_note}, {end_note}. "
                f"Escalation: {current_escalation}."
            )[:200]

        return (
            f"{n} episode(s): {first_finding} ({start_note}) "
            f"→ {last_finding} ({end_note}) — {arc}. "
            f"Escalation: {current_escalation}."
        )[:200]

    # ── Recent context ────────────────────────────────────────────────────────

    def build_recent_context(
        self,
        findings_snapshot: List[dict],
        current_minute: Optional[int],
        window_minutes: int = 5,
    ) -> str:
        """
        Summarise findings from the last `window_minutes` minutes.
        ≤ 150 chars. Used to anchor the LLM to what just happened.
        """
        if not findings_snapshot:
            return ""

        if current_minute is None:
            # Fall back to last 10% of findings
            recent = findings_snapshot[-(max(1, len(findings_snapshot) // 10)):]
        else:
            threshold = current_minute - window_minutes
            recent = [
                f for f in findings_snapshot
                if f.get("minute") is not None and f["minute"] >= threshold
            ]

        if not recent:
            recent = findings_snapshot[-3:]  # always show at least last 3

        type_counts: Dict[str, int] = {}
        peak_sev = "low"
        sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        for f in recent:
            ft = f.get("type", "unknown")
            type_counts[ft] = type_counts.get(ft, 0) + 1
            if sev_order.get(f.get("severity", "low"), 0) > sev_order.get(peak_sev, 0):
                peak_sev = f.get("severity", "low")

        dominant_types = sorted(type_counts.items(), key=lambda x: -x[1])
        findings_str = ", ".join(
            f"{ft.replace('_', ' ')} (×{c})" if c > 1 else ft.replace("_", " ")
            for ft, c in dominant_types[:3]
        )

        window_label = (
            f"Last {window_minutes} min" if current_minute is not None else "Recent"
        )
        return f"{window_label}: {peak_sev.upper()} — {findings_str}"[:150]

    # ── Relevant episode retrieval ─────────────────────────────────────────────

    def find_relevant_episodes(
        self,
        episodes: List[PlayerEpisode],
        current_findings: List[str],
        current_anomaly_type: Optional[str] = None,
        top_n: int = MAX_RELEVANT_EPISODES,
    ) -> List[PlayerEpisode]:
        """
        Select prior episodes most relevant to the current anomaly.

        Relevance heuristics (descending priority):
        1. Episode dominant finding matches current finding type exactly
        2. Episode dominant finding overlaps with current finding set
        3. Episode had worsening trend (always informative for context)
        4. Episode had high/critical severity
        5. More recent episodes preferred (tiebreak)

        Excludes the most recent (ongoing) episode — that's already covered
        by `recent_context`.
        """
        if not episodes:
            return []

        target_types = set(current_findings)
        if current_anomaly_type:
            target_types.add(current_anomaly_type)

        # Score each episode (lower index = older = deprioritized)
        scored: List[Tuple[float, PlayerEpisode]] = []
        closed_episodes = [ep for ep in episodes if not ep.is_ongoing()]

        for i, ep in enumerate(closed_episodes):
            ep_types = set(ep.dominant_findings)
            recency_weight = (i + 1) / max(len(closed_episodes), 1)  # 0→1 newer

            exact_match    = 2.0 if ep_types & target_types else 0.0
            trend_bonus    = 1.0 if ep.trend_direction == "worsening" else 0.0
            severity_bonus = {"critical": 1.0, "high": 0.6, "medium": 0.3, "low": 0.0}.get(
                ep.severity, 0.0
            )
            score = exact_match + trend_bonus + severity_bonus + recency_weight
            scored.append((score, ep))

        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:top_n]]

    # ── Main compression entry point ───────────────────────────────────────────

    def compress(
        self,
        episodes: List[PlayerEpisode],
        trend_summaries: Dict[str, str],
        recent_findings: List[dict],
        intervention_history: List[str],
        current_minute: Optional[int],
        current_escalation: str,
        current_finding_types: Optional[List[str]] = None,
        historical_episodes: Optional[List[dict]] = None,
        # ── Risk trajectory overrides (from ActiveRiskTrajectory) ─────────────
        # When provided, these replace the episode-derived persistence_windows
        # and active_pattern so that cross-subtype risk accumulation is visible
        # to downstream policy even when semantic subtypes fluctuate.
        trajectory_total_risk_windows: Optional[int] = None,
        trajectory_pattern_persistence: Optional[int] = None,
        trajectory_active_pattern: Optional[str] = None,
    ) -> CompressedTemporalContext:
        """
        Build CompressedTemporalContext from all available symbolic memory.

        Parameters
        ----------
        episodes            : full episode list from MatchState
        trend_summaries     : pre-computed trend directions by axis
        recent_findings     : raw finding deque snapshot (NOT fed to LLM directly)
        intervention_history: logged coach actions
        current_minute      : current match minute (for recent context window)
        current_escalation  : current escalation level string
        current_finding_types : finding types from the current anomaly window
                                (used to select relevant prior episodes)

        Returns
        -------
        CompressedTemporalContext
            Token-efficient, LLM-ready. Contains NO raw telemetry.
        """
        trajectory  = self.build_trajectory_narrative(episodes, current_escalation)
        recent_ctx  = self.build_recent_context(recent_findings, current_minute)
        recurring   = self.detect_recurring_patterns(episodes)
        transitions = self.detect_state_transitions(episodes)

        relevant = self.find_relevant_episodes(
            episodes=episodes,
            current_findings=current_finding_types or [],
        )
        relevant_lines = [ep.compressed_summary() for ep in relevant]

        # ── Ongoing episode — single authoritative source for two fields ───────
        # Never infer persistence or active_pattern from string parsing.
        # Both values come directly from the ongoing episode object.
        ongoing_ep    = next((ep for ep in reversed(episodes) if ep.is_ongoing()), None)

        # ── persistence_windows: prefer trajectory total_risk_windows ─────────
        # ongoing_ep.persistence_duration counts only the current same-type group.
        # trajectory_total_risk_windows accumulates across all alert windows
        # this match, surviving subtype transitions.  This is the value that
        # feeds policy rules (substitute requires ≥ 8, reduce_load ≥ 4, etc.)
        if trajectory_total_risk_windows is not None:
            persistence_w = trajectory_total_risk_windows
        else:
            # Fallback for callers that don't yet supply the trajectory
            persistence_w = ongoing_ep.persistence_duration if ongoing_ep else 0

        # ── active_pattern: prefer trajectory when available ──────────────────
        # trajectory_active_pattern is the current pattern of the risk trajectory
        # (blanks when the player is clean).  Fall back to episode-derived value.
        active_pattern: Optional[str] = None
        if trajectory_active_pattern:
            active_pattern = trajectory_active_pattern.replace("_", " ")
        elif ongoing_ep and ongoing_ep.dominant_findings:
            active_pattern = ongoing_ep.dominant_findings[0].replace("_", " ")
        elif current_finding_types:
            active_pattern = current_finding_types[0].replace("_", " ")

        logger.warning(
            "CTX BUILD DEBUG | active_pattern=%r "
            "pattern_persistence=%d total_risk_windows=%d "
            "escalation=%s episodes=%d",
            active_pattern,
            trajectory_pattern_persistence if trajectory_pattern_persistence is not None
                else (ongoing_ep.persistence_duration if ongoing_ep else 0),
            trajectory_total_risk_windows if trajectory_total_risk_windows is not None else 0,
            current_escalation,
            len(episodes),
        )

        return CompressedTemporalContext(
            trajectory_narrative=trajectory,
            trend_summaries=trend_summaries,
            recent_context=recent_ctx,
            intervention_history=list(intervention_history),
            recurring_patterns=recurring,
            state_transitions=transitions,
            relevant_prior_episodes=relevant_lines,
            current_escalation=current_escalation,
            episode_count=len(episodes),
            persistence_windows=persistence_w,
            pattern_persistence_windows=(
                trajectory_pattern_persistence
                if trajectory_pattern_persistence is not None
                else (ongoing_ep.persistence_duration if ongoing_ep else 0)
            ),
            total_risk_windows=(
                trajectory_total_risk_windows
                if trajectory_total_risk_windows is not None
                else 0
            ),
            active_pattern=active_pattern,
            historical_episodes=historical_episodes or []
        )