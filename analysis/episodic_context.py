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
        Render as a labelled LLM-ready prompt block.
        Hard cap at MAX_CONTEXT_TOKENS chars; truncates gracefully.
        """
        if self.is_empty():
            return ""

        parts: List[str] = []

        parts.append(
            f"Match trajectory [{self.current_escalation.upper()}] "
            f"({self.episode_count} episode(s)):"
        )
        if self.trajectory_narrative:
            parts.append(f"  Arc: {self.trajectory_narrative}")

        if self.trend_summaries:
            trend_str = "; ".join(
                f"{k}: {v}"
                for k, v in self.trend_summaries.items()
                if v not in ("stable", "insufficient data")
            )
            if trend_str:
                parts.append(f"  Trends: {trend_str}")

        if self.recurring_patterns:
            parts.append(
                "  Recurring patterns: " + "; ".join(self.recurring_patterns[:3])
            )

        if self.state_transitions:
            parts.append(
                "  Transitions: " + "; ".join(self.state_transitions[:3])
            )

        if self.relevant_prior_episodes:
            parts.append("  Prior episodes relevant to current anomaly:")
            for ep_line in self.relevant_prior_episodes[:MAX_RELEVANT_EPISODES]:
                parts.append(f"    • {ep_line}")

        if self.recent_context:
            parts.append(f"  Recent context: {self.recent_context}")

        if self.intervention_history:
            parts.append(
                "  Interventions: " + "; ".join(self.intervention_history[-3:])
            )

        block = "\n".join(parts)

        # Hard truncate to budget (prefer clean line boundaries)
        if len(block) > MAX_CONTEXT_TOKENS:
            block = block[:MAX_CONTEXT_TOKENS].rsplit("\n", 1)[0] + "\n  [context truncated]"

        return block


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

        Strategy:
        - Group consecutive findings where the dominant type stays the same
          (within EPISODE_MERGE_WINDOW_MIN minutes).
        - A change of dominant type triggers a new episode.
        - Episodes that share the same dominant type as an existing episode
          are merged with it (persistence extension), not duplicated.

        Returns the updated full episode list (existing + new).
        """
        if not findings_snapshot:
            return existing_episodes or []

        episodes: List[PlayerEpisode] = list(existing_episodes or [])
        next_idx = max((ep.episode_index for ep in episodes), default=0) + 1

        # Group findings into runs by dominant type
        groups: List[List[dict]] = []
        current_group: List[dict] = []
        current_dominant: Optional[str] = None

        for f in findings_snapshot:
            ftype = f.get("type", "unknown")
            fmin  = f.get("minute")
            prev_min = current_group[-1].get("minute") if current_group else None

            # Start new group if: type changed, or time gap exceeds merge window
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

        # Convert groups to episodes — skip groups already covered by existing episodes
        for group in groups:
            dominant = group[0].get("type", "unknown")
            start_m  = group[0].get("minute")
            end_m    = group[-1].get("minute")
            severity = max(
                (f.get("severity", "low") for f in group),
                key=lambda s: {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(s, 0),
            )
            confidences = [f.get("confidence", 0.5) for f in group]
            peak_conf   = max(confidences) if confidences else 0.5

            # Anomaly score range: use average over episode length as proxy
            n_scores = len(anomaly_scores_snapshot)
            peak_score = max(anomaly_scores_snapshot[-len(group):]) if n_scores >= len(group) else 0.0

            trend_dir = self._infer_episode_trend(group)
            response  = self._infer_episode_response(group, findings_snapshot)

            # Try to extend an existing episode rather than creating a duplicate
            matched = False
            for ep in reversed(episodes):
                if (
                    ep.dominant_findings
                    and ep.dominant_findings[0] == dominant
                    and ep.is_ongoing()
                ):
                    ep.end_minute = end_m
                    ep.persistence_duration += len(group)
                    ep.peak_anomaly_score = max(ep.peak_anomaly_score, peak_score)
                    ep.peak_confidence = max(ep.peak_confidence, peak_conf)
                    ep.trend_direction = trend_dir
                    ep.response = response
                    matched = True
                    break

            if not matched:
                ep = PlayerEpisode(
                    episode_index=next_idx,
                    start_minute=start_m,
                    end_minute=end_m if len(groups) > 1 else None,  # last group is ongoing
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
                next_idx += 1

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
        Looks for the state label on the most recent finding of the same type.
        """
        if not group:
            return "unknown"
        dominant = group[-1].get("type", "unknown")
        # Check if same type recurs after the group in full history
        group_set = {id(f) for f in group}
        later = [
            f for f in full_history
            if id(f) not in group_set and f.get("type") == dominant
        ]
        if later:
            last_state = later[-1].get("state", "active")
            if last_state in ("resolving",):
                return "resolved"
            if last_state in ("escalating",):
                return "escalated"
            return "persisted"

        # No later occurrences — check final state in group
        last_state = group[-1].get("state", "active")
        if last_state == "resolving":
            return "resolved"
        if last_state == "escalating":
            return "escalated"
        if last_state == "stabilizing":
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
        trajectory = self.build_trajectory_narrative(episodes, current_escalation)
        recent_ctx = self.build_recent_context(recent_findings, current_minute)
        recurring  = self.detect_recurring_patterns(episodes)
        transitions = self.detect_state_transitions(episodes)

        relevant = self.find_relevant_episodes(
            episodes=episodes,
            current_findings=current_finding_types or [],
        )
        relevant_lines = [ep.compressed_summary() for ep in relevant]

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
        )