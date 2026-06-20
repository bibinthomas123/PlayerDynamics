"""
Gap-Aware Windowing — PlayerDynamics

Closes the second real-data blocker found in the resampler validation:
SequenceWindowBuilder.build_from_session() (analysis/anomaly_detection.py)
slides over consecutive DataFrame ROWS with zero awareness of elapsed wall-
clock time. If a player's resampled events_df has a real tracking gap
(substitution, off-court period) between two consecutive rows, a nominal
"8-step / 120-second" window built across that gap can actually span far
longer (the resampler validation observed up to ~3420s) -- two physically
disconnected tracking segments silently spliced into one training example.

This module does NOT modify SequenceWindowBuilder, BaselineBuilder,
PatternAnalysisEngine, or any model code. It is a thin, additive layer that:

    1. Replicates build_from_session()'s OWN windowing math (same
       window_steps, same default stride) purely to label which row-range
       each window would draw from -- build_from_session() itself returns
       only (sequence, mask) numpy arrays and discards which source rows
       (and therefore which timestamps) produced them.
    2. Checks the REAL timestamps in that row-range for any consecutive gap
       exceeding SequenceWindowConfig.gap_threshold_s.
    3. Drops (default) or flags windows that fail that check.

Pipeline position
------------------
    KinexonResampler.resample()  (Phase 1 -- per-player events_df)
                |
                v
    detect_window_gaps()              <- this module, NEW
                |
                v
    build_from_session_gap_aware()    <- this module, NEW (wraps, does not
                |                         modify, SequenceWindowBuilder.build_from_session)
                v
    build_training_sequences_gap_aware()  <- this module, NEW (mirrors, does
                |                             not modify, PatternAnalysisEngine.
                |                             build_training_sequences's
                v                             per-session grouping loop)
    [ train_all_models() callers can opt into this instead of the
      gap-blind build_training_sequences(), once torch/model training is
      actually exercised -- not wired into train_all_models() itself here,
      per "do not train models yet" ]

Why DROP, not flag (task 3's "choose the safer option")
------------------------------------------------------------
Both modes are implemented (`mode="drop"` / `mode="flag"`), but DROP is the
default and the one used in this phase's validation. Flagging only changes
the data if every future consumer remembers to check the flag before using
a window -- a single careless training loop that iterates `(seq, mask)`
pairs without inspecting a third element silently trains on spliced,
physically-discontinuous data, with no error and no signal that anything
went wrong. Dropping fails closed: a discontinuous window simply does not
exist for any later code to forget to check. This mirrors this codebase's
own established convention elsewhere (e.g. CoachSituation's "fails closed,
never guesses" -- analysis/coach_situation.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import CONFIG, SequenceWindowConfig

if TYPE_CHECKING:
    from analysis.anomaly_detection import PatternAnalysisEngine, SequenceWindowBuilder

logger = logging.getLogger(__name__)


@dataclass
class GapAuditResult:
    """Diagnostics for one player's (or one session's) gap-filtering pass."""
    player_id: Optional[int]
    n_windows_before: int
    n_windows_after: int
    n_dropped: int
    gap_seconds: List[float] = field(default_factory=list)  # one entry per DROPPED window: its largest internal gap

    @property
    def largest_gap_s(self) -> float:
        return max(self.gap_seconds) if self.gap_seconds else 0.0


def detect_window_gaps(
    events_df: pd.DataFrame,
    window_steps: int,
    stride: int,
    gap_threshold_s: float,
    ts_col: str = "ts",
) -> List[Tuple[bool, float]]:
    """
    Returns one (is_gap_free, largest_internal_gap_seconds) tuple per window,
    in the EXACT same order/count that
    SequenceWindowBuilder.build_from_session(events_df, stride) would
    produce from the same events_df -- replicates that method's own
    `for start in range(0, n - window_steps + 1, stride)` loop, but only
    ever reads `ts_col`; it does not touch any feature column and does not
    call build_from_session.

    is_gap_free is True when every consecutive pair of rows inside the
    window has a timestamp delta <= gap_threshold_s.
    """
    df = events_df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    if n < window_steps:
        return []

    ts = pd.to_datetime(df[ts_col], utc=True)
    results: List[Tuple[bool, float]] = []
    for start in range(0, n - window_steps + 1, stride):
        end = start + window_steps
        deltas = ts.iloc[start:end].diff().dt.total_seconds().dropna()
        max_gap = float(deltas.max()) if len(deltas) else 0.0
        results.append((max_gap <= gap_threshold_s, max_gap))
    return results


def build_from_session_gap_aware(
    window_builder: "SequenceWindowBuilder",
    events_df: pd.DataFrame,
    stride: Optional[int] = None,
    gap_threshold_s: Optional[float] = None,
    mode: str = "drop",
    ts_col: str = "ts",
) -> Tuple[List[Any], GapAuditResult]:
    """
    Wraps SequenceWindowBuilder.build_from_session() (called UNMODIFIED,
    exactly as PatternAnalysisEngine itself calls it) with a post-hoc,
    timestamp-based gap check.

    mode="drop" (default, recommended -- see module docstring):
        returns only the (sequence, mask) pairs whose window is gap-free.
    mode="flag":
        returns every (sequence, mask, is_gap_free) triple -- caller is
        responsible for checking the third element before use.

    Returns (windows, audit) where audit is a GapAuditResult with
    before/after counts and the gap sizes of every dropped window.
    """
    cfg: SequenceWindowConfig = CONFIG.window
    if stride is None:
        stride = window_builder.window_steps
    if gap_threshold_s is None:
        gap_threshold_s = cfg.gap_threshold_s
    if mode not in ("drop", "flag"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'drop' or 'flag')")

    sorted_df = events_df.sort_values(ts_col).reset_index(drop=True)
    windows = window_builder.build_from_session(sorted_df, stride=stride)
    gap_info = detect_window_gaps(sorted_df, window_builder.window_steps, stride, gap_threshold_s, ts_col)

    if len(windows) != len(gap_info):
        # Defensive: build_from_session's own windowing math must match
        # detect_window_gaps' replication of it exactly. A mismatch here
        # means the two have drifted apart (e.g. a future change to
        # build_from_session's stride/window_steps handling) and the gap
        # labels can no longer be trusted to align with the windows --
        # fail loudly rather than silently mislabel.
        raise RuntimeError(
            f"windowing math mismatch: build_from_session produced {len(windows)} windows, "
            f"detect_window_gaps replicated {len(gap_info)} -- cannot align gap labels to windows"
        )

    dropped_gaps: List[float] = []
    kept: List[Any] = []
    for (seq, mask), (is_gap_free, gap_s) in zip(windows, gap_info):
        if is_gap_free:
            kept.append((seq, mask, True) if mode == "flag" else (seq, mask))
        else:
            dropped_gaps.append(gap_s)
            if mode == "flag":
                kept.append((seq, mask, False))

    n_after = sum(1 for ok, _ in gap_info if ok) if mode == "drop" else len(windows)
    audit = GapAuditResult(
        player_id=None,
        n_windows_before=len(windows),
        n_windows_after=n_after,
        n_dropped=len(dropped_gaps),
        gap_seconds=dropped_gaps,
    )
    return kept, audit


def build_training_sequences_gap_aware(
    pattern_engine: "PatternAnalysisEngine",
    events_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    mode: str = "drop",
) -> Tuple[List[Any], List[GapAuditResult]]:
    """
    Gap-aware counterpart to PatternAnalysisEngine.build_training_sequences().
    Mirrors that method's own per-session_id grouping loop exactly (filter
    events_df by session_id, call the windower once per session) but routes
    through build_from_session_gap_aware() instead of calling
    window_builder.build_from_session() raw.

    build_training_sequences() itself is left completely untouched -- this
    is a parallel function, not a patch, so existing callers/tests of the
    original keep its exact current (gap-blind) behaviour.
    """
    all_pairs: List[Any] = []
    audits: List[GapAuditResult] = []

    for session in sessions_df.itertuples(index=False):
        session_id = int(session.session_id)
        sess_ev = events_df.loc[events_df["session_id"] == session_id]
        if sess_ev.empty:
            continue
        windows, audit = build_from_session_gap_aware(
            pattern_engine.window_builder, sess_ev, mode=mode
        )
        audit.player_id = getattr(session, "player_id", None)
        audits.append(audit)
        for w in windows:
            all_pairs.append((*w, session_id))

    return all_pairs, audits
