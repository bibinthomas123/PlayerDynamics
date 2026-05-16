"""
analysis/live_window_accumulator.py
────────────────────────────────────
Separates raw telemetry ingestion from inference.

Instead of running the full analysis pipeline on every incoming packet,
the accumulator buffers events per player and only emits a window when
enough events have been collected.  The serve loop then calls
`pipeline.process_live_window(window)` exactly once per emitted window,
creating a discrete, non-overlapping inference cadence.

Architecture it enforces
────────────────────────
    raw telemetry packets
            ↓
    LiveWindowAccumulator.push()
            ↓
    completed window  (or None — still accumulating)
            ↓
    pipeline.process_live_window(window)   ← ONE inference per window

Why this matters
────────────────
With window_size=24 and stride=24 (non-overlapping), 1 092 telemetry
packets produce ≈ 45 inference cycles instead of 1 092.  This eliminates:

  • alert duplication from overlapping inferences on near-identical buffers
  • fake persistence increments on every packet
  • exploding anomaly trajectory lengths
  • motif / escalation reinforcement without new information
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, Deque, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class LiveWindowAccumulator:
    """
    Buffers per-player telemetry packets and emits fixed-size inference
    windows.

    Parameters
    ----------
    window_size : int
        Number of telemetry events required to form one inference window.
        Must match the sequence length expected by the trained model
        (default 24).
    stride : int
        How many events to discard from the front of the buffer after
        emitting a window.  Set equal to ``window_size`` for
        non-overlapping windows (recommended during initial stabilisation).
        Values smaller than ``window_size`` create overlapping windows and
        reintroduce the leakage this class is designed to prevent.
    """

    def __init__(
        self,
        window_size: int = 24,
        stride: int = 24,
        ignore_time_gaps: bool = False,        # set True for batch/replay data
        ignore_session_boundaries: bool = False,  # set True for batch/replay data
    ) -> None:

        self.ignore_time_gaps = ignore_time_gaps
        self.ignore_session_boundaries = ignore_session_boundaries
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if stride < window_size and stride > 1:
            logger.warning(
                "LiveWindowAccumulator: stride=%d < window_size=%d — "
                "windows will overlap and may reintroduce inference leakage. "
                "Use stride=window_size for non-overlapping windows.",
                stride,
                window_size,
            )

        self.window_size = window_size
        self.stride = stride

        # Per-player ring buffers.  maxlen caps memory use; once full,
        # old events are automatically dropped from the left.
        self._buffers: Dict[str, Deque[dict]] = {}

        # Counters for diagnostics / logging
        self._events_seen: Dict[str, int] = {}
        self._windows_emitted: Dict[str, int] = {}
        self._current_session: Dict[int, str] = {}
        self._last_ts: Dict[int, pd.Timestamp] = {}
        self._reset_flags: Dict[str, bool] = {}
    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def push(
        self,
        player_id: str,
        event: dict,
    ) -> Optional[List[dict]]:
        """
        Append one telemetry event for *player_id*.

        Returns
        -------
        list[dict]
            A completed inference window.
        None
            Window not yet complete.
        """

        # ─────────────────────────────────────────────
        # Canonicalize player identity
        # Prevent:
        #   "7" vs 7 vs "007"
        # fragmentation bugs.
        # ─────────────────────────────────────────────
        player_id = str(player_id).strip()

        # ─────────────────────────────────────────────
        # Per-player state init
        # ─────────────────────────────────────────────
        if player_id not in self._buffers:
            # logger.info(
            #     "ACCUMULATOR INIT | player=%s",
            #     player_id,
            # )

            self._buffers[player_id] = deque(maxlen=self.window_size)
            self._events_seen[player_id] = 0
            self._windows_emitted[player_id] = 0
            self._reset_flags[player_id] = True

        # ─────────────────────────────────────────────
        # Canonical session identity
        #
        # IMPORTANT:
        # session_id and match_id must NOT alternate.
        # Prefer match_id globally.
        # ─────────────────────────────────────────────
        incoming_session = (
            event.get("match_id") or event.get("session_id") or "default_session"
        )

        incoming_session = str(incoming_session)

        prev_session = self._current_session.get(player_id)

        # ─────────────────────────────────────────────
        # Session boundary reset
        # Disabled when ignore_session_boundaries=True
        # (e.g. historical replay with interleaved sessions)
        # ─────────────────────────────────────────────
        if (
            not self.ignore_session_boundaries
            and prev_session is not None
            and incoming_session != prev_session
        ):
            logger.warning(
                "BUFFER RESET | reason=session_change "
                "player=%s old=%s new=%s buf_len=%d",
                player_id,
                prev_session,
                incoming_session,
                len(self._buffers[player_id]),
            )

            self._buffers[player_id].clear()
            self._reset_flags[player_id] = True

        self._current_session[player_id] = incoming_session

        # ─────────────────────────────────────────────
        # Temporal discontinuity reset
        # ─────────────────────────────────────────────
        if not self.ignore_time_gaps:

            ts_raw = event.get("ts")

            if ts_raw is None:
                logger.warning(
                    "EVENT MISSING TS | player=%s",
                    player_id,
                )
            else:
                try:
                    from pandas import to_datetime

                    curr_ts = pd.to_datetime(ts_raw, utc=True)

                    prev_ts = self._last_ts.get(player_id)

                    if prev_ts is not None:

                        if not isinstance(prev_ts, pd.Timestamp):
                            prev_ts = pd.to_datetime(prev_ts, utc=True)

                        gap_s = (curr_ts - prev_ts).total_seconds()

                        # Negative time movement
                        if gap_s < 0:
                            logger.warning(
                                "BUFFER RESET | reason=time_reversal "
                                "player=%s gap=%.3fs",
                                player_id,
                                gap_s,
                            )

                            self._buffers[player_id].clear()

                        # Large telemetry gap
                        elif gap_s > 60:

                            logger.warning(
                                "BUFFER RESET | reason=time_gap "
                                "player=%s gap=%.1fs buf_len=%d",
                                player_id,
                                gap_s,
                                len(self._buffers[player_id]),
                            )

                            self._buffers[player_id].clear()

                    self._last_ts[player_id] = curr_ts

                except Exception as exc:

                    logger.warning(
                        "TIMESTAMP PARSE FAILED | player=%s ts=%r err=%s",
                        player_id,
                        ts_raw,
                        exc,
                    )

        # ─────────────────────────────────────────────
        # Normal accumulation
        # ─────────────────────────────────────────────
        buf = self._buffers[player_id]

        buf.append(event)

        self._events_seen[player_id] += 1

        # Debug visibility
        # if len(buf) in (1, 5, 10, 15, 20, self.window_size):
            # logger.info(
            #     "BUFFER STATUS | player=%s size=%d/%d session=%s",
            #     player_id,
            #     len(buf),
            #     self.window_size,
            #     incoming_session,
            # )

        # Not enough events yet
        if len(buf) < self.window_size:
            return None

        # ─────────────────────────────────────────────
        # Snapshot complete window
        # ─────────────────────────────────────────────
        window: List[dict] = list(buf)

        # ─────────────────────────────────────────────
        # Advance by stride
        # Non-overlapping windows:
        # window_size == stride
        # ─────────────────────────────────────────────
        n_drop = min(
            self.stride,
            len(buf),
        )

        for _ in range(n_drop):
            buf.popleft()

        self._windows_emitted[player_id] += 1

        # logger.info(
        #     "WINDOW EMIT | player=%s size=%d " "total_windows=%d total_events=%d",
        #     player_id,
        #     len(window),
        #     self._windows_emitted[player_id],
        #     self._events_seen[player_id],
        # )

        return window

    def reset(self, player_id: str) -> None:
        """Discard the buffer for a single player (e.g. on session change)."""
        self._buffers.pop(player_id, None)
        self._events_seen.pop(player_id, None)
        self._windows_emitted.pop(player_id, None)

    def reset_all(self) -> None:
        """Discard all buffers (e.g. on match restart)."""
        self._buffers.clear()
        self._events_seen.clear()
        self._windows_emitted.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return a snapshot of accumulation progress for all players."""
        return {
            pid: {
                "buffered": len(buf),
                "needed": max(0, self.window_size - len(buf)),
                "events_seen": self._events_seen.get(pid, 0),
                "windows_emitted": self._windows_emitted.get(pid, 0),
            }
            for pid, buf in self._buffers.items()
        }
    
    def consume_reset_flag(
    self,
    player_id: str,
) -> bool:
        return self._reset_flags.pop(player_id, False)