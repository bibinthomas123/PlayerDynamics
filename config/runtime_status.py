"""
runtime_status.py — lightweight in-process tracker for the `run` dashboard.

RedisStreamProducer.publish() calls record_publish() after every successful
XADD; _run_status_display() in main.py reads the snapshot to draw the live
health panel without touching Redis more than once per refresh cycle.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional


class RuntimeStatusTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream_last: Dict[str, float] = {}
        self._stream_count: Dict[str, int] = {}
        self.start_time: float = time.time()
        self.mode: str = "unknown"
        self.match_id: str = ""
        self.match_label: str = ""
        self.speed: float = 1.0

    def record_publish(self, stream: str) -> None:
        with self._lock:
            t = time.time()
            self._stream_last[stream] = t
            self._stream_count[stream] = self._stream_count.get(stream, 0) + 1

    def stream_last(self, stream: str) -> Optional[float]:
        with self._lock:
            return self._stream_last.get(stream)

    def stream_count(self, stream: str) -> int:
        with self._lock:
            return self._stream_count.get(stream, 0)

    def last_publish_any(self) -> Optional[float]:
        with self._lock:
            return max(self._stream_last.values()) if self._stream_last else None


_instance: Optional[RuntimeStatusTracker] = None
_lock = threading.Lock()


def get_tracker() -> RuntimeStatusTracker:
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = RuntimeStatusTracker()
    return _instance


def reset_tracker() -> RuntimeStatusTracker:
    global _instance
    with _lock:
        _instance = RuntimeStatusTracker()
        return _instance
