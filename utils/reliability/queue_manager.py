"""
Priority-Aware Queue Manager
Ensures deterministic backpressure and graceful degradation of non-critical services.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, Any, Optional, List, Callable
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

class Priority(IntEnum):
    INFERENCE = 0        # Critical: Must be processed immediately
    ALERT_STATE = 1       # High: Alert transitions must be monotonic
    TELEMETRY_INTEGRITY = 2 # Medium: TVL checks
    SHAP = 3              # Low: Explainability is an enhancement
    LLM_SUMMARY = 4       # Lowest: Natural language is non-critical

@dataclass
class QueueTask:
    task_id: str
    priority: Priority
    payload: Any
    timestamp: float = field(default_factory=time.time)
    deadline: float = 0.0 # 0.0 = no deadline

class BoundedPriorityQueue:
    """
    A deterministic queue with priority-based shedding.

    Under overload, the queue drops tasks starting from the lowest priority
    (LLM Summary) to protect the core inference path.
    """
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues: Dict[Priority, asyncio.Queue] = {
            p: asyncio.Queue() for p in Priority
        }
        self._current_size = 0
        self._dropped_counts: Dict[Priority, int] = {p: 0 for p in Priority}

    async def enqueue(self, task: QueueTask):
        """
        Enqueues a task. If the queue is full, it sheds the lowest priority
        tasks until space is made or the task itself is shed.
        """
        if self._current_size >= self.max_size:
            if not await self._shed_to_make_room(task.priority):
                logger.warning("Queue OVERLOAD: Dropping task %s (Priority %s)",
                                task.task_id, task.priority.name)
                self._dropped_counts[task.priority] += 1
                return False

        await self._queues[task.priority].put(task)
        self._current_size += 1
        return True

    async def dequeue(self) -> Optional[QueueTask]:
        """
        Dequeues the highest priority task available.
        """
        for p in Priority:
            q = self._queues[p]
            if not q.empty():
                task = await q.get()
                self._current_size -= 1
                return task
        return None

    async def _shed_to_make_room(self, incoming_priority: Priority) -> bool:
        """
        Attempts to drop tasks of lower priority than the incoming task.
        """
        # Try to drop from the lowest priority upwards
        for p in reversed(Priority):
            if p > incoming_priority:
                q = self._queues[p]
                if not q.empty():
                    try:
                        q.get_nowait()
                        self._current_size -= 1
                        self._dropped_counts[p] += 1
                        return True
                    except asyncio.QueueEmpty:
                        continue
        return False

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "current_size": self._current_size,
            "utilization": self._current_size / self.max_size,
            "dropped": self._dropped_counts,
            "status": "OVERLOADED" if self._current_size >= self.max_size else "HEALTHY"
        }
