"""
Deterministic State & Mutation Framework
Ensures exactly-once semantics, event-time ordering, and replay-safe state transitions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, Optional, List, Callable, Generic, TypeVar, Set
import hashlib
import json
import logging
from copy import deepcopy
from typing import Any


logger = logging.getLogger(__name__)

T = TypeVar("T")

class MutationStatus(Enum):
    PENDING = auto()
    COMMITTED = auto()
    REJECTED = auto()

@dataclass(frozen=True)
class EventContract:
    """Immutable event schema to prevent schema drift during replay."""
    event_id: str
    player_id: int
    timestamp: datetime
    payload: Dict[str, Any]
    version: str = "1.0"

    def fingerprint(self) -> str:
        """Deterministic hash of the event for idempotency checks."""
        content = json.dumps(self.payload, sort_keys=True)
        return hashlib.sha256(f"{self.event_id}:{content}".encode()).hexdigest()

@dataclass
class StateMutation:
    """A versioned change to system state."""
    mutation_id: str
    target_object: str
    previous_version: int
    new_version: int
    change_set: Dict[str, Any]
    event_id: str  # The causal event that triggered this mutation
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

class MutationJournal:
    """
    Append-only journal of all state mutations.
    Allows for full audit reconstruction and crash-safe recovery.
    """
    def __init__(self):
        self._journal: List[StateMutation] = []
        self._processed_events: Set[str] = set()

    def commit(self, mutation: StateMutation) -> bool:
        """Exactly-once commit of a state mutation."""
        if mutation.event_id in self._processed_events:
            logger.info("Duplicate event %s detected. Skipping mutation.", mutation.event_id)
            return False

        self._journal.append(mutation)
        self._processed_events.add(mutation.event_id)
        return True

    def get_history(self, target_object: str) -> List[StateMutation]:
        return [m for m in self._journal if m.target_object == target_object]

    def rebuild_state(self, target_object: str, initial_state: Any) -> Any:
        """
        Rebuild object state by replaying all committed mutations
        in chronological order.

        Supported operations:
            - set
            - update
            - append
            - increment
            - delete
        """

        current_state = deepcopy(initial_state)

        history = sorted(
            self.get_history(target_object),
            key=lambda m: m.timestamp
        )

        for mutation in history:

            op = mutation.operation
            changes = mutation.change_set

            try:

                # ─────────────────────────────────────────────
                # SET
                # Replace/add dictionary keys
                # ─────────────────────────────────────────────
                if op == "set":

                    if not isinstance(current_state, dict):
                        raise TypeError(
                            f"'set' operation requires dict state "
                            f"(got {type(current_state).__name__})"
                        )

                    for key, value in changes.items():
                        current_state[key] = deepcopy(value)

                # ─────────────────────────────────────────────
                # UPDATE
                # Recursive dict merge
                # ─────────────────────────────────────────────
                elif op == "update":

                    if not isinstance(current_state, dict):
                        raise TypeError(
                            f"'update' operation requires dict state "
                            f"(got {type(current_state).__name__})"
                        )

                    for key, value in changes.items():

                        if (
                            key in current_state
                            and isinstance(current_state[key], dict)
                            and isinstance(value, dict)
                        ):
                            current_state[key].update(deepcopy(value))
                        else:
                            current_state[key] = deepcopy(value)

                # ─────────────────────────────────────────────
                # APPEND
                # Append items to list state
                # ─────────────────────────────────────────────
                elif op == "append":

                    if not isinstance(current_state, list):
                        raise TypeError(
                            f"'append' operation requires list state "
                            f"(got {type(current_state).__name__})"
                        )

                    if isinstance(changes, list):
                        current_state.extend(deepcopy(changes))
                    else:
                        current_state.append(deepcopy(changes))

                # ─────────────────────────────────────────────
                # INCREMENT
                # Numeric increments
                # ─────────────────────────────────────────────
                elif op == "increment":

                    if not isinstance(current_state, dict):
                        raise TypeError(
                            f"'increment' operation requires dict state "
                            f"(got {type(current_state).__name__})"
                        )

                    for key, delta in changes.items():

                        if key not in current_state:
                            current_state[key] = 0

                        current_state[key] += delta

                # ─────────────────────────────────────────────
                # DELETE
                # Remove dictionary keys
                # ─────────────────────────────────────────────
                elif op == "delete":

                    if not isinstance(current_state, dict):
                        raise TypeError(
                            f"'delete' operation requires dict state "
                            f"(got {type(current_state).__name__})"
                        )

                    keys = changes

                    if not isinstance(keys, list):
                        keys = [keys]

                    for key in keys:
                        current_state.pop(key, None)

                # ─────────────────────────────────────────────
                # UNKNOWN OP
                # ─────────────────────────────────────────────
                else:
                    raise ValueError(
                        f"Unknown mutation operation: {op}"
                    )

            except Exception as exc:

                logger.exception(
                    "Failed replaying mutation "
                    "event_id=%s target=%s operation=%s",
                    mutation.event_id,
                    mutation.target_object,
                    mutation.operation,
                )

                raise RuntimeError(
                    f"Replay failed for mutation {mutation.event_id}"
                ) from exc

        return current_state

class TemporalCausalityGuard:
    """
    Enforces event-time monotonicity.
    Prevents out-of-order packets from corrupting state.
    """
    def __init__(self):
        self._last_event_time: Dict[int, datetime] = {}

    def validate_sequence(self, player_id: int, event_time: datetime) -> bool:
        last_time = self._last_event_time.get(player_id)
        if last_time and event_time < last_time:
            return False # Out of order

        self._last_event_time[player_id] = event_time
        return True
