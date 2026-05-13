"""
Hardened Alert Management State Machine.
Deterministic transitions to prevent alert fragmentation and silent failures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Set
from datetime import datetime, timezone
from enum import IntEnum, Enum, auto
import logging

logger = logging.getLogger(__name__)

class AlertLevel(IntEnum):
    NONE = 0
    WARNING = 1
    SUSTAINED = 2
    CRITICAL = 3
    HOLD = 4        # Telemetry loss / Degraded state
    SAFE_MODE = 5   # System-wide scientific invalidation

@dataclass
class AlertState:
    """Deterministic state for a player-alert pair."""
    level: AlertLevel = AlertLevel.NONE
    first_triggered_at: Optional[datetime] = None
    last_triggered_at: Optional[datetime] = None
    persistence_count: int = 0
    recovery_count: int = 0
    episode_id: int = 0
    telemetry_confidence: float = 1.0

    def transition(self, new_level: AlertLevel) -> bool:
        if self.level == new_level:
            return False
        self.level = new_level
        return True

class AlertManager:
    """
    Deterministic Finite State Machine (FSM) for Alert Management.

    Guarantees:
    - Hysteresis: Prevents rapid toggling between states.
    - Event-Time Semantics: State transitions based on data-time, not processing-time.
    - Telemetry-Loss Hold: State shifts to HOLD during blackouts, not recovery.
    """
    def __init__(
        self,
        min_persistence: int = 3,
        cooldown_seconds: int = 300,
        escalation_threshold: int = 10,
        recovery_threshold: int = 3
    ):
        self.min_persistence = min_persistence
        self.cooldown_seconds = cooldown_seconds
        self.escalation_threshold = escalation_threshold
        self.recovery_threshold = recovery_threshold

        # State storage: player_id -> { alert_type -> AlertState }
        self._states: Dict[int, Dict[str, AlertState]] = {}
        self._global_safe_mode = False

    def set_safe_mode(self, active: bool):
        self._global_safe_mode = active
        if active:
            logger.critical("AlertManager: ENTERING GLOBAL SAFE MODE. All alerts suppressed/flagged.")

    def process_signal(self, player_id: int, alert_type: str,
                      signal_active: bool,
                      confidence: float = 1.0,
                      event_time: Optional[datetime] = None) -> AlertLevel:
        """
        Deterministic state transition based on current signal and telemetry health.
        """
        now = event_time or datetime.now(tz=timezone.utc)

        if self._global_safe_mode:
            return AlertLevel.SAFE_MODE

        player_map = self._states.setdefault(player_id, {})
        state = player_map.setdefault(alert_type, AlertState())

        # 1. Telemetry Health Check
        if confidence < 0.4:
            # Telemetry failure -> State HOLD. We do NOT recover alerts during blackouts.
            state.level = AlertLevel.HOLD
            return AlertLevel.HOLD

        # 2. Signal Processing (Sustenance & Recovery)
        if signal_active:
            state.persistence_count += 1
            state.recovery_count = 0

            # Trigger/Escalate
            if state.persistence_count >= self.min_persistence:
                target_level = AlertLevel.WARNING
                if state.persistence_count >= self.escalation_threshold:
                    target_level = AlertLevel.CRITICAL

                # Hysteresis: Avoid flapping if already at a higher or equal level
                if target_level > state.level:
                    state.transition(target_level)
                    state.first_triggered_at = state.first_triggered_at or now
                    state.episode_id += 1

                state.last_triggered_at = now
        else:
            # Recovery Logic
            state.persistence_count = 0
            state.recovery_count += 1

            if state.recovery_count >= self.recovery_threshold:
                if state.level != AlertLevel.NONE:
                    state.transition(AlertLevel.NONE)
                    state.first_triggered_at = None

        return state.level

    def get_state(self, player_id: int, alert_type: str) -> AlertState:
        return self._states.get(player_id, {}).get(alert_type, AlertState())

    def clear_player(self, player_id: int):
        if player_id in self._states:
            del self._states[player_id]
