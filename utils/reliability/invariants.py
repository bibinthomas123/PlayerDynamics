"""
Runtime invariant enforcement for the sports anomaly detection platform.
Ensures that the system never silently continues after scientific or operational corruption.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, Optional, List
import logging

from utils.reliability.safe_mode import safe_mode, SafeModeLevel

logger = logging.getLogger(__name__)

class InvariantSeverity(Enum):
    INFO = auto()     # Logged, no action
    WARNING = auto()   # Logged, alert operators, continue
    CRITICAL = auto()  # Immediate SAFE MODE trigger, stop inference
    FATAL = auto()    # Immediate process termination

@dataclass
class InvariantViolation:
    invariant_id: str
    severity: InvariantSeverity
    timestamp: datetime
    message: str
    context: Dict[str, Any]
    version: str = "1.0"

class SafeModeTrigger(Exception):
    """Exception raised to force the system into Safe Mode."""
    def __init__(self, violation: InvariantViolation):
        self.violation = violation
        super().__init__(violation.message)

class SystemInvariantGuard:
    """
    Machine-enforced guard for operational and scientific invariants.

    Invariants are checked at critical pipeline junctions. Any violation must be
    structurally logged and, depending on severity, can trigger Safe Mode.
    """
    def __init__(self):
        self.violations_log: List[InvariantViolation] = []
        self.safe_mode_active = False
        self.active_level = 0 # 0: Normal, 1: Explainability degraded, 2: Telemetry degraded, 3: Scientific invalidation

    def check(self, invariant_id: str, condition: bool, severity: InvariantSeverity,
               message: str, context: Dict[str, Any]) -> None:
        """
        Evaluates an invariant condition. If False, logs a violation and potentially triggers Safe Mode.
        """
        if condition:
            return

        violation = InvariantViolation(
            invariant_id=invariant_id,
            severity=severity,
            timestamp=datetime.now(tz=timezone.utc),
            message=message,
            context=context
        )
        self.violations_log.append(violation)

        # Structural Logging
        log_payload = {
            "event": "INVARIANT_VIOLATION",
            "id": invariant_id,
            "severity": severity.name,
            "msg": message,
            "ctx": context,
            "ts": violation.timestamp.isoformat()
        }

        if severity == InvariantSeverity.INFO:
            logger.info("Invariant [%s] INFO: %s", invariant_id, message)
        elif severity == InvariantSeverity.WARNING:
            logger.warning("Invariant [%s] WARNING: %s | Context: %s", invariant_id, message, context)
        elif severity == InvariantSeverity.CRITICAL:
            logger.error("Invariant [%s] CRITICAL: %s | TRIGGERING SAFE MODE", invariant_id, message)
            self.safe_mode_active = True
            self.active_level = 3
            safe_mode.set_level(SafeModeLevel.LEVEL_3)
            raise SafeModeTrigger(violation)
        elif severity == InvariantSeverity.FATAL:
            logger.critical("Invariant [%s] FATAL: %s | TERMINATING PROCESS", invariant_id, message)
            raise SystemExit(1)

    def reset_safe_mode(self) -> None:
        """Manually reset safe mode after operator intervention."""
        logger.info("Safe mode reset by operator.")
        self.safe_mode_active = False
        self.active_level = 0

# Global Guard Instance
guard = SystemInvariantGuard()
