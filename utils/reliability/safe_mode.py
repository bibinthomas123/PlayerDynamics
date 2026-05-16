"""
Safe Mode Controller
Implements graded degradation levels to ensure scientific validity under stress.
"""
from __future__ import annotations
from enum import IntEnum
from typing import Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SafeModeLevel(IntEnum):
    NORMAL  = 0   # Full system operational
    LEVEL_1 = 1   # Explainability degraded (SHAP/LLM suppressed)
    LEVEL_2 = 2   # Telemetry degraded (Inference flagged, limited alerts)
    LEVEL_3 = 3   # Scientific invalidation (Freeze calibration, critical alerts only)


class SafeModeController:
    """
    Manages the operational state of the system.

    Coordinates the degradation of features based on the highest active
    invariant violation or system stress signal.

    trigger(id, level) raises the system level if the new level is higher.
    clear(id)          removes a trigger; level drops to the next highest.
    set_level(level)   forces a level directly (used by InvariantGuard).
    """

    def __init__(self) -> None:
        self._current_level: SafeModeLevel = SafeModeLevel.NORMAL
        self._active_triggers: Set[str] = set()
        self._trigger_levels: Dict[str, SafeModeLevel] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def trigger(self, trigger_id: str, level: SafeModeLevel) -> None:
        """Register a trigger that requires *level* or above."""
        logger.warning(
            "SafeMode trigger: %s → %s (was %s)",
            trigger_id, level.name, self._current_level.name,
        )
        self._active_triggers.add(trigger_id)
        self._trigger_levels[trigger_id] = level
        self._update_level()

    def clear(self, trigger_id: str) -> None:
        """Remove a trigger and recalculate the current level."""
        if trigger_id in self._active_triggers:
            self._active_triggers.discard(trigger_id)
            self._trigger_levels.pop(trigger_id, None)
            self._update_level()
            logger.info(
                "SafeMode cleared: %s → level now %s",
                trigger_id, self._current_level.name,
            )

    def set_level(self, level: SafeModeLevel) -> None:
        """Force-set the system level directly (used by InvariantGuard)."""
        logger.critical("SafeMode forced to: %s", level.name)
        self._current_level = level

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def level(self) -> SafeModeLevel:
        return self._current_level

    def is_feature_enabled(self, feature_id: str, required_level: SafeModeLevel) -> bool:
        """
        Return True when the feature is allowed to run.

        A feature requiring *required_level* is disabled once the system
        reaches that level.  Example: SHAP requires LEVEL_1 — disabled
        when current level is LEVEL_1 or higher.
        """
        return self._current_level < required_level

    def is_mutation_allowed(self, target_object: str) -> bool:
        """
        Return False at LEVEL_3 to freeze all calibration / state updates.
        """
        if self._current_level >= SafeModeLevel.LEVEL_3:
            logger.warning(
                "SafeMode LEVEL_3: mutation blocked for '%s'", target_object
            )
            return False
        return True

    def status(self) -> dict:
        """Diagnostic snapshot."""
        return {
            "level":    self._current_level.name,
            "triggers": {k: v.name for k, v in self._trigger_levels.items()},
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _update_level(self) -> None:
        if not self._active_triggers:
            self._current_level = SafeModeLevel.NORMAL
        else:
            self._current_level = max(
                self._trigger_levels.values(),
                default=SafeModeLevel.NORMAL,
            )
        logger.info("SafeMode level updated → %s", self._current_level.name)


# Module-level singleton — imported by all subsystems.
safe_mode = SafeModeController()