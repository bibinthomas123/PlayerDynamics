"""
Safe Mode Controller
Implements graded degradation levels to ensure scientific validity under stress.
"""
from __future__ import annotations
from enum import IntEnum, auto
from typing import Set, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SafeModeLevel(IntEnum):
    NORMAL = 0       # Full system operational
    LEVEL_1 = 1      # Explainability degraded (SHAP/LLM suppressed)
    LEVEL_2 = 2      # Telemetry degraded (Inference flagged, limited alerts)
    LEVEL_3 = 3      # Scientific invalidation (Freeze calibration, critical alerts only)

class SafeModeController:
    """
    Manages the operational state of the system.

    Coordinates the degradation of features based on the highest active
    invariant violation or system stress signal.
    """
    def __init__(self):
        self._current_level = SafeModeLevel.NORMAL
        self._active_triggers: Set[str] = set()
        self._overrides: Dict[str, SafeModeLevel] = {}

        # Add to __init__:
        self._trigger_levels: Dict[str, SafeModeLevel] = {}

        # Replace trigger() to store the level:
        def trigger(self, trigger_id: str, level: SafeModeLevel):
            logger.warning("Safe Mode Triggered: %s -> Level %s", trigger_id, level.name)
            self._active_triggers.add(trigger_id)
            self._trigger_levels[trigger_id] = level
            self._update_level()

        # Replace clear() to remove the stored level:
        def clear(self, trigger_id: str):
            if trigger_id in self._active_triggers:
                self._active_triggers.remove(trigger_id)
                self._trigger_levels.pop(trigger_id, None)
                self._update_level()

        # Replace _update_level() to actually compute the max:
        def _update_level(self):
            if not self._active_triggers:
                self._current_level = SafeModeLevel.NORMAL
                return
            self._current_level = max(self._trigger_levels.values(), default=SafeModeLevel.NORMAL)
            
        # The system level is the maximum of all active triggers
        # In a real system, we'd look up the trigger_id's associated level
        # For now, we assume triggers are passed with their levels.
        # We'll track the current max level among all active triggers.
        # Since triggers are set via trigger(), we'll just store the state.
        pass

    @property
    def level(self) -> SafeModeLevel:
        return self._current_level

    def set_level(self, level: SafeModeLevel):
        """Force set the system level (used by InvariantGuard)."""
        logger.critical("System level forced to: %s", level.name)
        self._current_level = level

    def is_feature_enabled(self, feature_id: str, required_level: SafeModeLevel) -> bool:
        """
        Determines if a feature should run based on current Safe Mode level.
        Example: SHAP requires LEVEL_1. If level is LEVEL_1, SHAP is disabled.
        """
        return self._current_level < required_level

    def is_mutation_allowed(self, target_object: str) -> bool:
        """
        Freezes mutable subsystems during high-level Safe Mode.
         Calibration and model weights must be frozen at LEVEL_3.
        """
        if self._current_level >= SafeModeLevel.LEVEL_3:
            # Only allow read-only operations; block all calibration/state updates
            return False
        return True

# Global Controller
safe_mode = SafeModeController()
