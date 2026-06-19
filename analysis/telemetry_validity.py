"""
Telemetry Validity Layer (TVL)
Hardens the pipeline against corrupted or implausible sports telemetry.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TelemetryStatus(Enum):
    VALID = auto()     # High confidence, suitable for all inference
    DEGRADED = auto()  # Minor issues, inference allowed but flagged
    INVALID = auto()   # Plausibility failure, MUST NOT trigger physiological alerts
    UNKNOWN = auto()   # Insufficient data to determine status

@dataclass
class ValidityMetrics:
    status: TelemetryStatus
    confidence: float  # 0.0 to 1.0
    mask_completeness: float
    jitter_ms: float
    plausibility_score: float
    issues: list[str]

class TelemetryValidityLayer:
    """
    Analyzes raw event streams for physical plausibility and sensor health.
    Prevents corrupted data from poisoning adaptive calibration.
    """
    def __init__(self, replay_mode: bool = False):
        # Per-player health tracking
        self._player_health: Dict[int, Dict[str, Any]] = {}
        self.replay_mode = replay_mode

        # Physical constants for plausibility checks
        self.MAX_SPEED_MS = 13.5        # ~48 km/h - elite sprint peak (Mbappe-class burst)
        self.MAX_HR_BPM = 220.0         # Human physiological ceiling
        self.MAX_ACCEL_MS2 = 15.0       # Peak acceleration for athlete change-of-direction
        self.MIN_HR_BPM = 30.0          # Rest/Bradycardia floor

    def validate_event(self, player_id: int, event: Dict[str, Any]) -> ValidityMetrics:
        """
        Performs a multi-stage validity check on a single event.
        """
        issues = []
        confidence = 1.0

        # 1. Mask Completeness
        # Assume event has a set of expected keys

        # Field names must match SequenceWindowBuilder._extract_features() output:
        # - "is_sprint" (not "sprint_flag" — that's internal to the feature array)
        # - "distance_delta_m" is what the data generator emits;
        #    SequenceWindowBuilder computes "distance_delta" internally from x/y.
        #    TVL operates on the raw event dict, so check the generator field name.
        expected_keys = {"speed_ms", "heart_rate_bpm", "distance_delta_m", "is_sprint"}
        present_keys = {k for k in expected_keys if event.get(k) is not None}
        completeness = len(present_keys) / len(expected_keys)

        if completeness < 0.75:
            return ValidityMetrics(
                status=TelemetryStatus.INVALID,
                confidence=0.0,
                mask_completeness=completeness,
                jitter_ms=0.0,
                plausibility_score=0.0,
                issues=[f"low_completeness_{completeness:.2f}"],
            )

        # 2. Physical Plausibility
        spd = event.get("speed_ms", 0.0)
        accel = abs(event.get("accel", 0.0))

        if accel > 12.0:
            issues.append(f"implausible_accel_{accel:.2f}")
            confidence = 0.0

        if spd > self.MAX_SPEED_MS:
            overage = (spd - self.MAX_SPEED_MS) / self.MAX_SPEED_MS
            if overage > 0.20:          # >20% over ceiling → hard reject
                issues.append(f"implausible_speed_{spd:.2f}")
                confidence = 0.0
            else:                        # borderline → degrade, don't discard
                issues.append(f"borderline_speed_{spd:.2f}")
                confidence = max(0.0, confidence - 0.25)

        hr_raw = event.get("heart_rate_bpm")
        if hr_raw is None:
            # Sensor absent (wearable not worn/synced) — not a bad reading.
            # Reduce confidence but stay VALID; movement-based inference proceeds.
            confidence = max(0.0, confidence - 0.20)
            issues.append("hr_sensor_absent")
        else:
            hr = float(hr_raw)
            if hr > self.MAX_HR_BPM or hr < self.MIN_HR_BPM:
                issues.append(f"implausible_hr_{hr:.0f}")
                confidence = 0.0

        # 3. Temporal Monotonicity (if timestamp present)
        ts_raw = event.get("ts") or event.get("timestamp")
        ts = None
        if ts_raw is not None:
            try:
                import pandas as pd
                ts = pd.to_datetime(ts_raw, utc=True)
            except Exception:
                pass  # unparseable timestamp — skip monotonicity check

        if ts is not None:
            prev_ts = self._get_prev_ts(player_id)
            if prev_ts is not None:
                try:
                    dt = (ts - prev_ts).total_seconds()
                    if dt <= 0:
                        if self.replay_mode:
                            # Timestamp reversals are expected in replay streams
                            # (interleaved sessions replayed at CPU speed).
                            # Degrade rather than invalidate so inference is not
                            # silently dropped. Use a distinct issue marker so
                            # audit/evaluation can distinguish replay disorder
                            # from genuine live sensor corruption.
                            issues.append("replay_non_monotonic_timestamp")
                            confidence = min(confidence, 0.7)
                        else:
                            issues.append("non_monotonic_timestamp")
                            confidence = 0.0
                    elif dt > 5.0:
                        if self.replay_mode:
                            # Large forward gaps are expected in replay streams
                            # (consecutive events may be days/seasons apart).
                            # No confidence penalty — this is stream ordering,
                            # not a sensor failure. Distinct marker preserves
                            # audit separability from live gap events.
                            issues.append(f"replay_timestamp_gap_{dt:.2f}s")
                        else:
                            issues.append(f"timestamp_gap_{dt:.2f}s")
                            confidence -= 0.3
                except Exception:
                    pass  # incompatible types — skip
            self._update_ts(player_id, ts)

        # Determine overall status
        if confidence <= 0.0:
            status = TelemetryStatus.INVALID
        elif confidence < 0.8:
            status = TelemetryStatus.DEGRADED
        else:
            status = TelemetryStatus.VALID
 
        if status != TelemetryStatus.VALID:
            # Separate genuine live issues from expected replay noise.
            # Only live issues warrant a WARNING — replay gaps and timestamp
            # reversals are structural properties of the replay stream, not
            # sensor failures, and must not pollute the operator warning channel.
            live_issues = [i for i in issues if not i.startswith("replay_")]
            if live_issues:
                logger.warning(
                    "Telemetry degraded | player=%d | status=%s | issues=%s",
                    player_id, status.name, live_issues,
                )
            else:
                logger.debug(
                    "Telemetry degraded | player=%d | status=%s | issues=%s",
                    player_id, status.name, issues,
                )
        
        return ValidityMetrics(
            status=status,
            confidence=max(0.0, confidence),
            mask_completeness=completeness,
            jitter_ms=0.0, # Requires window-level analysis
            plausibility_score=confidence,
            issues=issues
        )

    def reset_player(self, player_id: int) -> None:
        """Clear all per-player temporal state (call on session boundary reset).

        _player_health[player_id] is the single dict that holds all mutable
        per-player tracking — currently only "last_ts". Dropping the whole
        entry is safe and future-proof: if new keys are added to the dict
        later they are cleared automatically rather than silently carrying
        stale values into the next session.
        """
        self._player_health.pop(player_id, None)

    def _get_prev_ts(self, player_id: int) -> Optional[float]:
        return self._player_health.get(player_id, {}).get("last_ts")

    def _update_ts(self, player_id: int, ts: Any) -> None:
        if player_id not in self._player_health:
            self._player_health[player_id] = {}
        self._player_health[player_id]["last_ts"] = ts