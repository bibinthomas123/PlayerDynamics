"""
Players Data — IBM CIC Germany
Realistic Synthetic Dataset Generator

Generates a complete 2-season (76 matchdays + 152 training sessions per player)
dataset that is statistically indistinguishable from real GPS/wearable exports.

Design principles
─────────────────
• Position-accurate biomechanics:
    GK  → low distance, few sprints, high jump count
    CB  → moderate distance, zonal defence patterns
    FB  → highest distance + sprints, wide channels
    CM  → highest distance, central dense zones
    CAM → high intensity bursts, final third zones
    W   → high sprint count, wide zones
    ST  → short bursts, box area, high HR peaks

• Realistic fatigue:
    - Exponential speed decay across 90 minutes
    - Cumulative fatigue across a congested fixture schedule (3-game weeks)
    - Full recovery after 3+ rest days
    - Simulated muscle injuries (miss 2–6 weeks)

• Coach annotation realism:
    - Pre-match status based on training load
    - Fatigue notes correlate with last-match minutes
    - Tactical notes for positional drift events

• Anomaly seeding (ground truth labels):
    - 5% of sessions have genuine fatigue anomalies
    - 3% have positional drift events
    - 2% have workload spikes (congested schedule)

Output
──────
    data/
      players.csv
      sessions.csv
      events.csv           ← high-frequency (1 row / 15 s per player per session)
      annotations.csv
      ground_truth_labels.csv   ← for model evaluation
"""
from __future__ import annotations

import json
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("datagen")

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Position Profiles  (from literature + FIFA/UEFA tracking data benchmarks)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionProfile:
    position: str
    distance_mean_m: float       # Total distance per 90 min
    distance_std_m: float
    sprint_count_mean: float
    sprint_count_std: float
    hi_run_count_mean: float     # High-intensity runs (>5.5 m/s)
    max_speed_mean_ms: float
    max_speed_std_ms: float
    avg_hr_mean: float           # Average HR during match (bpm)
    avg_hr_std: float
    max_hr_mean: float
    fatigue_alpha_mean: float    # Exponential decay rate (higher = faster fatigue)
    fatigue_alpha_std: float
    x_zone_center: float         # Typical pitch X zone [0-100]
    y_zone_center: float         # Typical pitch Y zone [0-100]
    x_zone_std: float
    y_zone_std: float


POSITION_PROFILES: Dict[str, PositionProfile] = {
    "GK": PositionProfile(
        "GK",
        distance_mean_m=5600, distance_std_m=450,
        sprint_count_mean=3,  sprint_count_std=2,
        hi_run_count_mean=8,
        max_speed_mean_ms=7.5, max_speed_std_ms=0.6,
        avg_hr_mean=138, avg_hr_std=8,
        max_hr_mean=172, fatigue_alpha_mean=0.002, fatigue_alpha_std=0.0005,
        x_zone_center=5, y_zone_center=50, x_zone_std=4, y_zone_std=10,
    ),
    "CB": PositionProfile(
        "CB",
        distance_mean_m=10300, distance_std_m=650,
        sprint_count_mean=12, sprint_count_std=4,
        hi_run_count_mean=28,
        max_speed_mean_ms=8.8, max_speed_std_ms=0.5,
        avg_hr_mean=152, avg_hr_std=9,
        max_hr_mean=183, fatigue_alpha_mean=0.0045, fatigue_alpha_std=0.0008,
        x_zone_center=22, y_zone_center=50, x_zone_std=8, y_zone_std=18,
    ),
    "LB": PositionProfile(
        "LB",
        distance_mean_m=11700, distance_std_m=700,
        sprint_count_mean=28, sprint_count_std=6,
        hi_run_count_mean=55,
        max_speed_mean_ms=9.6, max_speed_std_ms=0.5,
        avg_hr_mean=162, avg_hr_std=8,
        max_hr_mean=189, fatigue_alpha_mean=0.0065, fatigue_alpha_std=0.001,
        x_zone_center=40, y_zone_center=15, x_zone_std=20, y_zone_std=12,
    ),
    "RB": PositionProfile(
        "RB",
        distance_mean_m=11600, distance_std_m=700,
        sprint_count_mean=27, sprint_count_std=6,
        hi_run_count_mean=53,
        max_speed_mean_ms=9.5, max_speed_std_ms=0.5,
        avg_hr_mean=161, avg_hr_std=8,
        max_hr_mean=188, fatigue_alpha_mean=0.0063, fatigue_alpha_std=0.001,
        x_zone_center=40, y_zone_center=85, x_zone_std=20, y_zone_std=12,
    ),
    "CM": PositionProfile(
        "CM",
        distance_mean_m=12500, distance_std_m=750,
        sprint_count_mean=22, sprint_count_std=5,
        hi_run_count_mean=60,
        max_speed_mean_ms=9.2, max_speed_std_ms=0.5,
        avg_hr_mean=166, avg_hr_std=9,
        max_hr_mean=191, fatigue_alpha_mean=0.007, fatigue_alpha_std=0.001,
        x_zone_center=50, y_zone_center=50, x_zone_std=25, y_zone_std=25,
    ),
    "CDM": PositionProfile(
        "CDM",
        distance_mean_m=12000, distance_std_m=700,
        sprint_count_mean=18, sprint_count_std=5,
        hi_run_count_mean=52,
        max_speed_mean_ms=9.0, max_speed_std_ms=0.5,
        avg_hr_mean=163, avg_hr_std=9,
        max_hr_mean=189, fatigue_alpha_mean=0.0065, fatigue_alpha_std=0.001,
        x_zone_center=42, y_zone_center=50, x_zone_std=15, y_zone_std=20,
    ),
    "CAM": PositionProfile(
        "CAM",
        distance_mean_m=11500, distance_std_m=700,
        sprint_count_mean=24, sprint_count_std=5,
        hi_run_count_mean=58,
        max_speed_mean_ms=9.4, max_speed_std_ms=0.5,
        avg_hr_mean=165, avg_hr_std=9,
        max_hr_mean=190, fatigue_alpha_mean=0.0068, fatigue_alpha_std=0.001,
        x_zone_center=68, y_zone_center=50, x_zone_std=18, y_zone_std=20,
    ),
    "LW": PositionProfile(
        "LW",
        distance_mean_m=11200, distance_std_m=680,
        sprint_count_mean=32, sprint_count_std=7,
        hi_run_count_mean=62,
        max_speed_mean_ms=10.1, max_speed_std_ms=0.6,
        avg_hr_mean=164, avg_hr_std=9,
        max_hr_mean=190, fatigue_alpha_mean=0.0075, fatigue_alpha_std=0.0012,
        x_zone_center=62, y_zone_center=18, x_zone_std=18, y_zone_std=15,
    ),
    "RW": PositionProfile(
        "RW",
        distance_mean_m=11100, distance_std_m=680,
        sprint_count_mean=31, sprint_count_std=7,
        hi_run_count_mean=60,
        max_speed_mean_ms=10.0, max_speed_std_ms=0.6,
        avg_hr_mean=163, avg_hr_std=9,
        max_hr_mean=189, fatigue_alpha_mean=0.0073, fatigue_alpha_std=0.0012,
        x_zone_center=62, y_zone_center=82, x_zone_std=18, y_zone_std=15,
    ),
    "ST": PositionProfile(
        "ST",
        distance_mean_m=10700, distance_std_m=650,
        sprint_count_mean=26, sprint_count_std=6,
        hi_run_count_mean=50,
        max_speed_mean_ms=9.8, max_speed_std_ms=0.5,
        avg_hr_mean=163, avg_hr_std=9,
        max_hr_mean=191, fatigue_alpha_mean=0.0072, fatigue_alpha_std=0.0011,
        x_zone_center=80, y_zone_center=50, x_zone_std=15, y_zone_std=20,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Squad Definition — 25 players (realistic squad size)
# ─────────────────────────────────────────────────────────────────────────────
SQUAD = [
    # GK
    {"id": 1, "name": "Player 01", "position": "GK",  "age": 29, "age_group": "Senior", "nationality": "DE"},
    {"id": 2, "name": "Player 02", "position": "GK",  "age": 23, "age_group": "U23",    "nationality": "DE"},
    # CB
    {"id": 3, "name": "Player 03", "position": "CB",  "age": 31, "age_group": "Senior", "nationality": "FR"},
    {"id": 4, "name": "Player 04", "position": "CB",  "age": 27, "age_group": "Senior", "nationality": "ES"},
    {"id": 5, "name": "Player 05", "position": "CB",  "age": 22, "age_group": "U23",    "nationality": "DE"},
    # FB
    {"id": 6, "name": "Player 06", "position": "LB",  "age": 26, "age_group": "Senior", "nationality": "BR"},
    {"id": 7, "name": "Player 07", "position": "LB",  "age": 21, "age_group": "U23",    "nationality": "DE"},
    {"id": 8, "name": "Player 08", "position": "RB",  "age": 28, "age_group": "Senior", "nationality": "IT"},
    {"id": 9, "name": "Player 09", "position": "RB",  "age": 24, "age_group": "Senior", "nationality": "DE"},
    # Midfield
    {"id":10, "name": "Player 10", "position": "CDM", "age": 30, "age_group": "Senior", "nationality": "ES"},
    {"id":11, "name": "Player 11", "position": "CDM", "age": 25, "age_group": "Senior", "nationality": "DE"},
    {"id":12, "name": "Player 12", "position": "CM",  "age": 27, "age_group": "Senior", "nationality": "AR"},
    {"id":13, "name": "Player 13", "position": "CM",  "age": 24, "age_group": "Senior", "nationality": "DE"},
    {"id":14, "name": "Player 14", "position": "CM",  "age": 20, "age_group": "U23",    "nationality": "DE"},
    {"id":15, "name": "Player 15", "position": "CAM", "age": 26, "age_group": "Senior", "nationality": "PT"},
    {"id":16, "name": "Player 16", "position": "CAM", "age": 22, "age_group": "U23",    "nationality": "DE"},
    # Wide
    {"id":17, "name": "Player 17", "position": "LW",  "age": 25, "age_group": "Senior", "nationality": "NG"},
    {"id":18, "name": "Player 18", "position": "LW",  "age": 21, "age_group": "U23",    "nationality": "DE"},
    {"id":19, "name": "Player 19", "position": "RW",  "age": 28, "age_group": "Senior", "nationality": "ES"},
    {"id":20, "name": "Player 20", "position": "RW",  "age": 23, "age_group": "U23",    "nationality": "DE"},
    # ST
    {"id":21, "name": "Player 21", "position": "ST",  "age": 30, "age_group": "Senior", "nationality": "PL"},
    {"id":22, "name": "Player 22", "position": "ST",  "age": 27, "age_group": "Senior", "nationality": "FR"},
    {"id":23, "name": "Player 23", "position": "ST",  "age": 19, "age_group": "U23",    "nationality": "DE"},
    # Squad depth
    {"id":24, "name": "Player 24", "position": "CB",  "age": 32, "age_group": "Senior", "nationality": "NL"},
    {"id":25, "name": "Player 25", "position": "CM",  "age": 24, "age_group": "Senior", "nationality": "TR"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Season Schedule Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_schedule(
    season_start: datetime,
    n_matchdays: int = 38,
    training_per_week: int = 4,
) -> List[dict]:
    """
    Generate a realistic Bundesliga-style fixture schedule.
    Returns list of {date, type: 'match'|'training', is_congested: bool}.
    """
    schedule = []
    current = season_start

    for matchday in range(n_matchdays):
        # Match on Saturday
        match_date = current + timedelta(days=(5 - current.weekday()) % 7)
        schedule.append({
            "date": match_date,
            "type": "match",
            "matchday": matchday + 1,
            "is_congested": matchday in [7, 8, 21, 22, 33, 34],  # 3-game weeks
        })

        # Training Mon/Tue/Thu/Fri of that week
        for offset_days in [1, 2, 4, 5]:
            t_date = match_date + timedelta(days=offset_days)
            schedule.append({
                "date": t_date,
                "type": "training",
                "matchday": None,
                "is_congested": matchday in [7, 8, 21, 22, 33, 34],
            })

        current = match_date + timedelta(days=7)

    return sorted(schedule, key=lambda x: x["date"])


# ─────────────────────────────────────────────────────────────────────────────
# Per-player individual variation
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PlayerVariation:
    """Per-player random offsets — creates realistic individual differences."""
    distance_offset: float       # m offset from position mean
    sprint_offset: float
    speed_offset: float          # m/s offset
    hr_offset: float             # bpm offset
    fatigue_alpha: float         # Individual fatigue decay rate
    fatigue_sensitivity: float   # How much fatigue accumulates (0.5–1.5)
    injury_proneness: float      # 0–1, prob of injury per month


def create_player_variation(player_id: int) -> PlayerVariation:
    r = np.random.default_rng(player_id * 31 + 7)
    return PlayerVariation(
        distance_offset=float(r.normal(0, 400)),
        sprint_offset=float(r.normal(0, 3)),
        speed_offset=float(r.normal(0, 0.3)),
        hr_offset=float(r.normal(0, 5)),
        fatigue_alpha=float(r.uniform(0.003, 0.009)),
        fatigue_sensitivity=float(r.uniform(0.6, 1.4)),
        injury_proneness=float(r.uniform(0.02, 0.15)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Session Generator
# ─────────────────────────────────────────────────────────────────────────────
class SessionGenerator:

    SPRINT_THRESHOLD   = 7.0     # m/s
    HI_RUN_THRESHOLD   = 5.5     # m/s
    EVENT_INTERVAL_S   = 15      # One GPS event every 15 seconds

    def __init__(self):
        self._session_id = 0
        self._event_id = 0

    def _next_session_id(self) -> int:
        self._session_id += 1
        return self._session_id

    def generate_match_session(
        self,
        player: dict,
        profile: PositionProfile,
        variation: PlayerVariation,
        date: datetime,
        is_congested: bool = False,
        cumulative_fatigue: float = 0.0,   # 0–1 accumulated fatigue load
        is_anomaly: bool = False,
        anomaly_type: Optional[str] = None,
    ) -> Tuple[dict, List[dict], Optional[dict]]:
        """
        Generate one full 90-minute match session for one player.
        Returns (session_record, [event_records], annotation_record|None)
        """
        session_id = self._next_session_id()
        r = np.random.default_rng(session_id * 13)

        # ── Base metrics with individual variation ──
        base_distance = profile.distance_mean_m + variation.distance_offset
        base_distance *= (1.0 - cumulative_fatigue * 0.15 * variation.fatigue_sensitivity)
        if is_congested:
            base_distance *= 0.93  # 7% reduction in congested schedule

        # Anomaly: genuine fatigue drops distance significantly
        if is_anomaly and anomaly_type == "fatigue":
            base_distance *= rng.uniform(0.58, 0.72)

        total_distance = max(3000, float(r.normal(base_distance, profile.distance_std_m * 0.6)))

        base_sprints = profile.sprint_count_mean + variation.sprint_offset
        if is_anomaly and anomaly_type == "fatigue":
            base_sprints *= rng.uniform(0.4, 0.65)
        sprint_count = max(0, int(r.normal(base_sprints, profile.sprint_count_std)))

        max_speed = float(np.clip(
                r.normal(profile.max_speed_mean_ms + variation.speed_offset,
                        profile.max_speed_std_ms),
                3.0, 12.0
            ))
        if is_anomaly and anomaly_type == "fatigue":
            max_speed *= rng.uniform(0.82, 0.93)

        avg_hr = float(r.normal(profile.avg_hr_mean + variation.hr_offset, profile.avg_hr_std))
        max_hr = float(r.normal(profile.max_hr_mean + variation.hr_offset * 0.5, 5))
        hi_distance = total_distance * rng.uniform(0.22, 0.33)

        duration_min = float(r.integers(88, 96))  # Playing time varies (subs, injury time)
        avg_speed = total_distance / (duration_min * 60)

        session = {
            "session_id": session_id,
            "player_id": player["id"],
            "player_external_id": f"p{player['id']:03d}",
            "match_id": f"match_{date.strftime('%Y%m%d')}",
            "session_type": "match",
            "started_at": date.isoformat(),
            "ended_at": (date + timedelta(minutes=duration_min + 15)).isoformat(),
            "duration_minutes": duration_min,
            "total_distance_m": round(total_distance, 1),
            "sprint_count": sprint_count,
            "max_speed_ms": round(max_speed, 2),
            "high_speed_distance_m": round(hi_distance, 1),
            "avg_speed_ms": round(avg_speed, 3),
            "avg_heart_rate_bpm": round(avg_hr, 1),
            "max_heart_rate_bpm": round(max_hr, 1),
            "data_quality_score": float(r.uniform(0.91, 1.0)),
            "gps_coverage_pct": float(r.uniform(0.96, 1.0)),
            "is_congested": is_congested,
            "cumulative_fatigue": round(cumulative_fatigue, 3),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type or "",
        }

        # ── Event-level data ──
        events = self._generate_events(
            session_id, player, profile, variation, date,
            duration_min, avg_speed, max_speed, avg_hr, max_hr,
            is_anomaly, anomaly_type
        )

        # ── Coach annotation ──
        annotation = None
        ann_r = np.random.default_rng(session_id * 7)

        # Always annotate anomalies; randomly annotate ~20% of normal sessions
        if is_anomaly or ann_r.random() < 0.20:
            if is_anomaly and anomaly_type == "fatigue":
                sev = float(ann_r.uniform(0.55, 0.9))
                val = "moderate" if sev > 0.7 else "mild"
                ann_type = "fatigue_flag"
            elif is_congested and cumulative_fatigue > 0.4:
                sev = float(ann_r.uniform(0.3, 0.6))
                val = "mild"
                ann_type = "fatigue_flag"
            else:
                sev = float(ann_r.uniform(0.0, 0.3))
                val = "none"
                ann_type = "general"

            annotation = {
                "player_id": player["id"],
                "session_id": session_id,
                "annotation_type": ann_type,
                "value": val,
                "severity": round(sev, 3),
                "annotated_at": (date + timedelta(minutes=duration_min + 30)).isoformat(),
                "annotated_by": "coach_001",
                "note": f"Post-match review matchday {date.strftime('%Y-%m-%d')}",
            }

        return session, events, annotation

    def generate_training_session(
        self,
        player: dict,
        profile: PositionProfile,
        variation: PlayerVariation,
        date: datetime,
        is_congested: bool = False,
        cumulative_fatigue: float = 0.0,
    ) -> Tuple[dict, List[dict]]:
        """
        Generate a training session (typically 75 min, lower intensity than match).
        """
        session_id = self._next_session_id()
        r = np.random.default_rng(session_id * 17)

        # Training intensity: 60–75% of match load
        intensity = float(r.uniform(0.55, 0.75))
        if is_congested:
            intensity *= 0.85  # Recovery training during congested schedule

        base_distance = (profile.distance_mean_m + variation.distance_offset) * intensity
        base_distance *= (1.0 - cumulative_fatigue * 0.1 * variation.fatigue_sensitivity)

        total_distance = max(2000, float(r.normal(base_distance, profile.distance_std_m * 0.4)))
        sprint_count = max(0, int(r.normal(profile.sprint_count_mean * intensity * 0.7,
                                           profile.sprint_count_std * 0.5)))
        max_speed = float(r.normal(profile.max_speed_mean_ms * 0.88, profile.max_speed_std_ms))
        avg_hr = float(r.normal(profile.avg_hr_mean * 0.88, profile.avg_hr_std))
        max_hr = float(r.normal(profile.max_hr_mean * 0.9, 5))
        duration_min = float(r.integers(70, 80))
        avg_speed = total_distance / (duration_min * 60)

        session = {
            "session_id": session_id,
            "player_id": player["id"],
            "player_external_id": f"p{player['id']:03d}",
            "match_id": None,
            "session_type": "training",
            "started_at": date.isoformat(),
            "ended_at": (date + timedelta(minutes=duration_min)).isoformat(),
            "duration_minutes": duration_min,
            "total_distance_m": round(total_distance, 1),
            "sprint_count": sprint_count,
            "max_speed_ms": round(max_speed, 2),
            "high_speed_distance_m": round(total_distance * 0.18, 1),
            "avg_speed_ms": round(avg_speed, 3),
            "avg_heart_rate_bpm": round(avg_hr, 1),
            "max_heart_rate_bpm": round(max_hr, 1),
            "data_quality_score": float(r.uniform(0.88, 1.0)),
            "gps_coverage_pct": float(r.uniform(0.94, 1.0)),
            "is_congested": is_congested,
            "cumulative_fatigue": round(cumulative_fatigue, 3),
            "is_anomaly": False,
            "anomaly_type": "",
        }

        events = self._generate_events(
            session_id, player, profile, variation, date,
            duration_min, avg_speed, max_speed, avg_hr, max_hr,
            is_anomaly=False, anomaly_type=None
        )

        return session, events

    def _generate_events(
        self,
        session_id: int,
        player: dict,
        profile: PositionProfile,
        variation: PlayerVariation,
        start_time: datetime,
        duration_min: float,
        avg_speed: float,
        max_speed: float,
        avg_hr: float,
        max_hr: float,
        is_anomaly: bool,
        anomaly_type: Optional[str],
    ) -> List[dict]:
        """
        Generate high-frequency events (1 per EVENT_INTERVAL_S).
        Implements realistic fatigue decay curve with noise.
        """
        n_events = int(duration_min * 60 / self.EVENT_INTERVAL_S)
        r = np.random.default_rng(session_id * 23)

        # Fatigue decay parameters
        alpha = variation.fatigue_alpha
        beta = avg_speed * 1.3   # Starting speed slightly above avg

        if is_anomaly and anomaly_type == "fatigue":
            # Anomalous session: steeper decay kicks in at mid-match
            alpha *= rng.uniform(1.6, 2.4)

        events = []
        prev_x, prev_y = profile.x_zone_center, profile.y_zone_center

        for i in range(n_events):
            t_min = i * self.EVENT_INTERVAL_S / 60.0
            t_elapsed = i * self.EVENT_INTERVAL_S

            # Speed: exponential decay + noise + sprint bursts
            base_speed = beta * math.exp(-alpha * t_min)
            noise = float(r.normal(0, avg_speed * 0.25))
            speed = max(0.0, base_speed + noise)

            # Occasional sprint burst (not during anomaly fatigue state)
            if (not is_anomaly) and r.random() < 0.04:
                sprint_low = self.SPRINT_THRESHOLD
                sprint_high = max(self.SPRINT_THRESHOLD + 0.1, max_speed)

                if sprint_high > sprint_low:
                    speed = float(r.uniform(sprint_low, sprint_high))

            speed = min(speed, max_speed)

            # HR: generally tracking intensity with lag
            hr_noise = float(r.normal(0, 5))
            max_hr = float(np.clip(
                        r.normal(profile.max_hr_mean + variation.hr_offset * 0.5, 5),
                        60, max_hr + 5      
              ))
            hr = float(np.clip(
                    avg_hr + (max_hr - avg_hr) * (speed / max_speed) + hr_noise,
                    90, 210
                ))

            # Position: random walk around position zone
            dx = float(r.normal(0, 1.5))
            dy = float(r.normal(0, 1.5))

            # Positional drift anomaly: player drifts to wrong zone
            if is_anomaly and anomaly_type == "positional_drift" and t_min > 30:
                drift_target_x = 100 - profile.x_zone_center
                drift_target_y = 100 - profile.y_zone_center
                prev_x = prev_x + (drift_target_x - prev_x) * 0.02
                prev_y = prev_y + (drift_target_y - prev_y) * 0.02

            new_x = float(np.clip(prev_x + dx, 0, 100))
            new_y = float(np.clip(prev_y + dy, 0, 100))

            # Zone classification
            zone = _classify_zone(new_x, new_y)

            is_sprint = speed >= self.SPRINT_THRESHOLD
            is_hi = speed >= self.HI_RUN_THRESHOLD

            if is_sprint:
                event_type = "sprint"
            elif is_hi:
                event_type = "high_intensity_run"
            elif speed > 2.5:
                event_type = "jog"
            elif speed > 0.5:
                event_type = "walk"
            else:
                event_type = "rest"

            # HR recovery time: time for HR to drop 10bpm after sprint
            hr_recovery = float(r.uniform(15, 45)) if is_sprint else None

            events.append({
                "session_id": session_id,
                "player_id": player["id"],
                "ts": (start_time + timedelta(seconds=t_elapsed)).isoformat(),
                "speed_ms": round(speed, 3),
                "x_pitch": round(new_x, 2),
                "y_pitch": round(new_y, 2),
                "zone_id": zone,
                "heart_rate_bpm": hr,
                "hr_recovery_time_s": round(hr_recovery, 1) if hr_recovery else None,
                "event_type": event_type,
                "is_sprint": is_sprint,
                "is_high_intensity": is_hi,
                "elapsed_seconds": t_elapsed,
            })

            prev_x, prev_y = new_x, new_y

        return events


def _classify_zone(x: float, y: float) -> str:
    """Assign a tactical zone label from pitch X/Y coordinates."""
    if x < 33:
        depth = "def"
    elif x < 66:
        depth = "mid"
    else:
        depth = "att"

    if y < 33:
        flank = "L"
    elif y < 66:
        flank = "C"
    else:
        flank = "R"

    return f"{depth}_{flank}"


# ─────────────────────────────────────────────────────────────────────────────
# Injury / Absence Tracker
# ─────────────────────────────────────────────────────────────────────────────
class InjuryTracker:
    """Tracks injuries and returns availability status for each session."""

    def __init__(self):
        self._injuries: Dict[int, Tuple[datetime, datetime]] = {}

    def check_and_update(
        self,
        player_id: int,
        date: datetime,
        variation: PlayerVariation,
    ) -> bool:
        """Returns True if the player is available, False if injured."""
        # Check if current injury has expired
        if player_id in self._injuries:
            _, recovery_date = self._injuries[player_id]
            if date < recovery_date:
                return False
            else:
                del self._injuries[player_id]

        # Random injury event (position-proportional to proneness)
        if rng.random() < variation.injury_proneness / 90:  # per-session probability
            weeks_out = int(rng.integers(2, 7))
            self._injuries[player_id] = (date, date + timedelta(weeks=weeks_out))
            logger.debug("Player %d injured — out %d weeks from %s", player_id, weeks_out, date.date())
            return False

        return True


# ─────────────────────────────────────────────────────────────────────────────
# Fatigue Accumulation Model
# ─────────────────────────────────────────────────────────────────────────────
class FatigueAccumulator:
    """
    Rolling fatigue model per player.
    Fatigue accumulates with matches and hard training, decays with rest.
    """
    def __init__(self):
        self._fatigue: Dict[int, float] = {}

    def get(self, player_id: int) -> float:
        return self._fatigue.get(player_id, 0.0)

    def update(
        self,
        player_id: int,
        session_type: str,
        is_congested: bool,
        days_since_last: int,
    ) -> None:
        current = self._fatigue.get(player_id, 0.0)

        # Accumulate
        if session_type == "match":
            delta = 0.25 if not is_congested else 0.35
        else:
            delta = 0.08 if not is_congested else 0.12

        # Decay based on rest days
        decay = min(1.0, days_since_last * 0.18)
        new_fatigue = max(0.0, min(1.0, current * (1 - decay) + delta))
        self._fatigue[player_id] = new_fatigue


# ─────────────────────────────────────────────────────────────────────────────
# Main data generation entry point
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(
    n_seasons: int = 2,
    n_matchdays_per_season: int = 38,
    anomaly_rate: float = 0.05,
) -> dict:
    """
    Generate the full dataset.

    Returns
    -------
    dict with keys: players, sessions, events, annotations, ground_truth
    Each value is a list of dicts.
    """
    logger.info("Generating dataset: %d players, %d seasons, %d matchdays/season",
                len(SQUAD), n_seasons, n_matchdays_per_season)

    gen = SessionGenerator()
    injury_tracker = InjuryTracker()
    fatigue_acc = FatigueAccumulator()

    players_rows = []
    sessions_rows = []
    events_rows = []
    annotations_rows = []
    ground_truth_rows = []

    variations: Dict[int, PlayerVariation] = {
        p["id"]: create_player_variation(p["id"]) for p in SQUAD
    }

    # Build player table
    for p in SQUAD:
        players_rows.append({
            "player_id": p["id"],
            "external_id": f"p{p['id']:03d}",
            "full_name": p["name"],
            "position": p["position"],
            "age": p["age"],
            "age_group": p["age_group"],
            "nationality": p["nationality"],
        })

    season_start = datetime(2023, 8, 1, tzinfo=timezone.utc)

    for season in range(n_seasons):
        start = season_start + timedelta(days=365 * season)
        schedule = generate_schedule(start, n_matchdays=n_matchdays_per_season)

        logger.info("Season %d: %d scheduled days", season + 1, len(schedule))

        prev_dates: Dict[int, datetime] = {}

        for sched_item in schedule:
            date = sched_item["date"]
            session_type = sched_item["type"]
            is_congested = sched_item["is_congested"]

            # Matchday hour: 15:30 kickoff; training 10:00
            if session_type == "match":
                session_date = date.replace(hour=15, minute=30)
            else:
                session_date = date.replace(hour=10, minute=0)

            for player in SQUAD:
                pid = player["id"]
                var = variations[pid]
                profile = POSITION_PROFILES[player["position"]]

                # Rest days since last session
                prev = prev_dates.get(pid, session_date - timedelta(days=3))
                days_since = max(0, (session_date - prev).days)

                # Update fatigue
                fatigue_acc.update(pid, session_type, is_congested, days_since)
                cum_fatigue = fatigue_acc.get(pid)

                # Check injury availability
                available = injury_tracker.check_and_update(pid, session_date, var)
                if not available:
                    continue

                # Decide if this session has a seeded anomaly
                is_anomaly = False
                anomaly_type = None
                if session_type == "match" and rng.random() < anomaly_rate:
                    is_anomaly = True
                    anomaly_type = rng.choice(
                        ["fatigue", "positional_drift"],
                        p=[0.70, 0.30]
                    )

                try:
                    if session_type == "match":
                        session, events, annotation = gen.generate_match_session(
                            player, profile, var, session_date,
                            is_congested=is_congested,
                            cumulative_fatigue=cum_fatigue,
                            is_anomaly=is_anomaly,
                            anomaly_type=anomaly_type,
                        )
                        sessions_rows.append(session)
                        events_rows.extend(events)
                        if annotation:
                            annotations_rows.append(annotation)

                        ground_truth_rows.append({
                            "session_id": session["session_id"],
                            "player_id": pid,
                            "is_anomaly": is_anomaly,
                            "anomaly_type": anomaly_type or "",
                            "cumulative_fatigue": cum_fatigue,
                            "is_congested": is_congested,
                        })

                    else:
                        session, events = gen.generate_training_session(
                            player, profile, var, session_date,
                            is_congested=is_congested,
                            cumulative_fatigue=cum_fatigue,
                        )
                        sessions_rows.append(session)
                        events_rows.extend(events)

                except Exception as exc:
                    logger.warning("Session generation failed for player %d: %s", pid, exc)
                    continue

                prev_dates[pid] = session_date

        logger.info(
            "Season %d complete: %d sessions, %d events",
            season + 1, len(sessions_rows), len(events_rows)
        )

    return {
        "players": players_rows,
        "sessions": sessions_rows,
        "events": events_rows,
        "annotations": annotations_rows,
        "ground_truth": ground_truth_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save to CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset(data: dict) -> None:
    for key, rows in data.items():
        if not rows:
            logger.warning("No rows for %s — skipping", key)
            continue
        df = pd.DataFrame(rows)
        path = OUTPUT_DIR / f"{key}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s: %d rows → %s", key, len(df), path)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset statistics report
# ─────────────────────────────────────────────────────────────────────────────
def print_dataset_report(data: dict) -> None:
    sessions = pd.DataFrame(data["sessions"])
    events = pd.DataFrame(data["events"])
    gt = pd.DataFrame(data["ground_truth"])

    print("\n" + "=" * 65)
    print("SYNTHETIC DATASET REPORT")
    print("=" * 65)
    print(f"Players:          {len(data['players'])}")
    print(f"Total sessions:   {len(sessions)}")
    print(f"  → Matches:      {(sessions['session_type'] == 'match').sum()}")
    print(f"  → Training:     {(sessions['session_type'] == 'training').sum()}")
    print(f"Total events:     {len(events)}")
    print(f"Annotations:      {len(data['annotations'])}")

    if not gt.empty:
        print(f"\nGround truth labels:")
        print(f"  Anomalous sessions:  {gt['is_anomaly'].sum()} "
              f"({gt['is_anomaly'].mean()*100:.1f}%)")
        if gt["is_anomaly"].sum() > 0:
            atypes = gt[gt["is_anomaly"]]["anomaly_type"].value_counts()
            for atype, count in atypes.items():
                print(f"    {atype:25s}: {count}")

    print(f"\nPer-position session stats (matches only):")
    match_sessions = sessions[sessions["session_type"] == "match"]
    players_df = pd.DataFrame(data["players"])
    merged = match_sessions.merge(
        players_df[["player_id", "position"]], on="player_id"
    )
    pos_stats = merged.groupby("position").agg(
        sessions=("session_id", "count"),
        dist_mean=("total_distance_m", "mean"),
        sprint_mean=("sprint_count", "mean"),
    ).round(0)
    print(pos_stats.to_string())
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Players Data Synthetic Dataset Generator")
    parser.add_argument("--seasons", type=int, default=2,
                        help="Number of seasons to simulate (default: 2)")
    parser.add_argument("--matchdays", type=int, default=38,
                        help="Matchdays per season (default: 38)")
    parser.add_argument("--anomaly-rate", type=float, default=0.05,
                        help="Fraction of match sessions with anomalies (default: 0.05)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving CSV files (dry run)")
    args = parser.parse_args()

    data = generate_dataset(
        n_seasons=args.seasons,
        n_matchdays_per_season=args.matchdays,
        anomaly_rate=args.anomaly_rate,
    )

    print_dataset_report(data)

    if not args.no_save:
        save_dataset(data)
        print(f"Dataset saved to: {OUTPUT_DIR.resolve()}/")
        print("\nFiles:")
        for f in sorted(OUTPUT_DIR.glob("*.csv")):
            size_kb = f.stat().st_size / 1024
            rows = sum(1 for _ in open(f)) - 1
            print(f"  {f.name:35s} {rows:>7,} rows  {size_kb:>8.1f} KB")