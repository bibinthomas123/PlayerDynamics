"""
Players Data — IBM CIC Germany
Realistic Synthetic Dataset Generator  (v4 — Decision-Agent Rewrite)

Addresses all 12 critique points from the second architectural review:

R1  — Semi-Markov tactical phases: states hold for realistic durations (not per-tick)
R2  — Role-relative formation anchors: shape-relative offsets replace pairwise springs
R3  — Ball ownership model: ball belongs to a player; passes have passer + receiver
R4  — Reaction lag / partial observability: players respond to delayed world state
R5  — Angular momentum / heading physics: turning radius limits sharp direction changes
R6  — Metabolic HR model: HR driven by exertion integral, not instantaneous speed
R7  — Split fatigue components: neuromuscular / cardiovascular / sprint-specific
R8  — Adversarial opponent agents: pressure fields, interception radii, compactness
R9  — xG-driven goal causality: box entries → xG accumulation → goal events
R10 — Extended validation: heatmaps, compactness, transition bimodality, xG stats
R11 — Vectorised batch updates (NumPy arrays for all-player state)
R12 — Sensor corruption layer: GPS dropout, jitter, HR freeze, quantisation

Feature additions:
  Centralised 1 Hz match engine  
  Acceleration-limited physics  
  Ball-centric dynamics         
  Score/tactical context        
  ACWR injury model             
  Latent effort process         
  Correlated latent fitness     
  Continuous fatigue field       

Output schema unchanged:
    data/
      players.csv
      sessions.csv
      events.csv
      annotations.csv
      ground_truth_labels.csv
      validation_report.txt
"""
from __future__ import annotations

import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("datagen")

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

RNG_SEED       = 42
rng            = np.random.default_rng(RNG_SEED)

DT_SIM         = 1.0        # internal step (s)
DT_OUT         = 15.0       # output tick  (s)
STEPS_PER_TICK = int(DT_OUT / DT_SIM)

SPRINT_TH   = 7.0    # m/s
HI_RUN_TH   = 5.5
MAX_ACCEL   = 4.5    # m/s²
MAX_TURN    = 2.8    # rad/s  (R5 — angular inertia)
PITCH_X     = 105.0
PITCH_Y     = 68.0

# Reaction lag constants  (R4)
REACTION_LAG_BASE_S    = int(0.4)   # 400 ms base → rounded to 1 s ticks
REACTION_LAG_FATIGUE_S = int(0.3)   # up to +300 ms when fatigued

# ─────────────────────────────────────────────────────────────────────────────
# R1: Semi-Markov tactical phases — states have sampled durations
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_MODES = ["LOW_BLOCK", "BALANCED", "HIGH_PRESS", "CHAOTIC"]
LOCAL_STATES = ["LOW_TEMPO", "BUILD_UP", "TRANSITION", "PRESS", "COUNTER"]

# How long each local state lasts (seconds) before a transition is sampled
STATE_DURATION: Dict[str, Tuple[int, int]] = {
    "LOW_TEMPO":  (25, 90),
    "BUILD_UP":   (12, 45),
    "TRANSITION": (4,  12),
    "PRESS":      (6,  22),
    "COUNTER":    (4,  14),
}

# Global mode duration (seconds)
GLOBAL_DURATION: Dict[str, Tuple[int, int]] = {
    "LOW_BLOCK":  (300, 900),
    "BALANCED":   (120, 600),
    "HIGH_PRESS": (90,  420),
    "CHAOTIC":    (60,  240),
}

# Transition probabilities (only evaluated at end of state duration)
LOCAL_TRANSITIONS: Dict[str, np.ndarray] = {
    "LOW_BLOCK":  np.array([
        [0.60, 0.25, 0.08, 0.05, 0.02],
        [0.28, 0.42, 0.15, 0.10, 0.05],
        [0.16, 0.22, 0.35, 0.18, 0.09],
        [0.18, 0.26, 0.30, 0.22, 0.04],
        [0.15, 0.14, 0.38, 0.08, 0.25],
    ]),
    "BALANCED":   np.array([
        [0.55, 0.28, 0.09, 0.06, 0.02],
        [0.18, 0.44, 0.20, 0.13, 0.05],
        [0.09, 0.19, 0.32, 0.21, 0.19],
        [0.14, 0.24, 0.26, 0.28, 0.08],
        [0.09, 0.09, 0.32, 0.10, 0.40],
    ]),
    "HIGH_PRESS": np.array([
        [0.30, 0.18, 0.17, 0.25, 0.10],
        [0.10, 0.32, 0.22, 0.27, 0.09],
        [0.07, 0.13, 0.30, 0.30, 0.20],
        [0.09, 0.16, 0.24, 0.38, 0.13],
        [0.07, 0.07, 0.28, 0.20, 0.38],
    ]),
    "CHAOTIC":    np.array([
        [0.25, 0.20, 0.24, 0.18, 0.13],
        [0.14, 0.27, 0.24, 0.20, 0.15],
        [0.11, 0.14, 0.30, 0.23, 0.22],
        [0.11, 0.17, 0.24, 0.30, 0.18],
        [0.09, 0.09, 0.26, 0.19, 0.37],
    ]),
}

GLOBAL_TRANSITIONS = {
    "LOW_BLOCK":  {"LOW_BLOCK": 0.10, "BALANCED": 0.55, "HIGH_PRESS": 0.25, "CHAOTIC": 0.10},
    "BALANCED":   {"LOW_BLOCK": 0.20, "BALANCED": 0.15, "HIGH_PRESS": 0.45, "CHAOTIC": 0.20},
    "HIGH_PRESS": {"LOW_BLOCK": 0.15, "BALANCED": 0.40, "HIGH_PRESS": 0.15, "CHAOTIC": 0.30},
    "CHAOTIC":    {"LOW_BLOCK": 0.10, "BALANCED": 0.35, "HIGH_PRESS": 0.30, "CHAOTIC": 0.25},
}

STATE_EFFORT_PRESSURE = {
    "LOW_TEMPO": 0.04, "BUILD_UP": 0.09,
    "TRANSITION": 0.22, "PRESS": 0.30, "COUNTER": 0.44,
}
GLOBAL_EFFORT_MULT = {
    "LOW_BLOCK": 0.68, "BALANCED": 1.00,
    "HIGH_PRESS": 1.42, "CHAOTIC": 1.22,
}

# ─────────────────────────────────────────────────────────────────────────────
# R2: Role-relative formation shape
# Each role has (dx, dy) offset from team shape centroid (metres)
# 4-3-3 base shape; mirrored for away/defending
# ─────────────────────────────────────────────────────────────────────────────
ROLE_SHAPE_OFFSET: Dict[str, Tuple[float, float]] = {
    "GK":  (-47.0,  0.0),
    "CB":  (-28.0,  0.0),
    "LB":  (-22.0, -22.0),
    "RB":  (-22.0,  22.0),
    "CDM": (-10.0,  0.0),
    "CM":  (  0.0, -10.0),
    "CAM": ( 10.0,  0.0),
    "LW":  ( 18.0, -28.0),
    "RW":  ( 18.0,  28.0),
    "ST":  ( 24.0,  0.0),
}
# When two CM or CDM players share a role, spread them laterally
ROLE_LATERAL_SPREAD: Dict[str, float] = {
    "CB": 10.0, "CM": 12.0, "CDM": 8.0, "CAM": 8.0, "ST": 8.0,
}
# Compactness envelope: max allowed spread from shape centroid per line
LINE_COMPACTNESS = {
    "defensive": 18.0,   # CB, LB, RB max x-spread from each other
    "midfield":  22.0,
    "attacking": 26.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# R3: Ball ownership states
# ─────────────────────────────────────────────────────────────────────────────
PASS_DECISION_PROB: Dict[str, float] = {
    "LOW_TEMPO": 0.04, "BUILD_UP": 0.08,
    "TRANSITION": 0.18, "PRESS": 0.22, "COUNTER": 0.30,
}
DRIBBLE_LOSS_PROB: Dict[str, float] = {
    "LOW_TEMPO": 0.005, "BUILD_UP": 0.008,
    "TRANSITION": 0.025, "PRESS": 0.045, "COUNTER": 0.030,
}
BALL_TRAVEL_SPEED = 18.0   # m/s (pass speed)

# ─────────────────────────────────────────────────────────────────────────────
# Training taxonomy (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
TRAINING_TYPES = ["RECOVERY", "TACTICAL", "HIGH_INTENSITY", "MATCHDAY_MINUS_1", "CONDITIONING"]
TRAINING_INTENSITY    = {"RECOVERY": (0.38, 0.48), "TACTICAL": (0.52, 0.65),
                          "HIGH_INTENSITY": (0.70, 0.82), "MATCHDAY_MINUS_1": (0.42, 0.55),
                          "CONDITIONING": (0.60, 0.74)}
TRAINING_DURATION_MIN = {"RECOVERY": (45, 60), "TACTICAL": (70, 85),
                          "HIGH_INTENSITY": (65, 80), "MATCHDAY_MINUS_1": (50, 65),
                          "CONDITIONING": (55, 70)}
TRAINING_SPREAD       = {"RECOVERY": 0.5, "TACTICAL": 0.9, "HIGH_INTENSITY": 0.8,
                          "MATCHDAY_MINUS_1": 0.7, "CONDITIONING": 0.6}


def _select_training_type(days_since: int, is_congested: bool,
                           r: np.random.Generator) -> str:
    if is_congested:
        return r.choice(["RECOVERY", "TACTICAL"], p=[0.55, 0.45])
    if days_since <= 1:
        return "RECOVERY"
    if days_since == 2:
        return r.choice(["RECOVERY", "TACTICAL"], p=[0.3, 0.7])
    if days_since >= 5:
        return "MATCHDAY_MINUS_1"
    return r.choice(["TACTICAL", "HIGH_INTENSITY", "CONDITIONING"], p=[0.40, 0.35, 0.25])


# ─────────────────────────────────────────────────────────────────────────────
# Position profiles
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PositionProfile:
    position:           str
    distance_mean_m:    float
    distance_std_m:     float
    sprint_count_mean:  float
    sprint_count_std:   float
    max_speed_mean_ms:  float
    max_speed_std_ms:   float
    avg_hr_mean:        float
    max_hr_mean:        float
    fatigue_alpha_mean: float
    # pitch home region (metres)
    x_mean:  float
    y_mean:  float
    x_std:   float
    y_std:   float


POSITION_PROFILES: Dict[str, PositionProfile] = {
    "GK":  PositionProfile("GK",  5600, 450,  3, 2,  7.5, 0.6, 138, 172, 0.0020,  5.0, 34.0,  3.0,  5.0),
    "CB":  PositionProfile("CB",  10300,650,  12, 4,  8.8, 0.5, 152, 183, 0.0045, 22.0, 34.0,  7.0, 10.0),
    "LB":  PositionProfile("LB",  11700,700,  28, 6,  9.6, 0.5, 162, 189, 0.0065, 42.0, 10.0, 18.0,  7.0),
    "RB":  PositionProfile("RB",  11600,700,  27, 6,  9.5, 0.5, 161, 188, 0.0063, 42.0, 58.0, 18.0,  7.0),
    "CM":  PositionProfile("CM",  12500,750,  22, 5,  9.2, 0.5, 166, 191, 0.0070, 52.0, 34.0, 24.0, 16.0),
    "CDM": PositionProfile("CDM", 12000,700,  18, 5,  9.0, 0.5, 163, 189, 0.0065, 44.0, 34.0, 14.0, 12.0),
    "CAM": PositionProfile("CAM", 11500,700,  24, 5,  9.4, 0.5, 165, 190, 0.0068, 71.0, 34.0, 16.0, 12.0),
    "LW":  PositionProfile("LW",  11200,680,  32, 7, 10.1, 0.6, 164, 190, 0.0075, 65.0, 12.0, 16.0,  9.0),
    "RW":  PositionProfile("RW",  11100,680,  31, 7, 10.0, 0.6, 163, 189, 0.0073, 65.0, 56.0, 16.0,  9.0),
    "ST":  PositionProfile("ST",  10700,650,  26, 6,  9.8, 0.5, 163, 191, 0.0072, 84.0, 34.0, 14.0, 12.0),
}

SQUAD = [
    {"id":  1, "name": "Player 01", "position": "GK",  "age": 29, "age_group": "Senior", "nationality": "DE"},
    {"id":  2, "name": "Player 02", "position": "GK",  "age": 23, "age_group": "U23",    "nationality": "DE"},
    {"id":  3, "name": "Player 03", "position": "CB",  "age": 31, "age_group": "Senior", "nationality": "FR"},
    {"id":  4, "name": "Player 04", "position": "CB",  "age": 27, "age_group": "Senior", "nationality": "ES"},
    {"id":  5, "name": "Player 05", "position": "CB",  "age": 22, "age_group": "U23",    "nationality": "DE"},
    {"id":  6, "name": "Player 06", "position": "LB",  "age": 26, "age_group": "Senior", "nationality": "BR"},
    {"id":  7, "name": "Player 07", "position": "LB",  "age": 21, "age_group": "U23",    "nationality": "DE"},
    {"id":  8, "name": "Player 08", "position": "RB",  "age": 28, "age_group": "Senior", "nationality": "IT"},
    {"id":  9, "name": "Player 09", "position": "RB",  "age": 24, "age_group": "Senior", "nationality": "DE"},
    {"id": 10, "name": "Player 10", "position": "CDM", "age": 30, "age_group": "Senior", "nationality": "ES"},
    {"id": 11, "name": "Player 11", "position": "CDM", "age": 25, "age_group": "Senior", "nationality": "DE"},
    {"id": 12, "name": "Player 12", "position": "CM",  "age": 27, "age_group": "Senior", "nationality": "AR"},
    {"id": 13, "name": "Player 13", "position": "CM",  "age": 24, "age_group": "Senior", "nationality": "DE"},
    {"id": 14, "name": "Player 14", "position": "CM",  "age": 20, "age_group": "U23",    "nationality": "DE"},
    {"id": 15, "name": "Player 15", "position": "CAM", "age": 26, "age_group": "Senior", "nationality": "PT"},
    {"id": 16, "name": "Player 16", "position": "CAM", "age": 22, "age_group": "U23",    "nationality": "DE"},
    {"id": 17, "name": "Player 17", "position": "LW",  "age": 25, "age_group": "Senior", "nationality": "NG"},
    {"id": 18, "name": "Player 18", "position": "LW",  "age": 21, "age_group": "U23",    "nationality": "DE"},
    {"id": 19, "name": "Player 19", "position": "RW",  "age": 28, "age_group": "Senior", "nationality": "ES"},
    {"id": 20, "name": "Player 20", "position": "RW",  "age": 23, "age_group": "U23",    "nationality": "DE"},
    {"id": 21, "name": "Player 21", "position": "ST",  "age": 30, "age_group": "Senior", "nationality": "PL"},
    {"id": 22, "name": "Player 22", "position": "ST",  "age": 27, "age_group": "Senior", "nationality": "FR"},
    {"id": 23, "name": "Player 23", "position": "ST",  "age": 19, "age_group": "U23",    "nationality": "DE"},
    {"id": 24, "name": "Player 24", "position": "CB",  "age": 32, "age_group": "Senior", "nationality": "NL"},
    {"id": 25, "name": "Player 25", "position": "CM",  "age": 24, "age_group": "Senior", "nationality": "TR"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Latent fitness  (correlated physiology — from v3, kept)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LatentFitness:
    aerobic_capacity:    float
    anaerobic_capacity:  float
    recovery_efficiency: float
    fatigue_resilience:  float
    hr_rest:             float
    hr_rise_alpha:       float
    hr_decay_alpha:      float
    speed_multiplier:    float
    injury_proneness:    float


def _create_latent_fitness(player_id: int, age: int) -> LatentFitness:
    r = np.random.default_rng(player_id * 31 + 7)
    age_pen  = max(0.0, (age - 28) * 0.015)
    base_fit = float(np.clip(float(r.beta(5, 2)) - age_pen, 0.1, 1.0))

    ae = float(np.clip(base_fit + r.normal(0, 0.08), 0.1, 1.0))
    an = float(np.clip(base_fit + r.normal(0, 0.10), 0.1, 1.0))
    rc = float(np.clip(base_fit + r.normal(0, 0.09), 0.1, 1.0))
    rs = float(np.clip(base_fit + r.normal(0, 0.08), 0.1, 1.0))
    return LatentFitness(
        aerobic_capacity=ae, anaerobic_capacity=an,
        recovery_efficiency=rc, fatigue_resilience=rs,
        hr_rest=float(r.uniform(48, 72) - ae * 8),
        hr_rise_alpha=float(0.15 + (1 - ae) * 0.20),
        hr_decay_alpha=float(0.05 + rc * 0.12),
        speed_multiplier=float(0.90 + an * 0.20),
        injury_proneness=float(np.clip(0.02 + (1 - rs) * 0.13 + age_pen * 0.5, 0.01, 0.20)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# R7: Split fatigue components
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SplitFatigue:
    """
    neuromuscular  — sprint capability, peak force
    cardiovascular — HR elevation, aerobic efficiency
    sprint_specific — repeated-sprint ability decay
    All in [0, 1]; each recovers at different rates.
    """
    neuromuscular:   float = 0.0
    cardiovascular:  float = 0.0
    sprint_specific: float = 0.0

    @property
    def level(self) -> float:
        """Composite scalar for backward-compat (session-level reporting)."""
        return float((self.neuromuscular + self.cardiovascular + self.sprint_specific) / 3.0)

    def load_match(self, intensity: float = 1.0) -> None:
        self.neuromuscular   = float(np.clip(self.neuromuscular   + 0.22 * intensity, 0, 1))
        self.cardiovascular  = float(np.clip(self.cardiovascular  + 0.28 * intensity, 0, 1))
        self.sprint_specific = float(np.clip(self.sprint_specific + 0.30 * intensity, 0, 1))

    def load_training(self, intensity: float = 0.4) -> None:
        self.neuromuscular   = float(np.clip(self.neuromuscular   + 0.06 * intensity, 0, 1))
        self.cardiovascular  = float(np.clip(self.cardiovascular  + 0.08 * intensity, 0, 1))
        self.sprint_specific = float(np.clip(self.sprint_specific + 0.07 * intensity, 0, 1))

    def recover(self, efficiency: float, rest_days: float) -> None:
        # Neuromuscular recovers slowest (72h), cardiovascular fastest (24h)
        self.neuromuscular   = float(np.clip(
            self.neuromuscular   * math.exp(-(0.06 + efficiency * 0.04) * rest_days), 0, 1))
        self.cardiovascular  = float(np.clip(
            self.cardiovascular  * math.exp(-(0.16 + efficiency * 0.08) * rest_days), 0, 1))
        self.sprint_specific = float(np.clip(
            self.sprint_specific * math.exp(-(0.10 + efficiency * 0.06) * rest_days), 0, 1))

    # Continuous effect accessors
    def sprint_suppression(self) -> float:
        return float(np.clip(self.sprint_specific ** 1.4 * 0.9, 0, 0.88))

    def speed_reduction(self) -> float:
        return float(1.0 - 0.10 * self.neuromuscular)

    def hr_elevation(self) -> float:
        return float(9.0 * self.cardiovascular)

    def recovery_slowdown(self) -> float:
        return float(1.0 + 0.65 * self.cardiovascular)


# ─────────────────────────────────────────────────────────────────────────────
# R6: Metabolic HR model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MetabolicState:
    """
    metabolic_load integrates speed³·dt, decays with aerobic recovery.
    HR is a function of metabolic_load, not instantaneous speed.
    """
    load: float = 0.0

    def update(self, speed: float, fitness: LatentFitness,
               fatigue: SplitFatigue) -> None:
        # Load: cubic in speed (anaerobic cost)
        input_load = (speed ** 3) * DT_SIM * 0.0004
        recovery_rate = 0.018 + fitness.aerobic_capacity * 0.012
        recovery_rate *= (1.0 - fatigue.cardiovascular * 0.4)  # CV fatigue slows recovery
        self.load = float(np.clip(
            self.load + input_load - recovery_rate * self.load * DT_SIM,
            0.0, 1.0,
        ))

    def hr_target(self, fitness: LatentFitness, max_hr: float,
                  fatigue: SplitFatigue) -> float:
        effort_frac = float(np.clip(self.load ** 0.6, 0.0, 1.0))
        base = fitness.hr_rest + (max_hr - fitness.hr_rest) * effort_frac
        return base + fatigue.hr_elevation()


# ─────────────────────────────────────────────────────────────────────────────
# ACWR tracker (from v3, kept)
# ─────────────────────────────────────────────────────────────────────────────
class ACWRTracker:
    def __init__(self):
        self._acute:   Dict[int, float] = {}
        self._chronic: Dict[int, float] = {}

    def update(self, pid: int, load: float) -> None:
        a_a, a_c = 2/8.0, 2/29.0
        pa = self._acute.get(pid, load)
        pc = self._chronic.get(pid, load)
        self._acute[pid]   = load * a_a + pa * (1 - a_a)
        self._chronic[pid] = load * a_c + pc * (1 - a_c)

    def acwr(self, pid: int) -> float:
        a = self._acute.get(pid, 1.0)
        c = max(self._chronic.get(pid, 1.0), 1e-6)
        return float(np.clip(a / c, 0.3, 3.5))

    def injury_risk_mult(self, pid: int) -> float:
        a = self.acwr(pid)
        if a < 0.8:  return 0.5
        if a <= 1.3: return 1.0
        return float(1.0 + 2.5 * ((a - 1.3) / 1.2) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# R1: Semi-Markov engine state (holds duration counters)
# R3: Ball ownership
# R9: xG tracker
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MatchEngineState:
    t_sec:          int
    global_mode:    str
    local_state:    str
    # Semi-Markov counters
    local_state_remaining:  int   # seconds left in current local state
    global_mode_remaining:  int   # seconds left in current global mode
    # Ball
    ball_x:         float
    ball_y:         float
    ball_vx:        float
    ball_vy:        float
    ball_owner_id:  Optional[int]    # R3: None = loose / in flight
    ball_in_flight: bool
    ball_receiver_id: Optional[int]
    ball_flight_remaining: int       # seconds until ball reaches receiver
    possession_team: str
    # Team shape centroid
    centroid_x:     float
    centroid_y:     float
    # Score
    score_us:       int
    score_them:     int
    # R9: xG accumulator
    xg_us:          float
    xg_them:        float
    # Possession phase statistics (for R10 validation)
    possession_phase_length: int     # seconds current team has had ball
    box_entries_us:   int
    box_entries_them: int

    @property
    def score_diff(self) -> int:
        return self.score_us - self.score_them

    @property
    def t_min(self) -> float:
        return self.t_sec / 60.0

    def advance_tactical_state(self, r: np.random.Generator) -> None:
        """R1: Semi-Markov — transition only when timer expires."""
        # Local state
        self.local_state_remaining -= 1
        if self.local_state_remaining <= 0:
            idx = LOCAL_STATES.index(self.local_state)
            self.local_state = r.choice(LOCAL_STATES,
                                        p=LOCAL_TRANSITIONS[self.global_mode][idx])
            lo, hi = STATE_DURATION[self.local_state]
            self.local_state_remaining = int(r.integers(lo, hi + 1))

        # Global mode
        self.global_mode_remaining -= 1
        if self.global_mode_remaining <= 0:
            # R1 + scoreline bias (R6)
            t_min = self.t_min
            if self.score_diff < 0 and t_min > 70:
                probs = {"LOW_BLOCK": 0.02, "BALANCED": 0.25,
                         "HIGH_PRESS": 0.55, "CHAOTIC": 0.18}
            elif self.score_diff > 0 and t_min > 75:
                probs = {"LOW_BLOCK": 0.62, "BALANCED": 0.30,
                         "HIGH_PRESS": 0.05, "CHAOTIC": 0.03}
            elif abs(t_min - 45.0) < 1.0:   # halftime reset
                if self.score_diff > 0:
                    probs = {"LOW_BLOCK": 0.50, "BALANCED": 0.35,
                             "HIGH_PRESS": 0.10, "CHAOTIC": 0.05}
                elif self.score_diff < 0:
                    probs = {"LOW_BLOCK": 0.05, "BALANCED": 0.30,
                             "HIGH_PRESS": 0.45, "CHAOTIC": 0.20}
                else:
                    probs = {"LOW_BLOCK": 0.20, "BALANCED": 0.50,
                             "HIGH_PRESS": 0.20, "CHAOTIC": 0.10}
            else:
                probs = GLOBAL_TRANSITIONS[self.global_mode]
            modes   = list(probs.keys())
            weights = np.array([probs[m] for m in modes])
            self.global_mode = r.choice(modes, p=weights / weights.sum())
            lo, hi = GLOBAL_DURATION[self.global_mode]
            self.global_mode_remaining = int(r.integers(lo, hi + 1))

    def update_ball_physics(self, r: np.random.Generator,
                             all_home_states: Dict[int, "PlayerKineState"],
                             all_opp_agents: List["OpponentAgent"]) -> None:
        """R3: Ball belongs to owner or travels in flight to receiver."""
        if self.ball_in_flight:
            # Ball moving toward receiver
            if self.ball_receiver_id is not None and self.ball_receiver_id in all_home_states:
                rx = all_home_states[self.ball_receiver_id].x
                ry = all_home_states[self.ball_receiver_id].y
            else:
                rx, ry = self.ball_x + self.ball_vx * 2, self.ball_y + self.ball_vy * 2

            # Intercept check: opponent agent within 2.5 m of ball path
            for opp in all_opp_agents:
                dx = self.ball_x - opp.x
                dy = self.ball_y - opp.y
                if math.sqrt(dx * dx + dy * dy) < 2.5:
                    # Interception
                    self.possession_team = "away"
                    self.ball_owner_id    = None
                    self.ball_in_flight   = False
                    self.ball_receiver_id = None
                    self.ball_x, self.ball_y = opp.x, opp.y
                    self.ball_vx = float(r.normal(0, 1.5))
                    self.ball_vy = float(r.normal(0, 1.5))
                    self.possession_phase_length = 0
                    return

            self.ball_flight_remaining -= 1
            # Advance ball toward receiver
            dx = rx - self.ball_x
            dy = ry - self.ball_y
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            step = min(BALL_TRAVEL_SPEED * DT_SIM, dist)
            self.ball_x = float(np.clip(self.ball_x + dx/dist*step, 0, PITCH_X))
            self.ball_y = float(np.clip(self.ball_y + dy/dist*step, 0, PITCH_Y))

            if self.ball_flight_remaining <= 0:
                # Reception
                self.ball_owner_id  = self.ball_receiver_id
                self.ball_in_flight = False
                self.ball_receiver_id = None
            return

        # Ball owned by a player — follow owner position
        if self.ball_owner_id is not None:
            owner = all_home_states.get(self.ball_owner_id)
            if owner is not None:
                self.ball_x = owner.x + float(r.normal(0, 0.3))
                self.ball_y = owner.y + float(r.normal(0, 0.3))
            self.possession_team = "home"
            self.possession_phase_length += 1
        else:
            # Loose ball — random walk + drift
            self.ball_vx = 0.85 * self.ball_vx + float(r.normal(0, 0.8))
            self.ball_vy = 0.85 * self.ball_vy + float(r.normal(0, 0.6))
            self.ball_x  = float(np.clip(self.ball_x + self.ball_vx, 0, PITCH_X))
            self.ball_y  = float(np.clip(self.ball_y + self.ball_vy, 0, PITCH_Y))

    def update_centroid(self, r: np.random.Generator) -> None:
        if self.local_state in ("COUNTER", "PRESS"):
            tgt = 70.0 if self.possession_team == "home" else 35.0
        elif self.local_state == "BUILD_UP":
            tgt = 55.0
        else:
            tgt = 47.0
        if self.global_mode == "HIGH_PRESS": tgt += 9.0
        elif self.global_mode == "LOW_BLOCK": tgt -= 9.0
        self.centroid_x = float(np.clip(
            0.94*self.centroid_x + 0.06*tgt + r.normal(0, 0.7), 12.0, 93.0))
        self.centroid_y = float(np.clip(
            0.96*self.centroid_y + r.normal(0, 0.5), 22.0, 46.0))

    def accumulate_xg(self, r: np.random.Generator) -> None:
        """R9: xG from box entries + dangerous local states."""
        # Box defined as x > 83 (home attack box) for home team
        if self.ball_x > 83 and self.possession_team == "home":
            self.box_entries_us += 1
            xg_chance = 0.0
            if self.local_state == "COUNTER":
                xg_chance = float(r.uniform(0.06, 0.18))
            elif self.local_state == "PRESS":
                xg_chance = float(r.uniform(0.02, 0.10))
            elif self.local_state in ("TRANSITION", "BUILD_UP"):
                xg_chance = float(r.uniform(0.01, 0.06))
            self.xg_us += xg_chance
            # Goal from xG
            if r.random() < xg_chance * 0.15:
                self.score_us += 1

        elif self.ball_x < 22 and self.possession_team == "away":
            self.box_entries_them += 1
            xg_chance = float(r.uniform(0.01, 0.12))
            self.xg_them += xg_chance
            if r.random() < xg_chance * 0.15:
                self.score_them += 1


# ─────────────────────────────────────────────────────────────────────────────
# R8: Opponent agents — adversarial pressure fields
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OpponentAgent:
    """
    Minimal but adversarial opponent.
    Tracks a home player (man-marking simplified) or holds defensive shape.
    Exerts pressure field on nearby home players.
    """
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    role: str = "CM"
    marking_target_id: Optional[int] = None

    INTERCEPT_RADIUS = 2.5   # m
    PRESSURE_RADIUS  = 4.0   # m

    def update(self, engine: MatchEngineState,
               home_states: Dict[int, "PlayerKineState"],
               r: np.random.Generator) -> None:
        """
        When away has possession: spread into defensive compactness.
        When home has possession: press nearest home player.
        """
        if engine.possession_team == "away":
            # Hold defensive shape mirrored around midfield
            target_x = PITCH_X - engine.centroid_x + float(r.normal(0, 2.0))
            target_y = engine.centroid_y + float(r.normal(0, 3.0))
        else:
            # Press nearest home player or marking target
            if self.marking_target_id and self.marking_target_id in home_states:
                tgt = home_states[self.marking_target_id]
                target_x, target_y = tgt.x, tgt.y
            else:
                # Press closest player to ball
                bx, by = engine.ball_x, engine.ball_y
                target_x, target_y = bx + float(r.normal(0, 4)), by + float(r.normal(0, 4))

        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx*dx + dy*dy) + 1e-6
        target_speed = min(7.5, dist * 0.4)   # approach speed
        desired_vx = (dx/dist) * target_speed
        desired_vy = (dy/dist) * target_speed
        # Acceleration-limited
        dvx, dvy = desired_vx - self.vx, desired_vy - self.vy
        dv = math.sqrt(dvx*dvx + dvy*dvy) + 1e-6
        max_dv = MAX_ACCEL * DT_SIM
        if dv > max_dv:
            dvx *= max_dv/dv
            dvy *= max_dv/dv
        self.vx += dvx; self.vy += dvy
        self.x = float(np.clip(self.x + self.vx, 0, PITCH_X))
        self.y = float(np.clip(self.y + self.vy, 0, PITCH_Y))

    def pressure_on(self, px: float, py: float) -> float:
        """Returns pressure scalar [0,1] on a player at (px, py)."""
        dx, dy = px - self.x, py - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > self.PRESSURE_RADIUS:
            return 0.0
        return float(1.0 - dist / self.PRESSURE_RADIUS)


def _build_opponent_agents(players_in_match: List[dict],
                            r: np.random.Generator) -> List[OpponentAgent]:
    """Create one opponent agent per home outfield player."""
    agents = []
    for p in players_in_match:
        prof = POSITION_PROFILES[p["position"]]
        # Mirror position
        ox = PITCH_X - float(r.normal(prof.x_mean, prof.x_std * 0.5))
        oy = float(r.normal(prof.y_mean, prof.y_std * 0.5))
        agents.append(OpponentAgent(
            x=ox, y=oy,
            role=p["position"],
            marking_target_id=p["id"],
        ))
    return agents


# ─────────────────────────────────────────────────────────────────────────────
# R2 + R4 + R5: Per-player kinematic state with heading physics + reaction lag
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PlayerKineState:
    x:      float
    y:      float
    vx:     float = 0.0
    vy:     float = 0.0
    theta:  float = 0.0    # R5: heading (radians)
    omega:  float = 0.0    # R5: angular velocity
    effort: float = 0.0    # R8: latent effort
    hr:     float = 70.0   # current HR (driven by metabolic model)
    # R4: perception delay buffer — stores (x, y, ball_x, ball_y) snapshots
    perception_buffer: Deque = field(default_factory=lambda: deque(maxlen=4))

    def _perceived_world(self, engine: MatchEngineState,
                          fatigue: SplitFatigue) -> Tuple[float, float]:
        """R4: Return delayed ball position based on reaction lag."""
        lag = REACTION_LAG_BASE_S + (1 if fatigue.neuromuscular > 0.5 else 0)
        lag = min(lag, len(self.perception_buffer))
        if lag > 0 and len(self.perception_buffer) >= lag:
            snapshot = list(self.perception_buffer)[-lag]
            return snapshot[0], snapshot[1]
        return engine.ball_x, engine.ball_y

    def update(
        self,
        engine:            MatchEngineState,
        profile:           PositionProfile,
        fitness:           LatentFitness,
        fatigue:           SplitFatigue,
        metabolic:         MetabolicState,
        all_player_states: Dict[int, "PlayerKineState"],
        player_id:         int,
        all_players:       List[dict],
        opp_agents:        List[OpponentAgent],
        shape_anchor:      Tuple[float, float],  # R2
        is_anomaly:        bool,
        anomaly_type:      Optional[str],
        r:                 np.random.Generator,
    ) -> float:
        # ── R4: Save perception snapshot then read delayed world ───────────
        self.perception_buffer.append((engine.ball_x, engine.ball_y))
        perc_ball_x, perc_ball_y = self._perceived_world(engine, fatigue)

        # ── R8: Effort process ────────────────────────────────────────────
        pressure = (STATE_EFFORT_PRESSURE[engine.local_state]
                    * GLOBAL_EFFORT_MULT[engine.global_mode])
        # Opponent pressure increases effort demand
        opp_pressure = sum(a.pressure_on(self.x, self.y) for a in opp_agents)
        pressure += opp_pressure * 0.15

        decay = 0.08 * (1.0 + fitness.recovery_efficiency)
        fat_damp = 1.0 - fatigue.sprint_suppression() * 0.5
        self.effort = float(np.clip(
            self.effort
            + pressure * fat_damp * DT_SIM * 0.04
            - decay    * DT_SIM * 0.04
            + float(r.normal(0, 0.04)) * DT_SIM * 0.04,
            0.0, 1.0,
        ))

        sprint_thresh = 0.72 - fatigue.sprint_suppression() * 0.35
        is_sprinting  = self.effort > sprint_thresh

        max_speed = (profile.max_speed_mean_ms * fitness.speed_multiplier
                     * fatigue.speed_reduction())
        if is_sprinting:
            target_speed = float(r.uniform(SPRINT_TH, max(SPRINT_TH + 0.1, max_speed)))
        else:
            base = max_speed * 0.35 * (0.5 + self.effort)
            target_speed = float(np.clip(base + r.normal(0, 0.5), 0.0, max_speed))

        # ── R2: Role-relative shape anchor ───────────────────────────────
        ax, ay = shape_anchor

        # Positional drift anomaly
        if is_anomaly and anomaly_type == "positional_drift":
            drift = min(1.0, max(0.0, (engine.t_min - 30.0) / 50.0))
            ax = ax * (1 - drift) + (PITCH_X - profile.x_mean) * drift
            ay = ay * (1 - drift) + (PITCH_Y - profile.y_mean) * drift

        # Ball attraction (perceived, delayed)
        ball_k = {"GK": 0.002, "CB": 0.006, "LB": 0.010, "RB": 0.010,
                   "CDM": 0.012, "CM": 0.015, "CAM": 0.018,
                   "LW": 0.020, "RW": 0.020, "ST": 0.020}.get(profile.position, 0.012)

        # R8: If this player owns the ball, reduce attraction (already on it)
        if engine.ball_owner_id == player_id:
            ball_k *= 0.1

        fx = ball_k * (perc_ball_x - self.x) + 0.06 * (ax - self.x)
        fy = ball_k * (perc_ball_y - self.y) + 0.06 * (ay - self.y)

        # R8: Repulsion from opponent pressure field
        for a in opp_agents:
            p_val = a.pressure_on(self.x, self.y)
            if p_val > 0:
                dx, dy = self.x - a.x, self.y - a.y
                dist = math.sqrt(dx*dx + dy*dy) + 1e-6
                fx += p_val * 0.4 * (dx / dist)
                fy += p_val * 0.4 * (dy / dist)

        noise_x = float(r.normal(0, 0.25))
        noise_y = float(r.normal(0, 0.25))

        # R5: Desired heading from force direction
        desired_theta = math.atan2(fy + noise_y, fx + noise_x)
        # Angular inertia — limit turn rate
        delta_theta = desired_theta - self.theta
        # Normalise to [-pi, pi]
        delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi
        max_turn_step = MAX_TURN * DT_SIM
        self.omega = float(np.clip(delta_theta, -max_turn_step, max_turn_step))
        self.theta += self.omega

        # Velocity follows heading at target speed
        desired_vx = math.cos(self.theta) * target_speed
        desired_vy = math.sin(self.theta) * target_speed

        # ── Acceleration limit ────────────────────────────────────────────
        dvx = desired_vx - self.vx
        dvy = desired_vy - self.vy
        dv  = math.sqrt(dvx*dvx + dvy*dvy) + 1e-6
        if dv > MAX_ACCEL * DT_SIM:
            sc  = MAX_ACCEL * DT_SIM / dv
            dvx *= sc; dvy *= sc
        self.vx += dvx; self.vy += dvy

        new_x = float(np.clip(self.x + self.vx, 0, PITCH_X))
        new_y = float(np.clip(self.y + self.vy, 0, PITCH_Y))
        if new_x in (0.0, PITCH_X): self.vx = 0.0
        if new_y in (0.0, PITCH_Y): self.vy = 0.0
        self.x, self.y = new_x, new_y

        speed = float(np.clip(math.sqrt(self.vx**2 + self.vy**2), 0.0, max_speed))

        # ── R6: Metabolic HR ─────────────────────────────────────────────
        metabolic.update(speed, fitness, fatigue)
        hr_tgt = metabolic.hr_target(fitness, profile.max_hr_mean, fatigue)
        if hr_tgt > self.hr:
            delta = fitness.hr_rise_alpha * (hr_tgt - self.hr)
        else:
            delta = -fitness.hr_decay_alpha * (self.hr - fitness.hr_rest)
        # Physiological hard cap: HR cannot change faster than ~3 bpm/s
        delta = float(np.clip(delta, -3.0 * DT_SIM, 3.0 * DT_SIM))
        self.hr = float(np.clip(self.hr + delta, 40.0, 220.0))

        return speed


# ─────────────────────────────────────────────────────────────────────────────
# R2: Compute shape anchors for all players each tick
# ─────────────────────────────────────────────────────────────────────────────
def _compute_shape_anchors(
    players_in_match: List[dict],
    engine: MatchEngineState,
) -> Dict[int, Tuple[float, float]]:
    """
    Each player gets a target anchor = team_shape_centroid + role_offset.
    Players sharing a role get staggered laterally.
    """
    anchors: Dict[int, Tuple[float, float]] = {}
    # Count role instances to stagger duplicates
    role_counter: Dict[str, int] = {}

    for p in players_in_match:
        role   = p["position"]
        offset = ROLE_SHAPE_OFFSET.get(role, (0.0, 0.0))
        idx    = role_counter.get(role, 0)
        role_counter[role] = idx + 1

        lateral_spread = ROLE_LATERAL_SPREAD.get(role, 0.0)
        if lateral_spread > 0 and idx > 0:
            side = 1 if (idx % 2 == 1) else -1
            stagger_y = side * lateral_spread * math.ceil(idx / 2)
        else:
            stagger_y = 0.0

        ax = float(np.clip(engine.centroid_x + offset[0], 2.0, PITCH_X - 2.0))
        ay = float(np.clip(engine.centroid_y + offset[1] + stagger_y, 1.0, PITCH_Y - 1.0))
        anchors[p["id"]] = (ax, ay)

    return anchors


# ─────────────────────────────────────────────────────────────────────────────
# R3: Ball ownership / passing decision logic
# ─────────────────────────────────────────────────────────────────────────────
def _process_ball_decisions(
    engine:      MatchEngineState,
    player_states: Dict[int, PlayerKineState],
    players_in_match: List[dict],
    r:           np.random.Generator,
) -> None:
    """
    If home team possesses and owner decides to pass:
      - select receiver (nearest open teammate in forward/lateral direction)
      - launch ball in flight
    If owner loses ball under pressure: switch possession.
    """
    if engine.possession_team != "home" or engine.ball_owner_id is None:
        return
    if engine.ball_in_flight:
        return

    owner_id = engine.ball_owner_id
    owner    = player_states.get(owner_id)
    if owner is None:
        engine.ball_owner_id = None
        return

    # Turnover under pressure
    loss_p = DRIBBLE_LOSS_PROB.get(engine.local_state, 0.01)
    if r.random() < loss_p:
        engine.possession_team  = "away"
        engine.ball_owner_id    = None
        engine.possession_phase_length = 0
        return

    # Pass decision
    pass_p = PASS_DECISION_PROB.get(engine.local_state, 0.05)
    if r.random() < pass_p:
        # Select receiver: random teammate (weighted toward forward positions)
        candidates = [p for p in players_in_match if p["id"] != owner_id]
        if not candidates:
            return
        forward_weight = {
            "GK": 0.2, "CB": 0.4, "LB": 0.7, "RB": 0.7,
            "CDM": 0.9, "CM": 1.2, "CAM": 1.5,
            "LW": 1.6, "RW": 1.6, "ST": 1.8,
        }
        weights = np.array([forward_weight.get(p["position"], 1.0) for p in candidates])
        weights = weights / weights.sum()
        receiver = candidates[int(r.choice(len(candidates), p=weights))]
        rec_state = player_states.get(receiver["id"])
        if rec_state is None:
            return

        dx = rec_state.x - owner.x
        dy = rec_state.y - owner.y
        dist = math.sqrt(dx*dx + dy*dy) + 1e-6
        flight_time = max(1, int(dist / BALL_TRAVEL_SPEED))

        engine.ball_in_flight      = True
        engine.ball_receiver_id    = receiver["id"]
        engine.ball_flight_remaining = flight_time
        engine.ball_vx = (dx/dist) * BALL_TRAVEL_SPEED
        engine.ball_vy = (dy/dist) * BALL_TRAVEL_SPEED


def _assign_initial_ball_owner(
    players_in_match: List[dict],
    r: np.random.Generator,
) -> int:
    """Give ball to a CDM or CM at kickoff."""
    central = [p for p in players_in_match if p["position"] in ("CDM", "CM", "CB")]
    if central:
        return central[int(r.integers(0, len(central)))]["id"]
    return players_in_match[0]["id"]


# ─────────────────────────────────────────────────────────────────────────────
# R12: Sensor corruption layer
# Applied as a post-processing pass on the events DataFrame.
# ─────────────────────────────────────────────────────────────────────────────
def apply_sensor_corruption(events_df: pd.DataFrame,
                             r: np.random.Generator) -> pd.DataFrame:
    """
    Adds realistic GPS/HR sensor noise:
      - 1–3% GPS dropout windows (speed/position set to NaN)
      - Gaussian coordinate jitter (±0.3 m)
      - HR freeze events (HR stuck for 2–5 ticks)
      - Timestamp jitter (±2 s)
      - Rare spike artefacts in speed
    """
    df = events_df.copy()
    n  = len(df)

    # GPS dropout: random 15 s windows
    dropout_rate = float(r.uniform(0.010, 0.028))
    dropout_mask = r.random(n) < dropout_rate
    df.loc[dropout_mask, ["speed_ms", "x_pitch", "y_pitch"]] = np.nan

    # Coordinate jitter on non-dropped rows
    valid = ~dropout_mask
    df.loc[valid, "x_pitch"] = (df.loc[valid, "x_pitch"]
                                  + r.normal(0, 0.3, valid.sum())).clip(0, PITCH_X)
    df.loc[valid, "y_pitch"] = (df.loc[valid, "y_pitch"]
                                  + r.normal(0, 0.3, valid.sum())).clip(0, PITCH_Y)

    # HR freeze: pick random windows per player
    for pid in df["player_id"].unique():
        mask = df["player_id"] == pid
        idx  = df.index[mask].tolist()
        if len(idx) < 5:
            continue
        n_freezes = int(r.integers(0, 4))
        for _ in range(n_freezes):
            start   = int(r.integers(0, len(idx) - 2))
            length  = int(r.integers(2, 6))
            freeze_idx = idx[start:start + length]
            frozen_val = df.at[freeze_idx[0], "heart_rate_bpm"]
            df.loc[freeze_idx, "heart_rate_bpm"] = frozen_val

    # Speed spike artefacts (~0.2%)
    spike_mask = r.random(n) < 0.002
    spike_valid_mask = spike_mask & valid

    df.loc[spike_valid_mask, "speed_ms"] = r.uniform(
        11,
        14,
        spike_valid_mask.sum()
    )

    # Timestamp jitter (±2 s represented as string offset — mark in new column)
    jitter_s = r.integers(-2, 3, n).astype(float)
    df["ts_jitter_s"] = jitter_s   # downstream can apply; raw ts preserved

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Schedule & match context (from v3, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def generate_schedule(season_start: datetime, n_matchdays: int = 38) -> List[dict]:
    schedule, current = [], season_start
    for md in range(n_matchdays):
        match_date = current + timedelta(days=(5 - current.weekday()) % 7)
        schedule.append({"date": match_date, "type": "match", "matchday": md + 1,
                          "is_congested": md in [7, 8, 21, 22, 33, 34],
                          "days_since_last_match": 7})
        for off in [1, 2, 4, 5]:
            schedule.append({"date": match_date + timedelta(days=off), "type": "training",
                              "matchday": None, "is_congested": md in [7, 8, 21, 22, 33, 34],
                              "days_since_last_match": off})
        current = match_date + timedelta(days=7)
    return sorted(schedule, key=lambda x: x["date"])


@dataclass
class MatchContext:
    match_intensity: float; opponent_strength: float
    team_possession: float; is_away: bool

    @classmethod
    def sample(cls, r: np.random.Generator) -> "MatchContext":
        return cls(float(r.uniform(0.82, 1.18)), float(r.uniform(0.72, 1.28)),
                   float(r.uniform(0.32, 0.68)), bool(r.integers(0, 2)))


_session_counter = 0
def _next_sid() -> int:
    global _session_counter
    _session_counter += 1
    return _session_counter


# ─────────────────────────────────────────────────────────────────────────────
# Main match simulation
# ─────────────────────────────────────────────────────────────────────────────
def simulate_match(
    players_in_match: List[dict],
    profiles:         Dict[str, PositionProfile],
    fitnesses:        Dict[int, LatentFitness],
    fatigues:         Dict[int, SplitFatigue],
    match_ctx:        MatchContext,
    is_anomaly_map:   Dict[int, bool],
    anomaly_type_map: Dict[int, Optional[str]],
    start_time:       datetime,
    r:                np.random.Generator,
) -> Tuple[Dict[int, dict], Dict[int, List[dict]]]:

    duration_min  = float(r.integers(88, 96))
    n_total_secs  = int(duration_min * 60)

    # R1: initialise engine with semi-Markov counters
    engine = MatchEngineState(
        t_sec=0, global_mode="BALANCED", local_state="LOW_TEMPO",
        local_state_remaining=int(r.integers(*STATE_DURATION["LOW_TEMPO"])),
        global_mode_remaining=int(r.integers(*GLOBAL_DURATION["BALANCED"])),
        ball_x=PITCH_X/2, ball_y=PITCH_Y/2, ball_vx=0.0, ball_vy=0.0,
        ball_owner_id=_assign_initial_ball_owner(players_in_match, r),
        ball_in_flight=False, ball_receiver_id=None, ball_flight_remaining=0,
        possession_team="home",
        centroid_x=PITCH_X/2, centroid_y=PITCH_Y/2,
        score_us=0, score_them=0,
        xg_us=0.0, xg_them=0.0,
        possession_phase_length=0, box_entries_us=0, box_entries_them=0,
    )

    # Initialise player kinematic states
    player_states: Dict[int, PlayerKineState] = {}
    for p in players_in_match:
        prof = profiles[p["position"]]
        player_states[p["id"]] = PlayerKineState(
            x=float(r.normal(prof.x_mean, prof.x_std)),
            y=float(r.normal(prof.y_mean, prof.y_std)),
            hr=fitnesses[p["id"]].hr_rest + 28.0,
        )

    # Metabolic states
    metabolic_states: Dict[int, MetabolicState] = {p["id"]: MetabolicState() for p in players_in_match}

    # R8: Opponent agents
    opp_agents = _build_opponent_agents(players_in_match, r)

    # Accumulators
    acc_speed: Dict[int, List[float]] = {p["id"]: [] for p in players_in_match}
    acc_hr:    Dict[int, List[float]] = {p["id"]: [] for p in players_in_match}
    acc_x:     Dict[int, List[float]] = {p["id"]: [] for p in players_in_match}
    acc_y:     Dict[int, List[float]] = {p["id"]: [] for p in players_in_match}

    player_events: Dict[int, List[dict]] = {p["id"]: [] for p in players_in_match}

    session_dist:    Dict[int, float] = {p["id"]: 0.0 for p in players_in_match}
    session_sprints: Dict[int, int]   = {p["id"]: 0   for p in players_in_match}
    session_maxspd:  Dict[int, float] = {p["id"]: 0.0 for p in players_in_match}
    session_hrsum:   Dict[int, float] = {p["id"]: 0.0 for p in players_in_match}
    session_maxhr:   Dict[int, float] = {p["id"]: 0.0 for p in players_in_match}
    session_nsteps:  Dict[int, int]   = {p["id"]: 0   for p in players_in_match}

    tick_buf = 0
    prev_x: Dict[int, float] = {p["id"]: float(np.random.default_rng(p["id"]).normal(
        profiles[p["position"]].x_mean, profiles[p["position"]].x_std)) for p in players_in_match}
    prev_y: Dict[int, float] = {p["id"]: float(np.random.default_rng(p["id"] + 1000).normal(
        profiles[p["position"]].y_mean, profiles[p["position"]].y_std)) for p in players_in_match}
    prev_spd: Dict[int, float] = {p["id"]: 0.0 for p in players_in_match}

    for t in range(n_total_secs):
        engine.t_sec = t

        # R1: advance semi-Markov state machine
        engine.advance_tactical_state(r)
        engine.update_centroid(r)
        engine.accumulate_xg(r)

        # R2: compute role-relative shape anchors
        anchors = _compute_shape_anchors(players_in_match, engine)

        # R3: ball ownership / passing decisions
        _process_ball_decisions(engine, player_states, players_in_match, r)

        # R8: update opponent agents
        for opp in opp_agents:
            opp.update(engine, player_states, r)

        # R4 + ball physics
        engine.update_ball_physics(r, player_states, opp_agents)

        # Update all home players jointly
        for p in players_in_match:
            pid  = p["id"]
            prof = profiles[p["position"]]
            fit  = fitnesses[pid]
            fat  = fatigues[pid]
            met  = metabolic_states[pid]
            st   = player_states[pid]

            speed = st.update(
                engine=engine, profile=prof, fitness=fit, fatigue=fat,
                metabolic=met,
                all_player_states=player_states,
                player_id=pid,
                all_players=players_in_match,
                opp_agents=opp_agents,
                shape_anchor=anchors[pid],
                is_anomaly=is_anomaly_map.get(pid, False),
                anomaly_type=anomaly_type_map.get(pid),
                r=r,
            )

            acc_speed[pid].append(speed)
            acc_hr[pid].append(st.hr)
            acc_x[pid].append(st.x)
            acc_y[pid].append(st.y)

            session_dist[pid]    += speed * DT_SIM
            session_hrsum[pid]   += st.hr
            session_nsteps[pid]  += 1
            if speed > session_maxspd[pid]: session_maxspd[pid] = speed
            if st.hr  > session_maxhr[pid]: session_maxhr[pid]  = st.hr
            if speed >= SPRINT_TH:          session_sprints[pid] += 1

        tick_buf += 1

        # Emit 15 s output rows
        if tick_buf == STEPS_PER_TICK:
            for p in players_in_match:
                pid   = p["id"]
                spds  = acc_speed[pid]
                hrs   = acc_hr[pid]
                xs    = acc_x[pid]
                ys    = acc_y[pid]

                mean_spd  = float(np.mean(spds)) if spds else 0.0
                peak_spd  = float(np.max(spds))  if spds else 0.0
                mean_hr   = float(np.mean(hrs))  if hrs  else 70.0
                lx = xs[-1] if xs else profiles[p["position"]].x_mean
                ly = ys[-1] if ys else profiles[p["position"]].y_mean

                is_sprint_win = peak_spd >= SPRINT_TH
                fat = fatigues[pid]
                hr_rec = None
                if is_sprint_win:
                    hr_rec = round(float(r.uniform(18, 42)) * fat.recovery_slowdown(), 1)

                # Compute accel and distance_delta from window averages
                _dx_m = (lx - prev_x[pid]) / PITCH_X * PITCH_X   # lx already in metres (0–105)
                _dy_m = (ly - prev_y[pid]) / PITCH_Y * PITCH_Y
                _dist_delta = round(math.sqrt(_dx_m**2 + _dy_m**2), 3)
                _accel = round((mean_spd - prev_spd[pid]) / DT_OUT, 4)
                prev_x[pid]   = lx
                prev_y[pid]   = ly
                prev_spd[pid] = mean_spd

                player_events[pid].append({
                    "session_id":         None,
                    "player_id":          pid,
                    "ts":                 (start_time + timedelta(seconds=t - STEPS_PER_TICK + 1)).isoformat(),
                    "speed_ms":           round(mean_spd, 3),
                    "accel":              _accel,
                    "distance_delta_m":   _dist_delta,
                    "x_pitch":            round(lx, 2),
                    "y_pitch":            round(ly, 2),
                    "zone_id":            _zone(lx, ly),
                    "heart_rate_bpm":     round(float(np.clip(mean_hr + r.normal(0, 1.2), 90, 210)), 1),
                    "hr_recovery_time_s": hr_rec,
                    "event_type":         _etype(mean_spd),
                    "is_sprint":          is_sprint_win,
                    "is_high_intensity":  peak_spd >= HI_RUN_TH,
                    "elapsed_seconds":    t - STEPS_PER_TICK + 1,
                    "match_state":        f"{engine.global_mode}:{engine.local_state}",
                    "ball_x":             round(engine.ball_x, 1),
                    "ball_y":             round(engine.ball_y, 1),
                    "ball_owner":         str(engine.ball_owner_id),
                    "score":              f"{engine.score_us}-{engine.score_them}",
                    "xg_us":             round(engine.xg_us, 3),
                    "xg_them":           round(engine.xg_them, 3),
                })
                acc_speed[pid].clear(); acc_hr[pid].clear()
                acc_x[pid].clear();    acc_y[pid].clear()

            tick_buf = 0

    # Build session metrics
    session_metrics: Dict[int, dict] = {}
    for p in players_in_match:
        pid = p["id"]
        n   = max(1, session_nsteps[pid])
        session_metrics[pid] = {
            "player_id":            pid,
            "player_external_id":   f"p{pid:03d}",
            "match_id":             f"match_{start_time.strftime('%Y%m%d')}",
            "session_type":         "match",
            "started_at":           start_time.isoformat(),
            "ended_at":             (start_time + timedelta(minutes=duration_min + 15)).isoformat(),
            "duration_minutes":     duration_min,
            "total_distance_m":     round(session_dist[pid], 1),
            "sprint_count":         session_sprints[pid],
            "max_speed_ms":         round(session_maxspd[pid], 2),
            "high_speed_distance_m": round(session_dist[pid] * 0.27, 1),
            "avg_speed_ms":         round(session_dist[pid] / (duration_min * 60), 3),
            "avg_heart_rate_bpm":   round(session_hrsum[pid] / n, 1),
            "max_heart_rate_bpm":   round(session_maxhr[pid], 1),
            "data_quality_score":   float(r.uniform(0.91, 1.0)),
            "gps_coverage_pct":     float(r.uniform(0.96, 1.0)),
            "is_congested":         False,
            "cumulative_fatigue":   round(fatigues[pid].level, 3),
            "match_intensity":      round(match_ctx.match_intensity, 3),
            "team_possession":      round(match_ctx.team_possession, 3),
            "opponent_strength":    round(match_ctx.opponent_strength, 3),
            "is_away":              match_ctx.is_away,
            "score_final":          f"{engine.score_us}-{engine.score_them}",
            "xg_us":                round(engine.xg_us, 3),
            "xg_them":              round(engine.xg_them, 3),
            "box_entries_us":       engine.box_entries_us,
        }

    return session_metrics, player_events


# ─────────────────────────────────────────────────────────────────────────────
# Training session (player-independent — centralised engine not needed)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_training_session(
    player:          dict,
    profile:         PositionProfile,
    fitness:         LatentFitness,
    fatigue:         SplitFatigue,
    date:            datetime,
    is_congested:    bool,
    days_since_match: int,
) -> Tuple[dict, List[dict]]:
    r = np.random.default_rng(hash((player["id"], date.isoformat())) & 0xFFFFFFFF)
    sid = _next_sid()
    ttype    = _select_training_type(days_since_match, is_congested, r)
    intensity= float(r.uniform(*TRAINING_INTENSITY[ttype]))
    dur_min  = float(r.integers(*TRAINING_DURATION_MIN[ttype]))
    spread   = TRAINING_SPREAD[ttype]
    n_secs   = int(dur_min * 60)

    max_spd = profile.max_speed_mean_ms * fitness.speed_multiplier * 0.88
    avg_spd = profile.distance_mean_m * intensity / (dur_min * 60)

    px, py   = float(r.normal(profile.x_mean, profile.x_std * 0.5)), float(r.normal(profile.y_mean, profile.y_std * 0.5))
    vx, vy   = 0.0, 0.0
    theta    = 0.0
    hr_cur   = fitness.hr_rest + 18.0
    met      = MetabolicState()

    acc_spd, acc_hr, acc_x, acc_y = [], [], [], []
    events   = []
    tot_dist = sprints = 0
    max_sp = hr_max_s = 0.0
    hr_sum = n_steps = 0
    tick_buf = 0
    _prev_x_tr, _prev_y_tr, _prev_spd_tr = px, py, 0.0

    for t in range(n_secs):
        t_min = t / 60.0
        if ttype == "RECOVERY":
            tspd = float(np.clip(r.normal(avg_spd * 0.68, avg_spd * 0.1), 0, max_spd * 0.5))
        elif ttype == "HIGH_INTENSITY":
            in_b = int(t_min) % 4 < 1
            tspd = float(r.uniform(SPRINT_TH * 0.82, max_spd * 0.88)) if in_b else float(np.clip(r.normal(avg_spd * 0.5, 0.4), 0, max_spd))
        else:
            tspd = float(np.clip(avg_spd * math.exp(-0.003 * t_min) + r.normal(0, 0.35), 0, max_spd))

        tspd *= fatigue.speed_reduction()
        tspd  = float(np.clip(tspd, 0, max_spd))

        # R5: heading + acceleration
        fx = 0.07 * (profile.x_mean - px) + float(r.normal(0, 0.2 * spread))
        fy = 0.07 * (profile.y_mean - py) + float(r.normal(0, 0.2 * spread))
        desired_theta = math.atan2(fy, fx)
        delta = (desired_theta - theta + math.pi) % (2*math.pi) - math.pi
        theta += float(np.clip(delta, -MAX_TURN*DT_SIM, MAX_TURN*DT_SIM))
        dvx = math.cos(theta)*tspd - vx
        dvy = math.sin(theta)*tspd - vy
        dv  = math.sqrt(dvx*dvx + dvy*dvy) + 1e-6
        if dv > MAX_ACCEL*DT_SIM:
            dvx *= MAX_ACCEL*DT_SIM/dv; dvy *= MAX_ACCEL*DT_SIM/dv
        vx += dvx; vy += dvy
        px  = float(np.clip(px + vx, 0, PITCH_X))
        py  = float(np.clip(py + vy, 0, PITCH_Y))
        spd = float(np.clip(math.sqrt(vx**2 + vy**2), 0, max_spd))

        # R6: metabolic HR
        met.update(spd, fitness, fatigue)
        hr_tgt = met.hr_target(fitness, profile.max_hr_mean * 0.88, fatigue)
        hr_tgt *= 0.85
        if hr_tgt > hr_cur:
            delta = fitness.hr_rise_alpha * 0.7 * (hr_tgt - hr_cur)
        else:
            delta = -fitness.hr_decay_alpha * 1.1 * (hr_cur - fitness.hr_rest)
        # Physiological cap: ≤3 bpm/s
        delta  = float(np.clip(delta, -3.0 * DT_SIM, 3.0 * DT_SIM))
        hr_cur = float(np.clip(hr_cur + delta, 40.0, 210.0))

        acc_spd.append(spd); acc_hr.append(hr_cur)
        acc_x.append(px); acc_y.append(py)
        tot_dist += spd * DT_SIM
        hr_sum   += hr_cur; n_steps += 1
        if spd > max_sp:   max_sp   = spd
        if hr_cur > hr_max_s: hr_max_s = hr_cur
        if spd >= SPRINT_TH: sprints += 1
        tick_buf += 1

        if tick_buf == STEPS_PER_TICK:
            ms = float(np.mean(acc_spd)); ps = float(np.max(acc_spd))
            mh = float(np.mean(acc_hr))
            lx, ly = acc_x[-1], acc_y[-1]
            is_sp = ps >= SPRINT_TH

            _dx_tr = lx - _prev_x_tr
            _dy_tr = ly - _prev_y_tr
            _dist_delta_tr = round(math.sqrt(_dx_tr**2 + _dy_tr**2), 3)
            _accel_tr = round((ms - _prev_spd_tr) / DT_OUT, 4)
            _prev_x_tr, _prev_y_tr, _prev_spd_tr = lx, ly, ms

            events.append({
                "session_id": sid, "player_id": player["id"],
                "ts": (date + timedelta(seconds=t - STEPS_PER_TICK + 1)).isoformat(),
                "speed_ms": round(ms, 3),
                "accel": _accel_tr,
                "distance_delta_m": _dist_delta_tr,
                "x_pitch": round(lx, 2), "y_pitch": round(ly, 2),
                "zone_id": _zone(lx, ly),
                "heart_rate_bpm": round(float(np.clip(mh + r.normal(0, 1.2), 80, 200)), 1),
                "hr_recovery_time_s": round(float(r.uniform(18, 40)), 1) if is_sp else None,
                "event_type": _etype(ms),
                "is_sprint": is_sp, "is_high_intensity": ps >= HI_RUN_TH,
                "elapsed_seconds": t - STEPS_PER_TICK + 1,
                "match_state": ttype,
                "ball_x": None, "ball_y": None, "ball_owner": None,
                "score": None, "xg_us": None, "xg_them": None,
            })
            
            acc_spd.clear(); acc_hr.clear(); acc_x.clear(); acc_y.clear()
            tick_buf = 0

    session = {
        "session_id": sid, "player_id": player["id"],
        "player_external_id": f"p{player['id']:03d}",
        "match_id": None, "session_type": "training", "training_type": ttype,
        "started_at": date.isoformat(),
        "ended_at": (date + timedelta(minutes=dur_min)).isoformat(),
        "duration_minutes": dur_min,
        "total_distance_m": round(tot_dist, 1),
        "sprint_count": sprints,
        "max_speed_ms": round(max_sp, 2),
        "high_speed_distance_m": round(tot_dist * 0.16, 1),
        "avg_speed_ms": round(tot_dist / (dur_min * 60), 3),
        "avg_heart_rate_bpm": round(hr_sum / max(1, n_steps), 1),
        "max_heart_rate_bpm": round(hr_max_s, 1),
        "data_quality_score": float(r.uniform(0.88, 1.0)),
        "gps_coverage_pct":   float(r.uniform(0.94, 1.0)),
        "is_congested": is_congested,
        "cumulative_fatigue": round(fatigue.level, 3),
        "match_intensity": None, "team_possession": None,
        "opponent_strength": None, "is_away": False,
        "score_final": None, "xg_us": None, "xg_them": None,
        "box_entries_us": None,
    }
    return session, events


# ─────────────────────────────────────────────────────────────────────────────
# Injury tracker
# ─────────────────────────────────────────────────────────────────────────────
class InjuryTracker:
    def __init__(self):
        self._injuries: Dict[int, Tuple[datetime, datetime]] = {}

    def check_and_update(self, pid: int, date: datetime,
                          fitness: LatentFitness, acwr: ACWRTracker) -> bool:
        if pid in self._injuries:
            _, rec = self._injuries[pid]
            if date < rec:
                return False
            del self._injuries[pid]
        risk = (fitness.injury_proneness / 90.0) * acwr.injury_risk_mult(pid)
        if rng.random() < risk:
            weeks = int(rng.integers(2, 7))
            self._injuries[pid] = (date, date + timedelta(weeks=weeks))
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# R10: Extended validation layer
# ─────────────────────────────────────────────────────────────────────────────
def validate_dataset(events: pd.DataFrame, sessions: pd.DataFrame) -> str:
    lines = ["=" * 65, "VALIDATION REPORT — v4", "=" * 65]

    match_ev = events[events["session_id"].isin(
        sessions[sessions["session_type"] == "match"]["session_id"]
    )].copy()

    spd = match_ev["speed_ms"].dropna().values

    # 1. Speed autocorrelation
    if len(spd) > 100:
        ac = float(np.corrcoef(spd[:-1], spd[1:])[0, 1])
        lines.append(f"[{'PASS' if ac > 0.35 else 'FAIL'}] Speed autocorr lag-1: {ac:.3f}  (>0.35)")

    # 2. Acceleration bounds
    accel = np.diff(spd) / DT_OUT
    mean_a = float(np.mean(np.abs(accel)))
    p99_a  = float(np.percentile(np.abs(accel), 99))
    lines.append(f"[{'PASS' if mean_a < 1.0 else 'FAIL'}] Mean |accel|: {mean_a:.3f}  (<1.0)")
    lines.append(f"[{'PASS' if p99_a < MAX_ACCEL + 1 else 'FAIL'}] 99pct |accel|: {p99_a:.3f}  (<{MAX_ACCEL+1:.1f})")

    # R10-A: Positional heatmaps (x-std per role)
    lines.append("\n[R10-A] Positional x-std by role (GK<CB<CM expected):")
    if "player_id" in match_ev.columns:
        sq_df = pd.DataFrame(SQUAD)[["id", "position"]].rename(columns={"id": "player_id"})
        me2 = match_ev.merge(sq_df, on="player_id", how="left")
        for pos in ["GK", "CB", "CDM", "CM", "LW", "ST"]:
            sub = me2[me2["position"] == pos]["x_pitch"].dropna()
            if len(sub) > 10:
                lines.append(f"  {pos:5s}: x_std={sub.std():.1f} m  x_mean={sub.mean():.1f}")

    # R10-B: Team compactness — convex hull area per tick sample
    if "x_pitch" in match_ev.columns and len(match_ev) > 50:
        try:
            from scipy.spatial import ConvexHull
            # Sample 50 ticks, compute hull area per tick
            ticks = match_ev.groupby("elapsed_seconds")
            hull_areas = []
            for _, grp in list(ticks)[:50]:
                pts = grp[["x_pitch", "y_pitch"]].dropna().values
                if len(pts) >= 4:
                    try:
                        hull_areas.append(ConvexHull(pts).volume)  # volume = area in 2D
                    except Exception:
                        pass
            if hull_areas:
                lines.append(f"\n[R10-B] Team convex hull area: mean={np.mean(hull_areas):.0f} m²  "
                              f"std={np.std(hull_areas):.0f}")
        except ImportError:
            lines.append("\n[R10-B] scipy unavailable — skipping hull area check")

    # R10-C: Speed bimodality (settled vs transition)
    if len(spd) > 200:
        frac_hi = float(np.mean(spd > HI_RUN_TH))
        frac_lo = float(np.mean(spd < 2.0))
        lines.append(f"\n[R10-C] Speed bimodality: frac>HI={frac_hi:.3f}  frac<walk={frac_lo:.3f}")
        lines.append(f"  (expect ~0.06–0.15 HI, ~0.20–0.40 walk/rest)")

    # R10-D: Sprint burst clustering
    sprint_flags = (match_ev["is_sprint"].astype(int)).values
    runs = []
    cur  = 0
    for f in sprint_flags:
        if f == 1: cur += 1
        else:
            if cur > 0: runs.append(cur); cur = 0
    if runs:
        fano = float(np.var(runs)) / max(float(np.mean(runs)), 1e-6)
        lines.append(f"\n[{'PASS' if fano > 1.2 else 'WARN'}] Sprint Fano factor: {fano:.2f}  (>1.2 = clustered)")

    # R10-E: Possession duration distribution (from match states)
    if "match_state" in match_ev.columns:
        phase_col = match_ev["match_state"].str.split(":").str[-1]
        lo_tempo_frac = float((phase_col == "LOW_TEMPO").mean())
        counter_frac  = float((phase_col == "COUNTER").mean())
        lines.append(f"\n[R10-E] Tactical phase fractions:")
        lines.append(f"  LOW_TEMPO={lo_tempo_frac:.2f}  COUNTER={counter_frac:.2f}")
        lines.append(f"  (expect LOW_TEMPO>0.15, COUNTER<0.15)")

    # HR-speed lag correlation
    if "heart_rate_bpm" in match_ev.columns:
        hr_s = match_ev["heart_rate_bpm"].dropna().values
        spd_s = match_ev["speed_ms"].dropna().values
        min_len = min(len(hr_s), len(spd_s))
        if min_len > 50:
            hr_s = hr_s[:min_len]; spd_s = spd_s[:min_len]
            lag_cors = [float(np.corrcoef(spd_s[:-lg], hr_s[lg:])[0, 1]) for lg in range(1, 6)]
            best = int(np.argmax(lag_cors) + 1)
            lines.append(f"\n[{'PASS' if 1<=best<=4 else 'WARN'}] HR-speed best lag: {best} ticks = {best*15}s")

    # xG stats  (R10 / R9)
    if "xg_us" in sessions.columns:
        xg_match = sessions[sessions["session_type"] == "match"]["xg_us"].dropna()
        if len(xg_match) > 0:
            lines.append(f"\n[R10/R9] xG per match: mean={xg_match.mean():.2f}  max={xg_match.max():.2f}")

    # Sensor corruption check
    n_total = len(events)
    n_null  = events["speed_ms"].isna().sum()
    lines.append(f"\n[R12] GPS dropout rate: {n_null/max(n_total,1)*100:.2f}%  (expect 1–3%)")

    lines.append(f"\nRows — events: {len(events):,}   sessions: {len(sessions):,}")
    lines.append("=" * 65)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _etype(speed: float) -> str:
    if speed >= SPRINT_TH:  return "sprint"
    if speed >= HI_RUN_TH:  return "high_intensity_run"
    if speed >  2.5:        return "jog"
    if speed >  0.5:        return "walk"
    return "rest"


def _zone(x: float, y: float) -> str:
    d = "def" if x < 35.0 else ("mid" if x < 70.0 else "att")
    f = "L"   if y < 22.7 else ("C"   if y < 45.3 else "R")
    return f"{d}_{f}"


def _make_annotation(player, sid, dur_min, date, is_anomaly,
                      anomaly_type, is_congested, fatigue, r):
    if not (is_anomaly or r.random() < 0.20):
        return None
    if is_anomaly and anomaly_type == "fatigue":
        sev = float(r.uniform(0.45, 0.75)); val = "medium" if sev > 0.6 else "mild"
        ann = "fatigue_flag"
    elif is_congested and fatigue.level > 0.4:
        sev = float(r.uniform(0.25, 0.55)); val = "mild"; ann = "fatigue_flag"
    else:
        sev = float(r.uniform(0.0, 0.25)); val = "none"; ann = "general"
    return {"player_id": player["id"], "session_id": sid,
            "annotation_type": ann, "value": val, "severity": round(sev, 3),
            "annotated_at": (date + timedelta(minutes=dur_min + 30)).isoformat(),
            "annotated_by": "coach_001",
            "note": f"Post-match review {date.strftime('%Y-%m-%d')}"}


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(n_seasons: int = 2,
                     n_matchdays_per_season: int = 38,
                     anomaly_rate: float = 0.05) -> dict:
    logger.info("v4: %d players, %d seasons, %d matchdays/season",
                len(SQUAD), n_seasons, n_matchdays_per_season)

    injury_tracker = InjuryTracker()
    acwr_tracker   = ACWRTracker()

    fitnesses: Dict[int, LatentFitness] = {
        p["id"]: _create_latent_fitness(p["id"], p["age"]) for p in SQUAD
    }
    fatigues: Dict[int, SplitFatigue] = {p["id"]: SplitFatigue() for p in SQUAD}

    players_rows, sessions_rows, events_rows = [], [], []
    annotations_rows, ground_truth_rows = [], []

    for p in SQUAD:
        players_rows.append({
            "player_id": p["id"], "external_id": f"p{p['id']:03d}",
            "full_name": p["name"], "position": p["position"],
            "age": p["age"], "age_group": p["age_group"], "nationality": p["nationality"],
        })

    season_start  = datetime(2023, 8, 1, tzinfo=timezone.utc)
    match_contexts: Dict[str, MatchContext] = {}
    prev_dates:     Dict[int, datetime] = {}

    for season in range(n_seasons):
        start    = season_start + timedelta(days=365 * season)
        schedule = generate_schedule(start, n_matchdays_per_season)
        logger.info("Season %d: %d days scheduled", season + 1, len(schedule))

        for item in schedule:
            date      = item["date"]
            stype     = item["type"]
            congested = item["is_congested"]
            days_since= item.get("days_since_last_match", 3)

            session_date = date.replace(
                hour=15 if stype == "match" else 10,
                minute=30 if stype == "match" else 0,
            )

            if stype == "match":
                dk = date.strftime("%Y%m%d")
                if dk not in match_contexts:
                    match_contexts[dk] = MatchContext.sample(
                        np.random.default_rng(int(dk) % (2**31))
                    )
                mctx = match_contexts[dk]

                available = [
                    p for p in SQUAD
                    if injury_tracker.check_and_update(
                        p["id"], session_date, fitnesses[p["id"]], acwr_tracker
                    )
                ]
                if not available:
                    continue

                is_anom_map   = {p["id"]: rng.random() < anomaly_rate for p in available}
                anom_type_map = {
                    p["id"]: (str(rng.choice(["fatigue", "positional_drift"], p=[0.70, 0.30]))
                               if is_anom_map[p["id"]] else None)
                    for p in available
                }

                match_r = np.random.default_rng(
                    hash(session_date.isoformat()) & 0xFFFFFFFF
                )
                try:
                    session_metrics, player_events = simulate_match(
                        players_in_match=available, profiles=POSITION_PROFILES,
                        fitnesses=fitnesses, fatigues=fatigues, match_ctx=mctx,
                        is_anomaly_map=is_anom_map, anomaly_type_map=anom_type_map,
                        start_time=session_date, r=match_r,
                    )
                except Exception as exc:
                    logger.warning("Match failed %s: %s", session_date.date(), exc)
                    continue

                for p in available:
                    pid  = p["id"]
                    sid  = _next_sid()
                    met  = session_metrics[pid]
                    met["session_id"]  = sid
                    met["is_congested"] = congested
                    for ev in player_events[pid]:
                        ev["session_id"] = sid

                    sessions_rows.append(met)
                    events_rows.extend(player_events[pid])

                    ann = _make_annotation(
                        p, sid, met["duration_minutes"], session_date,
                        is_anom_map[pid], anom_type_map[pid],
                        congested, fatigues[pid], match_r,
                    )
                    if ann: annotations_rows.append(ann)

                    ground_truth_rows.append({
                        "session_id": sid, "player_id": pid,
                        "is_anomaly": is_anom_map[pid],
                        "anomaly_type": anom_type_map[pid] or "",
                        "cumulative_fatigue": fatigues[pid].level,
                        "is_congested": congested,
                        "acwr": round(acwr_tracker.acwr(pid), 3),
                    })

                    load = met["total_distance_m"] / 1000.0
                    acwr_tracker.update(pid, load)
                    prev = prev_dates.get(pid, session_date - timedelta(days=3))
                    rest = max(0, (session_date - prev).days)
                    fatigues[pid].recover(fitnesses[pid].recovery_efficiency, rest)
                    fatigues[pid].load_match(1.4 if congested else 1.0)
                    prev_dates[pid] = session_date

            else:  # training
                for p in SQUAD:
                    pid = p["id"]
                    if not injury_tracker.check_and_update(pid, session_date, fitnesses[pid], acwr_tracker):
                        continue
                    prev = prev_dates.get(pid, session_date - timedelta(days=3))
                    rest = max(0, (session_date - prev).days)
                    fatigues[pid].recover(fitnesses[pid].recovery_efficiency, rest)
                    try:
                        session, evs = simulate_training_session(
                            p, POSITION_PROFILES[p["position"]],
                            fitnesses[pid], fatigues[pid],
                            session_date, congested, days_since,
                        )
                    except Exception as exc:
                        logger.warning("Training failed pid %d: %s", pid, exc)
                        continue
                    sessions_rows.append(session)
                    events_rows.extend(evs)
                    acwr_tracker.update(pid, session["total_distance_m"] / 1000.0)
                    fatigues[pid].load_training(float(TRAINING_INTENSITY.get(
                        session.get("training_type", "TACTICAL"), (0.5, 0.6))[0]))
                    prev_dates[pid] = session_date

        logger.info("Season %d done: %d sessions, %d events",
                    season + 1, len(sessions_rows), len(events_rows))

    return {"players": players_rows, "sessions": sessions_rows,
            "events": events_rows, "annotations": annotations_rows,
            "ground_truth_labels": ground_truth_rows}


# ─────────────────────────────────────────────────────────────────────────────
# Save + report
# ─────────────────────────────────────────────────────────────────────────────
def save_dataset(data: dict, apply_corruption: bool = True, output_dir: Optional[Path] = None) -> None:
    """
    output_dir=None (default): writes to OUTPUT_DIR (PlayerDynamics/data),
    unchanged behaviour for every existing caller.

    Previously this parameter did not exist at all -- cmd_generate's
    --data-dir flag was validated/echoed in its summary output but never
    actually passed here, so every `generate` run silently wrote into
    OUTPUT_DIR regardless of --data-dir. That collided with the real
    Kinexon export files which also live in PlayerDynamics/data by default
    (positions.csv, statistics.csv survive a collision since their
    filenames differ from the synthetic five-CSV set, but events.csv does
    not -- a `generate` run silently overwrites the real Kinexon events.csv
    with synthetic data). Passing output_dir now makes --data-dir effective.
    """
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    events_df = pd.DataFrame(data["events"])
    if apply_corruption and not events_df.empty:
        logger.info("Applying sensor corruption layer (R12)…")
        events_df = apply_sensor_corruption(events_df, rng)

    for key, rows in data.items():
        if not rows:
            logger.warning("No rows for %s — skipping", key)
            continue
        df = pd.DataFrame(rows) if key != "events" else events_df
        path = target_dir / f"{key}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s: %d rows → %s", key, len(df), path)


def print_dataset_report(data: dict) -> None:
    sessions = pd.DataFrame(data["sessions"])
    events   = pd.DataFrame(data["events"])
    gt       = pd.DataFrame(data.get("ground_truth_labels", []))

    print("\n" + "=" * 65)
    print("SYNTHETIC DATASET REPORT  (v4 — Decision-Agent Rewrite)")
    print("=" * 65)
    print(f"Players:        {len(data['players'])}")
    print(f"Sessions:       {len(sessions)}")
    print(f"  → Matches:    {(sessions['session_type']=='match').sum()}")
    print(f"  → Training:   {(sessions['session_type']=='training').sum()}")
    if "training_type" in sessions.columns:
        for tt, cnt in sessions[sessions["session_type"]=="training"]["training_type"].value_counts().items():
            print(f"    {tt:20s}: {cnt}")
    print(f"Events:         {len(events):,}")

    if not gt.empty and "is_anomaly" in gt.columns:
        print(f"\nGround truth:")
        print(f"  Anomalous: {gt['is_anomaly'].sum()} ({gt['is_anomaly'].mean()*100:.1f}%)")
        if gt["is_anomaly"].any():
            for at, cnt in gt[gt["is_anomaly"]]["anomaly_type"].value_counts().items():
                print(f"    {at:25s}: {cnt}")
        if "acwr" in gt.columns:
            print(f"  Avg ACWR: {gt['acwr'].mean():.2f}")

    ms = sessions[sessions["session_type"]=="match"]
    if not ms.empty:
        sq_df  = pd.DataFrame(data["players"])
        merged = ms.merge(sq_df[["player_id","position"]], on="player_id", how="left")
        print("\nPer-position match stats:")
        print(merged.groupby("position").agg(
            sessions=("session_id","count"),
            dist_mean=("total_distance_m","mean"),
            sprint_mean=("sprint_count","mean"),
        ).round(0).to_string())

    if not events.empty and not sessions.empty:
        report = validate_dataset(events, sessions)
        print("\n" + report)
        (OUTPUT_DIR / "validation_report.txt").write_text(report)

    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Players Data v4")
    p.add_argument("--seasons",      type=int,   default=2)
    p.add_argument("--matchdays",    type=int,   default=38)
    p.add_argument("--anomaly-rate", type=float, default=0.05)
    p.add_argument("--no-save",      action="store_true")
    p.add_argument("--no-corruption",action="store_true")
    args = p.parse_args()

    data = generate_dataset(
        n_seasons=args.seasons,
        n_matchdays_per_season=args.matchdays,
        anomaly_rate=args.anomaly_rate,
    )
    print_dataset_report(data)
    if not args.no_save:
        save_dataset(data, apply_corruption=not args.no_corruption)
        print(f"\nDataset → {OUTPUT_DIR.resolve()}/")
        for f in sorted(OUTPUT_DIR.glob("*.csv")):
            kb   = f.stat().st_size / 1024
            rows = sum(1 for _ in open(f)) - 1
            print(f"  {f.name:35s} {rows:>7,} rows  {kb:>8.1f} KB")