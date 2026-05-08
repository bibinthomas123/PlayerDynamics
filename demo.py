"""
Players Data — IBM CIC Germany
End-to-End Pipeline Demo

Demonstrates the full analysis pipeline with realistic synthetic data.
using the same data structures that real GPS/API/WS feeds would produce.

Run with:
   python demo.py
"""
from __future__ import annotations

import logging
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from scipy.integrate import trapezoid
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis.orchestrator import PlayersDataAnalysisPipeline
from explainability.xai_layer import SHAPExplanation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("demo")

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

players_df = pd.read_csv(DATA_DIR / "players.csv")
sessions_df = pd.read_csv(DATA_DIR / "sessions.csv")
events_df = pd.read_csv(DATA_DIR / "events.csv")
annotations_df = pd.read_csv(DATA_DIR / "annotations.csv")


# Fix timestamps
events_df["ts"] = pd.to_datetime(events_df["ts"])
sessions_df["started_at"] = pd.to_datetime(sessions_df["started_at"])
sessions_df["ended_at"] = pd.to_datetime(sessions_df["ended_at"])
annotations_df["annotated_at"] = pd.to_datetime(annotations_df["annotated_at"])


def build_features(events_df):
    return events_df.groupby("session_id").agg(
        window_distance_m=("speed_ms", lambda x: trapezoid(x, dx=15)),
        window_avg_speed_ms=("speed_ms", "mean"),
        window_sprint_count=("is_sprint", "sum"),
        heart_rate_bpm=("heart_rate_bpm", "mean"),
    ).reset_index()

# ─────────────────────────────────────────────
# Realistic session data generator
# (Represents real historical data that would come from DB in production)
# ─────────────────────────────────────────────

def make_session_history(
    player_id: int,
    n_sessions: int = 30,
    distance_mean: float = 10500,
    sprint_mean: float = 25,
    rng_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic historical session data for one player.
    Mimics real GPS + wearable data over 30 sessions.
    """
    rng = np.random.default_rng(rng_seed)
    now = datetime.now(tz=timezone.utc)

    session_rows = []
    event_rows = []
    annotation_rows = []

    for i in range(n_sessions):
        session_id = player_id * 1000 + i
        started_at = now - timedelta(days=n_sessions - i, hours=int(rng.integers(10, 20)))
        duration_min = float(rng.integers(85, 97))

        # Session-level metrics with realistic variability
        total_dist = rng.normal(distance_mean, distance_mean * 0.08)
        sprint_count = max(0, int(rng.normal(sprint_mean, sprint_mean * 0.2)))
        max_speed = rng.normal(9.2, 0.5)
        hi_dist = rng.normal(total_dist * 0.28, total_dist * 0.04)
        avg_speed = total_dist / (duration_min * 60)

        session_rows.append({
            "session_id": session_id,
            "player_id": player_id,
            "started_at": started_at,
            "duration_minutes": duration_min,
            "total_distance_m": round(total_dist, 1),
            "sprint_count": sprint_count,
            "max_speed_ms": round(max_speed, 2),
            "high_speed_distance_m": round(hi_dist, 1),
            "avg_speed_ms": round(avg_speed, 3),
            "avg_heart_rate_bpm": float(rng.integers(138, 162)),
            "data_quality_score": float(rng.uniform(0.85, 1.0)),
        })

        # Event-level data (1 event per minute for speed)
        n_events = int(duration_min)
        for minute in range(n_events):
            # Fatigue curve: speed decays across the match
            fatigue_factor = np.exp(-0.008 * minute)
            base_speed = avg_speed * fatigue_factor * rng.uniform(0.7, 1.4)
            speed = max(0.0, base_speed)

            # Pitch position oscillates around player's typical zone
            x_center, y_center = {
                1: (20, 50), 2: (30, 25), 3: (30, 75),   # CB, LB, RB
                4: (50, 50), 5: (65, 50), 6: (65, 30),   # CM, CAM, LW
                7: (65, 70), 8: (85, 50),                  # RW, ST
            }.get(player_id % 8 + 1, (50, 50))

            x = np.clip(rng.normal(x_center, 8), 0, 100)
            y = np.clip(rng.normal(y_center, 10), 0, 100)

            event_rows.append({
                "session_id": session_id,
                "player_id": player_id,
                "ts": started_at + timedelta(minutes=minute),
                "speed_ms": round(speed, 2),
                "is_sprint": speed >= 7.0,
                "x_pitch": round(x, 2),
                "y_pitch": round(y, 2),
                "heart_rate_bpm": int(rng.integers(125, 175)),
            })

        # Occasional coach annotation
        if rng.random() < 0.25:
            annotation_rows.append({
                "session_id": session_id,
                "player_id": player_id,
                "annotation_type": "fatigue_flag",
                "value": rng.choice(["mild", "moderate", "none"]),
                "severity": float(rng.uniform(0.0, 0.6)),
                "annotated_at": started_at,
                "annotated_by": "coach_001",
            })

    sessions_df = pd.DataFrame(session_rows)
    events_df = pd.DataFrame(event_rows)
    annotations_df = pd.DataFrame(annotation_rows) if annotation_rows else pd.DataFrame()

    return sessions_df, events_df, annotations_df


def make_fatigued_event(external_id: str, baseline_dist: float) -> dict:
    """
    Simulate a live event for a fatigued player.
    Sprint count and distance are well below their personal baseline.
    """
    return {
        "player_external_id": external_id,
        "ts": datetime.now(tz=timezone.utc),
        "source": "ws",
        "latitude": 51.4775,
        "longitude": -0.0165,
        "speed_ms": 2.1,                            # Very slow for this player
        "window_sprint_count": 1,                   # Much below baseline
        "window_distance_m": baseline_dist * 0.41,  # 59% below baseline
        "window_avg_speed_ms": 2.3,
        "x_pitch": 55.0,
        "y_pitch": 28.0,
        "heart_rate_bpm": 185,                      # Elevated
        "hr_recovery_time_s": 62.0,                 # Poor recovery
        "data_quality_score": 0.97,
        "is_sprint": False,
        "is_high_intensity": False,
    }


def alert_handler(explanation: SHAPExplanation) -> None:
    """Callback invoked every time an alert is generated."""
    print("\n" + "=" * 70)
    print("🚨  PLAYERS DATA ALERT")
    print("=" * 70)
    print(f"Player:          {explanation.external_id}")
    print(f"Recommendation:  {explanation.recommendation_type.upper()}")
    print(f"Confidence:      {explanation.confidence * 100:.0f}%")
    print()
    print("📊  SHAP Contributions (top 5):")
    for c in explanation.top_contributions[:5]:
        bar = "+" * int(abs(c.shap_value) * 100) if c.shap_value > 0 else "-" * int(abs(c.shap_value) * 100)
        bar = bar[:30]
        sign = "▲" if c.shap_value >= 0 else "▽"
        print(f"  {sign} {c.human_label[:40]:40s}  {c.formatted_value}")
        print(f"    SHAP: {c.shap_value:+.4f}  {bar}")
    print()
    print(f"💬  EXPLANATION:\n  {explanation.nlg_summary}")
    print()
    print(f"↔️  COUNTERFACTUAL:\n  {explanation.counterfactual}")
    print()
    print("Coach actions: [Accept] [Override] [Add note]")
    print("=" * 70)


# ─────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("PLAYERS DATA — IBM CIC Germany, Group 11 / 2B")
    print("Explainable Player Pattern Analysis — Analysis Pipeline Demo")
    print("=" * 70 + "\n")

    # ── 1. Initialise pipeline ──
    pipeline = PlayersDataAnalysisPipeline()
    pipeline.set_alert_callback(alert_handler)

    # ── 2. Register squad ──
    squad = [
        dict(player_id=1, external_id="p001", name="Player 1 (GK)",  position="GK",  age=28, age_group="Senior"),
        dict(player_id=2, external_id="p002", name="Player 2 (CB)",  position="CB",  age=24, age_group="Senior"),
        dict(player_id=3, external_id="p003", name="Player 3 (CB)",  position="CB",  age=30, age_group="Senior"),
        dict(player_id=4, external_id="p004", name="Player 4 (LB)",  position="LB",  age=22, age_group="U23"),
        dict(player_id=5, external_id="p005", name="Player 5 (RB)",  position="RB",  age=25, age_group="Senior"),
        dict(player_id=6, external_id="p006", name="Player 6 (CM)",  position="CM",  age=27, age_group="Senior"),
        dict(player_id=7, external_id="p007", name="Player 7 (CAM)", position="CAM", age=26, age_group="Senior"),
        dict(player_id=8, external_id="p008", name="Player 8 (LW)",  position="LW",  age=21, age_group="U23"),
        dict(player_id=9, external_id="p009", name="Player 9 (RW)",  position="RW",  age=23, age_group="Senior"),
        dict(player_id=10,external_id="p010", name="Player 10 (ST)", position="ST",  age=29, age_group="Senior"),
    ]

    for p in squad:
        pipeline.register_player(**p)

    print(f"✓ Registered {len(squad)} players\n")

    print("Loading historical data from CSVs...")

    for p in squad:
        pid = p["player_id"]

        # ── Filter player data ──
        player_sessions = sessions_df[sessions_df["player_id"] == pid].copy()

        if player_sessions.empty:
            continue

        # Limit to last 30 sessions (important)
        player_sessions = player_sessions.sort_values("started_at").tail(30)

        player_events = events_df[
            events_df["session_id"].isin(player_sessions["session_id"])
        ].copy()

        player_annotations = annotations_df[
            annotations_df["session_id"].isin(player_sessions["session_id"])
        ]

        features_df = build_features(player_events)

        # ── Merge features into sessions ──
        player_sessions = player_sessions.merge(
            features_df,
            on="session_id",
            how="left"
        )

        # ── Fill missing (safety) ──
        player_sessions["window_distance_m"] = player_sessions["window_distance_m"].fillna(0)
        player_sessions["window_avg_speed_ms"] = player_sessions["window_avg_speed_ms"].fillna(0)
        player_sessions["window_sprint_count"] = player_sessions["window_sprint_count"].fillna(0)
        player_sessions["heart_rate_bpm"] = player_sessions["heart_rate_bpm"].fillna(120)

        # ── Load into pipeline ──
        pipeline.load_historical_data(
            player_id=pid,
            sessions_df=player_sessions,
            events_df=player_events,
            annotations_df=player_annotations,
        )

    print("✓ Historical data loaded from dataset\n")



    # ── 4. Compute baselines ──
    print("Computing personal baselines (28-day window)...")
    baselines = pipeline.compute_baselines(window_days=28)
    print(f"✓ Baselines computed for {len(baselines)} players\n")

    for pid, b in baselines.items():
        player = pipeline.registry.get(pid)
        print(
            f"  {player['name']:25s} "
            f"dist_mean={b.distance_mean:7.0f}m  "
            f"sprint_mean={b.sprint_count_mean:5.1f}  "
            f"fatigue_r2={b.fatigue_r_squared or 0:.3f}"
        )

    # ── 5. Train Isolation Forest models ──
    print("\nTraining Isolation Forest models (per-player)...")
    pipeline.train_all_models()
    print("✓ All models trained\n")

    # ── 6. Simulate live events ──
    print("Simulating live match events...\n")

    # Normal events for most players
    normal_players = ["p001", "p002", "p003", "p004", "p005",
                      "p006", "p008", "p009", "p010"]
    for ext_id in normal_players:
        player = pipeline.registry.get_by_external_id(ext_id)
        if player is None:
            continue
        b = player["baseline"]
        if b is None:
            continue
        normal_event = {
            "player_external_id": ext_id,
            "ts": datetime.now(tz=timezone.utc),
            "source": "ws",
            "speed_ms": 5.8,
            "window_sprint_count": int(b.sprint_count_mean * 0.9),
            "window_distance_m": b.distance_mean * 0.85,
            "window_avg_speed_ms": 4.2,
            "x_pitch": b.avg_x or 50.0,
            "y_pitch": b.avg_y or 50.0,
            "heart_rate_bpm": 155,
            "hr_recovery_time_s": 22.0,
            "data_quality_score": 0.98,
            "is_sprint": False,
            "is_high_intensity": True,
        }
        result = pipeline.process_live_event(normal_event, segment_index=0)
        if result:
            print(f"  Alert for {ext_id}: {result.recommendation_type}")

    # Player 7 (CAM) — simulate fatigue scenario
    player7 = pipeline.registry.get(7)
    b7 = player7["baseline"]

    print("\nSimulating Player 7 (CAM) fatigue scenario (min 60–75)...")
    fatigued_event = make_fatigued_event("p007", b7.distance_mean)
    result = pipeline.process_live_event(fatigued_event, segment_index=0)  # 60–75 min segment

    # ── 7. Simulate coach override ──
    if result:
        print("\nSimulating coach override for Player 7...")
        # The alert was shown but the coach overrides (tactical reason, not fatigue)
        pipeline.log_coach_decision(
            inference_id=pipeline._inference_id_counter,
            player_id=7,
            decision="override",
            coach_id="coach_hauptmann",
            coach_note="Player 7 is holding position — tactical instruction, not fatigue.",
        )
        print("✓ Override logged with context snapshot\n")

    # ── 8. Override summary ──
    summary = pipeline.get_override_summary()
    print(f"Override summary: {summary}\n")

    # ── 9. Recalibration ──
    print("Running recalibration pipeline...")
    recal_results = pipeline.recalibrate(trigger_reason="demo_run")
    if recal_results:
        for r in recal_results:
            print(f"  Recalibrated player={r['player_id']}: {r['notes']}")
    else:
        print("  No recalibration adjustments needed (insufficient overrides — expected for demo)")

    # ── 10. Fairness audit ──
    print("\nRunning fairness audit...")
    audit_report = pipeline.run_fairness_audit()
    print(audit_report)

    # ── 11. Inference log ──
    inference_log = pipeline.get_inference_log()
    print(f"\nInference log: {len(inference_log)} records captured")
    if not inference_log.empty and "recommendation_type" in inference_log.columns:
        counts = inference_log["recommendation_type"].value_counts()
        for rec_type, count in counts.items():
            print(f"  {rec_type}: {count}")



if __name__ == "__main__":
    main()
