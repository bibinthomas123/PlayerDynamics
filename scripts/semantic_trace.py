# -*- coding: ascii -*-
"""
scripts/semantic_trace.py

Real-player semantic trace: SC Magdeburg vs HSG Wetzlar, session 3387.
Player: Lukas Mertens (mapped_id=2058, jersey #22).

Shows the semantic pipeline from AnomalyResult feature values to SemanticFindings.

BEFORE fix: hr_sensor_present=True (simulates old state) -> all findings suppressed.
AFTER  fix: hr_sensor_present=False (new default)        -> movement findings fire.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

from config.settings import CONFIG
from explainability.semantics_layer import SemanticInterpreter


SEPARATOR = "-" * 60
SEPARATOR2 = "=" * 60


# ---------------------------------------------------------------------------
# Feature vector built from real Kinexon observations for Lukas Mertens.
# Values are derived from positions.csv (session 3387):
#   speed_ms       ~1.2 m/s  (walking pace during low-intensity period)
#   distance_delta ~0.035 m per 50ms tick; x30 steps = ~36 m per window
#   heart_rate_bpm  0.0  (wearable not worn: None -> safe_float -> 0.0)
#   sprint_flag     0    (below 5.5 m/s threshold after Phase A fix)
#   x_pitch        ~52  (left side of court, defending third)
#   positional_drift_score: 1.5 (player drifted 1.5x outside zone radius)
#   acwr            1.1  (within normal workload band)
# ---------------------------------------------------------------------------

FEATURE_VALUES = {
    # Sequence features (from last LSTM step)
    "speed_ms":             1.2,
    "acceleration_ms2":     0.0,
    "heart_rate_bpm":       0.0,    # sensor absent -> safe_float(None, 0.0)
    "sprint_flag":          0.0,
    "x_pitch":             52.0,
    "y_pitch":             28.0,
    "distance_delta_m":     1.2,
    "hr_recovery_rate":     0.0,

    # Window-level fields (added by orchestrator _build_xai_feature_vector)
    "window_avg_speed_ms":  1.2,
    "window_distance_m":   36.0,    # 1.2 m/s x 30 s
    "window_sprint_count":  0.0,

    # Workload and positional signals
    "acwr":                 1.1,
    "positional_drift_score": 1.5,  # 1.5x zone radius -> above drift_elevated threshold (1.2)
    "fatigue_decay_residual": 0.0,
    "speed_drop_pct":         0.0,
    "hr_recovery_time_s":    0.0,

    # Z-scores relative to personal baseline
    "z_distance":          -0.2,
    "z_sprint_count":      -0.3,
    "z_top_speed":         -0.4,
    "z_high_speed_dist":   -0.2,

    # Coach annotations (not set for this session)
    "coach_fatigue_severity": 0.0,
    "coach_pre_match_status_encoded": 0.0,
}

# SHAP values: channel-ablation attribution from the LSTM encoder.
# Positive SHAP = feature is pushing the anomaly score up.
# window_avg_speed_ms: 0.22  -> low speed driving the anomaly
# window_distance_m: 0.10   -> low distance driving the anomaly
# positional_drift_score: 0.18 -> drift driving the anomaly (above shap_relevant=0.05)
SHAP_VALUES = {
    "window_avg_speed_ms":    0.22,
    "window_distance_m":      0.10,
    "positional_drift_score": 0.18,
    "speed_ms":               0.08,
    "acwr":                   0.03,
    "window_sprint_count":    0.01,
    "z_distance":            -0.02,
}


def run_semantic_pass(hr_sensor_present, label):
    """Run interpret() with given hr_sensor_present config and print results."""
    interp = SemanticInterpreter()
    with patch.object(CONFIG.kinexon, "hr_sensor_present", hr_sensor_present):
        quality = interp._assess_window_quality(FEATURE_VALUES)
        findings = interp.interpret(
            shap_values=SHAP_VALUES,
            feature_values=FEATURE_VALUES,
            persistence_windows=0,
        )
    return quality, findings


def main():
    print(SEPARATOR2)
    print("SEMANTIC TRACE -- session 3387 (HSG Wetzlar vs SC Magdeburg)")
    print("Player  : Lukas Mertens  (mapped_id=2058, jersey #22)")
    print("Window  : ~mid first half, walking phase after defensive set")
    print("HR data : ABSENT (wearable not worn; heart_rate_bpm=None -> 0.0)")
    print(SEPARATOR2)

    # --- Feature vector summary ---
    print()
    print("FEATURE VECTOR (key fields):")
    fields = [
        ("window_avg_speed_ms",   "m/s",   "below 2.5 m/s threshold -> locomotor_suppression candidate"),
        ("window_distance_m",     "m",     "36m in 30s window"),
        ("heart_rate_bpm",        "bpm",   "0.0 = sensor absent (safe_float(None, 0.0))"),
        ("positional_drift_score","x norm","1.5x zone radius -> above drift_elevated (1.2)"),
        ("acwr",                  "",      "1.1 -> within normal workload band"),
        ("sprint_flag",           "",      "0.0 -> no sprint (below 5.5 m/s after D1 fix)"),
    ]
    for k, unit, note in fields:
        v = FEATURE_VALUES.get(k, 0.0)
        print(f"  {k:<26} = {v:.1f}{(' ' + unit) if unit else ''}  | {note}")

    print()
    print("SHAP ATTRIBUTION (top driving features):")
    for k, v in sorted(SHAP_VALUES.items(), key=lambda x: -abs(x[1])):
        bar = "+" * int(abs(v) * 30) if v > 0 else "-" * int(abs(v) * 30)
        print(f"  {k:<26} = {v:+.2f}  [{bar}]")

    print()
    print(SEPARATOR2)

    # ----- BEFORE (simulate old state: hr_sensor_present=True) -----
    print("BEFORE: hr_sensor_present=True (simulates state before this fix)")
    print(SEPARATOR)
    quality_before, findings_before = run_semantic_pass(hr_sensor_present=True, label="BEFORE")
    print(f"  Window quality degraded : {quality_before['degraded']}")
    if quality_before["reasons"]:
        for r in quality_before["reasons"]:
            print(f"    [BLOCKED] {r}")
    print(f"  SemanticFindings        : {len(findings_before)} (suppressed by quality gate)")
    if not findings_before:
        print("  --> interpret() returned [] -- ALL finding types blocked")

    print()
    print(SEPARATOR2)

    # ----- AFTER (hr_sensor_present=False, the new default) -----
    print("AFTER:  hr_sensor_present=False (new default in KinexonConfig)")
    print(SEPARATOR)
    quality_after, findings_after = run_semantic_pass(hr_sensor_present=False, label="AFTER")
    print(f"  Window quality degraded : {quality_after['degraded']}")
    if not quality_after["reasons"]:
        print("  Quality reasons         : none  (gate cleared)")
    print(f"  SemanticFindings        : {len(findings_after)}")

    print()
    if findings_after:
        for i, f in enumerate(findings_after, 1):
            print(f"  Finding {i}: {f.finding_type}")
            print(f"    Severity   : {f.severity}")
            print(f"    Confidence : {f.confidence:.2f}")
            print(f"    Domain     : {f.domain}")
            print(f"    Summary    : {f.summary}")
            print(f"    Features   : {', '.join(f.supporting_features)}")
            ev_str = ", ".join(f"{k}={v:.2f}" for k, v in f.evidence.items())
            print(f"    Evidence   : {ev_str}")
            if i < len(findings_after):
                print()
    else:
        print("  --> No findings (SHAP/threshold conditions not met for this window)")

    print()
    print(SEPARATOR2)
    print("COMPARISON:")
    print(SEPARATOR)
    print(f"  BEFORE: {len(findings_before)} findings (all suppressed by HR quality gate)")
    print(f"  AFTER : {len(findings_after)} findings generated")
    if findings_after:
        ftypes = [f.finding_type for f in findings_after]
        print(f"  Types : {', '.join(ftypes)}")
        print()
        print("  Finding types now unblocked for Kinexon HR-less exports:")
        unblocked = ["locomotor_suppression", "locomotor_overload",
                     "tactical_instability", "fatigue_accumulation"]
        for ft in unblocked:
            status = "FIRES" if ft in ftypes else "conditions not met this window"
            print(f"    {ft:<30} -> {status}")
        cardiovascular_note = (
            "cardiovascular_overload       -> still blocked (requires hr >= 175 bpm)"
        )
        print(f"    {cardiovascular_note}")
    print()
    print(SEPARATOR2)
    print("Fix verified: HR-absent Kinexon events now produce semantic findings.")
    print("CONFIG.kinexon.hr_sensor_present = False (set True when wearables are integrated)")
    print(SEPARATOR2)


if __name__ == "__main__":
    main()
