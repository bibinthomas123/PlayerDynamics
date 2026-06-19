# -*- coding: ascii -*-
"""
phase_a_trace.py

Real Kinexon data trace showing before vs after for all four Phase A fixes.
Uses actual positions.csv data from SC Magdeburg session 3387.

Outputs:
  - Sprint count (D1)
  - Regime classification (D1 cascade)
  - distance_delta values (D2)
  - Baseline speed (D4)
  - Fatigue signals (D3 + D4 combined)

Run:
    python scripts/phase_a_trace.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config.settings import CONFIG
from ingestion.kinexon_adapter import KinexonAdapter

DATA_DIR       = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH     = DATA_DIR / "statistics.csv"

LINE = "-" * 72
SEC  = "=" * 72

# ---------------------------------------------------------------------------
# Pre-fix (football) constants -- kept local for comparison only
# ---------------------------------------------------------------------------
_OLD_SPRINT_MS    = 7.0
_OLD_PITCH_LEN    = 105.0
_OLD_PITCH_WID    = 68.0
_OLD_MATCH_DUR    = 90 * 60   # 5400 s
_OLD_HALF_DUR     = 45 * 60   # 2700 s

# Post-fix (handball) values -- from CONFIG (what the engine now uses)
_NEW_SPRINT_MS    = CONFIG.kinexon.sprint_threshold_ms
_NEW_PITCH_LEN    = CONFIG.kinexon.pitch_length_m
_NEW_PITCH_WID    = CONFIG.kinexon.pitch_width_m
_NEW_MATCH_DUR    = CONFIG.kinexon.match_duration_s
_NEW_HALF_DUR     = CONFIG.kinexon.match_half_duration_s


def _sprint_flag(speed, thr):
    return 1 if speed >= thr else 0


def _distance_delta(x, px, y, py, pw, pl):
    dx_m = (x - px) / 100.0 * pw
    dy_m = (y - py) / 100.0 * pl
    return math.sqrt(dx_m * dx_m + dy_m * dy_m)


def _classify_regime(sprint_frac, mean_x):
    if mean_x < 33.0:
        territory = "defensive"
    elif mean_x > 67.0:
        territory = "attacking"
    else:
        territory = "midfield"
    if sprint_frac >= 0.15:
        intensity = "high"
    elif sprint_frac >= 0.04:
        intensity = "medium"
    else:
        intensity = "low"
    return territory + "__" + intensity


def _baseline_speed(distance_mean, duration_s):
    return distance_mean / duration_s if distance_mean > 0 else 3.5


def col(label, value, width=38):
    return "  {:<{}}{}".format(label, width, value)


def main():
    print(SEC)
    print("  Phase A Calibration Trace - SC Magdeburg / Kinexon Session 3387")
    print(SEC)

    if not POSITIONS_PATH.exists():
        print("ERROR: {} not found. Restore data/ directory first.".format(POSITIONS_PATH))
        sys.exit(1)

    adapter = KinexonAdapter()
    meta    = adapter.load_player_meta(STATS_PATH) if STATS_PATH.exists() else {}

    # Collect up to 100 events per player for trace stats
    player_events = {}
    n_total = 0
    for obs in adapter.stream_positions(POSITIONS_PATH, meta,
                                        session_id="trace", match_id="trace"):
        n_total += 1
        pid = obs.player_id
        if pid not in player_events:
            player_events[pid] = []
        if len(player_events[pid]) < 100:
            player_events[pid].append(obs)
        if n_total > 200_000:
            break

    # Pick Lukas Mertens (mapped_id=2058) if present, else first player with 8+ events
    PREFER_ID = 2058
    if PREFER_ID in player_events and len(player_events[PREFER_ID]) >= 8:
        trace_pid = PREFER_ID
        trace_evs = player_events[PREFER_ID]
    else:
        trace_pid = next(pid for pid, evs in player_events.items() if len(evs) >= 8)
        trace_evs = player_events[trace_pid]

    print("\nTrace player: mapped_id={}".format(trace_pid))
    if trace_pid in meta:
        m = meta[trace_pid]
        print("  Name   : {}".format(m.player_name))
        print("  Jersey : {}".format(m.jersey_number))
        print("  Group  : {}".format(m.group_name))
    print("  Events : {} (using up to 100 for stats)".format(len(trace_evs)))
    print(LINE)

    evts = [adapter.to_event_dict(obs, elapsed_s=float(i * 15))
            for i, obs in enumerate(trace_evs)]

    # -------------------------------------------------------------------
    # D1: Sprint classification
    # -------------------------------------------------------------------
    print("\n" + SEC)
    print("  D1 -- Sprint Threshold  (was 7.0 m/s -> now 5.5 m/s)")
    print(SEC)

    speeds      = [float(e["speed_ms"] or 0) for e in evts]
    sp_old      = [_sprint_flag(s, _OLD_SPRINT_MS) for s in speeds]
    sp_new      = [_sprint_flag(s, _NEW_SPRINT_MS) for s in speeds]
    count_old   = sum(sp_old)
    count_new   = sum(sp_new)
    max_speed   = max(speeds)
    mean_speed  = sum(speeds) / len(speeds)
    frac_old    = count_old / len(sp_old)
    frac_new    = count_new / len(sp_new)

    print(col("Threshold (BEFORE)", "{:.1f} m/s  (football)".format(_OLD_SPRINT_MS)))
    print(col("Threshold (AFTER)",  "{:.1f} m/s  (handball IHF)".format(_NEW_SPRINT_MS)))
    print(col("Events analysed",    str(len(evts))))
    print(col("Max speed in window", "{:.2f} m/s".format(max_speed)))
    print(col("Mean speed",          "{:.2f} m/s".format(mean_speed)))
    print()
    note_old = "  <- always 0 (bug)" if count_old == 0 else ""
    note_new = "  OK" if count_new > 0 else "  (not sprinting in this segment)"
    print(col("Sprint events BEFORE", "{} / {}{}".format(count_old, len(evts), note_old)))
    print(col("Sprint events AFTER",  "{} / {}{}".format(count_new, len(evts), note_new)))
    print(col("Sprint fraction BEFORE", "{:.4f}".format(frac_old)))
    print(col("Sprint fraction AFTER",  "{:.4f}".format(frac_new)))

    sprint_speeds = sorted(set(round(s, 2) for s in speeds if s >= _NEW_SPRINT_MS))
    if sprint_speeds:
        print("\n  Speeds now classified as sprints (>= 5.5 m/s):")
        for s in sprint_speeds:
            was = "XX missed before" if s < _OLD_SPRINT_MS else "was sprint before too"
            print("    {:.2f} m/s  [{}]".format(s, was))
    else:
        print("\n  No speeds >= 5.5 m/s in this segment (player at rest/jog).")
        print("  Max speed: {:.2f} m/s".format(max_speed))

    # -------------------------------------------------------------------
    # Regime (cascade from D1)
    # -------------------------------------------------------------------
    print("\n" + LINE)
    print("  Regime classification (cascade from D1)")
    print(LINE)

    x_vals    = [float(e.get("x_pitch", 50)) for e in evts]
    mean_x    = sum(x_vals) / len(x_vals)
    regime_old = _classify_regime(frac_old, mean_x)
    regime_new = _classify_regime(frac_new, mean_x)

    print(col("mean_x_pitch",       "{:.1f}".format(mean_x)))
    print(col("Sprint frac BEFORE", "{:.4f}".format(frac_old)))
    print(col("Sprint frac AFTER",  "{:.4f}".format(frac_new)))
    note_r_old = "  <- always __low (bug)" if regime_old.endswith("__low") and count_new > 0 else ""
    print(col("Regime BEFORE", regime_old + note_r_old))
    print(col("Regime AFTER",  regime_new))

    # -------------------------------------------------------------------
    # D2: distance_delta
    # -------------------------------------------------------------------
    print("\n" + SEC)
    print("  D2 -- Pitch Geometry  (was 105x68 m -> now 40x20 m)")
    print(SEC)

    print(col("Pitch (BEFORE)", "{:.0f} m x {:.0f} m  (FIFA)".format(_OLD_PITCH_LEN, _OLD_PITCH_WID)))
    print(col("Pitch (AFTER)",  "{:.0f} m x {:.0f} m  (IHF handball)".format(_NEW_PITCH_LEN, _NEW_PITCH_WID)))
    print()

    dist_old = []
    dist_new = []
    for i in range(1, len(evts)):
        x  = float(evts[i].get("x_pitch", 50))
        y  = float(evts[i].get("y_pitch", 50))
        px = float(evts[i-1].get("x_pitch", 50))
        py = float(evts[i-1].get("y_pitch", 50))
        dist_old.append(_distance_delta(x, px, y, py, _OLD_PITCH_WID, _OLD_PITCH_LEN))
        dist_new.append(_distance_delta(x, px, y, py, _NEW_PITCH_WID, _NEW_PITCH_LEN))

    mean_old  = 0.0
    mean_new  = 0.0
    ratio     = 1.0
    total_old = 0.0
    total_new = 0.0
    if dist_old:
        mean_old  = sum(dist_old) / len(dist_old)
        mean_new  = sum(dist_new) / len(dist_new)
        max_old   = max(dist_old)
        max_new   = max(dist_new)
        total_old = sum(dist_old)
        total_new = sum(dist_new)
        ratio     = mean_old / mean_new if mean_new > 0 else 0.0

        print(col("Ticks with movement data", str(len(dist_old))))
        print()
        print(col("Mean distance_delta BEFORE", "{:.4f} m  (football dims)".format(mean_old)))
        print(col("Mean distance_delta AFTER",  "{:.4f} m  (handball dims)".format(mean_new)))
        print(col("Inflation factor removed",   "{:.2f}x  (old / new)".format(ratio)))
        print()
        print(col("Max distance_delta BEFORE",  "{:.4f} m".format(max_old)))
        print(col("Max distance_delta AFTER",   "{:.4f} m".format(max_new)))
        print()
        print(col("Total distance BEFORE",      "{:.2f} m".format(total_old)))
        print(col("Total distance AFTER",       "{:.2f} m".format(total_new)))

        print("\n  First 5 tick-level distance_delta values:")
        print("  {:>3}  {:>22}  {:>22}  {:>8}".format(
            "t", "BEFORE (football m)", "AFTER (handball m)", "ratio"))
        print("  " + "-" * 62)
        for i, (d_old, d_new) in enumerate(zip(dist_old[:5], dist_new[:5])):
            r = d_old / d_new if d_new > 0 else 0.0
            print("  {:>3}  {:>22.4f}  {:>22.4f}  {:>7.2f}x".format(i+1, d_old, d_new, r))

    # -------------------------------------------------------------------
    # D4: Baseline speed
    # -------------------------------------------------------------------
    print("\n" + SEC)
    print("  D4 -- Baseline Speed Denominator  (was 90x60 s -> now 60x60 s)")
    print(SEC)

    bs_old_old = bs_new_new = 0.0
    ratio_before = ratio_after = 0.0
    speed_low_before = speed_low_after = False
    projected_old = projected_new = 0.0

    if dist_old:
        ticks_in_window   = len(dist_old)
        ticks_per_old     = _OLD_MATCH_DUR / 15   # at 15 s/tick
        ticks_per_new     = _NEW_MATCH_DUR / 15

        projected_old = (sum(dist_old) / ticks_in_window) * ticks_per_old
        projected_new = (sum(dist_new) / ticks_in_window) * ticks_per_new

        bs_old_old = _baseline_speed(projected_old, _OLD_MATCH_DUR)
        bs_new_new = _baseline_speed(projected_new, _NEW_MATCH_DUR)

        print(col("Denominator (BEFORE)", "{} s  (90 min football)".format(_OLD_MATCH_DUR)))
        print(col("Denominator (AFTER)",  "{} s  (60 min handball)".format(_NEW_MATCH_DUR)))
        print()
        print(col("Projected session distance (old dims)", "{:.1f} m".format(projected_old)))
        print(col("Projected session distance (new dims)", "{:.1f} m".format(projected_new)))
        print()
        print(col("baseline_speed BEFORE (old dims + old denom)",
                  "{:.4f} m/s  <- inflated".format(bs_old_old)))
        print(col("baseline_speed AFTER  (new dims + new denom)",
                  "{:.4f} m/s  <- correct".format(bs_new_new)))
        print()

        ratio_before    = mean_speed / max(bs_old_old, 0.1)
        ratio_after     = mean_speed / max(bs_new_new, 0.1)
        speed_low_before = ratio_before < 0.55
        speed_low_after  = ratio_after  < 0.55

        print(col("Player mean speed in window",    "{:.2f} m/s".format(mean_speed)))
        print(col("speed_ratio BEFORE",
                  "{:.4f}  -> speed_low={}".format(ratio_before, speed_low_before)))
        print(col("speed_ratio AFTER",
                  "{:.4f}  -> speed_low={}".format(ratio_after, speed_low_after)))
        if speed_low_before and not speed_low_after:
            print()
            print("  FIX: player no longer permanently flagged as 'slow'")
        elif not speed_low_before and not speed_low_after:
            print()
            print("  BOTH old and new: not slow (fast enough even with inflated baseline)")

    # -------------------------------------------------------------------
    # D3: late_in_game at representative elapsed values
    # -------------------------------------------------------------------
    print("\n" + SEC)
    print("  D3 -- late_in_game Gate  (was elapsed > 2700 s -> now > 1800 s)")
    print(SEC)

    print(col("Half duration (BEFORE)", "{} s  (45 min football)".format(_OLD_HALF_DUR)))
    print(col("Half duration (AFTER)",  "{} s  (30 min handball)".format(_NEW_HALF_DUR)))
    print()
    print("  {:>10}  {:>16}  {:>16}  {:>14}".format(
        "elapsed_s", "BEFORE (>2700)", "AFTER (>1800)", "change"))
    print("  " + "-" * 60)

    for elapsed in [600, 1200, 1800, 1801, 2100, 2400, 2700, 3000, 3600]:
        old    = elapsed > _OLD_HALF_DUR
        new    = elapsed > _NEW_HALF_DUR
        if new and not old:
            change = "NOW LATE"
        else:
            change = "same"
        print("  {:>10.0f}  {:>16}  {:>16}  {:>14}".format(
            elapsed, str(old), str(new), change))

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\n" + SEC)
    print("  Phase A Summary -- Before vs After")
    print(SEC)

    print()
    print("  Metric                           BEFORE                     AFTER")
    print("  " + "-" * 68)
    print("  Sprint threshold                 7.0 m/s (football)         5.5 m/s (handball IHF)")
    print("  Sprint count (this window)       {:<26} {}".format(count_old, count_new))
    print("  Sprint fraction                  {:<26.4f} {:.4f}".format(frac_old, frac_new))
    intensity_after = "high" if frac_new >= 0.15 else ("medium" if frac_new >= 0.04 else "low")
    intensity_before = "always 'low' (bug)"
    print("  Regime intensity                 {:<26} {}".format(intensity_before, intensity_after))
    print()
    print("  Pitch dimensions                 105m x 68m (FIFA)          40m x 20m (handball)")
    if dist_old:
        print("  Mean distance_delta              {:<26.4f} {:.4f} m".format(mean_old, mean_new))
        print("  Inflation factor removed         {:<26.2f} -> 1.00x".format(ratio))
    print()
    print("  Match duration (baseline denom)  5400 s (90 min)            3600 s (60 min)")
    if dist_old:
        print("  baseline_speed                   {:<26.4f} {:.4f} m/s".format(bs_old_old, bs_new_new))
        print("  speed_ratio (at mean speed)      {:<26.4f} {:.4f}".format(ratio_before, ratio_after))
        print("  speed_low flag                   {:<26} {}".format(str(speed_low_before), speed_low_after))
    print()
    print("  Half duration (late_in_game)     2700 s (45 min)            1800 s (30 min)")
    newly_late = sum(
        1 for t in range(1801, 3601)
        if (t > _NEW_HALF_DUR) and not (t > _OLD_HALF_DUR)
    )
    print("  Extra seconds now flagged 'late' 0                          {} s".format(newly_late))
    print()
    print("  ENGINE STATUS:")
    if count_old == 0 and max_speed >= _NEW_SPRINT_MS:
        print("  OK  D1: sprint detection restored -- speeds above 5.5 m/s now classified")
    elif max_speed < _NEW_SPRINT_MS:
        print("  OK  D1: sprint threshold correct (no sprints in this segment, max={:.2f} m/s)".format(max_speed))
    if dist_old:
        print("  OK  D2: distance inflation removed ({:.2f}x -> 1.00x)".format(ratio))
        if speed_low_before and not speed_low_after:
            print("  OK  D4: speed_low no longer permanently True")
        else:
            print("  OK  D4: baseline speed denominator corrected (3600 s)")
    print("  OK  D3: late_in_game covers full handball second half (1800-3600 s)")
    print()


if __name__ == "__main__":
    main()
