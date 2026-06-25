"""
kinexon_trace.py

Real-player trace: Kinexon positions.csv → feature vector → mask → model input → AnomalyResult.

Demonstrates the before/after behaviour of the is_real fix.

Run:
    python scripts/kinexon_trace.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timezone

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
from config.settings import KinexonConfig, CONFIG
from ingestion.kinexon_adapter import KinexonAdapter
from analysis.anomaly_detection import SequenceWindowBuilder, N_SEQUENCE_FEATURES, SEQUENCE_FEATURE_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH     = DATA_DIR / "statistics.csv"

TRACE_PLAYER_ID = None   # will take first player found
WINDOW_SIZE     = CONFIG.window.window_steps   # default: 8

LINE = "-" * 72


def _col(label, value, width=30):
    return f"  {label:<{width}}{value}"


def main():
    print(LINE)
    print("Kinexon Real-Player Inference Trace")
    print(LINE)

    if not POSITIONS_PATH.exists():
        print(f"ERROR: {POSITIONS_PATH} not found.")
        sys.exit(1)

    adapter = KinexonAdapter()

    # Load player metadata
    meta = adapter.load_player_meta(STATS_PATH) if STATS_PATH.exists() else {}
    print(f"Players in metadata: {len(meta)}")

    # Stream positions to collect one player's events
    player_events: dict[int, list] = {}
    n_total = 0

    for obs in adapter.stream_positions(POSITIONS_PATH, meta, session_id="trace", match_id="trace"):
        n_total += 1
        pid = obs.player_id
        if pid not in player_events:
            player_events[pid] = []
        if len(player_events[pid]) < WINDOW_SIZE + 4:   # small buffer
            player_events[pid].append(obs)
        # Stop once we have enough for one player
        all_ready = all(len(evs) >= WINDOW_SIZE for evs in player_events.values())
        if all_ready and n_total > 50_000:
            break

    # Pick the first player that has enough data
    trace_pid = None
    for pid, evs in player_events.items():
        if len(evs) >= WINDOW_SIZE:
            trace_pid = pid
            trace_evs = evs[:WINDOW_SIZE]
            break

    if trace_pid is None:
        print("No player has enough events yet. Increase buffer or check data.")
        sys.exit(1)

    print(f"\nTrace player: mapped_id={trace_pid}")
    if trace_pid in meta:
        m = meta[trace_pid]
        print(f"  Name       : {m.player_name}")
        print(f"  Jersey     : {m.jersey_number}")
        print(f"  Group      : {m.group_name}")
    print(f"  Events used: {len(trace_evs)} (window_size={WINDOW_SIZE})")
    print(LINE)

    # ── Stage 1: RawPlayerObservation ────────────────────────────────────────
    print("\nSTAGE 1  RawPlayerObservation (first 3 events)")
    for i, obs in enumerate(trace_evs[:3]):
        raw = adapter.to_raw_observation(obs)
        print(f"  [{i}] ts={raw.ts.strftime('%H:%M:%S.%f')[:12]}"
              f"  speed={obs.speed_ms:.2f} m/s"
              f"  x={obs.x_m:.2f}m  y={obs.y_m:.2f}m"
              f"  hr={raw.heart_rate_bpm!r}"
              f"  sprint={obs.sprint_flag}"
              f"  dist_delta={obs.distance_delta_m:.3f}m")
    print(LINE)

    # ── Stage 2: Event dicts ─────────────────────────────────────────────────
    print("\nSTAGE 2  Event dicts (to_event_dict)")
    event_dicts = []
    for i, obs in enumerate(trace_evs):
        evt = adapter.to_event_dict(obs, elapsed_s=float(i * 15))
        event_dicts.append(evt)
    e0 = event_dicts[0]
    print(f"  Keys       : {sorted(e0.keys())}")
    print(f"  speed_ms   : {e0['speed_ms']}")
    print(f"  hr_bpm     : {e0['heart_rate_bpm']!r}  (None = sensor absent)")
    print(f"  is_sprint  : {e0['is_sprint']}")
    print(f"  is_real?   : speed_ms is not None = {e0['speed_ms'] is not None}")
    print(LINE)

    # ── Stage 3: build_live_window ────────────────────────────────────────────
    print("\nSTAGE 3  build_live_window  (after is_real fix)")
    builder = SequenceWindowBuilder()
    seq, mask = builder.build_live_window(event_dicts)

    print(f"  seq.shape  : {seq.shape}  (window_steps={WINDOW_SIZE}, n_features={N_SEQUENCE_FEATURES})")
    print(f"  mask.shape : {mask.shape}")
    print(f"  mask       : {mask.tolist()}")
    print(f"  mask True? : {mask.all()}  (all real ticks)")
    print()
    print("  Feature matrix (8 features x window_steps rows):")
    header = "  " + "  ".join(f"{n[:10]:>10}" for n in SEQUENCE_FEATURE_NAMES)
    print(header)
    for t in range(WINDOW_SIZE):
        row_vals = "  ".join(f"{seq[t, f]:>10.4f}" for f in range(N_SEQUENCE_FEATURES))
        print(f"  t={t}  {row_vals}")
    print(LINE)

    # ── Stage 4: Feature statistics ──────────────────────────────────────────
    print("\nSTAGE 4  Feature statistics over window")
    for fi, name in enumerate(SEQUENCE_FEATURE_NAMES):
        vals = seq[:, fi]
        print(_col(name, f"min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}"))
    print()
    real_count = mask.sum()
    all_zeros = (seq.sum() == 0.0)
    print(_col("Real ticks (mask True)", f"{real_count} / {WINDOW_SIZE}"))
    print(_col("All-zero sequence?", str(all_zeros)))
    print(LINE)

    # ── Stage 5: BEFORE fix demonstration (simulated) ────────────────────────
    print("\nSTAGE 5  BEFORE fix (simulated: is_real required HR)")
    before_real = [
        (e.get("speed_ms") is not None and e.get("heart_rate_bpm") is not None)
        for e in event_dicts
    ]
    before_mask = np.array(before_real, dtype=bool)
    print(f"  Before mask: {before_mask.tolist()}")
    print(f"  Real ticks : {before_mask.sum()} / {WINDOW_SIZE}   (ZERO real ticks)")
    print(f"  Consequence: _masked_mse numerator=0  denominator=1e-8  raw_loss=0.0")
    print(f"  Consequence: is_anomaly=False forever, drift & workload still run")
    print(LINE)

    # ── Stage 6: AFTER fix summary ───────────────────────────────────────────
    print("\nSTAGE 6  AFTER fix (is_real = speed_ms is not None)")
    after_real = [e.get("speed_ms") is not None for e in event_dicts]
    after_mask = np.array(after_real, dtype=bool)
    print(f"  After mask : {after_mask.tolist()}")
    print(f"  Real ticks : {after_mask.sum()} / {WINDOW_SIZE}")

    # Simulate masked MSE with real data (random reconstruction for illustration)
    rng = np.random.default_rng(42)
    recon = seq + rng.normal(0, 0.5, seq.shape).astype(np.float32)
    m = after_mask.astype(np.float32)[:, np.newaxis]   # (T, 1)
    sq = (seq - recon) ** 2
    raw_loss_sim = float((sq * m).sum() / (m.sum() * N_SEQUENCE_FEATURES + 1e-8))
    print(f"  Simulated reconstruction loss  : {raw_loss_sim:.6f}  (non-zero)")
    print(f"  (Real model output depends on trained weights)")
    print(LINE)

    # ── Stage 7: AnomalyResult fields preview ────────────────────────────────
    print("\nSTAGE 7  AnomalyResult fields (from last sequence row)")
    last = seq[-1]
    fv_preview = {SEQUENCE_FEATURE_NAMES[i]: float(last[i]) for i in range(N_SEQUENCE_FEATURES)}
    fv_preview["mask_completeness"] = float(after_mask.mean())
    print("  Feature vector from sequence[-1]:")
    for k, v in fv_preview.items():
        meaningful = "OK" if v != 0.0 else "was 0 before fix"
        print(_col(f"  {k}", f"{v:.4f}   [{meaningful}]"))
    print(LINE)

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\nFINAL VERDICT")
    print()
    print("  BEFORE fix:")
    print("    mask_completeness = 0.0")
    print("    feature_vector[speed_ms]  = 0.0  (padded)")
    print("    feature_vector[x_pitch]   = 0.0  (padded)")
    print("    raw_loss  = 0.0  (masked MSE over 0 real ticks)")
    print("    is_anomaly = False  (always)")
    print("    drift / workload = live outputs (unaffected)")
    print()
    print("  AFTER fix:")
    print(f"    mask_completeness = {float(after_mask.mean()):.2f}")
    print(f"    feature_vector[speed_ms]  = {fv_preview['speed_ms']:.4f}  (real)")
    print(f"    feature_vector[x_pitch]   = {fv_preview['x_pitch']:.4f}  (real)")
    print(f"    feature_vector[hr_bpm]    = {fv_preview['heart_rate_bpm']:.4f}  (0.0 = sensor absent)")
    print(f"    raw_loss = model output over {WINDOW_SIZE} real ticks  (meaningful)")
    print(f"    is_anomaly = threshold-dependent  (can fire now)")
    print(f"    drift / workload = live outputs  (unchanged)")
    print()
    print("  Can the existing engine generate meaningful scores from Kinexon?")
    print("  AFTER fix: YES (movement-based). HR feature contributes 0 to reconstruction.")
    print("  Without a trained model: raw_loss=0.0 (engine stub). Train first.")
    print(LINE)


if __name__ == "__main__":
    main()
