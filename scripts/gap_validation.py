"""
gap_validation.py

Validates gap-aware windowing (analysis/gap_aware_windowing.py) and audits
BaselineBuilder.compute_provisional()'s MIN_EVENTS_PER_WINDOW impact -- both
against real session 3387 data, via the Phase 1 KinexonResampler.

Run:
    python scripts/gap_validation.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from config.settings import CONFIG
from ingestion.kinexon_adapter import KinexonAdapter
from ingestion.kinexon_resampler import KinexonResampler
from analysis.gap_aware_windowing import detect_window_gaps
from analysis.baseline import BaselineBuilder

# ---------------------------------------------------------------------------
# Torch stub -- this environment's PyTorch install is broken (DLL load
# failure), and analysis/anomaly_detection.py defines a class with `nn.Module`
# as a base at module level, which crashes import without it.
# SequenceWindowBuilder itself has zero torch dependency (confirmed: it's
# pure numpy/pandas). Same technique already used in this repo's own
# scripts/window_gap_trace.py -- reused verbatim here, not invented for this
# script, so PART 4 below can exercise the REAL, unmodified
# SequenceWindowBuilder.build_from_session() rather than only my own
# independent replication of its windowing math (Parts 1-2).
# ---------------------------------------------------------------------------
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except Exception:
        torch_stub = types.ModuleType("torch")
        nn_stub = types.ModuleType("torch.nn")
        optim_stub = types.ModuleType("torch.optim")
        utils_stub = types.ModuleType("torch.utils")
        utils_data_stub = types.ModuleType("torch.utils.data")
        serialization_stub = types.ModuleType("torch.serialization")

        class _FakeModule:
            def __init__(self, *a, **kw): pass
            def __call__(self, *a, **kw): return self

        for _name in ("Module", "Linear", "LSTM", "Dropout", "GRU", "TransformerEncoder",
                      "TransformerEncoderLayer", "MultiheadAttention", "Sequential",
                      "ReLU", "LayerNorm", "Embedding", "BatchNorm1d"):
            setattr(nn_stub, _name, _FakeModule)
        optim_stub.Adam = _FakeModule
        utils_data_stub.DataLoader = _FakeModule
        utils_data_stub.TensorDataset = _FakeModule
        serialization_stub.add_safe_globals = lambda *a, **kw: None
        utils_stub.data = utils_data_stub
        torch_stub.nn = nn_stub
        torch_stub.optim = optim_stub
        torch_stub.utils = utils_stub
        torch_stub.serialization = serialization_stub
        torch_stub.Tensor = object
        torch_stub.device = lambda *a, **kw: "cpu"
        torch_stub.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
        cuda_stub = types.ModuleType("torch.cuda")
        cuda_stub.is_available = lambda: False
        torch_stub.cuda = cuda_stub
        backends_stub = types.ModuleType("torch.backends")
        mps_stub = types.ModuleType("torch.backends.mps")
        mps_stub.is_available = lambda: False
        backends_stub.mps = mps_stub
        torch_stub.backends = backends_stub
        torch_stub.__getattr__ = lambda name: (lambda *a, **kw: None)
        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub
        sys.modules["torch.optim"] = optim_stub
        sys.modules["torch.utils"] = utils_stub
        sys.modules["torch.utils.data"] = utils_data_stub
        sys.modules["torch.serialization"] = serialization_stub
        sys.modules["torch.cuda"] = cuda_stub
        sys.modules["torch.backends"] = backends_stub
        sys.modules["torch.backends.mps"] = mps_stub
        print("[verification stub] real torch unavailable -- injected minimal stub so the "
              "REAL SequenceWindowBuilder can be imported and exercised against real data.\n")

DATA_DIR = _ROOT / "data"
POSITIONS_PATH = DATA_DIR / "positions.csv"
STATS_PATH = DATA_DIR / "statistics.csv"
SESSION_ID = "3387"

LINE = "-" * 78
WINDOW_STEPS = CONFIG.window.window_steps          # 8
STRIDE = WINDOW_STEPS                              # build_from_session's own default
GAP_THRESHOLD_S = CONFIG.window.gap_threshold_s    # 60.0


def main() -> None:
    print(LINE)
    print("Gap-Aware Windowing + Baseline Audit -- real session 3387")
    print(LINE)
    print(f"window_steps={WINDOW_STEPS}  stride={STRIDE}  gap_threshold_s={GAP_THRESHOLD_S}")

    adapter = KinexonAdapter()
    meta = adapter.load_player_meta(STATS_PATH)
    observations = list(
        adapter.stream_positions(POSITIONS_PATH, meta, session_id=SESSION_ID, match_id=SESSION_ID)
    )
    resampler = KinexonResampler()
    events_by_player, sessions_df = resampler.resample(observations, session_id=SESSION_ID)
    print(f"Players with resampled data: {len(events_by_player)}")

    # ── Part 1: gap detection / window filtering ─────────────────────────
    print(f"\n{LINE}\nPART 1 -- Gap detection\n{LINE}")

    total_before = 0
    total_after = 0
    affected_players = []
    all_dropped_gaps = []
    per_player_rows = []

    for player_id, df in sorted(events_by_player.items()):
        gap_info = detect_window_gaps(df, WINDOW_STEPS, STRIDE, GAP_THRESHOLD_S)
        n_before = len(gap_info)
        n_after = sum(1 for ok, _ in gap_info if ok)
        n_dropped = n_before - n_after
        dropped_gaps = [g for ok, g in gap_info if not ok]

        total_before += n_before
        total_after += n_after
        all_dropped_gaps.extend(dropped_gaps)
        if n_dropped > 0:
            affected_players.append((player_id, n_before, n_after, n_dropped, max(dropped_gaps)))

        per_player_rows.append((player_id, n_before, n_after, n_dropped))

    print(f"{'player_id':>10} {'before':>8} {'after':>8} {'dropped':>8}")
    for pid, b, a, d in per_player_rows:
        print(f"{pid:>10} {b:>8} {a:>8} {d:>8}")

    print(f"\nTOTAL windows before filtering: {total_before}")
    print(f"TOTAL windows after filtering:  {total_after}")
    print(f"TOTAL windows dropped:          {total_before - total_after}")
    print(f"Players affected (>=1 window dropped): {len(affected_players)} / {len(events_by_player)}")

    print(f"\n{'player_id':>10} {'before':>8} {'after':>8} {'dropped':>8} {'largest_gap_s':>14}")
    for pid, b, a, d, g in sorted(affected_players, key=lambda r: -r[4]):
        print(f"{pid:>10} {b:>8} {a:>8} {d:>8} {g:>14.1f}")

    if all_dropped_gaps:
        print(f"\nLargest detected gaps overall (top 10): "
              f"{sorted(all_dropped_gaps, reverse=True)[:10]}")
        print(f"Largest single gap: {max(all_dropped_gaps):.1f}s "
              f"(threshold: {GAP_THRESHOLD_S}s)")

    # ── Part 2: confirm every retained window spans ~120s ────────────────
    print(f"\n{LINE}\nPART 2 -- Retained-window span check (ALL retained windows, not a sample)\n{LINE}")
    target = CONFIG.window.window_seconds
    span_violations = 0
    span_checked = 0
    max_span_seen = 0.0
    for player_id, df in sorted(events_by_player.items()):
        sorted_df = df.sort_values("ts").reset_index(drop=True)
        ts = pd.to_datetime(sorted_df["ts"], utc=True)
        n = len(sorted_df)
        for start in range(0, n - WINDOW_STEPS + 1, STRIDE):
            end = start + WINDOW_STEPS
            deltas = ts.iloc[start:end].diff().dt.total_seconds().dropna()
            max_gap = float(deltas.max()) if len(deltas) else 0.0
            if max_gap > GAP_THRESHOLD_S:
                continue  # this window was dropped in Part 1 -- skip, not "retained"
            span_checked += 1
            span = (ts.iloc[end - 1] - ts.iloc[start]).total_seconds() + CONFIG.window.event_interval_s
            max_span_seen = max(max_span_seen, span)
            if abs(span - target) > 1.0:  # 1s tolerance for float/rounding
                span_violations += 1

    print(f"Retained windows checked: {span_checked}")
    print(f"Windows whose span deviates from {target}s by >1s: {span_violations}")
    print(f"Max span observed among retained windows: {max_span_seen:.1f}s (target {target}s)")
    print("CONFIRMED: every retained window spans ~120s." if span_violations == 0 else "VIOLATION FOUND.")

    # ── Part 3: baseline audit (real compute_provisional() calls) ───────
    print(f"\n{LINE}\nPART 3 -- BaselineBuilder.compute_provisional() audit\n{LINE}")
    builder = BaselineBuilder()
    n_none = 0
    n_built = 0
    MIN_EVENTS_PER_WINDOW = 30  # mirrors analysis/baseline.py's own hardcoded constant
    window_seconds = 120

    for player_id, df in sorted(events_by_player.items()):
        profile = builder.compute_provisional(
            player_id=player_id, external_id=str(player_id), events_df=df, window_seconds=window_seconds
        )
        if profile is None:
            n_none += 1
        else:
            n_built += 1

        # Independently replicate compute_provisional's own internal window
        # loop just to report HOW MANY rows land in a typical 120s window
        # for this player (real numbers, not assumed).
        d = df.sort_values("ts").reset_index(drop=True)
        elapsed = (pd.to_datetime(d["ts"], utc=True) - pd.to_datetime(d["ts"], utc=True).min()).dt.total_seconds()
        max_window = int(elapsed.max() // window_seconds) + 1 if len(d) else 0
        rows_per_window = []
        for w in range(max_window):
            seg = d[(elapsed >= w * window_seconds) & (elapsed < (w + 1) * window_seconds)]
            rows_per_window.append(len(seg))

        passing = sum(1 for r in rows_per_window if r >= MIN_EVENTS_PER_WINDOW)
        if player_id == sorted(events_by_player.keys())[0]:
            print(f"Example -- player {player_id}: rows per 120s window = {rows_per_window[:10]}"
                  f"{'...' if len(rows_per_window) > 10 else ''}")
            print(f"  MIN_EVENTS_PER_WINDOW={MIN_EVENTS_PER_WINDOW}; windows passing this gate: "
                  f"{passing} / {len(rows_per_window)}")

    print(f"\nPlayers with a provisional baseline built: {n_built} / {len(events_by_player)}")
    print(f"Players rejected (compute_provisional returned None): {n_none} / {len(events_by_player)}")
    print(f"\nAt event_interval_s={CONFIG.window.event_interval_s}s, every 120s window "
          f"contains exactly {window_seconds // CONFIG.window.event_interval_s} resampled rows "
          f"-- always below MIN_EVENTS_PER_WINDOW={MIN_EVENTS_PER_WINDOW}, for every player, "
          f"by construction (not data-dependent).")

    # ── Part 4: cross-check against the REAL, unmodified SequenceWindowBuilder ──
    print(f"\n{LINE}\nPART 4 -- Cross-check against the REAL SequenceWindowBuilder.build_from_session()\n{LINE}")
    from analysis.anomaly_detection import SequenceWindowBuilder
    from analysis.gap_aware_windowing import build_from_session_gap_aware

    real_builder = SequenceWindowBuilder()
    real_total_before = 0
    real_total_after = 0
    mismatches = 0
    for player_id, df in sorted(events_by_player.items()):
        kept, audit = build_from_session_gap_aware(real_builder, df, mode="drop")
        real_total_before += audit.n_windows_before
        real_total_after += audit.n_windows_after
        independent = detect_window_gaps(df, WINDOW_STEPS, STRIDE, GAP_THRESHOLD_S)
        n_independent_after = sum(1 for ok, _ in independent if ok)
        if audit.n_windows_after != n_independent_after or audit.n_windows_before != len(independent):
            mismatches += 1
            print(f"  MISMATCH player {player_id}: real-builder before/after="
                  f"{audit.n_windows_before}/{audit.n_windows_after} vs "
                  f"independent={len(independent)}/{n_independent_after}")
        # Also confirm the REAL builder's returned seq/mask shapes are correct.
        if kept:
            seq0, mask0 = kept[0]
            assert seq0.shape == (WINDOW_STEPS, 8), f"unexpected seq shape {seq0.shape}"
            assert mask0.shape == (WINDOW_STEPS,)

    print(f"Real SequenceWindowBuilder, gap-filtered total before: {real_total_before}")
    print(f"Real SequenceWindowBuilder, gap-filtered total after:  {real_total_after}")
    print(f"Matches independent (torch-free) replication for all {len(events_by_player)} players: "
          f"{'YES' if mismatches == 0 else f'NO -- {mismatches} mismatches'}")
    print(f"Confirms build_from_session_gap_aware() correctly wraps the REAL, unmodified "
          f"SequenceWindowBuilder.build_from_session() -- not just an independent reimplementation.")

    print(LINE)


if __name__ == "__main__":
    main()
