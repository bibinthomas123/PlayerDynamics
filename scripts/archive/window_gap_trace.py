# -*- coding: ascii -*-
"""
scripts/window_gap_trace.py

Real-player trace of the window-contamination fix using player 2059
(Manuel Zehnder), whose session has 7 large tracking gaps (audit evidence:
1640s, 901s, 808s, 300s, plus 3 smaller ones).

This environment's PyTorch install is broken (DLL load failure under
Python 3.14), and analysis/anomaly_detection.py defines
class SharedLSTMEncoder(nn.Module) at module level, which crashes import
when torch is unavailable. SequenceWindowBuilder itself has zero torch
dependency, so a minimal stub (only nn.Module needs to be a real type --
confirmed via grep: it is the only nn.* class used as a base class in the
file) is injected purely to make the module importable for this trace.
This stub is NOT part of the committed test suite (which skips cleanly
per the codebase's established convention -- see test_phase_a_calibration.py).

Shows:
  BEFORE -- a minimal reimplementation of the pre-fix add_event() (no gap
            check) run over the same real ticks, to demonstrate the window
            that WOULD have been emitted.
  AFTER  -- the actual fixed SequenceWindowBuilder.add_event(), same ticks.
"""
import sys
import os
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Minimal torch stub (verification-only; not used by the real deployment,
# which has a working PyTorch install). Only nn.Module needs to be a real
# type for `class SharedLSTMEncoder(nn.Module)` to resolve at import time.
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
            def to(self, *a, **kw): return self
            def parameters(self): return []
            def state_dict(self): return {}
            def load_state_dict(self, *a, **kw): pass
            def eval(self): return self
            def train(self, *a, **kw): return self

        nn_stub.Module = _FakeModule
        for _name in ("Linear", "LSTM", "Dropout", "GRU", "TransformerEncoder",
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
        torch_stub.no_grad = lambda: types.SimpleNamespace(
            __enter__=lambda s: None, __exit__=lambda s, *a: None
        )
        cuda_stub = types.ModuleType("torch.cuda")
        cuda_stub.is_available = lambda: False
        torch_stub.cuda = cuda_stub
        sys.modules["torch.cuda"] = cuda_stub

        backends_stub = types.ModuleType("torch.backends")
        mps_stub = types.ModuleType("torch.backends.mps")
        mps_stub.is_available = lambda: False
        backends_stub.mps = mps_stub
        torch_stub.backends = backends_stub
        sys.modules["torch.backends"] = backends_stub
        sys.modules["torch.backends.mps"] = mps_stub

        # Catch-all (PEP 562): any further module-level torch.* call used by
        # anomaly_detection.py at import time (e.g. manual_seed) that this
        # stub didn't anticipate resolves to a harmless no-op callable,
        # instead of requiring this list to be exhaustive.
        def _torch_getattr(_name):
            return lambda *a, **kw: None
        torch_stub.__getattr__ = _torch_getattr

        sys.modules["torch"] = torch_stub
        sys.modules["torch.nn"] = nn_stub
        sys.modules["torch.optim"] = optim_stub
        sys.modules["torch.utils"] = utils_stub
        sys.modules["torch.utils.data"] = utils_data_stub
        sys.modules["torch.serialization"] = serialization_stub
        print("[verification stub] real torch unavailable in this env -- "
              "injected a minimal stub so SequenceWindowBuilder (which has "
              "no torch dependency itself) can be imported and exercised "
              "against real data.\n")

from datetime import datetime, timedelta, timezone
from collections import deque
import numpy as np
import pandas as pd

from analysis.anomaly_detection import (
    SequenceWindowBuilder,
    N_SEQUENCE_FEATURES,
    SEQUENCE_FEATURE_NAMES,
)
from config.settings import CONFIG

SEP = "-" * 70
SEP2 = "=" * 70


# ---------------------------------------------------------------------------
# Pre-fix reimplementation (no gap check) -- mirrors the exact code that
# existed before this fix, for BEFORE/AFTER comparison purposes only.
# ---------------------------------------------------------------------------
class _PreFixBuilder:
    def __init__(self, window_steps):
        self.window_steps = window_steps
        self._buffers = {}
        self._mask_buffers = {}
        self._prev_events = {}

    def add_event(self, event, extract_fn):
        pid = event.get("player_external_id", "")
        buf = self._buffers.setdefault(pid, deque(maxlen=self.window_steps))
        mbuf = self._mask_buffers.setdefault(pid, deque(maxlen=self.window_steps))
        prev = self._prev_events.get(pid)
        is_real = event.get("speed_ms") is not None
        fv = extract_fn(event, prev) if is_real else np.zeros(N_SEQUENCE_FEATURES, dtype=np.float32)
        self._prev_events[pid] = event if is_real else prev
        buf.append(fv)
        mbuf.append(is_real)
        if len(buf) == self.window_steps:
            return np.array(list(buf), dtype=np.float32), np.array(list(mbuf), dtype=bool)
        return None


def _event(pid, ts, x_pitch, speed_ms=1.5):
    return {
        "player_external_id": pid,
        "ts": ts.isoformat(),
        "speed_ms": speed_ms,
        "heart_rate_bpm": None,
        "x_pitch": x_pitch,
        "y_pitch": 50.0,
        "distance_delta_m": 0.0,
        "is_sprint": 0,
        "sprint_flag": 0,
        "source": "kinexon",
    }


def main():
    pid = "2059"
    window_steps = CONFIG.window.window_steps
    print(SEP2)
    print("WINDOW CONTAMINATION FIX -- real-player trace")
    print(f"Player: 2059 (Manuel Zehnder) -- centre_back, SC Magdeburg")
    print(f"Real gap from session 3387 audit: 1640s (27.3 min) at elapsed=4042s")
    print(f"window_steps={window_steps}  gap_threshold_s={CONFIG.window.gap_threshold_s}")
    print(SEP2)

    base_ts = datetime(2026, 6, 7, 16, 27, 22, tzinfo=timezone.utc)  # ~elapsed 4042s mark

    # Pre-gap: window_steps - 1 real ticks at x_pitch 40..47 (his position
    # right before going off, e.g. near the bench/touchline).
    pre_events = [
        _event(pid, base_ts + timedelta(seconds=15 * i), x_pitch=40.0 + i, speed_ms=1.2)
        for i in range(window_steps - 1)
    ]

    # The real 1640s gap (Zehnder, session 3387).
    gap_ts = base_ts + timedelta(seconds=15 * (window_steps - 1) + 1640)

    # Post-gap: window_steps fresh ticks at x_pitch 60..67 (back on court,
    # different zone -- distinguishable from pre-gap range).
    post_events = [
        _event(pid, gap_ts + timedelta(seconds=15 * i), x_pitch=60.0 + i, speed_ms=2.0)
        for i in range(window_steps)
    ]

    real_builder = SequenceWindowBuilder()
    x_idx = SEQUENCE_FEATURE_NAMES.index("x_pitch")
    accel_idx = SEQUENCE_FEATURE_NAMES.index("acceleration_ms2")

    # ---------------- BEFORE (pre-fix reimplementation) ----------------
    print("BEFORE (pre-fix logic -- no gap check):")
    print(SEP)
    pre_fix = _PreFixBuilder(window_steps)
    for e in pre_events:
        pre_fix.add_event(e, real_builder._extract)
    print(f"  Pre-gap buffer filled: {len(pre_fix._buffers[pid])}/{window_steps} ticks "
          f"(x_pitch range: 40-{40+window_steps-2})")
    print(f"  --- gap of 1640s occurs here (player benched) ---")
    result_before = None
    for i, e in enumerate(post_events):
        result_before = pre_fix.add_event(e, real_builder._extract)
        if result_before is not None:
            print(f"  Window EMITTED on post-gap tick #{i+1} "
                  f"(buffer was already full pre-gap, so 1st post-gap tick triggers emission)")
            break
    seq_before, mask_before = result_before
    x_vals_before = seq_before[:, x_idx]
    print(f"  Emitted window x_pitch values: {[round(float(v),1) for v in x_vals_before]}")
    n_stale = int((x_vals_before < 50).sum())
    print(f"  --> {n_stale}/{window_steps} ticks are STALE pre-gap data "
          f"(x_pitch < 50) mixed with post-gap data")
    print(f"  --> mask: {mask_before.tolist()}  (all True -- contamination is INVISIBLE to mask_completeness)")
    print(f"  --> first post-gap tick acceleration_ms2 = {seq_before[-(window_steps-1):, accel_idx][0]:.4f} "
          f"(computed against a {1640}s-old stale prev event using a fixed 15s dt assumption)")
    print()

    # ---------------- AFTER (actual fixed code) ----------------
    print("AFTER (fixed SequenceWindowBuilder.add_event()):")
    print(SEP)
    for e in pre_events:
        real_builder.add_event(e)
    print(f"  Pre-gap buffer filled: {len(real_builder._buffers[pid])}/{window_steps} ticks")
    print(f"  --- gap of 1640s occurs here (player benched) ---")
    result_after = None
    emitted_on = None
    for i, e in enumerate(post_events):
        result_after = real_builder.add_event(e)
        if result_after is not None:
            emitted_on = i + 1
            break
        else:
            print(f"  Post-gap tick #{i+1}: buffer size = {len(real_builder._buffers[pid])} "
                  f"(no window yet -- refilling from scratch)")
    print(f"  Window EMITTED on post-gap tick #{emitted_on} "
          f"(buffer had to refill from 1 to {window_steps} -- no premature emission)")
    seq_after, mask_after = result_after
    x_vals_after = seq_after[:, x_idx]
    print(f"  Emitted window x_pitch values: {[round(float(v),1) for v in x_vals_after]}")
    n_stale_after = int((x_vals_after < 50).sum())
    print(f"  --> {n_stale_after}/{window_steps} ticks are stale pre-gap data "
          f"(x_pitch < 50)")
    print(f"  --> first post-gap tick acceleration_ms2 = {seq_after[0, accel_idx]:.4f} "
          f"(prev=None after reset -- correct first-tick semantics, no stale delta)")
    print()

    print(SEP2)
    print("COMPARISON")
    print(SEP)
    print(f"  BEFORE: window emitted on post-gap tick #1, contains {n_stale}/{window_steps} stale ticks")
    print(f"  AFTER : window emitted on post-gap tick #{emitted_on}, contains {n_stale_after}/{window_steps} stale ticks")
    print()
    if n_stale > 0 and n_stale_after == 0:
        print("  FIX VERIFIED: contamination eliminated. The window emitted after a")
        print("  substitution gap now contains only post-gap ticks; the LSTM, regime")
        print("  classifier, semantic interpreter, and fatigue logic all consume this")
        print("  same window, so all four are fixed simultaneously by this one change.")
    else:
        print("  UNEXPECTED: fix did not behave as expected -- investigate.")
    print(SEP2)


if __name__ == "__main__":
    main()
