"""
OLD (8-feature, events.csv-disabled) vs NEW (32-feature, events.csv-enabled)
training comparison for session 3387.

Standalone, read-only analysis script -- does not modify any production
training/inference code. Trains the shared backbone twice in-process (once
per config, since per-player calibration state is not persisted to
shared_backbone.pt -- see scripts/evaluate_pilot_model.py's own docstring
for why retraining is required to reproduce it) and reports:

  1. Reconstruction-loss distribution for each config (same descriptive
     stats as `main.py evaluate --data-source kinexon`).
  2. True per-feature SHAP attribution (channel-ablation, same method
     explainability/xai_layer.py._explain_sequence_shap() uses internally)
     averaged over a sample of windows -- but reported for ALL
     SEQUENCE_FEATURE_NAMES (32 in the NEW config), not just the curated
     15-name coach-facing subset that production XAI exposes.

Run from the PlayerDynamics project root:
    env/Scripts/python.exe scripts/feature_importance_comparison.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CONFIG, SEQUENCE_FEATURE_NAMES as SFN
from main import _load_kinexon_frames, _build_pipeline, _register_and_load_kinexon_players
from analysis.gap_aware_windowing import build_training_sequences_gap_aware

DATA_DIR = Path("data")
SESSION_ID = "3387"
N_SAMPLE_WINDOWS_PER_PLAYER = 5
RNG = np.random.default_rng(42)


def _dist(values: np.ndarray) -> dict:
    s = pd.Series(values)
    return {
        "min": float(s.min()), "p25": float(s.quantile(.25)), "median": float(s.median()),
        "p75": float(s.quantile(.75)), "p90": float(s.quantile(.90)), "max": float(s.max()),
        "mean": float(s.mean()), "n": int(len(s)),
    }


def _channel_shap(model, sequence: np.ndarray, mask: np.ndarray, background: np.ndarray, player_id: int) -> np.ndarray:
    """Full per-feature SHAP attribution over all F=len(SFN) channels.

    Same channel-ablation method as explainability/xai_layer.py's
    XAILayer._explain_sequence_shap(), reimplemented here (read-only
    analysis script) to expose ALL feature channels by name, not just
    the curated 15-name coach-facing subset that production XAI returns.
    """
    T, F = sequence.shape
    seq_norm = model.normaliser.transform(sequence[np.newaxis])[0]
    bg_norm = model.normaliser.transform(background)
    bg_mean = bg_norm.mean(axis=0)

    base_loss = float(model.reconstruction_loss_for_shap(
        player_id=player_id, sequences_norm=seq_norm[np.newaxis].astype(np.float32), mask=mask,
    )[0])

    perturbed = np.stack(
        [np.where(np.eye(F, dtype=bool)[fi][np.newaxis, :], bg_mean, seq_norm) for fi in range(F)],
        axis=0,
    ).astype(np.float32)
    ablated = model.reconstruction_loss_for_shap(player_id=player_id, sequences_norm=perturbed, mask=mask)
    return (ablated - base_loss).astype(np.float32)


def run_config(use_event_features: bool) -> dict:
    label = "NEW_32feat" if use_event_features else "OLD_8feat_equiv"
    print(f"\n=== {label} (use_event_features={use_event_features}) ===", flush=True)

    CONFIG.scoring.pilot_mode = True

    events_by_player, sessions_df, meta = _load_kinexon_frames(
        DATA_DIR, SESSION_ID, use_event_features=use_event_features
    )
    pipeline = _build_pipeline(Path(f"models_compare/{'new' if use_event_features else 'old'}_scratch"))
    _register_and_load_kinexon_players(pipeline, events_by_player, sessions_df, meta)

    baselines = pipeline.compute_baselines(window_days=28, use_provisional_fallback=True)
    print(f"  baselines: {len(baselines)} players")

    result = pipeline.train_all_models(use_gap_aware_windowing=True)
    shared_info = result.get("shared_model", {})
    print(f"  trained: n_players={shared_info.get('n_players')} n_windows={shared_info.get('n_windows')}")

    engine = pipeline.pattern_engine
    shared_model = engine._shared_model

    losses = []
    shap_rows = []
    for pid in baselines:
        df = events_by_player[pid].sort_values("ts").reset_index(drop=True)
        player_sessions = sessions_df[sessions_df["player_id"] == pid]
        windows, _audit = build_training_sequences_gap_aware(
            pattern_engine=engine, events_df=df, sessions_df=player_sessions,
        )
        if not windows:
            continue

        p = pipeline.registry.get(pid)
        background = p.get("sequence_background")

        sample_idx = RNG.choice(
            len(windows), size=min(N_SAMPLE_WINDOWS_PER_PLAYER, len(windows)), replace=False
        )
        for wi in sample_idx:
            seq, mask, _sid = windows[wi]
            seq_norm = shared_model.normaliser.transform(seq[np.newaxis])[0]
            loss = float(shared_model.reconstruction_loss_for_shap(
                player_id=pid, sequences_norm=seq_norm[np.newaxis].astype(np.float32), mask=mask,
            )[0])
            losses.append(loss)

            if background is not None and len(background) >= 2:
                shap_f = _channel_shap(shared_model, seq, mask, background, pid)
                shap_rows.append(shap_f)

    losses_arr = np.array(losses, dtype=np.float64)
    shap_arr = np.array(shap_rows, dtype=np.float64) if shap_rows else np.zeros((0, len(SFN)))

    mean_abs_shap = (
        np.abs(shap_arr).mean(axis=0) if len(shap_arr) else np.zeros(len(SFN))
    )
    importance = sorted(
        zip(SFN, mean_abs_shap.tolist()), key=lambda kv: kv[1], reverse=True
    )

    return {
        "label": label,
        "use_event_features": use_event_features,
        "n_players": len(baselines),
        "n_windows_total": shared_info.get("n_windows"),
        "n_windows_sampled_for_shap": len(shap_rows),
        "reconstruction_loss_distribution": _dist(losses_arr),
        "mean_abs_shap_by_feature": importance,
    }


def main() -> None:
    old_result = run_config(use_event_features=False)
    new_result = run_config(use_event_features=True)

    out = {"old": old_result, "new": new_result}
    out_path = Path("models_compare/feature_importance_comparison.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("\n\n========== SUMMARY ==========")
    print(f"OLD reconstruction loss: {old_result['reconstruction_loss_distribution']}")
    print(f"NEW reconstruction loss: {new_result['reconstruction_loss_distribution']}")
    print("\nTop 10 features by mean |SHAP| -- NEW (32-feature) config:")
    for name, val in new_result["mean_abs_shap_by_feature"][:10]:
        print(f"  {name:28s} {val:.6f}")
    print("\nTop 10 features by mean |SHAP| -- OLD (8-feature-equivalent) config:")
    for name, val in old_result["mean_abs_shap_by_feature"][:10]:
        print(f"  {name:28s} {val:.6f}")
    print(f"\nFull results written -> {out_path}")


if __name__ == "__main__":
    main()
