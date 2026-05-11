"""
Players Data — SHAP compatibility shim.

In environments where the `shap` library is not installed,
this module provides a lightweight fallback that preserves
the full explanation interface using feature-magnitude proxies.

When `shap` is installed, this module is not used — the real
shap.KernelExplainer runs instead (see xai_layer.py).
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import shap as _real_shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning(
        "shap library not installed. Falling back to magnitude-proxy explanations. "
        "Install with: pip install shap"
    )


def compute_shap_values(
    predict_fn,
    feature_vector: np.ndarray,
    background_data: Optional[np.ndarray] = None,
    n_background: int = 50,
) -> Tuple[np.ndarray, float]:
    """
    Compute SHAP values for a single feature vector.

    Returns
    -------
    (shap_values, base_value)
    """
    n_fv = len(feature_vector)  # authoritative feature count

    if SHAP_AVAILABLE and background_data is not None:
        try:
            # Guard: background must have the same column count as the feature vector.
            # When the background was built from flattened sequences (T*F cols) but the
            # feature vector has only F cols, KernelExplainer raises an index OOB error.
            if background_data.shape[1] != n_fv:
                raise ValueError(
                    f"Background column count ({background_data.shape[1]}) != "
                    f"feature vector length ({n_fv}). Falling back to proxy."
                )
            bg = _real_shap.kmeans(background_data, min(n_background, len(background_data)))
            explainer = _real_shap.KernelExplainer(predict_fn, bg)
            vals = explainer.shap_values(feature_vector.reshape(1, -1), silent=True)
            if isinstance(vals, list):
                vals = vals[0]
            return vals[0], float(explainer.expected_value)
        except Exception as exc:
            logger.warning("SHAP computation failed, using fallback: %s", exc)

    # ── Fallback: magnitude proxy ──
    # Approximate SHAP contributions proportional to |feature| deviation from zero.
    # This preserves the explanation shape and sign while lacking mathematical SHAP guarantees.
    # The proxy predict_fn operates on the feature_vector directly (not background columns).
    fv = feature_vector.copy()
    try:
        base_value = float(predict_fn(np.zeros(n_fv, dtype=np.float32).reshape(1, -1))[0])
        total_effect = float(predict_fn(fv.reshape(1, -1))[0]) - base_value
    except Exception:
        # predict_fn may be incompatible with the feature dimensionality; use safe defaults
        base_value = 0.0
        total_effect = float(np.abs(fv).mean())

    # Distribute total effect proportionally by |feature value|
    magnitudes = np.abs(fv)
    total_mag = magnitudes.sum()
    if total_mag > 0:
        proxy_shap = (magnitudes / total_mag) * total_effect * np.sign(fv)
    else:
        proxy_shap = np.zeros_like(fv)

    return proxy_shap, base_value


def build_kmeans_background(data: np.ndarray, k: int = 50) -> np.ndarray:
    """
    Return k representative background samples for SHAP KernelExplainer.

    shap.kmeans wraps sklearn KMeans.  Two failure modes:
      A) n_clusters > n_unique_rows → sklearn finds fewer clusters than requested
         → shap.DenseData raises "# of weights must match data matrix!"
      B) n_clusters > len(data)     → trivially wrong request

    deduplicate first, then cap k to the number of unique rows.
    If shap.kmeans still raises (e.g. degenerate 1-point cluster), fall back
    to a random subsample — an equally valid (if less optimal) background.
    """
    # Deduplicate rows so sklearn KMeans never finds fewer clusters than k
    unique_data = np.unique(data, axis=0)
    k_safe = min(k, len(unique_data))

    if SHAP_AVAILABLE:
        try:
            summary = _real_shap.kmeans(unique_data, k_safe)
            return summary.data
        except Exception as exc:
            logger.warning(
                "shap.kmeans failed (k=%d, unique_rows=%d): %s — using random subsample",
                k_safe, len(unique_data), exc,
            )

    # Fallback: random subsample (no replacement; safe because k_safe <= len(unique_data))
    idx = np.random.choice(len(unique_data), k_safe, replace=False)
    return unique_data[idx]