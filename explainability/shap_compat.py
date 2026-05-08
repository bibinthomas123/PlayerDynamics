"""
Players Data — SHAP compatibility shim.

In environments where the `shap` library is not installed,
this module provides a lightweight fallback that preserves
the full explanation interface using feature-magnitude proxies.

When `shap` IS installed, this module is not used — the real
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
    if SHAP_AVAILABLE and background_data is not None:
        try:
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
    base_value = float(predict_fn(np.zeros_like(feature_vector).reshape(1, -1))[0])
    fv = feature_vector.copy()
    total_effect = float(predict_fn(fv.reshape(1, -1))[0]) - base_value

    # Distribute total effect proportionally by |feature value|
    magnitudes = np.abs(fv)
    total_mag = magnitudes.sum()
    if total_mag > 0:
        proxy_shap = (magnitudes / total_mag) * total_effect * np.sign(fv)
    else:
        proxy_shap = np.zeros_like(fv)

    return proxy_shap, base_value


def build_kmeans_background(data: np.ndarray, k: int = 50) -> np.ndarray:
    """Return k representative background samples via k-means centroids."""
    if SHAP_AVAILABLE:
        summary = _real_shap.kmeans(data, min(k, len(data)))
        return summary.data
    # Fallback: random subsample
    idx = np.random.choice(len(data), min(k, len(data)), replace=False)
    return data[idx]
