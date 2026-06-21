"""
PlayerAnalyticsEvent — PlayerDynamics

The coach-facing wire projection of AnomalyResult (analysis/anomaly_detection.py),
published onto analytics.players exactly as Possession/TeamState/CoachInsight/
CoachSituation are published onto their own analytics.* streams by
MatchOrchestrator (analysis/match_orchestrator.py).

Why a separate, slimmer dataclass instead of publishing AnomalyResult directly
--------------------------------------------------------------------------------
AnomalyResult is an internal ML object, not a wire contract: it carries raw
numpy arrays (raw_sequence, raw_mask), nested non-trivial objects
(signals: List[Signal], policy_resolution: Optional[PolicyResolution]), and
fields explicitly marked repr=False / underscore-prefixed as internal-only
(_xai_kwargs, base_explanation, semantic_state). dataclasses.asdict() --
which ingestion/stream_codec.py's encode() relies on for every other
analytics.* dataclass -- cannot JSON-serialise any of that. PlayerAnalyticsEvent
is a deliberate, flat, JSON-safe projection of only the fields a coach-facing
consumer needs, exactly mirroring how CoachInsight/CoachSituation are slim
projections over TeamStateTrend/Possession rather than raw internal state.

Determinism
------------
to_player_analytics_event() is a pure function of an existing AnomalyResult --
no new computation, no new model, no new thresholds. It only selects and
flattens fields already present on the object PatternAnalysisEngine produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.anomaly_detection import AnomalyResult


@dataclass
class PlayerAnalyticsEvent:
    """One coach-facing player-analytics observation, ready for analytics.players."""
    player_id: int
    external_id: str
    match_id: Optional[str]
    timestamp: datetime

    anomaly_score: float
    is_anomaly: bool
    confidence: float
    alert_level: str  # AlertLevel enum member name, e.g. "WARNING" / "CRITICAL" / "NONE"

    model_type: str
    model_version: str
    baseline_mode: str  # "historical" | "provisional" -- see AnomalyResult.baseline_mode

    fatigue_flag: bool
    positional_drift_flag: bool
    workload_flag: bool
    workload_status: str

    signal_types: List[str] = field(default_factory=list)
    recommendation_type: Optional[str] = None
    nlg_summary: str = ""
    persistence_windows: int = 0


def to_player_analytics_event(
    result: "AnomalyResult", match_id: Optional[str] = None
) -> PlayerAnalyticsEvent:
    """Pure projection of an AnomalyResult onto the wire-safe PlayerAnalyticsEvent shape."""
    return PlayerAnalyticsEvent(
        player_id=result.player_id,
        external_id=result.external_id,
        match_id=match_id,
        timestamp=result.ts,
        anomaly_score=result.anomaly_score,
        is_anomaly=result.is_anomaly,
        confidence=result.confidence,
        alert_level=result.alert_level.name,
        model_type=result.model_type,
        model_version=result.model_version,
        baseline_mode=result.baseline_mode,
        fatigue_flag=result.fatigue_flag,
        positional_drift_flag=result.positional_drift_flag,
        workload_flag=result.workload_flag,
        workload_status=result.workload_status,
        signal_types=list(result.signal_types),
        recommendation_type=result.recommendation_type,
        nlg_summary=result.nlg_summary,
        persistence_windows=result.persistence_windows,
    )


@dataclass
class PilotPlayerAnalyticsEvent:
    """
    Coach-facing PILOT analytics observation for analytics.players.

    Separate from PlayerAnalyticsEvent above (which is the production,
    is_anomaly-gated shape, currently never published in practice because
    is_anomaly is always False on single-session pilot data -- see the
    calibration/persistence audits this preceded). This event is published
    unconditionally for every scored pilot window, not gated on is_anomaly.

    Deliberately excludes is_anomaly and alert_level: both remain
    experimental (EMA smoothing suppresses every real signal on this
    dataset before either field can mean anything -- see
    analysis/anomaly_detection.py AlertManager / pilot-mode calibration
    work). Exposes only fields already produced by the existing trained
    model and inference engine: reconstruction loss, confidence, the
    effective threshold, a raw (pre-smoothing) threshold-breach flag, and
    real SHAP attributions. No new computation, model, or threshold is
    introduced by this dataclass or its projection function below.
    """
    player_id: int
    external_id: str
    player_name: str
    position: str
    match_id: Optional[str]
    timestamp: datetime

    reconstruction_loss: float
    confidence: float
    # None when no calibration source (per-player or pilot-pooled) is
    # usable for this window -- not float("inf"): JSON has no Infinity
    # literal, so json.dumps would emit the bare token `Infinity`, which is
    # NOT valid JSON and would fail JSON.parse() on the wire.
    threshold: Optional[float]
    raw_threshold_breach: bool

    # Real SHAP attributions (analysis.anomaly_detection's
    # SharedBackboneAutoencoder.reconstruction_loss_for_shap via
    # explainability.xai_layer.XAILayer._explain_sequence_shap), already
    # sorted by |value| descending and truncated to the top N by the
    # caller -- this dataclass does not re-sort or re-truncate.
    top_shap_features: List[Dict[str, Any]] = field(default_factory=list)

    model_version: str = "unknown"
    regime: str = ""
    # Which calibration source produced `threshold` for this window --
    # "per_player", "pilot_pooled", or "none". See
    # InferenceEngine._resolve_tracker(). Purely informational.
    tracker_source: str = "none"

    # ── Real per-window telemetry (added for the coach-facing workload/
    # baseline-deviation dashboard panels) ─────────────────────────────────
    # All four below are computed by SUMMING/AVERAGING the real per-tick
    # values already present across this window's own sequence array
    # (SEQUENCE_FEATURE_NAMES columns over window_steps timesteps) -- not a
    # new model output, not a new measurement. distance_delta_m and
    # sprint_flag are read at every timestep and aggregated by the caller
    # (scripts/publish_pilot_analytics.py); previously
    # analysis/orchestrator.py's _build_xai_feature_vector() only read the
    # LAST timestep and multiplied/scaled it as an approximation -- this
    # event uses the genuine sum/mean across the real window instead, which
    # is strictly more accurate and still introduces no new computation
    # beyond arithmetic over already-real values.
    window_distance_m: float = 0.0
    window_avg_speed_ms: float = 0.0
    window_sprint_ticks: int = 0  # count of the window's own ticks at/above sprint threshold (0..window_steps)

    # Baseline z-scores (PlayerBaselineProfile.zscore(), unmodified) for this
    # window's own aggregates against the player's own (historical or
    # provisional-fallback) baseline -- "how far this window is from this
    # player's own normal", purely descriptive, no new threshold or model.
    baseline_distance_z: float = 0.0
    baseline_speed_z: float = 0.0
    baseline_sprint_z: float = 0.0

    # Session-level real totals (KinexonResampler._session_summary_row()),
    # repeated on every event for this player+session -- a session-wide
    # context number, not a new per-window measurement.
    session_total_distance_m: float = 0.0
    session_high_speed_distance_m: float = 0.0


def to_pilot_player_analytics_event(
    result: "AnomalyResult",
    *,
    player_name: str,
    position: str,
    threshold: float,
    top_shap_features: List[Dict[str, Any]],
    model_version: str,
    regime: str,
    tracker_source: str,
    match_id: Optional[str] = None,
    window_distance_m: float = 0.0,
    window_avg_speed_ms: float = 0.0,
    window_sprint_ticks: int = 0,
    baseline_distance_z: float = 0.0,
    baseline_speed_z: float = 0.0,
    baseline_sprint_z: float = 0.0,
    session_total_distance_m: float = 0.0,
    session_high_speed_distance_m: float = 0.0,
) -> PilotPlayerAnalyticsEvent:
    """
    Pure projection of an AnomalyResult (plus the small set of pilot-only
    diagnostics not carried on AnomalyResult itself -- threshold, SHAP,
    player identity, real per-window telemetry) onto the wire-safe
    PilotPlayerAnalyticsEvent shape.

    threshold may be float("inf") on input (player never reached a usable
    calibration source) -- converted to None on the output dataclass
    (JSON has no Infinity literal) and raw_threshold_breach is forced
    False in that case rather than evaluating a comparison against infinity.

    The window_*/baseline_*/session_* parameters default to 0.0/0 rather
    than being required, so any existing caller built before these fields
    existed keeps working unchanged -- this projection function's signature
    change is purely additive.
    """
    is_calibrated = threshold != float("inf")
    raw_breach = bool(result.anomaly_score > threshold) if is_calibrated else False
    return PilotPlayerAnalyticsEvent(
        player_id=result.player_id,
        external_id=result.external_id,
        player_name=player_name,
        position=position,
        match_id=match_id,
        timestamp=result.ts,
        reconstruction_loss=result.anomaly_score,
        confidence=result.confidence,
        threshold=threshold if is_calibrated else None,
        raw_threshold_breach=raw_breach,
        top_shap_features=top_shap_features,
        model_version=model_version,
        regime=regime,
        tracker_source=tracker_source,
        window_distance_m=window_distance_m,
        window_avg_speed_ms=window_avg_speed_ms,
        window_sprint_ticks=window_sprint_ticks,
        baseline_distance_z=baseline_distance_z,
        baseline_speed_z=baseline_speed_z,
        baseline_sprint_z=baseline_sprint_z,
        session_total_distance_m=session_total_distance_m,
        session_high_speed_distance_m=session_high_speed_distance_m,
    )
