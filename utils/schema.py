"""
Players Data — IBM CIC Germany
Database Schema  (SQLAlchemy ORM + raw DDL)

Tables exactly as specified in the proposal:
  players, sessions, events, annotations, override_logs,
  recalibration_history, fairness_audit_log
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey,
    Index, Integer, JSON, String, Text, UniqueConstraint,
    func, text
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────
class AnnotationType(str, enum.Enum):
    FATIGUE_FLAG = "fatigue_flag"
    TACTICAL_NOTE = "tactical_note"
    PRE_MATCH_STATUS = "pre_match_status"
    INJURY_CONCERN = "injury_concern"
    GENERAL = "general"


class CoachDecision(str, enum.Enum):
    ACCEPT = "accept"
    OVERRIDE = "override"
    DEFER = "defer"


class RecommendationType(str, enum.Enum):
    SUBSTITUTION = "substitution"
    FATIGUE_ALERT = "fatigue_alert"
    POSITIONAL_DRIFT = "positional_drift"
    WORKLOAD_WARNING = "workload_warning"
    ANOMALY_FLAG = "anomaly_flag"


# ─────────────────────────────────────────────
# Core Tables
# ─────────────────────────────────────────────
class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(64), unique=True, nullable=False, comment="Sportradar / wearable device ID")
    full_name = Column(String(128), nullable=False)
    position = Column(String(32), nullable=False)          # GK, CB, LB, RB, CM, CAM, LW, RW, ST
    age = Column(Integer, nullable=False)
    nationality = Column(String(64))
    age_group = Column(String(16))                          # U18, U21, Senior
    squad_number = Column(Integer)
    is_active = Column(Boolean, default=True)

    # Relationships
    sessions = relationship("Session", back_populates="player", lazy="dynamic")
    annotations = relationship("CoachAnnotation", back_populates="player", lazy="dynamic")
    baseline_profiles = relationship("PlayerBaseline", back_populates="player", lazy="dynamic")
    override_logs = relationship("OverrideLog", back_populates="player", lazy="dynamic")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index("ix_players_position", "position"),
        Index("ix_players_age_group", "age_group"),
    )


class Session(Base):
    """One training session or match for one player."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    match_id = Column(String(64), comment="External match/event ID from data provider")
    session_type = Column(String(16), nullable=False)      # "match" | "training"
    started_at = Column(DateTime(timezone=True), nullable=False)
    ended_at = Column(DateTime(timezone=True))
    duration_minutes = Column(Float)

    # Aggregate metrics (computed after session)
    total_distance_m = Column(Float)
    high_speed_distance_m = Column(Float)
    sprint_count = Column(Integer)
    avg_speed_ms = Column(Float)
    max_speed_ms = Column(Float)
    avg_heart_rate_bpm = Column(Float)
    max_heart_rate_bpm = Column(Float)

    # Quality
    data_quality_score = Column(Float, default=1.0)        # 0–1, computed during ingestion
    gps_coverage_pct = Column(Float)

    player = relationship("Player", back_populates="sessions")
    events = relationship("PlayerEvent", back_populates="session", lazy="dynamic")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_sessions_player_started", "player_id", "started_at"),
        Index("ix_sessions_match", "match_id"),
    )


class PlayerEvent(Base):
    """High-frequency time-series event (GPS tick, HR reading, etc.)."""
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)

    # GPS
    latitude = Column(Float)
    longitude = Column(Float)
    speed_ms = Column(Float)
    acceleration_ms2 = Column(Float)

    # Derived positional
    x_pitch = Column(Float, comment="Normalized pitch X [0,100]")
    y_pitch = Column(Float, comment="Normalized pitch Y [0,100]")
    zone_id = Column(String(16), comment="Tactical zone label")

    # Wearable / biometric
    heart_rate_bpm = Column(Integer)
    hr_recovery_time_s = Column(Float)

    # Event classification
    event_type = Column(String(32))    # sprint, walk, jog, high_intensity_run, rest
    is_sprint = Column(Boolean, default=False)
    is_high_intensity = Column(Boolean, default=False)

    # Sliding window aggregates (pre-computed at ingestion for fast inference)
    window_sprint_count = Column(Integer)
    window_distance_m = Column(Float)
    window_avg_speed_ms = Column(Float)

    session = relationship("Session", back_populates="events")

    __table_args__ = (
        Index("ix_events_session_ts", "session_id", "ts"),
        Index("ix_events_player_ts", "player_id", "ts"),
    )


class CoachAnnotation(Base):
    """Manual coach inputs — first-class features alongside sensor data."""
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    annotation_type = Column(Enum(AnnotationType), nullable=False)
    value = Column(String(64))          # e.g., "mild" for fatigue, tactical note text
    note = Column(Text)
    severity = Column(Float)            # 0–1 coach-rated severity
    annotated_at = Column(DateTime(timezone=True), nullable=False)
    annotated_by = Column(String(128), comment="Coach username/ID")

    player = relationship("Player", back_populates="annotations")

    __table_args__ = (
        Index("ix_annotations_player_type", "player_id", "annotation_type"),
    )


class PlayerBaseline(Base):
    """Per-player rolling baseline profile — updated after each session."""
    __tablename__ = "player_baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False)
    window_days = Column(Integer, nullable=False)           # 7 or 28

    # Baseline statistics (mean, std for each key metric)
    avg_distance_m = Column(Float)
    std_distance_m = Column(Float)
    avg_sprint_count = Column(Float)
    std_sprint_count = Column(Float)
    avg_top_speed_ms = Column(Float)
    std_top_speed_ms = Column(Float)
    avg_high_speed_distance_m = Column(Float)
    std_high_speed_distance_m = Column(Float)

    # Fatigue decay curve parameters (exponential fit)
    fatigue_decay_alpha = Column(Float, comment="Decay rate coefficient")
    fatigue_decay_beta = Column(Float, comment="Decay amplitude")
    fatigue_r_squared = Column(Float, comment="Goodness of fit")

    # Positional norms
    avg_x_position = Column(Float)
    avg_y_position = Column(Float)
    position_std_radius = Column(Float)

    n_sessions_used = Column(Integer)

    player = relationship("Player", back_populates="baseline_profiles")

    __table_args__ = (
        Index("ix_baselines_player_window", "player_id", "window_days"),
        UniqueConstraint("player_id", "window_days", "computed_at", name="uq_baseline_player_window_ts"),
    )


class ModelInference(Base):
    """Log of every model output with full SHAP payload — fully auditable."""
    __tablename__ = "model_inferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    recommendation_type = Column(Enum(RecommendationType), nullable=False)
    confidence = Column(Float, nullable=False)              # 0–1
    triggered_at = Column(DateTime(timezone=True), nullable=False)

    # SHAP payload (JSON)
    shap_values = Column(JSON, nullable=False,
                         comment="Dict[feature_name -> shap_contribution]")
    feature_values = Column(JSON, nullable=False,
                            comment="Dict[feature_name -> raw_value]")
    baseline_value = Column(Float, comment="SHAP base value (expected model output)")
    counterfactual = Column(Text, comment="Plain-language counterfactual sentence")
    nlg_summary = Column(Text, comment="Template-generated plain-language explanation")

    # Model provenance
    model_version = Column(String(32), nullable=False)
    anomaly_score = Column(Float)                          # Isolation Forest score

    __table_args__ = (
        Index("ix_inferences_player_ts", "player_id", "triggered_at"),
        Index("ix_inferences_session", "session_id"),
    )


class OverrideLog(Base):
    """Every coach override is logged — primary learning signal for recalibration."""
    __tablename__ = "override_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inference_id = Column(Integer, ForeignKey("model_inferences.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    recommendation_type = Column(Enum(RecommendationType), nullable=False)
    decision = Column(Enum(CoachDecision), nullable=False)
    coach_id = Column(String(128), nullable=False)
    coach_note = Column(Text)
    overridden_at = Column(DateTime(timezone=True), nullable=False)

    # Context snapshot at time of override
    context_snapshot = Column(JSON, comment="Feature values at decision time")

    player = relationship("Player", back_populates="override_logs")

    __table_args__ = (
        Index("ix_overrides_player_type", "player_id", "recommendation_type"),
        Index("ix_overrides_session", "session_id"),
    )


class RecalibrationHistory(Base):
    """Audit trail for every model threshold / sensitivity adjustment."""
    __tablename__ = "recalibration_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True,
                       comment="NULL means squad-wide recalibration")
    recalibrated_at = Column(DateTime(timezone=True), nullable=False)
    trigger_reason = Column(String(128))                   # "weekly_cadence", "override_cluster"
    n_overrides_analyzed = Column(Integer)
    adjustments = Column(JSON, nullable=False,
                         comment="Dict[feature -> {old_threshold, new_threshold}]")
    model_version_before = Column(String(32))
    model_version_after = Column(String(32))
    notes = Column(Text)

    __table_args__ = (
        Index("ix_recal_player_ts", "player_id", "recalibrated_at"),
    )


class FairnessAuditLog(Base):
    """Output of fairness audit pipeline — detects systematic over/under-flagging."""
    __tablename__ = "fairness_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audited_at = Column(DateTime(timezone=True), nullable=False)
    attribute = Column(String(64), nullable=False)         # "position", "age_group", etc.
    group_label = Column(String(64), nullable=False)       # "GK", "U18", etc.
    flag_rate = Column(Float, nullable=False)
    squad_avg_flag_rate = Column(Float, nullable=False)
    disparity = Column(Float, nullable=False)              # flag_rate - squad_avg
    is_biased = Column(Boolean, nullable=False)
    action_taken = Column(Text)
    audit_window_days = Column(Integer)

    __table_args__ = (
        Index("ix_fairness_attr_ts", "attribute", "audited_at"),
    )
