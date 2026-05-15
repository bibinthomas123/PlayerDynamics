# Players Data — IBM CIC Germany
### Production-Grade Explainable Player Pattern Analysis for Real-Time Coaching

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Real-time anomaly detection platform for professional football. Transforms high-frequency telemetry (GPS, HR, accelerometry) into actionable coaching decisions using a shared-backbone LSTM autoencoder, regime-aware per-player calibration, and a multi-layer explainability suite — all engineered to maintain a **< 200 ms inference SLA** under distributed failure conditions.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Tech Stack](#tech-stack)
3. [File Map](#file-map)
4. [Quick Start](#quick-start)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [CLI Reference](#cli-reference)
8. [Data Schema](#data-schema)
9. [ML Pipeline](#ml-pipeline)
10. [Explainability (XAI)](#explainability-xai)
11. [Reliability & Hardening](#reliability--hardening)
12. [Fairness & Recalibration](#fairness--recalibration)
13. [Logging & Observability](#logging--observability)
14. [Exit Codes](#exit-codes)
15. [Known Limitations & Roadmap](#known-limitations--roadmap)
16. [References](#references)

---

## System Architecture

```
Telemetry Stream (GPS/REST/WS/MQTT)
          │
          ▼
  ┌─────────────────────────┐
  │  Ingestion Layer        │  GPS NMEA · SportRadar REST · WebSocket · MQTT (QoS 1)
  └────────────┬────────────┘
               │  RawPlayerObservation
               ▼
  ┌─────────────────────────┐
  │  Pre-Accumulation       │──→ [Reject timestamp reversals]
  │  Temporal Guard         │──→ [Detect epoch discontinuities]
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  LiveWindowAccumulator  │  Per-player ring buffer; stride = window_size
  │  (24-event windows)     │  Emits one non-overlapping window per 24 events,
  └────────────┬────────────┘  reducing overlap-induced persistence amplification
               │
               ▼
  ┌─────────────────────────┐
  │  Post-Window TVL        │──→ [Physical plausibility validation]
  │  Semantic Validation    │    VALID · DEGRADED · INVALID
  └────────────┬────────────┘
               │  List[dict] window
               ▼
  ┌────────────────────────────────────────────────────┐
  │  Pattern Analysis Engine                           │
  │  ┌──────────────────────────────────────────────┐  │
  │  │  SharedBackboneAutoencoder (LSTM + FiLM)     │  │
  │  │  · Shared encoder across all players         │  │
  │  │  · Per-player FiLM conditioning embeddings   │  │
  │  │  · Per-player normaliser (µ/σ per feature)   │  │
  │  └──────────────────────────────────────────────┘  │
  │  ┌──────────────────────────────────────────────┐  │
  │  │  RegimeAwareThresholdStore                   │  │
  │  │  9 regimes: Territory(3) × Intensity(3)      │  │
  │  │  · Per-regime DynamicThresholdTracker        │  │
  │  │  · Fallback to global tracker if under-cal   │  │
  │  └──────────────────────────────────────────────┘  │
  │  ┌──────────────────────────────────────────────┐  │
  │  │  Auxiliary Detectors                         │  │
  │  │  · FatigueCurveComparator  (speed decay fit) │  │
  │  │  · PositionalDriftAnalyzer (GPS centroid)    │  │
  │  │  · WorkloadTrendTracker    (ACWR 0.8–1.5)    │  │
  │  └──────────────────────────────────────────────┘  │
  └────────────────────┬───────────────────────────────┘
                       │  AnomalyResult
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │  Explainability Suite (XAI)                         │
  │  · Temporal Feature Ablation  (F+2 model calls)     │
  │  · SHAP KernelExplainer       (if shap installed)   │
  │  · SemanticInterpreter        (symbolic reasoning)  │
  │  · LLMNLGEngine (Qwen2.5:14b) ─→ TemplateNLGEngine │
  └────────────────────┬────────────────────────────────┘
                       │  SHAPExplanation + nlg_summary
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │  Alert FSM (AlertManager)                           │
  │  NONE → WARNING → SUSTAINED → CRITICAL              │
  │  HOLD  (telemetry blackout)                         │
  │  SAFE_MODE  (system-wide scientific invalidation)   │
  └────────────────────┬────────────────────────────────┘
                       │  Recommendation + NDJSON alert
                       ▼
              Coach Dashboard / stdout
                       │
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │  Feedback & Recalibration Loop                      │
  │  · Coach override logging (OverrideRecord)          │
  │  · FairnessMonitor (position · age_group · nation.) │
  │  · RecalibrationPipeline (7-day cadence)            │
  │  · MutationJournal  (versioned threshold audit)     │
  └─────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Machine Learning & AI

| Library / Model | Version | Role |
| :--- | :--- | :--- |
| **PyTorch** (`torch`, `torch.nn`, `torch.optim`, `DataLoader`) | ≥ 2.0 | Shared LSTM backbone, Transformer AE, batch training, checkpoint serialisation |
| **scikit-learn** | ≥ 1.3 | ROC-AUC / PR-AUC / precision@k evaluation; KMeans background summarisation for SHAP |
| **SciPy** (`stats.zscore`, `optimize.curve_fit`, `integrate.trapezoid`) | ≥ 1.11 | Z-score baselines, exponential fatigue curve fitting, trapezoid distance integration |
| **SHAP** (`KernelExplainer`, `shap.kmeans`) | ≥ 0.42 | Feature-level attribution (graceful magnitude-proxy fallback when unavailable) |
| **Qwen2.5:14b via Ollama** | Local HTTP | LLM NLG coaching summaries; configurable timeout, deterministic template fallback |

### Data & Numerics

| Library | Role |
| :--- | :--- |
| **NumPy** | Sequence windowing, proxy SHAP computation, batch array ops |
| **Pandas** | CSV I/O, timestamp parsing/coercion, rolling baseline aggregation |

### Ingestion & Networking

| Library | Protocol | Role |
| :--- | :--- | :--- |
| **aiohttp** | HTTP / REST | SportRadar / Opta API polling adapter; exponential-backoff retry |
| **websockets** | WebSocket | Live match event stream adapter |
| **asyncio-mqtt** (`aiomqtt`) | MQTT (QoS 1) | Wearable sensor bridge (HR, accelerometry) |
| **pynmea2** | NMEA 0183 | GPS sentence parsing from serial port or TCP/gpsd |
| **asyncio** | — | Single-event-loop async I/O for all ingestion adapters |

### Infrastructure & Persistence

| Component | Notes |
| :--- | :--- |
| **PostgreSQL** | Primary store; `psycopg2` for sync ORM, `asyncpg` for async paths |
| **SQLAlchemy** | ORM models — `Player`, `Session`, `PlayerEvent`, audit logs |

### Python Standard Library (key modules)

| Module | Usage |
| :--- | :--- |
| `argparse` | Five-subcommand CLI with typed arguments and defaults |
| `logging` | Structured logging; JSON formatter enabled by `JSON_LOGS=1` |
| `hashlib` | Event fingerprinting for exactly-once semantics |
| `threading` | `Lock` for `HardenedRollingThresholdStore` thread safety |
| `collections.deque` | `LiveWindowAccumulator` per-player ring buffers |
| `dataclasses` | All domain objects (`AnomalyResult`, `SHAPExplanation`, `WindowRegime`, etc.) |
| `time.monotonic` | Alert cooldown gate; SLA latency measurement |

### Optional / Graceful-Degradation Dependencies

| Library | Behaviour when absent |
| :--- | :--- |
| `shap` | Falls back to `shap_compat.py` magnitude-proxy attribution |
| `torch` | Stub mode — no inference, pipeline still importable |
| `sklearn` | ROC-AUC / PR-AUC disabled; `evaluate` exits 3 |
| `tqdm` | Progress bars replaced with `logger.info()` calls |
| `pynmea2` | GPS serial/TCP adapter disabled; REST + WS still work |
| `aiohttp` | REST polling adapter disabled |

---

## File Map

### Core Analysis

| File | Class / Entry Point | Responsibility |
| :--- | :--- | :--- |
| `main.py` | `main()` | Production CLI entrypoint (`generate · train · evaluate · serve · audit`) |
| `analysis/orchestrator.py` | `PlayersDataAnalysisPipeline` | Wires ingestion → TVL → ML → XAI → FSM → feedback; match lifecycle |
| `analysis/anomaly_detection.py` | `SharedBackboneAutoencoder`, `PatternAnalysisEngine` | LSTM AE training + inference, threshold calibration, positional drift |
| `analysis/baseline.py` | `BaselineBuilder`, `PlayerBaselineProfile` | 28-day rolling baselines, fatigue curve fitting, ACWR tracking |
| `analysis/regime.py` | `SessionRegimeClassifier`, `RegimeAwareThresholdStore` | 9-regime (Territory × Intensity) window classification and threshold routing |
| `analysis/match_state.py` | `MatchStateManager`, `SemanticMatchState` | Longitudinal match memory, motif detection, trend reasoning |
| `analysis/live_window_accumulator.py` | `LiveWindowAccumulator` | Per-player ring buffer; emits fixed-stride inference windows |

### Explainability

| File | Class | Responsibility |
| :--- | :--- | :--- |
| `explainability/xai_layer.py` | `XAILayer`, `LLMNLGEngine`, `TemplateNLGEngine` | Temporal feature ablation, SHAP routing, Qwen2.5:14b NLG |
| `explainability/semantics_layer.py` | `SemanticInterpreter` | Symbolic physiological reasoning — cardiovascular, locomotor, workload, tactical |
| `explainability/shap_compat.py` | `compute_shap_values`, `build_kmeans_background` | SHAP with magnitude-proxy fallback; background deduplication guard |

### Reliability Layer

| File | Class | Responsibility |
| :--- | :--- | :--- |
| `utils/reliability/invariants.py` | `SystemInvariantGuard` | Machine-enforced system invariants; triggers graded Safe Mode |
| `utils/reliability/safe_mode.py` | `SafeMode` | Four-level degradation: NORMAL → LEVEL_1 → LEVEL_2 → LEVEL_3 |
| `utils/reliability/telemetry_validity.py` | `TelemetryValidityLayer` | Physical plausibility gate (`VALID` / `DEGRADED` / `INVALID`) |
| `utils/reliability/determinism.py` | `MutationJournal`, `TemporalCausalityGuard` | Versioned calibration log, strict event-time monotonicity |
| `utils/reliability/calibration_store.py` | `HardenedRollingThresholdStore` | Quarantine buffers, drift monitoring, thread-safe threshold store |
| `utils/reliability/adaptation_engine.py` | `AdaptationEngine` | Crash-safe, versioned calibration updates |
| `utils/reliability/queue_manager.py` | `PriorityQueueManager` | Priority-aware backpressure; sheds LLM tasks before SHAP before inference |

### Ingestion & Infrastructure

| File | Class | Responsibility |
| :--- | :--- | :--- |
| `ingestion/pipeline.py` | `GPSIngestionAdapter`, `SportRadarAPIAdapter`, `IngestionPipeline` | NMEA/TCP GPS, REST polling, WebSocket events, MQTT sensor bridge |
| `config/settings.py` | `PlayersDataConfig` | All configuration via environment variables and typed dataclasses |
| `config/ollama_client.py` | `OllamaClient` | Async HTTP wrapper for Qwen2.5:14b; response caching, timeout guard |
| `utils/schema.py` | ORM models | SQLAlchemy models for `Player`, `Session`, `PlayerEvent`, audit log |
| `utils/ema.py` | `EMASmoother` | Exponential moving average for anomaly score smoothing (α = 0.25) |
| `utils/alert_manager.py` | `AlertManager` | Deterministic FSM with hysteresis, cooldown gate, Safe Mode propagation |
| `utils/evaluation/episodes.py` | `extract_episodes`, `match_episodes` | Binary → episode conversion; TP/FP/FN at episode level |

### Data

| File | Responsibility |
| :--- | :--- |
| `data/data_generator.py` | v4 Decision-Agent synthetic data simulator; realistic anomaly seeding |

---

## Quick Start

```bash
# 1. Generate 2 seasons of synthetic training data
python main.py generate --seasons 2 --matchdays 38

# 2. Train shared backbone + calibrate per-player thresholds
python main.py train --sessions-per-player 60

# 3. Evaluate against ground truth labels (CI gate: AUC >= 0.70)
python main.py evaluate --out metrics/eval.json --min-auc 0.70

# 4. Stream live inference (NDJSON in -> NDJSON alerts out)
cat live_events.jsonl | python main.py serve

# 5. Run fairness audit + recalibration check
python main.py audit --log logs/inference_log.jsonl
```

---

## Installation

```bash
# Core ML & data
pip install torch scikit-learn numpy pandas scipy shap

# Ingestion adapters
pip install aiohttp websockets asyncio-mqtt pynmea2

# Database drivers
pip install sqlalchemy psycopg2-binary asyncpg

# Optional: progress bars
pip install tqdm

# LLM backend — install Ollama separately, then pull the model
# https://ollama.com
ollama pull qwen2.5:14b
```

**Python 3.10+ required.** PyTorch CPU is sufficient for inference; GPU is recommended for training large squads.

---

## Configuration

All configuration is driven by environment variables and typed dataclasses in `config/settings.py`. The singleton `CONFIG = PlayersDataConfig()` is imported throughout the codebase.

### Key Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `players_data` | Database name |
| `DB_USER` | `postgres` | Database user |
| `DB_PASSWORD` | `` | Database password |
| `GPS_SERIAL_PORT` | `/dev/ttyUSB0` | Serial port for NMEA GPS |
| `GPS_TCP_HOST` | `None` | TCP host for gpsd / NMEA-over-TCP |
| `GPS_TCP_PORT` | `2947` | TCP port for gpsd |
| `SPORTRADAR_API_KEY` | `` | SportRadar API key |
| `LIVE_WS_URL` | `ws://localhost:8765` | Live match event WebSocket URL |
| `MQTT_BROKER` | `localhost` | MQTT broker host |
| `JSON_LOGS` | `0` | Set to `1` for structured JSON log output to stderr |
| `OLLAMA_NLG_TIMEOUT_S` | `0.1` | Timeout for Qwen2.5:14b NLG calls |

### Key Tunable Parameters (`config/settings.py`)

| Dataclass | Field | Default | Notes |
| :--- | :--- | :--- | :--- |
| `SequenceWindowConfig` | `window_seconds` | `120` | Rolling window length |
| `SequenceWindowConfig` | `step_seconds` | `15` | Must match `DT_OUT` in data generator |
| `LSTMAutoencoderConfig` | `hidden_size` | `64` | LSTM hidden units |
| `LSTMAutoencoderConfig` | `latent_dim` | `16` | Bottleneck dimension |
| `LSTMAutoencoderConfig` | `max_epochs` | `250` | With patience=20 early stopping |
| `AnomalyScoringConfig` | `mad_multiplier` | `5.0` | MAD multiplier for small calibration sets (<150 windows) |
| `AnomalyScoringConfig` | `threshold_quantile` | `0.995` | Quantile for large calibration sets (>=150 windows) |
| `AnomalyScoringConfig` | `score_ema_alpha` | `0.25` | EMA smoothing factor for anomaly scores |
| `SHAPConfig` | `n_background_samples` | `30` | Background samples for feature ablation |
| `FeedbackConfig` | `recalibration_cadence_days` | `7` | Scheduled recalibration interval |
| `FairnessConfig` | `flag_rate_disparity_threshold` | `0.15` | Max allowed flag-rate gap between groups |

---

## CLI Reference

All commands log to `stderr` and output machine-readable JSON to `stdout`.

### `generate` — Synthesise Training Data

```bash
python main.py generate [OPTIONS]

Options:
  --data-dir PATH          Output directory for CSVs              [default: data]
  --seasons INT            Number of seasons to simulate          [default: 2]
  --matchdays INT          Matchdays per season                   [default: 38]
  --anomaly-rate FLOAT     Fraction of sessions with seeded anomalies [default: 0.05]
  --no-corruption          Skip sensor corruption layer (cleaner, faster)
  --quiet                  Suppress per-position summary table
  --log-level LEVEL        DEBUG | INFO | WARNING | ERROR         [default: INFO]
```

**Output:** Five CSVs written to `--data-dir`: `players.csv`, `sessions.csv`, `events.csv`, `annotations.csv`, `ground_truth_labels.csv`.

Exits 1 if validation fails (zero anomalies seeded, missing columns, empty events table).

---

### `train` — Fit Model & Calibrate Thresholds

```bash
python main.py train [OPTIONS]

Options:
  --data-dir PATH              CSV source directory               [default: data]
  --model-dir PATH             Checkpoint output directory        [default: models]
  --sessions-per-player INT    Most-recent N sessions per player  [default: 60]
  --log-level LEVEL                                               [default: INFO]
```

**Writes:** `models/shared_backbone.pt`, `models/train_summary.json`, `models/serve_state.json`.

`serve_state.json` contains serialised per-player baselines and calibrated threshold distributions so `serve` can cold-start without retraining.

Exits 2 if training produces a degenerate model or the checkpoint is missing.

---

### `evaluate` — Score Against Ground Truth

```bash
python main.py evaluate [OPTIONS]

Options:
  --data-dir PATH          CSV source directory                   [default: data]
  --model-dir PATH         Checkpoint directory                   [default: models]
  --out PATH               Metrics output (JSON)                  [default: metrics/eval.json]
  --min-auc FLOAT          CI gate: exit 3 if mean ROC-AUC below [default: 0.60]
  --log-level LEVEL                                               [default: INFO]
```

**Metrics computed per player:** ROC-AUC, PR-AUC, precision@k, FP-per-90-min, TP/FP/FN/TN. Aggregated as micro (global TP/FP sums) and macro (per-player mean).

Exits 3 if mean ROC-AUC < `--min-auc` or no players produced evaluable windows.

---

### `serve` — Live Inference Stream

Reads newline-delimited JSON events from `stdin`. Emits NDJSON alerts to `stdout`. Writes a full inference log (including non-alert windows) to `logs/inference_log.jsonl`.

```bash
python main.py serve [OPTIONS]

Options:
  --model-dir PATH              Checkpoint directory              [default: models]
  --min-alert-windows INT       Consecutive anomalous windows before alert [default: 3]
  --max-latency-ms INT          SLA threshold; violations logged as WARNING [default: 200]
  --ignore-time-gaps            Disable time-gap buffer resets (use for batch replay)
  --ignore-session-boundaries   Disable session-boundary resets (use for interleaved replay)
  --replay-mode                 Replay-safe mode; implies --ignore-time-gaps and --ignore-session-boundaries
  --log-level LEVEL                                               [default: INFO]
```

In replay mode, continuity is inferred from temporal consistency rather than raw dataset session identifiers. This prevents historical replay streams containing interleaved sessions from triggering accumulator resets on every event.

Replay streams generated from historical datasets frequently contain interleaved events originating from many distinct original training sessions. Raw `session_id` transitions in these datasets do not necessarily represent live continuity boundaries and therefore cannot be treated as accumulator reset signals during replay.

**Input event fields** (NDJSON, one event per line):

| Field | Type | Required | Notes |
| :--- | :--- | :--- | :--- |
| `player_external_id` | `str` | Yes | Must match a registered player |
| `ts` | `str` (ISO 8601) | Yes | UTC timestamp |
| `match_id` / `session_id` | `str` | — | Used for session-boundary detection |
| `speed_ms` | `float` | Yes | Instantaneous speed in m/s |
| `heart_rate_bpm` | `int` | Yes | BPM |
| `x_pitch` | `float` | — | Normalised pitch X [0, 100] |
| `y_pitch` | `float` | — | Normalised pitch Y [0, 100] |
| `distance_delta_m` | `float` | — | Distance covered since last tick |
| `is_sprint` | `bool` | — | True if speed >= 7.0 m/s |
| `elapsed_seconds` | `float` | — | Seconds into session (used for fatigue enrichment) |

**Alert output payload** (NDJSON to stdout on alert):

```json
{
  "player_id": 7,
  "external_id": "p007",
  "recommendation_type": "substitution",
  "confidence": 0.923,
  "anomaly_score": 0.418,
  "fatigue_flag": true,
  "drift_flag": false,
  "workload_flag": false,
  "workload_status": "normal",
  "nlg_summary": "Muller shows 28% speed drop and elevated HR non-recovery...",
  "counterfactual": "Alert would clear if speed_ms increased by 1.2 m/s.",
  "top_features": [
    {"feature": "hr_recovery", "shap": 0.142, "value": -0.31, "label": "HR not recovering"},
    {"feature": "speed_ms",    "shap": 0.097, "value": 3.1,   "label": "Below normal speed"}
  ],
  "latency_ms": 47.3,
  "ts": "2025-09-14T19:42:11Z",
  "gate_windows": 4
}
```

**Recommendation priority ladder** (at most one per inference cycle):

| Priority | `recommendation_type` | Trigger condition |
| :---: | :--- | :--- |
| 1 | `substitution` | Anomaly + fatigue flag + confidence > 85% |
| 2 | `fatigue_alert` | Fatigue flag without high confidence |
| 3 | `positional_drift` | Tactical zone violation |
| 4 | `workload_warning` | ACWR outside [0.8, 1.5] safe band |
| 5 | `anomaly_flag` | Model anomaly + confidence > 75% |

---

### `audit` — Fairness Audit & Recalibration Check

```bash
python main.py audit [OPTIONS]

Options:
  --log PATH         Inference log path (NDJSON or JSON array)    [default: logs/inference_log.jsonl]
  --data-dir PATH    CSV directory (for player metadata)          [default: data]
  --out PATH         Audit report output (JSON)                   [default: metrics/audit.json]
  --log-level LEVEL                                               [default: INFO]
```

Checks for flag-rate disparity across three protected attributes: `position`, `age_group`, `nationality`. Triggers `RecalibrationPipeline` if >= 10 override records are present in the log.

Exits 5 if bias is detected in any protected group (flag-rate disparity > `fairness.flag_rate_disparity_threshold`).

---

## Data Schema

Five CSVs are produced by `generate` and consumed by `train` / `evaluate`:

| File | Key columns |
| :--- | :--- |
| `players.csv` | `player_id`, `external_id`, `full_name`, `position`, `age`, `age_group`, `nationality` |
| `sessions.csv` | `session_id`, `player_id`, `started_at`, `ended_at` |
| `events.csv` | `session_id`, `ts`, `speed_ms`, `heart_rate_bpm`, `x_pitch`, `y_pitch`, `distance_delta_m`, `is_sprint`, `elapsed_seconds` |
| `annotations.csv` | `session_id`, `annotated_at`, `annotation_type`, `note` |
| `ground_truth_labels.csv` | `session_id`, `is_anomaly` |

---

## ML Pipeline

### Sequence Features

Eight features extracted per 15-second tick, forming 8-step (120 s) windows:

| Index | Name | Description |
| :---: | :--- | :--- |
| 0 | `speed_ms` | Instantaneous speed (m/s) |
| 1 | `accel` | Acceleration (m/s²), clamped ±10 |
| 2 | `heart_rate_bpm` | HR (BPM) |
| 3 | `sprint_flag` | Binary; 1 if speed >= 7.0 m/s |
| 4 | `x_pitch` | Normalised pitch X [0, 100] |
| 5 | `y_pitch` | Normalised pitch Y [0, 100] |
| 6 | `distance_delta` | Euclidean displacement since last tick (m) |
| 7 | `hr_recovery` | Fractional HR change per tick, clipped [-1, 1] |

### Shared Backbone LSTM Autoencoder

- **Architecture:** Shared LSTM encoder → FiLM (Feature-wise Linear Modulation) per-player conditioning embedding → bottleneck (latent dim 16) → LSTM decoder.
- **Training:** All registered players jointly. Per-player embeddings are learned alongside shared weights. Per-player µ/σ normalisers applied before encoding.
- **Calibration split:** 80% training, 20% held-out calibration per player. For large calibration sets (>=150 windows): `quantile(losses, 0.995)`. For small sets (<150 windows): `median + 5.0 × MAD × 1.4826`.
- **Threshold routing:** At inference, `SessionRegimeClassifier` labels the window (Territory × Intensity → 9 possible keys). The corresponding regime tracker is used; falls back to global tracker when a regime has <5 calibration samples.
- **Score smoothing:** EMA with α=0.25 applied to per-window reconstruction losses before threshold comparison.

### Regime Classification (9 regimes)

Every 120-second window is classified on two axes:

| Axis | Class | Criterion |
| :--- | :--- | :--- |
| **Territory** | `defensive` | mean x_pitch < 33 |
| | `midfield` | 33 <= mean x_pitch <= 67 |
| | `attacking` | mean x_pitch > 67 |
| **Intensity** | `high` | sprint fraction >= 15% of window steps |
| | `medium` | 4% <= sprint fraction < 15% |
| | `low` | sprint fraction < 4% |

Each regime maintains its own `DynamicThresholdTracker`. This distinguishes "normal high-intensity pressing" from "abnormal physiological distress" during the same match phase.

### Transformer Autoencoder (Experimental)

Disabled in production (`CONFIG.active_model = "lstm"`). Pre-LN transformer encoder with sinusoidal positional encoding and validity-weighted pooling in the bottleneck. Requires >=30 sessions per player. Enable via `CONFIG.active_model = "transformer"`.

### Auxiliary Detectors

**Fatigue Curve Comparator** — Fits `speed(t) = β·exp(−α·t)` to each player's historical speed-vs-elapsed-time data (via `scipy.optimize.curve_fit`). Flags when the live speed residual falls more than one personal standard deviation below the expected curve, coinciding with a model anomaly.

**Positional Drift Analyzer** — Computes historical GPS centroid (`avg_x`, `avg_y`) and spread (`position_std_radius`). Flags when the player's recent median position deviates beyond `positional.zone_radius_meters` (default 5.0 m) for more than `positional.drift_fraction_threshold` (30%) of window ticks.

**Workload Trend Tracker (ACWR)** — Tracks the Acute-to-Chronic Workload Ratio (7-day / 28-day rolling distance). Flags when ACWR falls outside [0.8, 1.5], the established safe training load band.

---

## Explainability (XAI)

The XAI pipeline has four sequential layers with a strict separation of concerns:

```
Temporal Feature Ablation  ->  SemanticInterpreter  ->  MatchStateManager  ->  LLMNLGEngine
     (attribution only)         (symbolic findings)    (longitudinal memory)   (narration only)
```

### 1. Temporal Feature Ablation

Runs `F + 2 = 10` model calls per inference window (one per feature zeroed out, plus baseline and full-feature). Provides channel-level attribution within the 200 ms SLA (~30–50 ms on CPU). Used as the primary attribution method in production.

`shap.KernelExplainer` is used when the `shap` library is installed and background matrix dimensions match the feature vector. The `shap_compat.py` magnitude-proxy fallback is used otherwise, preserving the explanation interface.

### 2. Semantic Interpreter

Converts raw SHAP attributions into typed `SemanticFinding` objects across five domains:

| Domain | Features monitored |
| :--- | :--- |
| `cardiovascular_load` | `heart_rate_bpm`, `hr_recovery_time_s` |
| `locomotor_load` | `speed_ms`, `distance_delta`, `sprint_flag`, z-scores |
| `workload_balance` | ACWR, fatigue accumulation metrics |
| `tactical` | `x_pitch`, `y_pitch`, positional drift |
| `persistence` | Longitudinal recurrence patterns |

The LLM receives `SemanticFinding` objects and acts as narrator only — physiological reasoning lives in this symbolic layer, not in the prompt.

### 3. Match State

Accumulates `SemanticFinding` objects over the full match timeline. Provides motif detection (repeated finding patterns within a session) and trend reasoning (increasing/decreasing severity over time). `build_semantic_summary()` feeds the LLM prompt with longitudinal context.

### 4. NLG Engine

`LLMNLGEngine` calls `qwen2.5:14b` via Ollama with a `OLLAMA_NLG_TIMEOUT_S` timeout. On timeout or connection failure, `TemplateNLGEngine` provides a deterministic, sub-millisecond fallback that preserves the full explanation interface and maintains the SLA.

---

## Reliability & Hardening

### Telemetry Validity Layer (TVL)

Telemetry validation operates in two stages:

1. **Pre-accumulation temporal validation (event-level)** — detects timestamp reversals and epoch discontinuities before the event enters the accumulator, triggering epoch-scoped runtime resets when continuity cannot be preserved.
2. **Post-window semantic validation (window-level)** — physical plausibility checks run after a complete window is emitted.

**Pre-accumulation checks:**

| Check | Rejection condition |
| :--- | :--- |
| **Temporal monotonicity** | Non-monotonic timestamp → buffer reset; gap >60 s → buffer reset |

**Post-window checks:**

| Check | Rejection condition |
| :--- | :--- |
| **Mask completeness** | <75% of `{speed_ms, heart_rate_bpm, distance_delta_m, is_sprint}` present → `INVALID` |
| **Physical plausibility** | speed >13.5 m/s (+20% margin before hard reject), HR outside [30, 220] BPM, accel >12 m/s² → `INVALID` |
| **Temporal gap** | gap >5 s → confidence -0.3 |

Status values: `VALID` (confidence=1.0), `DEGRADED` (0.0–0.8), `INVALID` (0.0). Inference is blocked for `INVALID` events. `DEGRADED` events are inferred but flagged. The Alert FSM shifts to `HOLD` when event confidence <0.4.

### Alert Finite-State Machine

```
              signal_active (>= min_persistence windows)
   NONE  ────────────────────────────────────────────▶  WARNING
    ▲                                                      │
    │  recovery (>= recovery_threshold clear windows)      │ signal_active (>= escalation_threshold)
    │                                                      ▼
    └──────────────────────────────────────────────── CRITICAL
                    ◀──────── HOLD  (confidence < 0.4) ────────────
                    ◀──────── SAFE_MODE (system-wide) ─────────────
```

- **Hysteresis:** Transitions only escalate within an episode; de-escalation requires `recovery_threshold` (default 3) consecutive clear windows.
- **Alert family cooldown:** 20 s cooldown per player. Switching alert type resets the cooldown immediately, ensuring the first instance of a new type is never suppressed.
- **Episode tracking:** Each `NONE → WARNING` transition increments `episode_id`, enabling episode-level TP/FP/FN evaluation via `utils/evaluation/episodes.py`.

### Safe Mode Architecture (four levels)

| Level | Trigger | Features disabled |
| :--- | :--- | :--- |
| `NORMAL` | — | None |
| `LEVEL_1` | SHAP/LLM violation or TVL `DEGRADED` | SHAP explanation; LLM NLG |
| `LEVEL_2` | Invariant violation (e.g. model–threshold mismatch) | Above + adaptive calibration frozen |
| `LEVEL_3` | Critical invariant failure | Above + inference suspended; all alerts suppressed |

Safe Mode propagates from `SystemInvariantGuard` → `AlertManager.set_safe_mode()` → all downstream consumers.

### LiveWindowAccumulator

Buffers 24 raw telemetry packets per player before emitting one inference window (non-overlapping, `stride = window_size`):

- 1,092 telemetry packets → ~45 inference cycles instead of 1,092
- Reduces: alert duplication from near-identical overlapping buffers, fake persistence increments on every packet, exploding trajectory lengths, motif reinforcement without new information
- Resets automatically on confirmed continuity breaks:
  - session boundary transitions (live mode only)
  - timestamp discontinuities
  - epoch-scale temporal gaps
- Buffer resets propagate through a unified epoch-reset path that atomically clears:
  - EMA smoothing state
  - positional trajectory buffers
  - alert FSM persistence state
  - rolling match-state trajectories
  - temporal validity timestamp history
  - output cooldown gates
- `consume_reset_flag()` exposes reset propagation to the serve loop while preserving replay-safe operation.

### Exactly-Once Semantics & Determinism

**Event fingerprinting (`MutationJournal`):** Each calibration update is content-hashed. Idempotent replay: duplicate updates from Redis retries or worker crashes are silently dropped.

**Temporal Causality Guard:** Detects timestamp reversals and epoch discontinuities before accumulation, triggering epoch-scoped runtime resets when continuity cannot be preserved. Configurable strict/warn mode.

**Replay-safe state:** `serve_state.json` serialises baselines and threshold tracker state required for deterministic cold-start serving and replay reconstruction.

**Priority-aware backpressure (`PriorityQueueManager`):** Under load, tasks are shed in reverse priority — LLM summaries dropped first, then SHAP, then inference — ensuring the 200 ms SLA is preserved even when the LLM is slow or unavailable.

---

## Fairness & Recalibration

### Fairness Audit

`FairnessMonitor` computes flag-rate disparity across three protected attributes:

| Attribute | Groups examined |
| :--- | :--- |
| `position` | GK, CB, LB, RB, CM, AM, LW, RW, ST |
| `age_group` | U21, Senior, Veteran |
| `nationality` | All unique nationalities in the squad |

A group whose flag rate deviates more than `fairness.flag_rate_disparity_threshold` (default 15%) from the squad mean is flagged as biased. The audit command exits with code 5 and identifies the biased groups in the output JSON.

### Recalibration Pipeline

`RecalibrationPipeline` runs when >=10 coach override records (`OverrideRecord`) have been logged for a player within the recalibration window. Adjusts per-player thresholds by `feedback.threshold_adjustment_step` (default ±5%) and applies a `feedback.per_player_sensitivity_decay` (default 10%) to prevent runaway threshold drift. Default cadence: every 7 days.

All threshold adjustments are recorded in `MutationJournal` for full auditability and replay-safe reconstruction.

---

## Logging & Observability

### Log Formats

**Human-readable (default, to stderr):**
```
2025-09-14T19:42:11Z  INFO      players_data.main     Serve complete | events=1092 alerts=23 sla_violations=0
```

**Structured JSON** (set `JSON_LOGS=1`):
```json
{"ts": "2025-09-14T19:42:11Z", "level": "INFO", "logger": "players_data.main", "message": "ALERT player=p007  type=substitution  conf=0.92 latency=47.1 ms"}
```

### Inference Log (`logs/inference_log.jsonl`)

Written by `serve` for every processed window (not just alerts). Fields: `inference_id`, `player_id`, `external_id`, `session_id`, `recommendation_type`, `is_anomaly`, `anomaly_score`, `confidence`, `fatigue_flag`, `drift_flag`, `workload_flag`, `nlg_summary`, `ts`. This file is the input to `audit`.

### Key Log Events to Monitor

| Log pattern | Meaning |
| :--- | :--- |
| `ALERT player=… type=… conf=… latency=… ms` | Alert emitted to stdout |
| `SLA breach: player=… latency=…ms > 200ms` | Inference exceeded SLA; investigate model load |
| `BUFFER RESET reason=session_change` | LiveWindowAccumulator cleared on new session |
| `BUFFER RESET reason=time_gap gap=…s` | Timestamp gap >60 s detected; buffer cleared |
| `EPOCH RESET \| player=… reason=… cleared=[…]` | Unified runtime state reset triggered by continuity break |
| `Telemetry degraded player=… status=INVALID` | TVL rejected a sensor event |
| `AlertManager: ENTERING GLOBAL SAFE MODE` | System-wide alert suppression active |
| `SHAP computation failed, using fallback` | SHAP library error; magnitude-proxy used |
| `Slow Ollama call: model=… ms` | LLM exceeded timeout; template fallback triggered |

Epoch resets are atomic runtime isolation events. They prevent contamination between discontinuous telemetry epochs by clearing all transient runtime state associated with a player before new accumulation begins.

---

## Exit Codes

| Code | Command(s) | Condition |
| :---: | :--- | :--- |
| `0` | all | Success |
| `1` | `generate`, `train`, `audit` | Data or validation error (missing files, empty tables, parse failure) |
| `2` | `train`, `evaluate`, `serve` | Model error (not trained, corrupt checkpoint, zero windows) |
| `3` | `evaluate` | ROC-AUC below `--min-auc`, or no players produced evaluable windows |
| `4` | `serve` | Unhandled stream error |
| `5` | `audit` | Bias detected in a protected attribute group |

---

## Known Limitations & Roadmap

**Current limitations:**

- The 200 ms SLA applies to inference only. LLM NLG generation is asynchronous and decoupled via a configurable timeout with deterministic fallback.
- Temporal feature ablation explains the derived feature vector, not raw LSTM hidden states. True SHAP over the full sequence space would require ~2,000 model calls per window (~2–15 s), violating the SLA.
- `SessionRegimeClassifier` uses rule-based Territory × Intensity bins. Match phase (first/second half) is not included because elapsed-time context is not threaded through the calibration interface at training time.
- `PatternAnalysisEngine` is not thread-safe. Callers must serialise access per event loop or process. One engine per asyncio event loop or per process is the supported deployment model.
- `TransformerAutoencoder` is experimental and disabled in production.
- Historical replay streams may interleave telemetry originating from unrelated source sessions. Replay mode resolves continuity from timestamps rather than dataset session provenance.

**Roadmap:**

- Learned GMM regime detector to replace rule-based Territory × Intensity bins, enabling data-driven regime discovery.
- Async `PatternAnalysisEngine` with per-player actor isolation for horizontal scaling.
- SHAP over LSTM hidden states via integrated gradients (`GradientExplainer`) — eliminates the sequence-space dimensionality problem.
- Kafka consumer integration for multi-worker `serve` deployments.
- FastAPI wrapper exposing `process_window_direct()` as a REST endpoint for integration with external dashboards.
- Elapsed-time axis in regime classification (match phase as a third regime dimension).

---

## References

1. Rein & Memmert (2016) — Big data and tactical analysis in elite soccer. DOI: [10.1186/s40064-016-3108-2](https://doi.org/10.1186/s40064-016-3108-2)
2. Foteinakis et al. (2025) — Explainable ML for Basketball. DOI: [10.3390/app152312401](https://doi.org/10.3390/app152312401)
3. Odet et al. (2024) — ML and Explainability for Sports Outcome Prediction.
4. Pietraszewski et al. (2025) — AI in Sports Analytics systematic review. DOI: [10.3390/app15137254](https://doi.org/10.3390/app15137254)
5. Kranzinger et al. (2025) — Explainable AI in Sports Science. DOI: [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
6. Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions. DOI: [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
7. Hochreiter & Schmidhuber (1997) — Long Short-Term Memory.
8. Bai et al. (2018) — An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. DOI: [10.48550/arXiv.1803.01271](https://doi.org/10.48550/arXiv.1803.01271)
9. Caron & Muller (2023) — TacticalGPT: Uncovering the Potential of LLMs for Predicting Tactical Decisions in Professional Football.
10. Ferrara (2024) — Large Language Models for Wearable Sensor-Based Human Activity Recognition. DOI: [10.3390/s24155045](https://doi.org/10.3390/s24155045)
11. Yang (2024) — ChatPPG: Multi-Modal Alignment of Large Language Models for Time-Series Forecasting in Table Tennis.
12. Tian et al. (2025) — SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance. DOI: [10.48550/arXiv.2512.14121](https://doi.org/10.48550/arXiv.2512.14121)
13. Liu et al. (2024) — Smartboard: Visual Exploration of Team Tactics with LLM Agent. DOI: [10.1109/TVCG.2024.3456200](https://doi.org/10.1109/TVCG.2024.3456200)
14. Feli et al. (2025) — An LLM-Powered Agent for Physiological Data Analysis. DOI: [10.1109/EMBC58623.2025.11254428](https://doi.org/10.1109/EMBC58623.2025.11254428)
15. Xia et al. (2024) — SportQA: A Benchmark for Sports Understanding in Large Language Models. DOI: [10.18653/v1/2024.naacl-long.283](https://doi.org/10.18653/v1/2024.naacl-long.283)
16. Apostolou & Tjortjis (2019) — Sports Analytics algorithms for performance prediction. DOI: [10.1109/IISA.2019.8900754](https://doi.org/10.1109/IISA.2019.8900754)
17. Sarlis & Tjortjis (2020) — Sports analytics — Evaluation of basketball players and team performance. DOI: [10.1016/j.is.2020.101562](https://doi.org/10.1016/j.is.2020.101562)
18. Ghosh et al. (2023) — Sports analytics review: Artificial intelligence applications, emerging technologies, and algorithmic perspective. DOI: [10.1002/widm.1496](https://doi.org/10.1002/widm.1496)