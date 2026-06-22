# Players Data — IBM CIC Germany
### Production-Grade Explainable Player Pattern Analysis for Real-Time Coaching

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Real-time anomaly detection platform for professional football. Transforms high-frequency telemetry (GPS, HR, accelerometry) into actionable coaching decisions using a shared-backbone LSTM autoencoder, regime-aware per-player calibration, and a multi-layer explainability suite — all engineered to maintain a **< 200 ms inference SLA** under distributed failure conditions.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Event Lifecycle](#event-lifecycle)
3. [Tech Stack](#tech-stack)
4. [File Map](#file-map)
5. [Quick Start](#quick-start)
6. [Installation](#installation)
7. [Configuration](#configuration)
8. [CLI Reference](#cli-reference)
9. [Data Schema](#data-schema)
10. [ML Pipeline](#ml-pipeline)
11. [Explainability (XAI)](#explainability-xai)
12. [Temporal State Compression](#temporal-state-compression)
13. [Cache-Augmented Generation (Redis CAG)](#cache-augmented-generation-redis-cag)
14. [Reliability & Hardening](#reliability--hardening)
15. [Replay Consistency Guarantees](#replay-consistency-guarantees)
16. [Fairness & Recalibration](#fairness--recalibration)
17. [Kinexon Real-Data Pilot Pipeline](#kinexon-real-data-pilot-pipeline)
18. [Logging & Observability](#logging--observability)
19. [Exit Codes](#exit-codes)
20. [Known Limitations & Roadmap](#known-limitations--roadmap)
21. [References](#references)

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
  │  │  · FatigueCurveAnalyzer    (speed decay fit) │  │
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
                       │  SemanticFindings + SHAP attributions
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │  Redis CAG Layer                                    │  ◄── Cache-Augmented Generation
  │  · Per-player SHAP attribution cache                │
  │  · SemanticFinding history (sorted sets, TTL-gated) │
  │  · Augments SemanticInterpreter with cached context │
  │    without re-running SHAP over past windows        │
  └────────────────────┬────────────────────────────────┘
                       │  Augmented findings
                       ▼
  ┌─────────────────────────────────────────────────────┐
  │  Temporal State Compression                         │
  │  · Trajectory narrative builder                     │
  │  · Escalation summary encoder                       │
  │  · Episodic abstraction (episode_id-scoped)         │
  │  Compresses finding stream → structured LLM prompt  │
  └────────────────────┬────────────────────────────────┘
                       │  Compressed state + SHAPExplanation
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

## Event Lifecycle

A single telemetry event passes through ten distinct processing stages before reaching the coach. This diagram provides the mental model for navigating the codebase.

```
  Raw Telemetry Event (GPS · HR · accelerometry)
          │
          │  player_external_id, ts, speed_ms, heart_rate_bpm, …
          ▼
  ┌───────────────────────┐
  │  1. Validity Gate     │  Pre-accumulation timestamp guard
  │     (TVL)             │  Epoch discontinuity → buffer reset
  └──────────┬────────────┘  INVALID → dropped  |  DEGRADED → flagged
             │
             ▼
  ┌───────────────────────┐
  │  2. Sequence Window   │  LiveWindowAccumulator ring buffer
  │     (24-event stride) │  Emits one window per 24 raw packets
  └──────────┬────────────┘  Post-window plausibility re-check (TVL)
             │
             ▼
  ┌───────────────────────┐
  │  3. Shared Model      │  SharedBackboneAutoencoder
  │     (LSTM + FiLM)     │  Regime-routed threshold comparison
  └──────────┬────────────┘  EMA-smoothed anomaly score
             │
             ▼
  ┌───────────────────────┐
  │  4. Attribution       │  Temporal Feature Ablation (F+2 calls)
  │     (SHAP / Ablation) │  SHAP KernelExplainer when available
  └──────────┬────────────┘  Magnitude-proxy fallback (shap_compat)
             │
             ▼
  ┌───────────────────────┐
  │  5. Semantic Findings │  SemanticInterpreter
  │                       │  SHAP weights → typed SemanticFinding
  └──────────┬────────────┘  Domains: cardiovascular · locomotor ·
             │               workload · tactical · persistence
             ▼
  ┌───────────────────────┐
  │  6. Redis CAG         │  Augment current findings with
  │     (Context Cache)   │  cached SHAP history + prior findings
  └──────────┬────────────┘  Deterministic, zero-retrieval-latency
             │               per-player longitudinal context
             ▼
  ┌───────────────────────┐
  │  7. State Compression │  MatchStateManager
  │                       │  Finding stream → trajectory narrative
  └──────────┬────────────┘  Motif detection · escalation summary ·
             │               episodic abstraction (episode_id-scoped)
             ▼
  ┌───────────────────────┐
  │  8. Policy Engine     │  AlertManager FSM
  │     (Alert FSM)       │  Hysteresis · cooldown · Safe Mode
  └──────────┬────────────┘  Recommendation priority ladder
             │
             ▼
  ┌───────────────────────┐
  │  9. NLG Layer         │  LLMNLGEngine (Qwen2.5:14b, async)
  │                       │  Receives compressed state only —
  └──────────┬────────────┘  not raw telemetry or full history
             │               TemplateNLGEngine fallback (<1 ms)
             ▼
  ┌───────────────────────┐
  │  10. Coach Dashboard  │  NDJSON alert → stdout
  │                       │  nlg_summary · top_features ·
  └───────────────────────┘  counterfactual · latency_ms
```

**SLA boundary:** The 200 ms clock runs from stage 2 (window emission) through stage 8 (alert FSM output). Stages 9–10 are asynchronous and off the SLA clock.

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
| **Redis** | CAG backing store; per-player SHAP attribution cache and `SemanticFinding` history (sorted sets, TTL-gated); deterministic context augmentation without retrieval latency |

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
| `redis` | CAG disabled; `SemanticInterpreter` operates without cached history |

---

## File Map

### Core Analysis

| File | Class / Entry Point | Responsibility |
| :--- | :--- | :--- |
| `main.py` | `main()` | Production CLI entrypoint (`generate · train · evaluate · serve · audit`) |
| `analysis/orchestrator.py` | `PlayersDataAnalysisPipeline` | Wires ingestion → TVL → ML → XAI → CAG → compression → FSM → feedback; match lifecycle |
| `analysis/anomaly_detection.py` | `SharedBackboneAutoencoder`, `PatternAnalysisEngine` | LSTM AE training + inference, threshold calibration, positional drift |
| `analysis/baseline.py` | `BaselineBuilder`, `PlayerBaselineProfile` | 28-day rolling baselines, fatigue curve fitting, ACWR tracking |
| `analysis/regime.py` | `SessionRegimeClassifier`, `RegimeAwareThresholdStore` | 9-regime (Territory × Intensity) window classification and threshold routing |
| `analysis/match_state.py` | `MatchStateManager`, `SemanticMatchState` | Longitudinal match memory, motif detection, trend reasoning, state compression |
| `analysis/live_window_accumulator.py` | `LiveWindowAccumulator` | Per-player ring buffer; emits fixed-stride inference windows |
| `analysis/telemetry_validity.py` | `TelemetryValidityLayer` | Physical plausibility gate (`VALID` / `DEGRADED` / `INVALID`); replay-aware timestamp validation |

### Explainability

| File | Class | Responsibility |
| :--- | :--- | :--- |
| `explainability/xai_layer.py` | `XAILayer`, `LLMNLGEngine`, `TemplateNLGEngine` | Temporal feature ablation, SHAP routing, Qwen2.5:14b NLG |
| `explainability/semantics_layer.py` | `SemanticInterpreter` | Symbolic physiological reasoning — cardiovascular, locomotor, workload, tactical |
| `explainability/shap_compat.py` | `compute_shap_values`, `build_kmeans_background` | SHAP with magnitude-proxy fallback; background deduplication guard |
| `explainability/episodic_context.py` | `TemporalContextCompressor`, `CompressedTemporalContext`, `PlayerEpisode`, `TacticalEpisode` | Compresses `SemanticFinding` streams into trajectory narratives, escalation summaries, and episodic abstractions before LLM conditioning |

### Cache-Augmented Generation

| File | Class | Responsibility |
| :--- | :--- | :--- |
| `cag/redis_client.py` | `RedisCheckpointStore`, `EpisodeStore` | Per-player SHAP attribution cache and `SemanticFinding` history; sorted-set TTL management |
| `cag/redis_client.py` | `RedisPubSubClient`, `RedisConnectionPool` | Pub/sub event streaming; connection pool management |

### Reliability Layer

| File | Class | Responsibility |
| :--- | :--- | :--- |
| `utils/reliability/invariants.py` | `SystemInvariantGuard` | Machine-enforced system invariants; triggers graded Safe Mode |
| `utils/reliability/safe_mode.py` | `SafeModeController` | Four-level degradation: NORMAL → LEVEL_1 → LEVEL_2 → LEVEL_3 |
| `utils/reliability/determinism.py` | `MutationJournal`, `TemporalCausalityGuard` | Versioned calibration log, strict event-time monotonicity |
| `utils/reliability/calibration_store.py` | `HardenedRollingThresholdStore` | Quarantine buffers, drift monitoring, thread-safe threshold store |
| `utils/reliability/adaptation_engine.py` | `DeterministicCalibrationManager` | Crash-safe, versioned calibration updates |
| `utils/reliability/queue_manager.py` | `BoundedPriorityQueue` | Priority-aware backpressure; sheds LLM tasks before SHAP before inference |

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

### Kinexon Real-Data Pilot Pipeline

See [Kinexon Real-Data Pilot Pipeline](#kinexon-real-data-pilot-pipeline) for the full picture.

| File | Class / Entry Point | Responsibility |
| :--- | :--- | :--- |
| `ingestion/kinexon_adapter.py` | `KinexonAdapter` | Parses real Kinexon `positions.csv`/`statistics.csv` exports into `RawPlayerObservation`s |
| `ingestion/kinexon_resampler.py` | `KinexonResampler` | Resamples raw Kinexon ticks into 15 s buckets (8 base columns) |
| `ingestion/kinexon_events_features.py` | `merge_event_features()` | Merges 24 window-aggregated `events.csv` features onto the resampled data — the 8→32 feature completion step |
| `analysis/player_workload.py` | `compute_player_workload_windows`, `assign_workload_status` | Model-free, per-tick coach workload aggregation (distance, sprint/accel/decel load, workload status) |
| `analysis/player_workload_event.py` | `PlayerWorkloadEvent` | Model-free dataclass published to `analytics.player_workload` |
| `analysis/player_analytics_event.py` | `PilotPlayerAnalyticsEvent`, `to_pilot_player_analytics_event()` | Model-output dataclass published to `analytics.players` (reconstruction_loss, confidence, SHAP, regime) |
| `scripts/publish_player_workload.py` | `main()` | One-shot batch publisher, model-free, 32-feature loader |
| `scripts/publish_pilot_analytics.py` | `main()` | One-shot batch publisher, real promoted-checkpoint inference, 32-feature loader |
| `scripts/run_live_player_analytics.py` | `main()` | Continuous production runtime — paced replay through `LiveWindowAccumulator`, publishes incrementally |
| `scripts/evaluate_pilot_model.py` | `_build_pipeline_and_load()`, `_build_pipeline_and_train()` | Shared checkpoint-loading helper (`use_event_features` flag controls 8 vs. 32 features) and standalone evaluation report |

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

# 5. Replay historical data (interleaved multi-session streams)
cat historical_events.jsonl | python main.py serve --replay-mode

# 6. Run fairness audit + recalibration check
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

# CAG backing store
pip install redis

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
| `REDIS_HOST` | `localhost` | Redis host for CAG store |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_CAG_TTL_S` | `3600` | TTL for cached SHAP and SemanticFinding entries (seconds) |
| `GPS_SERIAL_PORT` | `/dev/ttyUSB0` | Serial port for NMEA GPS |
| `GPS_TCP_HOST` | `None` | TCP host for gpsd / NMEA-over-TCP |
| `GPS_TCP_PORT` | `2947` | TCP port for gpsd |
| `SPORTRADAR_API_KEY` | `` | SportRadar API key |
| `LIVE_WS_URL` | `ws://localhost:8765` | Live match event WebSocket URL |
| `MQTT_BROKER` | `localhost` | MQTT broker host |
| `JSON_LOGS` | `0` | Set to `1` for structured JSON log output to stderr |
| `OLLAMA_NLG_TIMEOUT_S` | `30.0` | Timeout for Qwen2.5:14b async NLG calls (off SLA clock) |

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
| `CompressionConfig` | `max_findings_per_episode` | `12` | Finding cap before episodic abstraction triggers |
| `CompressionConfig` | `trajectory_window_steps` | `5` | Window count for trajectory narrative construction |
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
  --replay-mode                 Replay-safe mode; implies --ignore-time-gaps and
                                --ignore-session-boundaries. Also relaxes TVL timestamp
                                validation: reversals and large gaps produce DEGRADED
                                (not INVALID) so inference is not silently dropped.
  --log-level LEVEL                                               [default: INFO]
```

**SLA model:** The 200 ms SLA covers inference only (LSTM forward pass + threshold comparison + state compression). LLM NLG generation runs asynchronously off the SLA clock via a thread pool with a 30 s timeout. Two latency figures are observable:

| Metric | What it covers | Where it appears |
| :--- | :--- | :--- |
| `latency_ms` in alert payload | Inference + compression (T1) | stdout NDJSON, inference log |
| Ollama call duration | Async NLG completion (T2) | `Slow Ollama call` WARNING in stderr |

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
  "recommendation_type": "substitute",
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
| 1 | `substitute` | Recurrent cross-match pattern + sustained persistence (≥4 windows) + high/critical escalation |
| 2 | `recovery_intervention` | Cardiovascular or recovery degradation finding, sustained (≥3 windows), high/critical severity |
| 3 | `workload_restriction` | Fatigue accumulation finding OR ACWR ≥ 1.30, sustained ≥2 windows |
| 4 | `tactical_adjustment` | Tactical instability finding, any severity |
| 5 | `performance_monitor` | Locomotor overload finding with worsening trend |
| 6 | `anomaly_flag` | Default fallback; no specific rule matched |

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

The `SemanticInterpreter` is augmented by the Redis CAG layer (see [Cache-Augmented Generation](#cache-augmented-generation-redis-cag)): before classifying current-window attributions, the interpreter retrieves cached SHAP results and prior `SemanticFinding` objects for the player, enabling trend-aware symbolic reasoning without recomputing past windows. The LLM receives `SemanticFinding` objects and acts as narrator only — physiological reasoning lives in this symbolic layer, not in the prompt.

### 3. Match State

Accumulates `SemanticFinding` objects over the full match timeline. Provides motif detection (repeated finding patterns within a session) and trend reasoning (increasing/decreasing severity over time). `build_semantic_summary()` feeds the state compression layer, which condenses the finding stream before LLM conditioning.

### 4. NLG Engine

`LLMNLGEngine` calls `qwen2.5:14b` via Ollama asynchronously (off the SLA clock) with a `OLLAMA_NLG_TIMEOUT_S` timeout (default 30 s). The LLM receives a **compressed state representation** from the `TemporalContextCompressor` — not raw telemetry or the full finding stream — ensuring prompt entropy is minimised and physiological reasoning remains in the symbolic layer. On timeout or connection failure, `TemplateNLGEngine` provides a deterministic, sub-millisecond fallback.

**NLG summary guarantee:** Every emitted alert carries a non-empty `nlg_summary`. Alerts where SHAP is on cooldown (60 s XAI cooldown between full SHAP runs per player) receive an immediate template summary backed by cached attribution context from Redis. Alerts where SHAP runs receive the richer LLM-backed summary via the async worker.

---

## Temporal State Compression

### Motivation

Naively feeding the LLM a full stream of `SemanticFinding` objects accumulates four compounding problems as a match progresses:

- **Prompt entropy** — unrelated findings from different match phases dilute the signal relevant to the current alert.
- **Repeated findings** — the same physiological pattern (e.g., `hr_recovery` below baseline) may appear in every window of a sustained episode, adding tokens without adding information.
- **Temporal redundancy** — findings from 70 minutes ago carry little diagnostic weight for a substitution decision at 85 minutes.
- **Alert flooding** — without compression, the LLM receives the same escalation narrative on every window of a sustained episode, producing near-identical summaries.

### Compression Pipeline

`TemporalContextCompressor` (in `explainability/episodic_context.py`) operates in three stages after `MatchStateManager` has accumulated findings for the current episode:

**1. Trajectory Narrative**
Constructs a structured summary of the player's physiological trajectory over the last `compression.trajectory_window_steps` (default 5) inference windows. Each named domain (`cardiovascular_load`, `locomotor_load`, etc.) is represented by its direction vector (stable / worsening / recovering) and peak severity, not by individual finding instances. This reduces a 5-window finding sequence to a single structured object per domain.

```
cardiovascular_load: worsening  (peak severity: HIGH, onset: window -3)
locomotor_load:      stable     (severity: MEDIUM)
tactical:            recovering (drift cleared at window -1)
```

**2. Escalation Summary**
Encodes the Alert FSM trajectory for the current episode as a compact descriptor:
`NONE → WARNING (w=2) → SUSTAINED (w=4) → gate_windows=6`. This gives the LLM the full escalation arc in a single token-efficient string, replacing per-window FSM state repetition.

**3. Episodic Abstraction**
When `compression.max_findings_per_episode` (default 12) is exceeded within a single `episode_id`, older findings are collapsed into a typed episode header: `[EPISODE_START: cardiovascular+locomotor, onset 00:74:12, initial_confidence 0.81]`. Only findings from the most recent 3 windows are passed verbatim. This preserves longitudinal behavioural structure — the LLM knows what kind of episode this is and when it started — while eliminating token-for-token repetition of resolved findings.

### What the LLM Receives

The compressed prompt contains:
- **Trajectory narrative** (domain → direction + severity): ~40–80 tokens
- **Escalation summary** (FSM arc): ~15 tokens
- **Episodic header** (if applicable): ~25 tokens
- **Current-window top SHAP features** (from ablation or cache): ~60 tokens
- **Counterfactual** (what would clear the alert): ~20 tokens

Total: ~160–200 tokens of structured context, regardless of match duration or episode length. Without compression, a 90-minute match with 5-window findings would accumulate ~2,700+ tokens of raw finding history.

### Integration with Redis CAG

The compression layer is cache-aware. Before building the trajectory narrative, `TemporalContextCompressor` queries `RedisCheckpointStore` / `EpisodeStore` for the player's cached SHAP attributions from the XAI cooldown period. This ensures that windows where full SHAP was not recomputed (due to the 60 s cooldown) still contribute their attribution signal to the trajectory narrative via the cached values, rather than appearing as gaps.

---

## Cache-Augmented Generation (Redis CAG)

### Design Rationale

The system implements Cache-Augmented Generation (CAG) — as opposed to Retrieval-Augmented Generation (RAG) — using Redis as the backing store. The distinction is consequential for a real-time inference pipeline:

| | CAG (this system) | RAG (not used) |
| :--- | :--- | :--- |
| **Retrieval** | Deterministic key lookup (`player_id:shap`, `player_id:findings`) | Approximate nearest-neighbour search |
| **Latency** | O(1) Redis GET / ZRANGE | Vector store query latency (5–50 ms typical) |
| **Correctness** | Exact cached artefacts; no retrieval error | Relevant documents may not be returned |
| **Domain** | Closed, structured (per-player physiological history) | Open, unstructured (general knowledge) |

For a closed, structured domain like per-player physiological findings, RAG's retrieval flexibility is unnecessary and its latency and retrieval error are unacceptable within the 200 ms SLA. CAG provides the right tradeoff.

### Cached Artefacts

**SHAP attribution cache (`player_id:shap:window_ts`)**
After each SHAP run, the 8-feature attribution vector is written to Redis with a `REDIS_CAG_TTL_S` TTL (default 3600 s). During the 60-second XAI cooldown between full SHAP runs per player, the `SemanticInterpreter` and `TemporalContextCompressor` read the most recent cached attribution rather than falling back to zero-weight attribution. This means the trajectory narrative always reflects real attribution signal, not silence.

**SemanticFinding history (`player_id:findings` sorted set)**
Each `SemanticFinding` emitted by the `SemanticInterpreter` is appended to a per-player Redis sorted set, scored by Unix timestamp. `EpisodeStore` retrieves the N most recent findings (default N=10) before interpreter runs, enabling trend-aware symbolic reasoning:

```python
# Without CAG: interpreter sees only current window
findings = interpreter.classify(current_shap, current_window)

# With CAG: interpreter sees current window + longitudinal context
cached_context = cag_store.get_recent_findings(player_id, n=10)
cached_shap    = cag_store.get_latest_shap(player_id)
findings = interpreter.classify(current_shap, current_window,
                                context=cached_context,
                                prior_shap=cached_shap)
```

This is the critical enabler for multi-window trend detection — the interpreter can classify a finding as `persistence` (recurrent pattern) rather than `first_occurrence` only because the cached history is available without reprocessing the `MatchStateManager` trajectory.

### Graceful Degradation

When Redis is unavailable, `RedisCheckpointStore` / `EpisodeStore` returns empty context objects. The `SemanticInterpreter` falls back to single-window classification, and the `TemporalContextCompressor` builds trajectory narratives from in-memory `MatchStateManager` state only. No alerts are suppressed; the finding quality degrades gracefully from trend-aware to window-local.

```
REDIS_CAG_AVAILABLE=True  →  trend-aware findings, full trajectory narrative
REDIS_CAG_AVAILABLE=False →  window-local findings, in-memory trajectory only
```

---

## Reliability & Hardening

### Telemetry Validity Layer (TVL)

Telemetry validation operates in two stages:

1. **Pre-accumulation temporal validation (event-level)** — detects timestamp reversals and epoch discontinuities before the event enters the accumulator, triggering epoch-scoped runtime resets when continuity cannot be preserved.
2. **Post-window semantic validation (window-level)** — physical plausibility checks run after a complete window is emitted.

**Pre-accumulation checks:**

| Check | Live behaviour | Replay behaviour (`--replay-mode`) |
| :--- | :--- | :--- |
| **Timestamp reversal** | `non_monotonic_timestamp` → INVALID, buffer reset | `replay_non_monotonic_timestamp` → DEGRADED (confidence 0.7, floored to 0.8) |
| **Timestamp gap > 60 s** | buffer reset | no reset (gaps expected between seasons) |

**Post-window checks:**

| Check | Live behaviour | Replay behaviour |
| :--- | :--- | :--- |
| **Mask completeness** | <75% of required fields → INVALID | same |
| **Physical plausibility** | speed >13.5 m/s (+20% margin), HR outside [30, 220], accel >12 m/s² → INVALID | same |
| **Timestamp gap > 5 s** | `timestamp_gap_*` → confidence -0.3 | `replay_timestamp_gap_*` → no penalty |

Replay-specific issue strings (`replay_*`) are distinct from live equivalents so audit queries on `non_monotonic_timestamp` or `timestamp_gap_*` continue to find only genuine sensor failures, not expected replay stream disorder.

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
- Resets automatically on confirmed continuity breaks (session boundary transitions, timestamp discontinuities, epoch-scale temporal gaps)
- Buffer resets propagate through a unified epoch-reset path that atomically clears EMA state, positional trajectory buffers, alert FSM persistence state, rolling match-state trajectories, TVL per-player timestamp history, and output cooldown gates

### SLA Measurement

The SLA timer (`t_start`) is set immediately after the accumulator emits a complete window. The 200 ms budget measures inference time — LSTM forward pass, threshold comparison, result assembly, and state compression — and is not inflated by accumulation time or asynchronous LLM NLG.

### Exactly-Once Semantics & Determinism

**Event fingerprinting (`MutationJournal`):** Each calibration update is content-hashed. Idempotent replay: duplicate updates are silently dropped.

**Temporal Causality Guard:** Detects timestamp reversals and epoch discontinuities before accumulation, triggering epoch-scoped runtime resets. Configurable strict/warn mode.

**Priority-aware backpressure (`BoundedPriorityQueue`):** Under load, tasks are shed in reverse priority — LLM summaries dropped first, then SHAP, then inference — ensuring the 200 ms SLA is preserved even when the LLM is slow or unavailable.

---

## Replay Consistency Guarantees

Replay consistency is a first-class design concern. Most sports AI systems process historical data without guaranteeing that the inference, alert, and explanation outputs produced during replay are bitwise-reproducible and semantically equivalent to what would have been produced in live operation. This system provides explicit guarantees across four layers.

### 1. Deterministic Event Ordering

The `TemporalCausalityGuard` enforces strict event-time monotonicity across all ingestion paths. In replay mode, timestamp reversals that are expected artefacts of interleaved multi-session streams are classified as `replay_non_monotonic_timestamp` (DEGRADED, confidence floored at 0.8) rather than triggering buffer resets. This preserves inference continuity through interleaved streams while keeping the live `non_monotonic_timestamp` marker clean for genuine sensor failure auditing.

The replay-specific issue taxonomy (`replay_non_monotonic_timestamp`, `replay_timestamp_gap_*`) is distinct from live equivalents at every layer — TVL classification, log emission, and audit query — so post-match analysis of replay logs cannot be contaminated by expected stream disorder.

### 2. Persistence Accumulation Semantics

In live mode, the `LiveWindowAccumulator` resets on session boundary transitions and timestamp gaps > 60 s. In `--replay-mode`, these resets are suppressed because historical streams routinely interleave events from unrelated source sessions, and session-boundary transitions in the stream do not represent genuine continuity breaks. The accumulator instead relies solely on the `TemporalCausalityGuard` for epoch-scoped resets, preserving the same accumulation semantics that governed alert persistence during live operation.

### 3. State Transition Integrity

The unified epoch-reset path ensures that when a continuity break does occur in replay, the full runtime state is cleared atomically: EMA smoothing state, positional trajectory buffers, Alert FSM persistence counters, rolling match-state trajectories, TVL timestamp history, Redis CAG context (player findings and SHAP cache), and output cooldown gates are all reset together. Partial state resets — where the FSM clears but the EMA does not, for example — are architecturally prevented by routing all resets through a single `reset_player()` call chain.

### 4. Telemetry Confidence Behaviour

The 0.8 confidence floor applied to `DEGRADED` replay events is propagated consistently through the full pipeline: from `TelemetryValidityLayer._effective_confidence()` through `AlertManager` (which gates on confidence < 0.4 for `HOLD`) and through the inference log (`confidence` field in every NDJSON entry). This means that post-match confidence distributions computed from the inference log accurately reflect the replay-time confidence behaviour, enabling reproducible threshold sensitivity analysis.

The `--replay-mode` flag is threaded from `cmd_serve` → `_build_pipeline(replay_mode)` → `PlayersDataAnalysisPipeline(replay_mode)` → `TelemetryValidityLayer(replay_mode)` → `process_window_direct(replay_mode)` → `_effective_confidence()` without duplicating policy logic at any layer.

### Replay vs. Live: Behavioural Differences Summary

| Behaviour | Live mode | Replay mode (`--replay-mode`) |
| :--- | :--- | :--- |
| Timestamp reversal | INVALID → buffer reset | DEGRADED (conf 0.8) → inference proceeds |
| Timestamp gap > 60 s | Buffer reset | No reset |
| Session boundary transition | Buffer reset | No reset |
| TVL issue label | `non_monotonic_timestamp` | `replay_non_monotonic_timestamp` |
| Audit query contamination | Genuine sensor failures only | Replay disorder isolated to `replay_*` labels |
| Confidence floor on DEGRADED | 0.0–0.8 (unclamped) | 0.8 (floored) |

These differences are intentional and documented. They ensure replay outputs are maximally useful for post-match analysis and debugging while preserving the integrity of live sensor-failure auditing.

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

## Kinexon Real-Data Pilot Pipeline

Everything above this section describes the synthetic-data CLI (`generate · train · evaluate · serve`). A separate, real-data pilot pipeline runs alongside it, ingesting actual Kinexon UWB tracking exports (handball, not football GPS) for `analytics.players` / `analytics.player_workload` Redis Streams consumed by the Backend `AnalyticsBridgeService` / Frontend "Player Analytics" tab. It reuses the same `SharedBackboneAutoencoder` / `PatternAnalysisEngine` / SHAP stack documented above — no separate model architecture.

### Data Source

Real Kinexon CSV exports under `data/` (`positions.csv`, `statistics.csv`, `events.csv`), loaded via `ingestion/kinexon_adapter.py` (`KinexonAdapter`) → `ingestion/kinexon_resampler.py` (`KinexonResampler`, 15 s buckets). `ingestion/kinexon_events_features.py::merge_event_features()` merges 24 additional window-aggregated event features onto the 8 resampled columns, matching the **32-feature** input the promoted checkpoint (`models/shared_backbone.pt`) is actually trained on. As of this pipeline's current data, exactly one real match exists (session `3387`, HSG Wetzlar vs. SC Magdeburg, 2026-06-07) — multi-match history is a designed-but-not-yet-implemented roadmap item (see below).

### Pipeline Scripts (`scripts/`)

| Script | Model? | Feature count | Mode | Output stream |
| :--- | :--- | :--- | :--- | :--- |
| `publish_player_workload.py` | No — model-free aggregation (`analysis/player_workload.py`) | 32 (loader-level only; not fed to a model) | Batch (one-shot) | `analytics.player_workload` |
| `publish_pilot_analytics.py` | Yes — loads promoted checkpoint, never retrains | 32 | Batch (one-shot) | `analytics.players` |
| `run_live_player_analytics.py` | Yes — loads promoted checkpoint once at startup, never retrains/reloads | 32 | **Continuous** — paced replay of real session ticks through `LiveWindowAccumulator`, publishing incrementally per completed window | `analytics.player_workload` and `analytics.players` |
| `evaluate_pilot_model.py` | Yes — trains (`_build_pipeline_and_train()`) or loads (`_build_pipeline_and_load()`) | 8 by default; `use_event_features=True` for the loader path | One-shot diagnostic report | none (writes `_pilot_eval_windows.csv`) |

All three model-driven entrypoints (`publish_pilot_analytics.py`, `run_live_player_analytics.py`, and the loader path in `evaluate_pilot_model.py` when invoked with `use_event_features=True`) now route through the **same 32-feature loader** (`evaluate_pilot_model.py::_build_pipeline_and_load(use_event_features=True)`), eliminating an earlier train/serve skew where the promoted checkpoint was scored on only 8 of its 32 trained inputs.

`main.py serve` (the synthetic-data CLI documented above) is **architecturally separate** from this pilot pipeline: it has no Kinexon loader, depends entirely on whatever JSON its stdin producer supplies, uses a mismatched `LiveWindowAccumulator(24, 24)` against the model's real `window_steps=8`, and never publishes to Redis — it writes to `logs/inference_log.jsonl` and drives the synthetic system's own alert/NLG narrative instead. It is not part of the Kinexon production path.

### Continuous Inference (`run_live_player_analytics.py`)

The canonical production runtime for `analytics.players`. Loads the promoted checkpoint once, then replays real per-tick session data in chronological order across all players (paced via `--tick-interval-seconds`), pushing each tick through the same `LiveWindowAccumulator` class `main.py serve` uses (configured at the model's real `window_size=8, stride=8`). On each completed window it runs one real inference and publishes immediately — no end-of-run batch dump. No live Kinexon hardware feed exists yet in this codebase (`tracking.events` has no producer), so this is a paced replay of recorded data rather than a stadium connection, but it is a genuine long-running process otherwise.

```bash
python scripts/run_live_player_analytics.py [--tick-interval-seconds 0.2] [--max-ticks N]
```

### Multi-Match Player History (designed, not implemented)

A canonical append-only `player_match_history` store (one record per `(player_id, match_id)`, never updated) has been designed to support trend analysis once additional matches are ingested — distance, sprint/acceleration/deceleration counts, workload metric summaries, and reconstruction-loss/confidence/anomaly summaries per player per match. Not yet built. With only one real match currently available, multi-match analytics (workload trend, ACWR, performance trend, match-to-match consistency) cannot be computed yet regardless of engineering effort — this is a data-volume limitation, not a missing-feature one. See [Known Limitations & Roadmap](#known-limitations--roadmap).

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

Written by `serve` for every processed window (not just alerts). Fields: `inference_id`, `player_id`, `external_id`, `session_id`, `recommendation_type`, `is_anomaly`, `anomaly_score`, `confidence`, `fatigue_flag`, `drift_flag`, `workload_flag`, `nlg_summary`, `compression_tokens`, `cag_hit`, `ts`.

`cag_hit: true` indicates the window used Redis-cached SHAP attributions (XAI cooldown was active). `compression_tokens` records the token count of the compressed state passed to the LLM, enabling prompt efficiency monitoring.

When async LLM NLG completes, an enriched entry is appended with `"_nlg_enrichment": true`, `nlg_summary_llm`, and full `shap_values`.

### Key Log Events to Monitor

| Log pattern | Meaning |
| :--- | :--- |
| `ALERT player=… type=… conf=… latency=… ms` | Alert emitted to stdout |
| `SLA breach: player=… latency=…ms > 200ms` | Inference exceeded SLA; investigate model load |
| `CAG hit: player=… shap_cached=True findings_cached=N` | SemanticInterpreter augmented from Redis |
| `CAG miss: player=… redis_unavailable` | Redis down; falling back to single-window classification |
| `STATE COMPRESSED: player=… tokens=… findings_collapsed=N` | Episodic abstraction triggered; N findings collapsed to header |
| `BUFFER RESET reason=session_change` | LiveWindowAccumulator cleared on new session |
| `EPOCH RESET \| player=… reason=… cleared=[…]` | Unified runtime state reset triggered by continuity break |
| `Telemetry degraded player=… status=INVALID issues=[…]` | TVL rejected event; only live sensor issues appear at WARNING |
| `AlertManager: ENTERING GLOBAL SAFE MODE` | System-wide alert suppression active |
| `SHAP computation failed, using fallback` | SHAP library error; magnitude-proxy used |
| `Slow Ollama call: model=… ms` | LLM NLG took longer than expected; alert already emitted |
| `circuit breaker tripped — switching to template NLG` | Ollama unavailable; template fallback active for 30 s |

Replay-specific TVL issues (`replay_non_monotonic_timestamp`, `replay_timestamp_gap_*`) are logged at DEBUG level only and do not appear in WARNING output during normal replay operation.

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

- The 200 ms SLA covers inference and state compression (T1). LLM NLG generation (T2) is asynchronous and decoupled via a 30 s timeout with deterministic template fallback. Both latencies are observable separately in logs and the inference log.
- Temporal feature ablation explains the derived feature vector, not raw LSTM hidden states. True SHAP over the full sequence space would require ~2,000 model calls per window (~2–15 s), violating the SLA.
- `SessionRegimeClassifier` uses rule-based Territory × Intensity bins. Match phase (first/second half) is not included because elapsed-time context is not threaded through the calibration interface at training time.
- `PatternAnalysisEngine` is not thread-safe. One engine per asyncio event loop or per process is the supported deployment model.
- `TransformerAutoencoder` is experimental and disabled in production.
- Redis CAG TTL is uniform across all artefact types. SHAP attributions and `SemanticFinding` objects have different useful lifetimes (SHAP: ~1 match; findings: ~1 session) that a tiered TTL policy would address.
- Historical replay streams may interleave telemetry from unrelated source sessions. Anomaly scores in replay mode will vary across gate windows as the stream cycles through different historical sessions.
- The Kinexon pilot pipeline (see [Kinexon Real-Data Pilot Pipeline](#kinexon-real-data-pilot-pipeline)) currently has exactly one real match (session 3387). Multi-match analytics (workload trend, ACWR, performance trend, match-to-match consistency) cannot be computed until additional matches are ingested — a data-volume limitation, not an engineering gap.
- `main.py serve` and the Kinexon pilot pipeline are architecturally separate runtimes with no shared data loader; `serve` does not publish to Redis and is not part of the `analytics.players` production path.

**Roadmap:**

- Learned GMM regime detector to replace rule-based Territory × Intensity bins, enabling data-driven regime discovery.
- Async `PatternAnalysisEngine` with per-player actor isolation for horizontal scaling.
- SHAP over LSTM hidden states via integrated gradients (`GradientExplainer`) — eliminates the sequence-space dimensionality problem.
- Kafka consumer integration for multi-worker `serve` deployments.
- FastAPI wrapper exposing `process_window_direct()` as a REST endpoint for integration with external dashboards.
- Elapsed-time axis in regime classification (match phase as a third regime dimension).
- Tiered Redis TTL policy: short TTL for SHAP attributions (match-scoped), longer TTL for compressed episodic abstractions (season-scoped post-match analysis).
- Redis Streams integration for distributed exactly-once event fingerprinting across multi-worker `serve` deployments.

---

## References

1. Rein & Memmert (2016) — Big data and tactical analysis in elite soccer; DOI: https://doi.org/10.1186/s40064-016-3108-2
2. Foteinakis et al. (2025) — Explainable ML for Basketball; DOI: https://doi.org/10.3390/app152312401
3. Odet et al. (2024) — ML and Explainability for Sports Outcome Prediction
4. Pietraszewski et al. (2025) — AI in Sports Analytics systematic review; DOI: https://doi.org/10.3390/app15137254
5. Kranzinger et al. (2025) — Explainable AI in Sports Science; DOI: https://doi.org/10.48550/arXiv.1705.07874
6. Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions; DOI: https://doi.org/10.48550/arXiv.1705.07874
7. Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
8. Bai et al. (2018) — An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling; DOI: https://doi.org/10.48550/arXiv.1803.01271
9. Caron & Müller (2023) — TacticalGPT: Uncovering the Potential of LLMs for Predicting Tactical Decisions in Professional Football
10. Ferrara (2024) — Large Language Models for Wearable Sensor-Based Human Activity Recognition; DOI: https://doi.org/10.3390/s24155045
11. Yang (2024) — ChatPPG: Multi-Modal Alignment of Large Language Models for Time-Series Forecasting in Table Tennis
12. Tian et al. (2025) — SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance; DOI: https://doi.org/10.48550/arXiv.2512.14121
13. Liu et al. (2024) — Smartboard: Visual Exploration of Team Tactics with LLM Agent; DOI: https://doi.org/10.1109/TVCG.2024.3456200
14. Feli et al. (2025) — An LLM-Powered Agent for Physiological Data Analysis; DOI: https://doi.org/10.1109/EMBC58623.2025.11254428
15. Xia et al. (2024) — SportQA: A Benchmark for Sports Understanding in Large Language Models; DOI: https://doi.org/10.18653/v1/2024.naacl-long.283
16. Apostolou & Tjortjis (2019) — Sports Analytics algorithms for performance prediction; DOI: https://doi.org/10.1109/IISA.2019.8900754
17. Sarlis & Tjortjis (2020) — Sports analytics — Evaluation of basketball players and team performance; DOI: https://doi.org/10.1016/j.is.2020.101562
18. Ghosh et al. (2023) — Sports analytics review: AI applications, emerging technologies, and algorithmic perspective; DOI: https://doi.org/10.1002/widm.1496
19. Chan et al. (2025) — Don't Do RAG: When Cache-Augmented Generation is Better than Retrieval Augmented Generation; DOI: https://doi.org/10.48550/arXiv.2412.15605
20. Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks; DOI: https://doi.org/10.48550/arXiv.2005.11401
21. Perez et al. (2018) — FiLM: Visual Reasoning with a General Conditioning Layer; AAAI 2018
22. Gabbett (2016) — The training-injury prevention paradox; DOI: https://doi.org/10.1136/bjsports-2015-095788