# Players Data — IBM CIC Germany
## Production-Grade Explainable Player Pattern Analysis for Real-Time Coaching

This repository implements a high-reliability, distributed real-time anomaly detection platform for professional sports. It transforms high-frequency telemetry (GPS, HR, Accelerometry) into actionable coaching insights using a shared-backbone LSTM autoencoder, regime-aware calibration, and a multi-layered explainability suite.

---

## 🛠️ System Overview & File Map

This is a pure analysis and data ingestion core. It is designed to be embedded into a larger distributed system (e.g., via a FastAPI wrapper or Kafka consumer).

### Core Logic
- `main.py`: **Production Entrypoint**. Provides the CLI for the entire pipeline (`generate`, `train`, `evaluate`, `serve`, `audit`).
- `analysis/orchestrator.py`: **The Brain**. Wires together ingestion, ML inference, XAI, and reliability guards.
- `analysis/anomaly_detection.py`: **ML Engine**. Implements the Shared Backbone LSTM, sequence building, and anomaly scoring.
- `analysis/baseline.py`: **Physiological Profiling**. Computes 28-day rolling baselines and fatigue curves per player.
- `analysis/regime.py`: **Contextual Logic**. Classifies windows into 9 tactical regimes to apply regime-specific thresholds.
- `explainability/xai_layer.py`: **Interpretation**. Manages SHAP attributions and LLM-based natural language summaries.
- `explainability/shap_compat.py`: **Resilience**. Provides a magnitude-proxy fallback when the `shap` library is unavailable.
- `feedback/recalibration.py`: **Human-in-the-Loop**. Implements the override logging and threshold adjustment pipeline.

### Reliability & Determinism (Hardening Layer)
- `utils/reliability/invariants.py`: **The Guard**. Enforces system invariants and triggers Safe Mode.
- `utils/reliability/safe_mode.py`: **Degradation Controller**. Manages the 4 levels of system fallback.
- `utils/reliability/telemetry_validity.py`: **The Gate**. Filters implausible or corrupted sensor data.
- `utils/reliability/determinism.py`: **Consistency**. Implements the Mutation Journal and Temporal Causality Guard.
- `utils/reliability/calibration_store.py`: **Integrity**. Hardened store with quarantine buffers and drift monitoring.
- `utils/reliability/adaptation_engine.py`: **Recovery**. Ensures crash-safe, versioned calibration updates.
- `utils/reliability/queue_manager.py`: **Backpressure**. Priority-aware shedding to preserve the <200ms SLA.

### Infrastructure & Support
- `ingestion/pipeline.py`: **Data Input**. Handles GPS (NMEA/TCP), REST APIs, WebSockets, and MQTT.
- `data_generator.py`: **Synthetic Engine**. v4 Decision-Agent simulator for creating realistic test datasets.
- `config/settings.py`: **Central Config**. Environment-variable driven configuration for all subsystems.
- `config/ollama_client.py`: **LLM Bridge**. Local HTTP wrapper for Qwen2.5:14b via Ollama.
- `utils/schema.py`: **Persistence**. SQLAlchemy ORM models for players, sessions, and audit logs.
- `utils/ema.py`: **Signal Processing**. Exponential Moving Average implementation for score smoothing.

---

## 🚀 Operational Reliability Guarantees

Unlike standard ML pipelines, this system is engineered for **Final Hardening**, ensuring scientific validity under real-world distributed failure conditions.

### 🛡️ The Reliability Layer
- **System Invariant Guard**: Machine-enforced invariants (e.g., monotonic event ordering, model-threshold compatibility) that trigger a graded **Safe Mode** upon violation.
- **Telemetry Validity Layer (TVL)**: A physical plausibility gate that distinguishes between `VALID`, `DEGRADED`, and `INVALID` telemetry, preventing sensor corruption from poisoning the model or triggering false alerts.
- **Safe Mode Architecture**: A four-level degradation system (`NORMAL` $\to$ `LEVEL_3`) that surgically disables features (like SHAP/LLM) or freezes calibration to preserve system integrity.
- **Deterministic Alert FSM**: Alert transitions are handled by a finite-state machine with hysteresis and "Hold" states to prevent alert fragmentation during telemetry blackouts.

### ⚙️ Distributed Determinism
- **Exactly-Once Semantics**: An idempotent mutation protocol using event fingerprinting to prevent duplicate state updates during Redis retries or worker crashes.
- **Replay-Safe Calibration**: All adaptive threshold updates are recorded in a versioned **Mutation Journal**, allowing the system to reconstruct the exact scientific state from a raw event stream.
- **Temporal Causality Guard**: Strict enforcement of event-time monotonicity to prevent out-of-order packets from corrupting player baselines.
- **Priority-Aware Backpressure**: A bounded priority queue that sheds low-priority tasks (LLM summaries $\to$ SHAP) to guarantee the **< 200ms inference SLA** for critical alerts.

---

## 🛠️ Architecture Overview

```
Telemetry Stream (GPS/REST/WS/MQTT)
          ↓
   Temporal Causality Guard ───→ [Reject Out-of-Order]
          ↓
   Telemetry Validity Layer ───→ [Reject Implausible]
          ↓
   Pattern Analysis Engine
   ├── SharedBackboneAutoencoder (LSTM, Shared Weights + Player Embeddings)
   ├── RegimeAwareThresholdStore (Territory × Intensity Calibration)
   └── HardenedRollingThresholdStore (Quarantine Buffers + Drift Monitoring)
          ↓
   Explainability Suite (XAI)
   ├── SHAP KernelExplainer (Real-window background sampling)
   ├── LLMNLGEngine (Qwen2.5:14b via Ollama local API)
   └── TemplateNLGEngine (Deterministic fallback)
          ↓
   Sustained Alert FSM ───────→ Coach Dashboard
          ↓
   Feedback Loop (Recalibration Pipeline) ───→ Threshold Adjustment
```

---

## 📖 Quick Start

### Installation
```bash
pip install torch scikit-learn numpy pandas scipy shap aiohttp websockets
```

### Production Workflow
```bash
# 1. Generate realistic synthetic data (v4 Decision-Agent rewrite)
python main.py generate --seasons 2 --matchdays 38

# 2. Train the shared backbone & calibrate per-player thresholds
python main.py train

# 3. Evaluate against ground truth labels
python main.py evaluate --out metrics/eval.json

# 4. Stream live inference (NDJSON in → alerts out)
cat live_events.jsonl | python main.py serve

# 5. Run fairness audit & recalibration analysis
python main.py audit --log logs/inference_log.jsonl
```

---

## 🧬 Core Scientific Design

### Personal Baselines & Shared Learning
The system utilizes a **Shared Backbone** LSTM trained across the entire squad to learn general physiological patterns, but employs **per-player embeddings** and **individual normalisers**. Thresholds are calibrated exclusively on each player's own held-out windows, ensuring that "anomaly" is defined relative to the individual's history, not a squad average.

### Regime-Conditioned Thresholds
To prevent false positives during tactical transitions, the system classifies every 120s window into one of 9 behavioral regimes (**Territory** [Defensive/Midfield/Attacking] $\times$ **Intensity** [Low/Medium/High]). Each regime maintains its own calibration distribution, allowing the model to distinguish between "normal high-intensity pressing" and "abnormal physiological distress."

### Explainability (XAI)
- **SHAP**: Uses actual model perturbations on real data windows to provide feature-level attribution.
- **LLM-NLG**: Qwen2.5:14b generates clinical, factual summaries. If the LLM fails or times out (> 2s), the system transparently falls back to a deterministic template engine to maintain the SLA.

---

## 📈 Production CLI Reference

| Command | Purpose | Critical Flags |
| :--- | :--- | :--- |
| `generate` | Create synthetic datasets | `--anomaly-rate` |
| `train` | Train shared model & calibrate | `--sessions-per-player` |
| `evaluate` | Compute ROC-AUC/PR metrics | `--min-auc` (CI Gating) |
| `serve` | High-throughput live inference | `--max-latency-ms` |
| `audit` | Fairness & recalibration audit | `--log` |

**Operational Exit Codes:**
- `0`: Success
- `1`: Data/Validation Error
- `3`: Evaluation failure (AUC below target)
- `5`: Fairness Audit failed (Protected group bias detected)

---

## ⚠️ Limitations & Roadmap
- **SLA**: The 200ms target is for inference; LLM generation is asynchronous and decoupled.
- **SHAP Proxy**: Current SHAP explains the derived feature vector, not the raw LSTM hidden states.
- **Future**: Integration of a learned GMM regime detector to replace rule-based bins.


## References

1. Rein & Memmert (2016) — Big data and tactical analysis in elite soccer; DOI: https://doi.org/10.1186/s40064-016-3108-2
2. Foteinakis et al. (2025) — Explainable ML for Basketball; DOI: https://doi.org/10.3390/app152312401
3. Odet et al. (2024) — ML and Explainability for Sports Outcome Prediction
4. Pietraszewski et al. (2025) — AI in Sports Analytics systematic review; DOI: https://doi.org/10.3390/app15137254
5. Kranzinger et al. (2025) — Explainable AI in Sports Science; DOI: https://doi.org/10.48550/arXiv.1705.07874
6. Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions; DOI: https://doi.org/10.48550/arXiv.1705.07874
7. Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
8. Bai et al. (2018) — An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling; DOI: https://doi.org/10.48550/arXiv.1803.01271
9. Matthew Caron & Oliver Müller (2023) - TacticalGPT: Uncovering the Potential of LLMs for Predicting Tactical Decisions in Professional Football
10. Emilio Ferrara (2024) - Large Language Models for Wearable Sensor-Based Human Activity Recognition, DOI: https://doi.org/10.3390/s24155045
11. GuangLiang Yang (2024) - ChatPPG: Multi-Modal Alignment of Large Language Models for Time-Series Forecasting in Table Tennis 
12. Wenbo Tian, Ruting Lin, Hongxian Zheng, Yaodong Yang, Geng Wu, Zihao Zhang and Zhang Zhang (2025) - SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance; DOI: https://doi.org/10.48550/arXiv.2512.14121
13. Ziao Liu, Xiao Xie, Moqi He, Wenshuo Zhao, Yihong Wu, Liqi Cheng (2024) - Smartboard: Visual Exploration of Team Tactics with LLM Agent; DOI: https://doi.org/10.1109/TVCG.2024.3456200
14. Mohammad Feli, Iman Azimi, Pasi Liljeberg, Amir M. Rahmani (2025) - An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation; DOI: https://doi.org/10.1109/EMBC58623.2025.11254428
15. Haotian Xia, Zhengbang Yang, Yuqing Wang, Rhys Tracy, Yun Zhao, Dongdong Huang, Zezhi Chen, Yan Zhu, Yuan-fang Wang, Weining Shen - SportQA: A Benchmark for Sports Understanding in Large Language Models (2024) ; DOI: https://doi.org/10.18653/v1/2024.naacl-long.283
