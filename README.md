# Players Data — IBM CIC Germany · Group 11 / 2B

**Explainable Player Pattern Analysis for Real-Time Coaching Decisions**

---

## What This Is

Production-level Python analysis engine for the Players Data HCAI project.  
No frontend. No backend API. Pure analysis + data ingestion core.

### Components

| Layer | Module | What It Does |
|---|---|---|
| **Data Ingestion** | `ingestion/pipeline.py` | GPS (NMEA/TCP/GPX), REST API (SportRadar), WebSocket live stream, MQTT wearables |
| **Data Generator** | `data_generator.py` | v4 decision-agent synthetic dataset; 12-point realism rewrite; sensor corruption layer |
| **Personal Baseline** | `analysis/baseline.py` | Per-player rolling baseline (7d/28d), exponential fatigue curve fitting, workload ACWR |
| **Anomaly Detection** | `analysis/anomaly_detection.py` | Shared-backbone LSTM autoencoder, per-player regime-aware thresholds, positional drift, feature engineering |
| **Regime Classification** | `analysis/regime.py` | Classifies each 120 s window into territory × intensity regime; routes threshold comparison to regime-specific calibration distribution |
| **XAI Layer** | `explainability/xai_layer.py` | SHAP (KernelExplainer / magnitude-proxy fallback), **Qwen2.5:14b LLM summaries** (template fallback), counterfactual generation, waterfall data |
| **LLM Client** | `config/ollama_client.py` | Thin wrapper around local Ollama HTTP API; used exclusively by `LLMNLGEngine` |
| **SHAP Shim** | `explainability/shap_compat.py` | Transparent fallback when `shap` is not installed; preserves full explanation interface via feature-magnitude proxy |
| **Feedback Loop** | `feedback/recalibration.py` | Override logging, weekly recalibration, per-player sensitivity adjustment, fairness audit |
| **Orchestrator** | `analysis/orchestrator.py` | Wires all components; `PlayersDataAnalysisPipeline` is the single production interface |
| **Production CLI** | `main.py` | Five subcommands: `generate`, `train`, `evaluate`, `serve`, `audit` |
| **Demo harness** | `demo.py` | Integration showcase for stakeholder presentations; not the production entrypoint |
| **DB Schema** | `utils/schema.py` | Full SQLAlchemy ORM: players, sessions, events, annotations, override\_logs, fairness\_audit\_log |
| **Config** | `config/settings.py` | All parameters via environment variables, zero hardcoding |

---

## Quick Start

```bash
pip install torch scikit-learn numpy pandas scipy shap aiohttp websockets

# 1. Generate synthetic data
python main.py generate --seasons 2 --matchdays 38

# 2. Train the model
python main.py train

# 3. Evaluate against ground truth
python main.py evaluate --out metrics/eval.json

# 4. Stream live inference (NDJSON in → alerts out)
cat live_events.jsonl | python main.py serve

# 5. Run fairness + recalibration audit
python main.py audit --log logs/inference_log.jsonl
```

For the interactive demo (stakeholder presentation):

```bash
python demo.py
```

---

## Architecture

```
GPS / REST API / WebSocket / MQTT
          ↓
   IngestionPipeline
   (NMEA parse · GPS→pitch coords · sprint classification · sliding-window aggregation)
          ↓
   BaselineBuilder  ←──────────────────────────────────────┐
   (per-player 28d rolling baseline, exponential fatigue   │
    curve, workload ACWR 7d/28d)                           │
          ↓                                                │
   PatternAnalysisEngine                                   │
   ├── SharedBackboneAutoencoder (LSTM, shared weights)    │
   │   └── per-player embedding + per-player normaliser    │
   ├── RegimeAwareThresholdStore (territory × intensity)   │
   │   └── DynamicThresholdTracker per regime + global     │
   │       fallback; MAD or quantile depending on n_calib  │
   ├── PositionalDriftAnalyzer                             │
   └── WorkloadTrendTracker (ACWR 7d/28d)                  │
          ↓                                                │
   XAILayer                                                │
   ├── SHAP KernelExplainer (30 background samples,        │
   │   drawn from real player windows — not synthetic)     │
   ├── LLMNLGEngine (qwen2.5:14b via Ollama, 2 s timeout) │
   │   └── TemplateNLGEngine (deterministic fallback)      │
   ├── CounterfactualGenerator                             │
   └── WaterfallData (Recharts/D3 compatible)              │
          ↓                                                │
   Coach UI ── [Accept] [Override] [Add note]              │
          ↓                                                │
   FeedbackStore → RecalibrationPipeline  ─────────────────┘
          ↓
   FairnessMonitor (position, age_group, nationality)
```

---

## Production CLI (`main.py`)

`main.py` is the production entrypoint. It exposes five independent subcommands, each with machine-readable exit codes for CI gating.

```
python main.py generate [--seasons N] [--matchdays N] [--anomaly-rate F]
python main.py train    [--data-dir PATH] [--model-dir PATH] [--sessions-per-player N]
python main.py evaluate [--data-dir PATH] [--model-dir PATH] [--out PATH] [--min-auc F]
python main.py serve    [--model-dir PATH] [--min-alert-windows N] [--max-latency-ms N]
python main.py audit    [--log PATH] [--data-dir PATH] [--out PATH]
```

**Exit codes**

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Validation / data error |
| 2 | Model error (not trained, corrupt checkpoint) |
| 3 | Evaluation failed (e.g. ROC-AUC below `--min-auc`) |
| 4 | Serve / stream error |
| 5 | Audit found bias in a protected attribute group |

**`serve` input/output contract**

`serve` reads newline-delimited JSON events from stdin and writes alert JSON to stdout. Logs go to stderr. This separation lets operators pipe alert output directly to a webhook or Kafka topic.

Minimum required fields per input event:

```json
{
  "player_external_id": "p007",
  "ts": "2024-01-15T20:31:00Z",
  "speed_ms": 4.2,
  "heart_rate_bpm": 162,
  "x_pitch": 55.0,
  "y_pitch": 34.0,
  "elapsed_seconds": 2100
}
```

Alerts are gated: a recommendation fires only after `--min-alert-windows` (default 3) consecutive flagged windows with the same recommendation type. SLA violations (> 200 ms per window) are logged as warnings but do not halt the stream.

Set `JSON_LOGS=1` to switch all log output to structured JSON for log aggregation pipelines.

---

## Data Generator — v4 Decision-Agent Rewrite

`data_generator.py` is a full architectural rewrite from v3. The simulation now runs a centralized match engine at 1 Hz for all players jointly, with 15 s GPS tick output (`DT_OUT = 15 s`).

**12 realism improvements applied**

| ID | Change |
|----|--------|
| R1 | Semi-Markov tactical phases — states hold for realistic sampled durations, not per-tick transitions |
| R2 | Role-relative formation anchors — shape-relative offsets replace pairwise spring constraints |
| R3 | Ball ownership model — ball belongs to a specific player; passes include passer + receiver |
| R4 | Reaction lag / partial observability — players respond to a delayed world state snapshot |
| R5 | Angular momentum / heading physics — turning radius caps sharp direction reversals |
| R6 | Metabolic HR model — HR driven by exertion integral, not instantaneous speed |
| R7 | Split fatigue components — neuromuscular, cardiovascular, and sprint-specific tracks |
| R8 | Adversarial opponent agents — pressure fields, interception radii, defensive compactness |
| R9 | xG-driven goal causality — box entries accumulate xG; goals emerge from that process |
| R10 | Extended validation — heatmaps, compactness metrics, transition bimodality, xG stats written to `validation_report.txt` |
| R11 | Vectorised batch updates — NumPy array operations replace Python loops for all-player state |
| R12 | Sensor corruption layer — GPS dropout, coordinate jitter, HR freeze, speed quantisation |

**Output files**

```
data/
  players.csv
  sessions.csv
  events.csv
  annotations.csv
  ground_truth_labels.csv   ← anomaly labels (not in sessions.csv)
  validation_report.txt     ← R10 automated realism checks
```

---

## Model Design

### Sequence windows

Each 120-second rolling window (8 steps × 15 s/tick) produces an 8-feature matrix:

| Feature | Description |
|---|---|
| `speed_ms` | Raw GPS speed (m/s) |
| `acceleration_ms2` | Δspeed / Δt |
| `heart_rate_bpm` | HR normalised per player |
| `sprint_flag` | Binary: 1 if speed ≥ 7.0 m/s |
| `x_pitch` | Normalised pitch X [0, 100] |
| `y_pitch` | Normalised pitch Y [0, 100] |
| `distance_delta_m` | True Euclidean displacement in metres (pitch-scaled: 105 m × 68 m) |
| `hr_recovery_rate` | ΔHR / Δt in bpm/s (positive = HR rising, negative = recovering) |

`hr_recovery_rate` replaced `hr_recovery_time_s`. The old field was `None` on non-sprint ticks and a positive float only on sprint ticks — a near-binary signal that caused the LSTM to collapse onto it. `hr_recovery_rate` is computed continuously on every tick from the HR delta, giving the model a genuine physiological signal without creating a sprint-presence indicator.

`distance_delta_m` converts normalised coordinates to real metres before computing displacement. The pitch is not square: each axis uses its own scale factor (`x × 1.05 m/unit`, `y × 0.68 m/unit`).

### Anomaly scoring

Live reconstruction loss is EMA-smoothed (α = 0.25) before threshold comparison. Calibration thresholds are built from EMA-smoothed losses too, so both sides of the comparison are drawn from the same distribution.

Threshold method depends on calibration set size:

- **n ≥ 150 windows** — `quantile(EMA_losses, 0.995)`
- **n < 150 windows** — `median + 5 × MAD × 1.4826` (MAD-based, robust on small samples)

The MAD multiplier was raised from 4.0 to 5.0 to prevent confidence saturation: at 4.0 nearly every live window exceeded the threshold on small calibration sets, producing 100% confidence on all events.

### Regime-aware thresholds

Each window is classified into one of 9 behavioral regimes (territory × intensity) using only signals present in the window itself:

- **Territory** (mean `x_pitch`): `defensive` < 33 · `midfield` · `attacking` > 67
- **Intensity** (sprint flag fraction): `low` < 4% · `medium` · `high` ≥ 15%

A `RegimeAwareThresholdStore` maintains a `DynamicThresholdTracker` per regime and a global fallback. At inference, the regime-specific threshold is used when that regime is calibrated; the global tracker is the fallback. This prevents cross-regime variance (e.g. a high-press window vs a possession-retention window) from inflating the anomaly rate.

---

## LLM Natural Language Generation

Alert summaries are generated by **Qwen2.5:14b running locally via Ollama**, replacing the previous deterministic template engine as the primary NLG path.

```
LLM primary path:   qwen2.5:14b (Ollama)  ← used when available, ≤ 2 s
Template fallback:  TemplateNLGEngine     ← deterministic, < 1 ms
```

`XAILayer.explain_from_dict` always tries the LLM first. If Ollama is unavailable or the call exceeds `OLLAMA_NLG_TIMEOUT_S` (default 2 s), the template engine runs transparently so the 200 ms serve SLA is never broken.

The LLM is given the player name, alert type, model confidence, workload status, and the top contributing features with their SHAP values. It is instructed to produce 2–3 sentences in a clinical, factual tone, reference specific metric values, and conclude with a concrete time-bound action.

**Setup**

```bash
# Install Ollama: https://ollama.com
ollama pull qwen2.5:14b

# Optional: override timeout
export OLLAMA_NLG_TIMEOUT_S=3
```

The template engine remains the fallback and is still used in environments without Ollama. All explanation interfaces (`shap_values`, `counterfactual`, `waterfall_data`) are identical regardless of which NLG path was used. The `shap_method` field in the explanation output records which path fired.

---

## SHAP Explanation Output

```
Recommendation: "Substitute Player 7 — fatigue risk (confidence: 83%)"

Alert summary (qwen2.5:14b):
  "Player 7 is showing clear signs of late-match fatigue: sprint count has
   dropped 2.4 SD below their personal 28-day baseline and distance output
   in the last 120 seconds is 18% below their rolling average, while HR
   recovery rate remains elevated at +0.6 bpm/s. Recommend substitution
   before 75 minutes."

Feature contributions:
  Sprint count, last 120 s:      −2.4 SD below personal baseline   [SHAP: +0.41]
  Distance covered, last 120 s:  −18% vs rolling average           [SHAP: +0.29]
  Coach pre-match annotation:    "mild"                             [SHAP: +0.17]
  HR recovery rate:              within normal range                [SHAP: −0.08]

Counterfactual: "If sprint count were within 1.0 SD of personal baseline,
                 this flag would not trigger."

Coach action: [Accept] [Override] [Add note]
```

Explanations are generated on alert only (not every event). SHAP background is sampled from the player's own calibration windows — not synthetic data — so attributions reflect the true physiological manifold for that player.

---

## Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=players_data
DB_USER=postgres
DB_PASSWORD=your_password

# Data Sources
SPORTRADAR_API_KEY=your_key
LIVE_WS_URL=ws://your-provider:8765
MQTT_BROKER=localhost
GPS_TCP_HOST=localhost
GPS_TCP_PORT=2947

# LLM (Ollama)
OLLAMA_NLG_TIMEOUT_S=2       # seconds before falling back to template NLG

# Logging
JSON_LOGS=1                  # emit structured JSON logs to stderr (unset = human-readable)
```

---

## Production Integration

```python
from analysis.orchestrator import PlayersDataAnalysisPipeline

pipeline = PlayersDataAnalysisPipeline()

# 1. Register squad
pipeline.register_player(player_id=7, external_id="p007", name="Player 7",
                         position="CAM", age=26, age_group="Senior")

# 2. Load historical data from your DB
pipeline.load_historical_data(player_id=7, sessions_df=sessions, events_df=events)

# 3. Compute baselines + train shared LSTM model
pipeline.compute_baselines(window_days=28)
pipeline.train_all_models()

# 4. Register alert callback
def on_alert(explanation):
    # explanation.nlg_summary     — LLM or template plain English
    # explanation.shap_values     — Dict[feature -> float]
    # explanation.counterfactual  — counterfactual sentence
    # explanation.waterfall_data  — Recharts-compatible list
    # explanation.shap_method     — "kernel_sequence_space" | "magnitude_proxy"
    send_to_coach_dashboard(explanation.to_dict())

pipeline.set_alert_callback(on_alert)

# 5. Process live events (from your WebSocket handler)
#    Alerts fire after 3 consecutive windows with the same recommendation type.
#    Each window covers 8 events (120 s ÷ 15 s/tick).
pipeline.process_live_event(normalized_event, segment_index=4)

# 6. Log coach decision
pipeline.log_coach_decision(inference_id=1, player_id=7,
                            decision="override", coach_id="coach_001",
                            coach_note="Tactical — not fatigue")

# 7. Start full live ingestion (asyncio)
import asyncio
asyncio.run(pipeline.run_live(enable_ws=True, enable_mqtt=True))
```

---

## Key Design Decisions

**Personal baselines only.** The LSTM is trained with a shared backbone across the squad but uses a per-player embedding and per-player normaliser. Thresholds are calibrated exclusively on each player's own held-out windows. Squad averages are never used for anomaly decisions.

**Regime-conditioned thresholds.** A high-press window and a low-block possession window have structurally different reconstruction losses. Pooling them into one threshold raises the false-positive rate during legitimate intensity transitions. Each behavioral regime (territory × intensity) gets its own calibration distribution.

**EMA parity.** Live scores are EMA-smoothed (α = 0.25) before threshold comparison. Calibration thresholds are computed from EMA-smoothed losses too, so the distributions on both sides of the comparison are matched.

**LLM for every alert.** `LLMNLGEngine` calls qwen2.5:14b via local Ollama with a structured sports-science prompt. The template engine is the automatic fallback on timeout or connection failure. No LLM call blocks inference beyond the 2 s NLG budget, which is separate from the 200 ms LSTM inference SLA.

**SHAP for every alert.** KernelExplainer runs with 30 background samples when `shap` is installed. A magnitude-proxy fallback is used otherwise. Background samples come from real player windows, not random noise.

**Coach annotations as first-class features.** `coach_fatigue_severity` and `coach_pre_match_status_encoded` are XAI inputs alongside sensor-derived features.

**Override loop recalibrates thresholds.** `log_coach_decision()` feeds `RecalibrationPipeline`, which adjusts per-player sensitivity and rebuilds thresholds after ≥ 10 overrides or on the 7-day cadence.

**Fairness audit.** `FairnessMonitor` checks flag-rate disparity by `position`, `age_group`, and `nationality` every 7 days. Disparity > 15% across groups triggers a logged action. The `audit` CLI subcommand exits with code 5 if bias is detected, enabling CI gates.

**< 200 ms inference SLA.** Latency is measured and logged on every `process_live_event()` call. SLA violations are logged as warnings and counted in the `serve` exit summary.

---

## Known Limitations

**SHAP explains a proxy, not the LSTM.** `KernelExplainer` operates on the 15-dimensional XAI feature vector, not on the LSTM's internal reconstruction mechanism. Attributions reflect which derived features correlate with anomalousness, not which raw timesteps drove the reconstruction error. Integrated gradients applied directly to the LSTM would be more faithful.

**LLM output is not formally verified.** Qwen2.5:14b summaries are constrained by prompt engineering, not by formal output validation. The model is instructed not to invent data, but hallucination cannot be excluded. All SHAP values and counterfactuals are deterministic and auditable regardless of which NLG path fired.

**Regime classification is rule-based.** Territory and intensity bins are fixed thresholds. A learned regime detector (e.g. GMM or k-means on the LSTM latent space) would be more adaptive but requires additional training data and complicates the calibration interface.

**No match-phase conditioning.** Half-time resets, substitutions, and possession changes share the same latent manifold. Adding `elapsed_seconds` as a third regime axis would require threading match context through the calibration path, which is currently sequence-only.

**Evaluation uses synthetic distributions.** Ground truth labels are seeded by the data generator. The model may learn simulator-specific patterns rather than genuine physiological anomalies. Real-world validation against expert-labelled match data is the next required step before production deployment.

---

## References

1. Rein & Memmert (2016) — Big data and tactical analysis in elite soccer
2. Foteinakis et al. (2025) — Explainable ML for Basketball
3. Odet et al. (2024) — ML and Explainability for Sports Outcome Prediction
4. Pietraszewski et al. (2025) — AI in Sports Analytics systematic review
5. Kranzinger et al. (2025) — Explainable AI in Sports Science
6. Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions
7. Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
8. Bai et al. (2018) — An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling