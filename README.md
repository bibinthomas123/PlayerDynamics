# Players Data — IBM CIC Germany · Group 11 / 2B

**Explainable Player Pattern Analysis for Real-Time Coaching Decisions**

---

## What This Is

Production-level Python analysis engine for the Players Data HCAI project.  
No frontend. No backend API. Pure analysis + data ingestion core.

### Components Built

| Layer | Module | What It Does |
|---|---|---|
| **Data Ingestion** | `ingestion/pipeline.py` | GPS (NMEA/TCP/GPX), REST API (SportRadar), WebSocket live stream, MQTT wearables |
| **Personal Baseline** | `analysis/baseline.py` | Per-player rolling baseline (7d/28d), exponential fatigue curve fitting, workload ACWR |
| **Anomaly Detection** | `analysis/anomaly_detection.py` | Isolation Forest per player, positional drift, fatigue comparator, feature engineering |
| **XAI Layer** | `explainability/xai_layer.py` | SHAP values, waterfall chart data, template NLG, counterfactual generation |
| **Feedback Loop** | `feedback/recalibration.py` | Override logging, weekly recalibration, per-player sensitivity, bias audit |
| **Orchestrator** | `analysis/orchestrator.py` | Wires all components; `PlayersDataAnalysisPipeline` is the single production interface |
| **DB Schema** | `utils/schema.py` | Full SQLAlchemy ORM: players, sessions, events, annotations, override_logs, fairness_audit_log |
| **Config** | `config/settings.py` | All parameters via environment variables, zero hardcoding |

---

## Quick Start

```bash
# Install dependencies
pip install scikit-learn numpy pandas scipy shap aiohttp websockets

# Run the end-to-end demo
cd players_data
python demo.py
```

---

## Architecture (exactly as in proposal)

```
GPS / REST API / WebSocket / MQTT
          ↓
   IngestionPipeline
   (normalize, quality-score, sliding-window aggregation)
          ↓
   BaselineBuilder  ←──────────────────────────────────┐
   (per-player 28d rolling baseline, fatigue curve)    │
          ↓                                             │
   PatternAnalysisEngine                               │
   ├── IsolationForest (per-player, not squad avg)     │
   ├── FatigueCurveComparator                          │
   ├── PositionalDriftAnalyzer                         │
   └── WorkloadTrendTracker (ACWR 7d/28d)              │
          ↓                                             │
   XAILayer                                            │
   ├── SHAP values (KernelExplainer / proxy fallback)  │
   ├── TemplateNLG (deterministic, no LLM)             │
   ├── CounterfactualGenerator                          │
   └── WaterfallData (Recharts/D3 compatible)          │
          ↓                                             │
   Coach UI ── [Accept] [Override] [Add Note]          │
          ↓                                             │
   FeedbackStore → RecalibrationPipeline ──────────────┘
          ↓
   FairnessMonitor (position, age_group, nationality)
```

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

# 3. Compute baselines + train models
pipeline.compute_baselines(window_days=28)
pipeline.train_all_models()

# 4. Register alert callback
def on_alert(explanation):
    # explanation.nlg_summary     — plain English
    # explanation.shap_values     — Dict[feature -> float]
    # explanation.counterfactual  — counterfactual sentence
    # explanation.waterfall_data  — Recharts-compatible list
    send_to_coach_dashboard(explanation.to_dict())

pipeline.set_alert_callback(on_alert)

# 5. Process live events (from your WebSocket handler)
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

## SHAP Explanation Output (matches proposal example exactly)

```
Recommendation: "Substitute Player 7 — fatigue risk (confidence: 83%)"

Feature contributions:
  Sprint count, min 60–75:        −2.4 below personal baseline   [SHAP: +0.41]
  Distance covered, last 15 min:  −18% vs rolling average        [SHAP: +0.29]
  Coach pre-match annotation:     "mild"                          [SHAP: +0.17]
  Heart rate recovery time:       within normal range             [SHAP: −0.08]

Counterfactual: "If sprint count were within 1.0 of personal baseline,
                 this flag would not trigger."

Coach action: [Accept] [Override] [Add note]
```

---

## Key Design Decisions (matching proposal)

- **Personal baselines only** — Isolation Forest trained per player, never squad averages
- **SHAP for every output** — KernelExplainer when `shap` is installed; magnitude-proxy fallback otherwise
- **No LLM** — All text from deterministic templates. Fully reproducible and auditable
- **Coach annotations as first-class features** — `coach_fatigue_severity`, `coach_pre_match_status_encoded` are model inputs
- **Override loop retrains** — `log_coach_decision()` feeds `RecalibrationPipeline` which adjusts thresholds and per-player sensitivity
- **Fairness audit** — `FairnessMonitor` checks flag rate disparity by position, age_group, nationality; alerts at >15% disparity
- **<200ms inference SLA** — Latency is measured and logged on every `process_live_event()` call

---

## References (from proposal)

1. Rein & Memmert (2016) — Big data and tactical analysis in elite soccer
2. Foteinakis et al. (2025) — Explainable ML for Basketball
3. Odet et al. (2024) — ML and Explainability for Sports Outcome Prediction
4. Pietraszewski et al. (2025) — AI in Sports Analytics systematic review
5. Kranzinger et al. (2025) — Explainable AI in Sports Science
6. Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions
7. Liu et al. (2008) — Isolation Forest
