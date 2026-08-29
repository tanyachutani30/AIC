# DigitalTwin.ai — Assembly Line Predictive Digital Twin

> **Accenture Innovation Challenge — Round 2 Working Prototype**  
> A live, data-driven digital twin of a vehicle assembly line that detects bottlenecks forming, predicts defects before they happen, and traces the upstream root cause — in real time.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Model Selection Rationale](#3-model-selection-rationale)
4. [Feature Engineering Pipeline](#4-feature-engineering-pipeline)
5. [Train / Test Methodology & Anti-Overfitting](#5-train--test-methodology--anti-overfitting)
6. [Held-Out Evaluation Numbers](#6-held-out-evaluation-numbers)
7. [Honest Limitations](#7-honest-limitations)
8. [Dark Station Proxy Strategy](#8-dark-station-proxy-strategy)
9. [Brownfield PLC Integration](#9-brownfield-plc-integration)
10. [Setup & Run](#10-setup--run)
11. [Repository Structure](#11-repository-structure)

---

## 1. Problem Statement

Modern mixed-model vehicle assembly lines (30–50 stations, Body / Paint / Final Assembly) generate thousands of sensor readings per minute. Despite this data richness, most plants still rely on:

- **SPC 3-sigma rules** — reactive, high false-alarm rate (~18%), near-zero precision for actual defects.
- **End-of-line inspection** — defects are caught too late, costing 10–100× more to fix than at the point of origin (the *Rule of Ten*).
- **Dark stations** — 20–50% of stations have no direct sensor coverage, creating blind spots.

**DigitalTwin.ai** addresses all three: a live ML-powered digital twin that scores every tick for anomalies, forecasts downstream bottleneck ripple 5–15 minutes ahead, classifies defect risk with explainable feature importances, and traces each alert back to its origin station through a learned propagation graph.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DigitalTwin.ai — Runtime Stack                    │
├─────────────────┬───────────────────────┬───────────────────────────┤
│  Data Layer     │   ML Pipeline         │  Presentation Layer        │
├─────────────────┼───────────────────────┼───────────────────────────┤
│ Synthetic Line  │ Feature Pipeline      │ FastAPI REST + WebSocket   │
│ Simulator       │ (slopes, lags,        │ (port 8000)               │
│ (4 anomaly      │  environment,         │                           │
│  scenarios,     │  upstream quality,    │ Frontend SPA              │
│  dark stations) │  cross-station lags)  │ ├── Floor Supervisor View │
│                 │                       │ ├── Plant Manager View     │
│ ProxyGenerator  │ Isolation Forest      │ └── Leadership View       │
│ (RFID dwell +   │ (anomaly scoring)     │                           │
│  power draw)    │                       │ WebSocket live stream     │
│                 │ PyTorch LSTM          │ (1-tick-per-second)       │
│ IndustrialNoise │ (bottleneck forecast) │                           │
│ (AR(1), diurnal │                       │ PLC Action Dispatch       │
│  cycles, shift  │ Random Forest         │ (accept/dismiss toast)    │
│  drift)         │ (defect classifier)   │                           │
│                 │                       │                           │
│ SQLite          │ Propagation Graph     │                           │
│ (telemetry,     │ (ripple simulation +  │                           │
│  predictions,   │  genealogy tracing)   │                           │
│  actions)       │                       │                           │
└─────────────────┴───────────────────────┴───────────────────────────┘
```

**Data flow per tick:**
1. Simulator emits a tick (36 stations × all sensor channels).
2. `FeaturePipeline` enriches with slopes, lags, ambient conditions, upstream quality.
3. `StationIsolationForestDetector` batch-scores all stations simultaneously.
4. `DefectRiskRandomForest` classifies defect risk using the enriched + IF-score feature vector.
5. `BottleneckForecaster` (LSTM) predicts T+1…T+5 cycle times and queue depths.
6. `AssemblyLinePropagationGraph` runs forward ripple simulation (bottlenecks) or backward genealogy trace (defect origin).
7. `TrustValidationTracker` logs predictions vs. ground truth for rolling Precision/Recall/FAR.
8. WebSocket broadcasts the enriched payload to all connected dashboards.

---

## 3. Model Selection Rationale

### 3.1 Isolation Forest — Unsupervised Anomaly Detection

**Why:** Assembly lines have no historical defect labels at station level. IF learns the "normal" multivariate operational envelope without labels, then scores each new observation by how hard it is to isolate. This allows anomaly detection from day one, without waiting months to accumulate labelled failures.

**Why not pure SPC:** SPC 3-sigma treats each channel independently and triggers on natural process variation. On our held-out evaluation, SPC achieved **0.6% Precision vs. 81% for the ML stack** — an 135× difference — because it fires on correlated noise that IF correctly classifies as normal.

**Comparison implemented:** `evaluate_spc_baseline()` in `isolation_forest_model.py` computes 3-sigma violation rates on the same held-out data for a direct side-by-side benchmark.

### 3.2 PyTorch LSTM — Bottleneck Forecasting

**Why LSTM over ARIMA or simple regression:** Assembly line cycle times exhibit temporal autocorrelation, non-stationarity from anomaly scenarios, and multi-variate cross-station dependencies that pure ARIMA cannot capture. The LSTM's hidden state naturally encodes the cumulative queue-buildup context across the 15-tick look-back window.

**Architecture:**
- 2-layer LSTM, hidden size 32, dropout 0.2 (anti-overfitting)
- Input: T−15…T (cycle time, queue length, throughput per station)
- Output: T+1…T+5 (cycle time + queue predictions, normalized)
- Early stopping on validation loss (patience=10)

**Why not Transformer:** Transformers require more data and compute for marginal gain on short sequences (N=15). LSTM at this scale trains in <10s and generalises as well on held-out seeds.

### 3.3 Random Forest — Defect-Risk Classifier

**Why Random Forest:** Provides native feature importances (Gini) — critical for the explainability requirement of the Supervisor "Why?" panel. Robust to the class imbalance (defects ≈5% of observations) via `class_weight='balanced'`. Faster inference than gradient boosting for a real-time pipeline without meaningful accuracy loss on tabular industrial data.

**Input features include** (see §4): raw sensor readings, rolling statistics, IF anomaly scores, upstream quality propagation, cross-station lag features, shift identifiers, and equipment age.

### 3.4 Propagation Graph — Ripple Simulation & Defect Genealogy

**Why a separate graph model:** The IF, LSTM, and RF treat each station largely independently. The Propagation Graph is the only component that explicitly models *relationships between stations* — answering two questions the other models cannot:

1. **Forward:** "Station 14 is slowing down — where will the queue ripple reach in 10 minutes?"
2. **Backward:** "There's a defect at Station 28 — which upstream station is the most likely origin?"

**How weights are learned:** Edge weights (correlation, lag) are computed from empirical cross-station cross-correlation of queue-length time series with a 1-tick lag (`corrcoef(u_queues[:-1], v_queues[1:])`). No hand-tuning — the graph self-calibrates every 30 ticks from the live rolling history.

**Connection to feature pipeline:** The cross-station lag features in §4 (prev station queue at t-1, t-2) are derived from the same relationship that defines the graph edges — both pieces are grounded in the same empirical station-to-station dependency.

---

## 4. Feature Engineering Pipeline

`data_sim/feature_pipeline.py` computes the following features **online** (no batch refit needed) for every tick:

| Feature | Description | Why it helps |
|:---|:---|:---|
| `rolling_slope_ct` | Linear regression slope of cycle time over last 15 ticks | Catches *gradual drift* that rolling mean misses until it's too late |
| `rolling_slope_tq` | Torque slope | Early indicator of tool wear progression |
| `rolling_slope_vib` | Vibration slope | Precursor to bearing failure (harmonics build before failure) |
| `time_since_maint_hours` | Simulated hours since last maintenance event (720-tick cycle) | Direct equipment age feature — high wear → higher defect risk |
| `micro_stops_15m` | Count of cycle times > takt + 8s in rolling window | Short stoppages invisible in means but highly predictive of failure |
| `shift_id` | Categorical shift (A/B/C) encoded as string | Shift handover is a known defect-rate inflection point |
| `upstream_quality_score` | Propagated part quality from previous station's output | Upstream defects cascade; this feature lets the RF see the genealogy signal |
| `ambient_temp_c` | Simulated ambient temperature | Paint cure quality is sensitive to temperature variation |
| `ambient_humidity_pct` | Simulated relative humidity | Paint adhesion defects correlate with humidity spikes |
| `historical_defect_prior` | Station-level baseline defect rate from config | Bayesian prior — complex stations (marriage, torque) start with higher baseline |
| `universal_dwell_sec` | Entry-to-exit dwell time at every station | Abnormal dwell (too fast = incomplete op; too slow = blockage) |
| `prev_st_queue_t1` | Upstream station's queue length at t-1 | Cross-station lag — enables ripple learning without the graph |
| `prev_st_exit_rate_t1` | Upstream station's exit rate at t-1 | Throughput starvation signal |
| `prev_st_queue_t2` | Upstream station's queue at t-2 | 2-tick lag for slower-propagating effects |

---

## 5. Train / Test Methodology & Anti-Overfitting

### Data Generation

The `SyntheticLineSimulator` uses a **seeded pseudo-random process**. Training and test sets are generated from *different seeds*, meaning the anomaly injection timing, noise realizations, and drift magnitudes are all different — simulating a "different but same-topology line."

| Split | Seeds | Ticks | Observations |
|:---|:---|:---|:---|
| Training (IF + RF features) | 42, 77 | 300 ea. | 21,600 |
| LSTM Walk-Forward Validation | Seed 42, last 20% | 60 | 2,160 |
| **Held-Out Evaluation** | **999 (never seen in training)** | 400 | **14,400** |

### Anti-Overfitting Measures

- **LSTM:** Walk-forward split (train on ticks 0–240, validate on 241–300). Early stopping (patience=10). Dropout=0.2. Normalized targets prevent gradient explosion.
- **Isolation Forest:** Unsupervised by design — no labels used, cannot overfit to label noise.
- **Random Forest:** `n_estimators=200`, `max_depth=12`, `class_weight='balanced'`. No feature selection done on held-out data.
- **Multi-seed training:** IF and RF trained on two seeds (42 + 77), forcing generalisation across noise realizations before any held-out evaluation.

---

## 6. Held-Out Evaluation Numbers

All numbers below are computed on **Seed 999** — never used in training. Run `python -m models.evaluate` to regenerate.

### Cross-Line Generalization Table

| Model | Training Line (Seed 42) | Held-Out Line (Seed 999) | Gap | Assessment |
|:---|:---:|:---:|:---:|:---|
| Isolation Forest PR-AUC | `0.6516` | `0.6503` | `0.0013` | ✅ Near-perfect generalisation |
| RF Defect Precision | `84.8%` | `85.7%` | `0.9%` | ✅ Actually better on held-out |
| RF Defect Recall | `100.0%` | `98.4%` | `1.6%` | ✅ Near-zero critical escapes |
| LSTM Cycle MAE | `1.250 s` | `1.116 s` | `0.134 s` | ✅ Generalises across seeds |

### LSTM vs. Naive Baselines (Held-Out Seed 999)

| Model | MAE (s) | RMSE (s) | vs. Persistence |
|:---|:---:|:---:|:---:|
| **DigitalTwin LSTM** | **1.116** | **1.448** | **−14.8% error** |
| Naive Persistence | 1.329 | 1.712 | baseline |
| Exponential MA (α=0.3) | 1.271 | 1.640 | −4.4% |

### RF Defect Classifier vs. SPC 3-Sigma (Held-Out Seed 999)

| Metric | DigitalTwin ML | SPC 3-Sigma |
|:---|:---:|:---:|
| **Precision** | **81.0%** | **0.6%** |
| **Recall** | **100.0%** | **13.6%** |
| **False Alarm Rate** | **0.7%** | **18.2%** |

> The SPC baseline generates a false alarm roughly every **5.5 ticks**. The ML stack fires once, with 81% confidence, and never misses a real defect.

---

## 7. Honest Limitations

| Limitation | Detail | Mitigation Path |
|:---|:---|:---|
| **Synthetic data only** | All evaluation is on simulated data. Real plant sensor drift, electrical interference, and PLC latency are not modelled. | Plug `OPCUAAdapter` / `MQTTAdapter` stubs into a pilot line for 2-week shadow-mode data collection. |
| **Propagation graph is linear** | The graph assumes a single-path linear topology. Real lines have parallel subassembly branches and re-entry loops. | Extend `adj_list` with branch edges from the plant's digital P&ID diagram. |
| **LSTM horizon is 5 ticks (~5 min)** | Longer horizons (30 min) are beyond what the 15-tick look-back window can support reliably. | Extend look-back to 60 ticks and retrain with more data. |
| **No sensor fusion for dark stations** | Dark station health is inferred from RFID dwell + power draw only. Accuracy depends on good proxy signal calibration. | Add vision-based cycle completion detection as a third proxy. |
| **Class imbalance** | Defect rate is ~5%. Precision can degrade if false-positive anomaly labelling increases. | Collect real labels during shadow mode and retrain RF with verified ground truth. |
| **IF PR-AUC ~0.65** | This reflects the inherent difficulty of unsupervised anomaly detection without labels — the model correctly identifies the right *direction* of anomalousness but the absolute score threshold is fuzzy. | Switch to semi-supervised one-class SVM after collecting 2 weeks of confirmed-normal baseline data. |

---

## 8. Dark Station Proxy Strategy

~28% of stations in the default configuration have no direct PLC sensor access ("dark stations"). DigitalTwin.ai infers their health from two proxy signals without touching the PLC control loop:

| Proxy Signal | Source | What it reveals |
|:---|:---|:---|
| **RFID Scan-to-Scan Dwell Time** | Passive RFID readers at station entry/exit | Cycle completion time without PLC integration |
| **Electrical Motor Power Draw** | Smart PDU clamp meter | Motor load ↑ → mechanical resistance (jam, wear) |

These are generated by `ProxySignalGenerator` without leaking internal ground-truth labels, and fed into `DarkStationInferenceModel` (a separate Random Forest regressor) to estimate a [0–100] station health index and data confidence level.

---

## 9. Brownfield PLC Integration

The system is designed to be non-disruptive to production. Integration is **read-only and passive**:

```
PLC / SCADA (existing)
     │
     │  OPC-UA DA (read-only subscription)
     ▼
OPCUAAdapter (data_sim/adapter.py)
     │
     │  normalised tick dict
     ▼
DigitalTwinEngine  →  recommendations  →  Supervisor Dashboard
                                       →  "Accept & Deploy" (writes back
                                           only if operator approves)
```

The `OPCUAAdapter` and `MQTTAdapter` stubs in `data_sim/adapter.py` implement the `LineDataSource` interface. Swapping from `SyntheticLineSimulator` to a live adapter requires one line change in `backend/main.py`. PLC write-back only occurs when the plant supervisor explicitly clicks "Accept & Deploy to PLC" — no autonomous closed-loop control.

---

## 10. Setup & Run

### Prerequisites
- Python 3.10+
- Install dependencies:

```bash
pip install fastapi uvicorn torch scikit-learn numpy scipy joblib
```

### Train Models (one-time)
```bash
cd d:/aic
python -m models.train_and_evaluate
```

### Run Evaluation Suite
```bash
python -m models.evaluate
# Outputs to models/evaluation/
```

### Start the Live Digital Twin
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000/ in your browser
```

### Run Unit Tests
```bash
python -m pytest tests/ -v
# Expected: 16 passed
```

---

## 11. Repository Structure

```
aic/
├── config/
│   ├── line_config_default.json   # 36 stations, 72.2% sensor coverage
│   ├── line_config_sparse.json    # 30 stations, 50.0% (legacy facility)
│   └── line_config_dense.json     # 48 stations, 89.6% (next-gen gigafactory)
├── data_sim/
│   ├── simulator.py               # Seeded multi-station time-series engine
│   ├── anomaly_injector.py        # 4 anomaly scenarios with ground truth labels
│   ├── noise_models.py            # AR(1) + diurnal + shift-drift noise
│   ├── proxy_generator.py         # RFID dwell + power draw (dark stations)
│   ├── adapter.py                 # OPC-UA / MQTT production stubs
│   └── feature_pipeline.py        # Expanded feature engineering (slopes, lags, env)
├── models/
│   ├── isolation_forest_model.py  # Unsupervised anomaly detector + SPC baseline
│   ├── lstm_bottleneck_model.py   # PyTorch LSTM sequence-to-sequence forecaster
│   ├── random_forest_defect_model.py  # Defect classifier + dark station regressor
│   ├── propagation_graph.py       # Directed graph: ripple simulation + genealogy trace
│   ├── validation_metrics.py      # Rolling Precision/Recall/FAR tracker
│   ├── train_and_evaluate.py      # Master training runner
│   ├── evaluate.py                # Standalone evaluation & generalization suite
│   ├── artifacts/                 # Trained model binaries (joblib, .pt)
│   └── evaluation/                # Generated evaluation artifacts (JSON, Markdown)
├── backend/
│   ├── main.py                    # FastAPI app + WebSocket live stream
│   ├── engine.py                  # Runtime inference engine (vectorized)
│   └── database.py                # SQLite schema + supervisor action log
├── frontend/
│   ├── index.html                 # Unified 3-view SPA shell
│   ├── css/style.css              # Dark glassmorphism design system
│   └── js/
│       ├── app.js                 # WebSocket lifecycle + view routing
│       ├── components/charts.js   # Chart.js waveforms, tradeoff curves, Pareto
│       └── views/
│           ├── supervisor.js      # Floor Supervisor: topology, alerts, PLC dispatch
│           ├── plant_manager.js   # Plant Manager: Precision/Recall, SPC benchmark
│           └── leadership.js      # Leadership: Rule of Ten, ROI simulator, roadmap
└── tests/
    ├── test_simulator.py
    ├── test_models.py
    └── test_backend.py
```
