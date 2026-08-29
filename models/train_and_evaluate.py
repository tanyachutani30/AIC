"""
DigitalTwin.ai - Multi-Line Training & Held-Out Evaluation Pipeline
Vectorized, high-speed multi-line training runner and held-out benchmark evaluator.
"""

import os
import json
import subprocess
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from data_sim.simulator import SyntheticLineSimulator
from models.isolation_forest_model import StationIsolationForestDetector
from models.lstm_bottleneck_model import BottleneckForecaster
from models.random_forest_defect_model import DefectRiskRandomForest, DarkStationInferenceModel
from models.propagation_graph import AssemblyLinePropagationGraph
from models.validation_metrics import TrustValidationTracker


def run_training_and_evaluation() -> Dict[str, Any]:
    print("=" * 60)
    print("DigitalTwin.ai — End-to-End Model Training & Validation")
    print("=" * 60)

    artifacts_dir = Path("models/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Generate Training Data from Multiple Seeds & Topologies
    # -------------------------------------------------------------
    print("\n[Step 1/5] Generating training time-series across multiple line seeds...")
    
    # Line 1: Default 36-station topology (seed 42, 300 ticks)
    sim1 = SyntheticLineSimulator("config/line_config_default.json", seed=42)
    ticks_line1 = [sim1.stream_next_tick() for _ in range(300)]
    
    # Line 2: Sparse 30-station topology (seed 77, 300 ticks)
    sim2 = SyntheticLineSimulator("config/line_config_sparse.json", seed=77)
    ticks_line2 = [sim2.stream_next_tick() for _ in range(300)]

    all_ticks = ticks_line1 + ticks_line2
    all_train_records = []
    for t in all_ticks:
        for st in t["stations"]:
            st_rec = dict(st)
            st_rec["tick"] = t["tick"]
            st_rec["line_id"] = t["line_id"]
            all_train_records.append(st_rec)

    print(f"  -> Generated {len(all_train_records)} multi-station telemetry observations.")

    # -------------------------------------------------------------
    # 2. Train Isolation Forest Anomaly Detector
    # -------------------------------------------------------------
    print("\n[Step 2/5] Training Isolation Forest on sensor-rich stations...")
    with open("config/line_config_default.json", "r") as f:
        cfg = json.load(f)
        hp = cfg.get("model_hyperparameters", {})
        if_hp = hp.get("isolation_forest", {})
        rf_hp = hp.get("random_forest", {})
        
    iforest = StationIsolationForestDetector(
        contamination=if_hp.get("contamination"), 
        n_estimators=if_hp.get("n_estimators"), 
        random_state=42
    )
    sensor_rich_train = [r for r in all_train_records if r.get("sensor_rich", False)]
    iforest.fit(sensor_rich_train)
    iforest.save(str(artifacts_dir / "isolation_forest.joblib"))
    print("  -> Isolation Forest trained & saved.")

    # -------------------------------------------------------------
    # 3. Train PyTorch LSTM Bottleneck Forecaster
    # -------------------------------------------------------------
    print("\n[Step 3/5] Training PyTorch LSTM Bottleneck Forecaster (walk-forward split)...")
    lstm_forecaster = BottleneckForecaster(seq_len=15, horizon=5, hidden_dim=32)
    
    st14_train = [t["stations"][13] for t in ticks_line1[:220]]
    st14_val = [t["stations"][13] for t in ticks_line1[220:]]
    
    loss_history = lstm_forecaster.train_model(st14_train, st14_val, epochs=25, batch_size=32, lr=0.005)
    lstm_forecaster.save(str(artifacts_dir / "bottleneck_lstm.pt"))
    print(f"  -> LSTM trained. Final Val Loss: {loss_history['val_loss'][-1]:.4f}")

    # -------------------------------------------------------------
    # 4. Train Random Forest Defect Classifier & Dark-Station Model
    # -------------------------------------------------------------
    print("\n[Step 4/5] Training Random Forest Defect Classifier & Dark-Station Model...")
    rf_defect = DefectRiskRandomForest(
        n_estimators=rf_hp.get("n_estimators"),
        max_depth=rf_hp.get("max_depth"),
        random_state=42
    )
    rf_dark = DarkStationInferenceModel(random_state=42)

    all_scores = iforest.batch_score_records(all_train_records)
    
    X_rf, y_rf = [], []
    idx_counter = 0
    for t in all_ticks:
        st_count = len(t["stations"])
        t_scores = all_scores[idx_counter : idx_counter + st_count]
        idx_counter += st_count

        for i, st in enumerate(t["stations"]):
            st_score = float(t_scores[i])
            up_score = float(t_scores[i - 1]) if i > 0 else 0.0
            
            feat_vec = rf_defect.build_feature_vector(
                record=st,
                st_iforest_score=st_score,
                upstream_iforest_score=up_score,
                nominal_ct=float(st.get("nominal_cycle_time", 60.0)),
                tool_age_hours=float(t["tick"] * 0.2 + 50.0)
            )
            is_defect = 1 if st["ground_truth"]["is_defect"] else 0
            X_rf.append(feat_vec)
            y_rf.append(is_defect)

    rf_defect.fit(np.array(X_rf, dtype=np.float32), np.array(y_rf, dtype=np.int32))
    rf_defect.save(str(artifacts_dir / "defect_rf.joblib"))
    print(f"  -> Defect Classifier trained. Top Feature Importances: {rf_defect.feature_importances_}")

    # Train Dark Station Model
    X_dark, y_dark = [], []
    for r in all_train_records:
        dwell = float(r.get("rfid_dwell_time_sec", 60.0))
        pwr = float(r.get("power_kw", 3.0))
        nominal = 60.0
        amb_noise = float(r.get("ambient_noise_db", 75.0))
        opt_ct = float(r.get("optical_estimated_cycle_time", 60.0))
        is_anom = r["ground_truth"]["is_anomaly"]
        sev = float(r["ground_truth"]["severity"])
        true_health = 100.0 - (sev * 70.0 if is_anom else 0.0)
        
        feat_dark = rf_dark.extract_proxy_features(dwell, pwr, nominal, amb_noise, opt_ct)
        X_dark.append(feat_dark[0])
        y_dark.append(true_health)

    rf_dark.fit(np.array(X_dark, dtype=np.float32), np.array(y_dark, dtype=np.float32))
    rf_dark.save(str(artifacts_dir / "dark_station_rf.joblib"))
    print("  -> Dark Station Inference Regressor trained & saved.")

    # -------------------------------------------------------------
    # 4.5 Train Propagation Graph Topology
    # -------------------------------------------------------------
    print("\n[Step 4.5] Calibrating Propagation Graph Topology...")
    prop_graph = AssemblyLinePropagationGraph(station_count=36)
    
    station_ts_map = {}
    for r in all_train_records:
        st_id = r["station_id"]
        if st_id not in station_ts_map:
            station_ts_map[st_id] = []
        station_ts_map[st_id].append(r)
        
    prop_graph.calibrate_weights_from_telemetry(station_ts_map)
    prop_graph.save(str(artifacts_dir / "propagation_graph.joblib"))
    print("  -> Propagation Graph calibrated & saved.")

    # -------------------------------------------------------------
    # 5. Held-Out Generalization Evaluation (Unseen Seed 999)
    # -------------------------------------------------------------
    print("\n[Step 5/5] Evaluating on Held-Out Unseen Line Instance (Seed 999)...")
    held_out_sim = SyntheticLineSimulator("config/line_config_default.json", seed=999)
    test_ticks = [held_out_sim.stream_next_tick() for _ in range(300)]

    tracker = TrustValidationTracker(window_size=600, default_alert_threshold=0.50)

    # 5a. LSTM vs Naive Baselines
    test_st14_series = [t["stations"][13] for t in test_ticks]
    lstm_eval = lstm_forecaster.evaluate_baselines_vs_lstm(test_st14_series)
    print(f"  -> LSTM Bottleneck Forecaster Evaluation on Held-Out Line:")
    print(f"     LSTM MAE: {lstm_eval['lstm']['mae']:.3f} s  |  Persistence MAE: {lstm_eval['naive_persistence']['mae']:.3f} s  |  EMA MAE: {lstm_eval['ema_baseline']['mae']:.3f} s")
    print(f"     Forecast Error Reduction vs Persistence: {lstm_eval['mae_reduction_pct']}%")

    # 5b. Vectorized Defect Classifier and Isolation Forest vs SPC Baseline
    all_test_records = [st for t in test_ticks for st in t["stations"]]
    test_scores = iforest.batch_score_records(all_test_records)
    
    X_test_rf = []
    y_test_defect = []
    spc_flags = []
    timestamps = []

    test_idx = 0
    for t in test_ticks:
        st_count = len(t["stations"])
        t_scores = test_scores[test_idx : test_idx + st_count]
        test_idx += st_count

        for i, st in enumerate(t["stations"]):
            st_score = float(t_scores[i])
            up_score = float(t_scores[i - 1]) if i > 0 else 0.0
            
            feat_vec = rf_defect.build_feature_vector(
                record=st,
                st_iforest_score=st_score,
                upstream_iforest_score=up_score,
                nominal_ct=float(st.get("nominal_cycle_time", 60.0)),
                tool_age_hours=float(t["tick"] * 0.2 + 50.0)
            )
            X_test_rf.append(feat_vec)
            is_defect = bool(st["ground_truth"]["is_defect"])
            y_test_defect.append(is_defect)
            timestamps.append(t["timestamp"])

            # SPC baseline
            is_spc, _ = iforest.evaluate_spc_baseline(st)
            spc_flags.append(is_spc)

    X_test_rf_mat = np.array(X_test_rf, dtype=np.float32)
    test_probs = rf_defect.model.predict_proba(X_test_rf_mat)[:, 1]

    # Log to tracker
    for prob, is_def, ts in zip(test_probs, y_test_defect, timestamps):
        tracker.log_defect_prediction(float(prob), is_def, ts)

    # Compute SPC metrics
    spc_tp = sum(1 for f, d in zip(spc_flags, y_test_defect) if f and d)
    spc_fp = sum(1 for f, d in zip(spc_flags, y_test_defect) if f and not d)
    spc_tn = sum(1 for f, d in zip(spc_flags, y_test_defect) if not f and not d)
    spc_fn = sum(1 for f, d in zip(spc_flags, y_test_defect) if not f and d)

    ml_metrics = tracker.get_current_metrics()
    spc_precision = spc_tp / (spc_tp + spc_fp) if (spc_tp + spc_fp) > 0 else 0.0
    spc_recall = spc_tp / (spc_tp + spc_fn) if (spc_tp + spc_fn) > 0 else 0.0
    spc_far = spc_fp / (spc_fp + spc_tn) if (spc_fp + spc_tn) > 0 else 0.0

    print(f"\n  -> Model vs SPC Baseline Comparison on Held-Out Test Line:")
    print(f"     DigitalTwin.ai ML Defect Precision: {ml_metrics['defect_classifier']['precision'] * 100:.1f}% | Recall: {ml_metrics['defect_classifier']['recall'] * 100:.1f}% | FAR: {ml_metrics['defect_classifier']['false_alarm_rate'] * 100:.1f}%")
    print(f"     Naive SPC 3-Sigma  Defect Precision: {spc_precision * 100:.1f}% | Recall: {spc_recall * 100:.1f}% | FAR: {spc_far * 100:.1f}%")

    tradeoff_curve = tracker.get_threshold_tradeoff_curve(task_type="defect")

    # Generate Full Evaluation Report
    report = {
        "evaluation_dataset": "Held-Out Test Line Instance Alpha (Seed 999, 36 Stations, 300 Ticks)",
        "models_summary": {
            "isolation_forest": {
                "role": "Unsupervised Multivariate Anomaly Detection",
                "training_labels_required": False,
                "n_estimators": 120,
                "contamination": 0.06
            },
            "lstm_forecaster": {
                "role": "Short-Horizon Bottleneck Forecaster (Sequence-to-Sequence)",
                "input_sequence_len": 15,
                "forecast_horizon_ticks": 5,
                "loss_curves": loss_history,
                "held_out_benchmark": lstm_eval
            },
            "random_forest_defect_classifier": {
                "role": "Defect-Risk Probability & Explainability",
                "feature_importances": rf_defect.feature_importances_,
                "held_out_metrics": ml_metrics["defect_classifier"]
            },
            "spc_baseline_benchmark": {
                "role": "Traditional Univariate 3-Sigma Control Limits",
                "precision": round(spc_precision, 3),
                "recall": round(spc_recall, 3),
                "false_alarm_rate": round(spc_far, 3)
            },
            "dark_station_inference": {
                "role": "Proxy Signal (RFID Dwell + Power) Health Estimator",
                "proxy_signals": ["RFID Scan-to-Scan Dwell Time", "Active Electric Motor Power Draw"]
            }
        },
        "threshold_tradeoff_curve": tradeoff_curve
    }

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_hash = "unknown"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"evaluation_report_{git_hash}_{timestamp}.json"
    report_path = artifacts_dir / report_filename
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    latest_path = artifacts_dir / "evaluation_latest.json"
    shutil.copy(report_path, latest_path)

    print(f"\n[Done] Evaluation report exported to: {report_path}")
    print(f"       and updated latest link at: {latest_path}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    run_training_and_evaluation()
