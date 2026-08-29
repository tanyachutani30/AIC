"""
DigitalTwin.ai - Real-Time Digital Twin Inference Engine
High-performance vectorized real-time ML scoring, bottleneck forecasting,
explainable root-cause attribution, and prescriptive recommendations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

from models.isolation_forest_model import StationIsolationForestDetector
from models.lstm_bottleneck_model import BottleneckForecaster
from models.random_forest_defect_model import DefectRiskRandomForest, DarkStationInferenceModel
from models.validation_metrics import TrustValidationTracker
from models.propagation_graph import AssemblyLinePropagationGraph
from data_sim.feature_pipeline import FeaturePipeline


class DigitalTwinEngine:
    """
    Core runtime engine executing real-time ML scoring, bottleneck forecasting,
    explainable root-cause attribution, and prescriptive recommendations.
    """

    def __init__(self, artifacts_dir: str = "models/artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.station_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.alert_threshold = 0.50
        self.tick_num = 0
        self._ambient_temp = 22.0
        self._ambient_humidity = 55.0
        self._shift_id = "SHIFT_A"

        self.iforest = StationIsolationForestDetector()
        self.lstm = BottleneckForecaster()
        self.rf_defect = DefectRiskRandomForest()
        self.rf_dark = DarkStationInferenceModel()
        self.validation_tracker = TrustValidationTracker(window_size=600, default_alert_threshold=self.alert_threshold)
        self.feature_pipeline = FeaturePipeline(window_size=15)
        self.prop_graph = AssemblyLinePropagationGraph(station_count=36)

        self._load_artifacts_if_present()

    def _load_artifacts_if_present(self) -> None:
        if (self.artifacts_dir / "isolation_forest.joblib").exists():
            try:
                self.iforest.load(str(self.artifacts_dir / "isolation_forest.joblib"))
                self.lstm.load(str(self.artifacts_dir / "bottleneck_lstm.pt"))
                self.rf_defect.load(str(self.artifacts_dir / "defect_rf.joblib"))
                self.rf_dark.load(str(self.artifacts_dir / "dark_station_rf.joblib"))
                print("[DigitalTwinEngine] All ML artifacts loaded successfully.")
            except Exception as e:
                print(f"[DigitalTwinEngine] Warning loading artifacts: {e}.")

    def set_alert_threshold(self, threshold: float) -> None:
        self.alert_threshold = float(threshold)
        self.validation_tracker.set_alert_threshold(self.alert_threshold)

    def process_raw_tick(self, raw_tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-performance vectorized enrichment of all stations in tick.
        """
        enriched_stations = []
        tick_num = raw_tick["tick"]
        self.tick_num = tick_num
        timestamp = raw_tick["timestamp"]
        line_id = raw_tick.get("line_id", "LINE_01_DEFAULT")
        stations = raw_tick["stations"]
        num_stations = len(stations)

        # Derive simulated shift & ambient from tick number
        shift_names = ["SHIFT_A", "SHIFT_B", "SHIFT_C"]
        self._shift_id = shift_names[(tick_num // 480) % 3]
        self._ambient_temp = 22.0 + 3.0 * np.sin(tick_num * 0.015)
        self._ambient_humidity = 55.0 + 10.0 * np.cos(tick_num * 0.01)

        # Apply expanded feature pipeline (slopes, lags, environment, upstream quality)
        stations = self.feature_pipeline.extract_enriched_features(
            current_tick_records=stations,
            tick_num=tick_num,
            ambient_temp=float(self._ambient_temp),
            ambient_humidity=float(self._ambient_humidity),
            shift_id=self._shift_id
        )

        # Update history with enriched records
        for st in stations:
            self.station_history[st["station_id"]].append(st)

        # Periodically calibrate propagation graph from 30-tick history
        if tick_num % 30 == 0 and tick_num > 30:
            st_ts_map = {sid: list(hist) for sid, hist in self.station_history.items()}
            self.prop_graph.calibrate_weights_from_telemetry(st_ts_map)

        # 1. Batch score Isolation Forest anomaly scores
        if self.iforest.is_fitted:
            anomaly_scores = self.iforest.batch_score_records(stations)
        else:
            anomaly_scores = np.zeros(num_stations, dtype=np.float32)

        # 2. Build feature matrix for Defect Classifier
        X_feats = []
        for idx, st in enumerate(stations):
            st_score = float(anomaly_scores[idx])
            up_score = float(anomaly_scores[idx - 1]) if idx > 0 else 0.0
            nominal_ct = float(st.get("nominal_cycle_time", 60.0))
            
            vec = self.rf_defect.build_feature_vector(
                record=st,
                st_iforest_score=st_score,
                upstream_iforest_score=up_score,
                nominal_ct=nominal_ct,
                tool_age_hours=float(tick_num * 0.2 + 50.0)
            )
            X_feats.append(vec)

        # Batch predict defect probabilities
        if self.rf_defect.is_fitted:
            X_mat = np.array(X_feats, dtype=np.float32)
            defect_probs = self.rf_defect.model.predict_proba(X_mat)[:, 1]
        else:
            defect_probs = np.full(num_stations, 0.05, dtype=np.float32)

        active_alerts = []

        for idx, st in enumerate(stations):
            st_id = st["station_id"]
            is_sensor_rich = st.get("sensor_rich", False)
            nominal_ct = float(st.get("nominal_cycle_time", 60.0))
            anomaly_score = float(anomaly_scores[idx])
            defect_prob = float(defect_probs[idx])

            # 3. LSTM Bottleneck Forecasting (on selected stations or when queue builds)
            history_seq = list(self.station_history[st_id])
            if self.lstm.model is not None:
                lstm_res = self.lstm.predict_forecast(history_seq)
            else:
                lstm_res = {
                    "forecast_cycle_times": [st["cycle_time"]] * 5,
                    "forecast_queues": [st["queue_len"]] * 5,
                    "is_bottleneck_predicted": st["queue_len"] >= 3,
                    "max_forecast_queue": st["queue_len"],
                    "lead_time_ticks": 5
                }

            # 4. Dark Station Proxy Health & Confidence
            dark_res = self.rf_dark.infer_health_and_confidence(
                dwell_time=float(st.get("rfid_dwell_time_sec", st["cycle_time"])),
                power_kw=float(st.get("power_kw", 3.0)),
                nominal_ct=nominal_ct,
                ambient_noise_db=float(st.get("ambient_noise_db", 75.0)),
                optical_estimated_cycle_time=float(st.get("optical_estimated_cycle_time", nominal_ct)),
                ambient_noise_rolling_std=float(st.get("noise_rolling_std", 0.0))
            )

            # 5. Alert Triggering & Prescriptive Recommendations
            is_defect_alert = (defect_prob >= self.alert_threshold and anomaly_score >= 35.0)
            is_bottleneck_alert = (lstm_res["is_bottleneck_predicted"] and st["queue_len"] >= 3)
            is_alert = is_defect_alert or is_bottleneck_alert

            recommended_action = None
            if is_bottleneck_alert:
                donor_st = min(stations, key=lambda s: s["queue_len"])
                recommended_action = {
                    "action_title": f"Execute Dynamic Line Balance: Station {donor_st['station_id']} → Station {st_id}",
                    "action_type": "OPTIMAL_REROUTE",
                    "target_station_id": st_id,
                    "details": f"Forecast predicts queue growing to {lstm_res['max_forecast_queue']} units. Shift 1 technician from Station {donor_st['station_id']} (queue: {donor_st['queue_len']}).",
                    "impact": f"Clear backlog {lstm_res['lead_time_ticks']} min before takt-time violation"
                }
            elif is_defect_alert:
                if anomaly_score > 55.0 and st.get("vibration_rms", 0) and st["vibration_rms"] > 2.5:
                    recommended_action = {
                        "action_title": f"Schedule Spindle Bearing Replacement (Station {st_id})",
                        "action_type": "PREVENTIVE_MAINTENANCE",
                        "target_station_id": st_id,
                        "details": f"Vibration harmonics ({st['vibration_rms']:.2f}g) indicate mechanical spindle fatigue.",
                        "impact": "Prevent $42,000 unplanned tooling breakdown during production shift"
                    }
                elif idx > 0 and float(anomaly_scores[idx - 1]) > 35.0:
                    recommended_action = {
                        "action_title": f"Divert Unit to In-Line Touchup (Upstream Station {st_id - 1} Propagation)",
                        "action_type": "LOCAL_REWORK_DIVERTER",
                        "target_station_id": st_id,
                        "details": f"Upstream Station {st_id - 1} dimensional clamp drift detected. Divert to Station {st_id} local rework bay.",
                        "impact": "Saves $90/unit scrap penalty (Defeating Rule of Ten at Assembly Level)"
                    }
                else:
                    recommended_action = {
                        "action_title": f"Calibrate Torque Tooling & Dwell Parameters (Station {st_id})",
                        "action_type": "PROCESS_CALIBRATION",
                        "target_station_id": st_id,
                        "details": f"Torque creeping above nominal band (current: {st.get('torque_nm', 50):.1f} Nm). Auto-recalibrate controller target.",
                        "impact": "Restores Cpk capability index to 1.67"
                    }

            if is_alert and recommended_action:
                # Backward genealogy trace for defect alerts
                current_anom_map = {stations[i]["station_id"]: float(anomaly_scores[i]) for i in range(len(stations))}
                genealogy = self.prop_graph.trace_backward_genealogy(st_id, current_anom_map)

                # Forward ripple projection for bottleneck alerts
                ripple = None
                if is_bottleneck_alert:
                    ct_delta = max(0.0, float(st.get("cycle_time", 60.0)) - float(st.get("nominal_cycle_time", 60.0)))
                    ripple = self.prop_graph.simulate_forward_ripple(st_id, slowdown_sec=ct_delta, horizon_minutes=10)

                active_alerts.append({
                    "station_id": st_id,
                    "station_name": st["name"],
                    "zone": st["zone"],
                    "defect_risk_pct": round(defect_prob * 100, 1),
                    "anomaly_score": round(anomaly_score, 1),
                    "forecast_queue": lstm_res["max_forecast_queue"],
                    "recommendation": recommended_action,
                    "feature_importances": self.rf_defect.feature_importances_,
                    "genealogy_trace": genealogy,
                    "ripple_projection": ripple
                })


            # Log validation
            gt = st.get("ground_truth", {})
            self.validation_tracker.log_defect_prediction(
                predicted_prob=defect_prob,
                ground_truth_defect=bool(gt.get("is_defect", False)),
                timestamp=timestamp
            )

            enriched_record = {
                **st,
                "anomaly_score": round(anomaly_score, 1),
                "defect_risk_prob": round(defect_prob, 3),
                "defect_risk_pct": round(defect_prob * 100, 1),
                "is_bottleneck_predicted": lstm_res["is_bottleneck_predicted"],
                "forecast_cycle_times": lstm_res["forecast_cycle_times"],
                "forecast_queues": lstm_res["forecast_queues"],
                "max_forecast_queue": lstm_res["max_forecast_queue"],
                "lead_time_ticks": lstm_res["lead_time_ticks"],
                "estimated_health_pct": dark_res["estimated_health_pct"],
                "data_confidence": dark_res["data_confidence"] if not is_sensor_rich else "High (Direct Telemetry)",
                "feature_importances": self.rf_defect.feature_importances_,
                "is_alert": is_alert,
                "recommended_action": recommended_action
            }
            enriched_stations.append(enriched_record)

        avg_throughput = float(np.mean([s["throughput_uph"] for s in enriched_stations]))
        avg_health = float(np.mean([s["estimated_health_pct"] for s in enriched_stations]))
        total_backlog = int(sum(s["queue_len"] for s in enriched_stations))

        quality_rate = max(0.85, 1.0 - float(np.mean([s["defect_risk_prob"] for s in enriched_stations])))
        performance_rate = min(1.0, avg_throughput / 60.0)
        oee_pct = round(0.96 * performance_rate * quality_rate * 100.0, 1)

        return {
            "tick": tick_num,
            "timestamp": timestamp,
            "line_id": line_id,
            "line_name": raw_tick.get("line_name", "Body & Assembly Line"),
            "station_count": len(enriched_stations),
            "nominal_takt_time_sec": raw_tick.get("nominal_takt_time_sec", 60.0),
            "plant_kpis": {
                "average_throughput_uph": round(avg_throughput, 1),
                "plant_health_index": round(avg_health, 1),
                "total_line_backlog_units": total_backlog,
                "estimated_oee_pct": oee_pct,
                "active_alert_count": len(active_alerts)
            },
            "active_alerts": active_alerts,
            "stations": enriched_stations
        }
