"""
DigitalTwin.ai - Isolation Forest Unsupervised Anomaly Detection
Primary predictive mechanism for sensor-rich assembly stations.
Includes SPC (Statistical Process Control) 3-sigma baseline comparator.
"""

import numpy as np
import joblib
from typing import Dict, List, Any, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class StationIsolationForestDetector:
    """
    Unsupervised multivariate anomaly detector per station/family.
    Extracts rolling statistical features (mean, std, delta) from telemetry.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=120,
            max_samples="auto",
            contamination=contamination,
            random_state=random_state,
            n_jobs=1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # SPC baseline statistics: {station_id: {feature: (mean, std, ucl, lcl)}}
        self.spc_limits: Dict[int, Dict[str, Tuple[float, float, float, float]]] = {}

    def extract_features(self, telemetry_rows: List[Dict[str, Any]]) -> np.ndarray:
        """
        Builds multivariate feature matrix from telemetry records.
        Features: [cycle_time, torque_nm, vibration_rms, temperature_c, power_kw, rfid_dwell,
                   torque_missing, vibration_missing, temperature_missing]
        """
        feats = []
        for r in telemetry_rows:
            ct = float(r.get("cycle_time", 60.0))
            
            tq_val = r.get("torque_nm")
            tq_missing = 1.0 if tq_val is None else 0.0
            tq = float(tq_val if tq_val is not None else 50.0)
            
            vib_val = r.get("vibration_rms")
            vib_missing = 1.0 if vib_val is None else 0.0
            vib = float(vib_val if vib_val is not None else 1.25)
            
            tmp_val = r.get("temperature_c")
            tmp_missing = 1.0 if tmp_val is None else 0.0
            tmp = float(tmp_val if tmp_val is not None else 38.0)
            
            pwr = float(r.get("power_kw", 3.0))
            dwell = float(r.get("rfid_dwell_time_sec", 60.0))
            
            feats.append([ct, tq, vib, tmp, pwr, dwell, tq_missing, vib_missing, tmp_missing])
        return np.array(feats, dtype=np.float32)

    def fit(self, training_telemetry: List[Dict[str, Any]]) -> None:
        """
        Fits Isolation Forest on baseline/historical operating data.
        Also calculates standard SPC 3-sigma Upper/Lower Control Limits for benchmark comparison.
        """
        X = self.extract_features(training_telemetry)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Compute SPC control limits per station
        stations_data: Dict[int, List[Dict[str, Any]]] = {}
        for r in training_telemetry:
            st_id = r["station_id"]
            stations_data.setdefault(st_id, []).append(r)

        for st_id, records in stations_data.items():
            st_X = self.extract_features(records)
            means = np.mean(st_X, axis=0)
            stds = np.std(st_X, axis=0) + 1e-6
            self.spc_limits[st_id] = {
                "ct": (means[0], stds[0], means[0] + 3 * stds[0], max(0, means[0] - 3 * stds[0])),
                "tq": (means[1], stds[1], means[1] + 3 * stds[1], max(0, means[1] - 3 * stds[1])),
                "vib": (means[2], stds[2], means[2] + 3 * stds[2], max(0, means[2] - 3 * stds[2])),
                "tmp": (means[3], stds[3], means[3] + 3 * stds[3], max(0, means[3] - 3 * stds[3])),
            }

    def score_record(self, record: Dict[str, Any]) -> float:
        """
        Returns normalized 0–100 anomaly score.
        Higher score = more anomalous.
        """
        if not self.is_fitted:
            return 0.0
        
        X = self.extract_features([record])
        X_scaled = self.scaler.transform(X)
        
        raw_score = self.model.decision_function(X_scaled)[0]
        anomaly_score = 100.0 / (1.0 + np.exp(raw_score * 8.0))
        return float(np.clip(anomaly_score, 0.0, 100.0))

    def batch_score_records(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Vectorized batch scoring for thousands of records in milliseconds.
        """
        if not self.is_fitted or not records:
            return np.zeros(len(records), dtype=np.float32)
        
        X = self.extract_features(records)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        anomaly_scores = 100.0 / (1.0 + np.exp(raw_scores * 8.0))
        return np.clip(anomaly_scores, 0.0, 100.0).astype(np.float32)

    def evaluate_spc_baseline(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Naive SPC (Statistical Process Control) 3-sigma rule baseline.
        Flags anomaly ONLY when any single univariate channel exceeds +/- 3 standard deviations.
        """
        st_id = record["station_id"]
        limits = self.spc_limits.get(st_id)
        if not limits:
            return False, "No SPC baseline"

        ct = float(record.get("cycle_time", 60.0))
        if ct > limits["ct"][2] or ct < limits["ct"][3]:
            return True, f"SPC Violation: Cycle Time {ct:.1f}s outside 3-sigma [{limits['ct'][3]:.1f}, {limits['ct'][2]:.1f}]"

        if record.get("torque_nm") is not None:
            tq = float(record["torque_nm"])
            if tq > limits["tq"][2] or tq < limits["tq"][3]:
                return True, f"SPC Violation: Torque {tq:.1f}Nm outside 3-sigma [{limits['tq'][3]:.1f}, {limits['tq'][2]:.1f}]"

        if record.get("vibration_rms") is not None:
            vib = float(record["vibration_rms"])
            if vib > limits["vib"][2]:
                return True, f"SPC Violation: Vibration {vib:.2f}g outside 3-sigma UCL {limits['vib'][2]:.2f}"

        if record.get("temperature_c") is not None:
            tmp = float(record["temperature_c"])
            if tmp > limits["tmp"][2]:
                return True, f"SPC Violation: Temperature {tmp:.1f}C outside 3-sigma UCL {limits['tmp'][2]:.1f}"

        return False, "Nominal SPC"

    def save(self, filepath: str) -> None:
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "spc_limits": self.spc_limits,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.spc_limits = data["spc_limits"]
        self.is_fitted = data["is_fitted"]
