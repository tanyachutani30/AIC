"""
DigitalTwin.ai - Random Forest Defect Classifier & Dark-Station Inference
Implements:
1. Multi-station Defect-Risk Classifier with explainable Gini feature importances.
2. Dark-Station Health & Confidence Estimator trained purely on cheap proxy signals.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def _get_rf_defaults() -> Dict[str, Any]:
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "line_config_default.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
            return cfg.get("model_hyperparameters", {}).get("random_forest", {})
    except Exception:
        return {"n_estimators": 100, "max_depth": 6}


FEATURE_NAMES = [
    "Station Anomaly Score (IForest)",
    "Upstream Station Anomaly (Genealogy)",
    "RFID Dwell Time Deviation",
    "Active Power Draw Deviation",
    "Instantaneous Cycle Time",
    "Vibration Level",
    "Tool Operating Age"
]


class DefectRiskRandomForest:
    """
    Random Forest Defect-Risk Classifier.
    Combines multivariate anomaly scores, genealogy propagation, and proxy deviations
    to predict unit failure probability before downstream QA.
    """

    def __init__(self, n_estimators: Optional[int] = None, max_depth: Optional[int] = None, random_state: int = 42):
        defaults = _get_rf_defaults()
        self.n_estimators = n_estimators if n_estimators is not None else defaults.get("n_estimators", 100)
        self.max_depth = max_depth if max_depth is not None else defaults.get("max_depth", 6)
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1
        )
        self.feature_names = FEATURE_NAMES
        self.feature_importances_: Dict[str, float] = {}
        self.is_fitted = False

    def build_feature_vector(
        self,
        record: Dict[str, Any],
        st_iforest_score: float,
        upstream_iforest_score: float,
        nominal_ct: float = 60.0,
        tool_age_hours: float = 120.0
    ) -> np.ndarray:
        """
        Extracts engineered tabular feature vector for defect classification.
        """
        ct = float(record.get("cycle_time", nominal_ct))
        dwell = float(record.get("rfid_dwell_time_sec", ct))
        dwell_dev = abs(dwell - nominal_ct)
        pwr = float(record.get("power_kw", 3.0))
        pwr_dev = abs(pwr - 3.2)
        vib = float(record.get("vibration_rms") if record.get("vibration_rms") is not None else 1.25)
        
        return np.array([
            st_iforest_score,
            upstream_iforest_score,
            dwell_dev,
            pwr_dev,
            ct,
            vib,
            tool_age_hours
        ], dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits Random Forest with tuned hyperparameters.
        """
        self.model.fit(X, y)
        self.is_fitted = True

        # Calculate normalized feature importances
        raw_importances = self.model.feature_importances_
        total = np.sum(raw_importances) + 1e-8
        self.feature_importances_ = {
            name: round(float(imp / total * 100.0), 1)
            for name, imp in zip(self.feature_names, raw_importances)
        }

    def predict_risk(self, feature_vector: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Returns (defect_probability, feature_importance_explanation).
        """
        if not self.is_fitted:
            return 0.05, {}
        
        X = feature_vector.reshape(1, -1)
        prob = self.model.predict_proba(X)[0][1]
        return float(prob), self.feature_importances_

    def save(self, filepath: str) -> None:
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances_,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.feature_importances_ = data["feature_importances"]
        self.is_fitted = data["is_fitted"]


class DarkStationInferenceModel:
    """
    Infers station health and assigns Data Confidence (High / Medium / Low)
    for uninstrumented stations using ONLY cheap proxy metrics (RFID dwell, power draw).
    """

    def __init__(self, random_state: int = 42):
        self.regressor = RandomForestRegressor(
            n_estimators=80,
            max_depth=5,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=1
        )
        self.is_fitted = False

    def extract_proxy_features(self, dwell_time: float, power_kw: float, nominal_ct: float, ambient_noise_db: float, optical_estimated_cycle_time: float) -> np.ndarray:
        dwell_dev = dwell_time - nominal_ct
        power_ratio = power_kw / 3.0
        return np.array([[dwell_time, power_kw, ambient_noise_db, optical_estimated_cycle_time, dwell_dev, power_ratio]], dtype=np.float32)

    def fit(self, X_proxies: np.ndarray, y_health: np.ndarray) -> None:
        """
        Trains regressor to predict hidden health score (0 - 100%).
        """
        self.regressor.fit(X_proxies, y_health)
        self.is_fitted = True

    def infer_health_and_confidence(
        self,
        dwell_time: float,
        power_kw: float,
        nominal_ct: float = 60.0,
        ambient_noise_db: float = 75.0,
        optical_estimated_cycle_time: float = 60.0,
        ambient_noise_rolling_std: float = 0.0
    ) -> Dict[str, Any]:
        """
        Returns estimated station health percentage and confidence tier.
        """
        if not self.is_fitted:
            # Fallback heuristic if not fitted
            dwell_dev = abs(dwell_time - nominal_ct)
            health = max(20.0, 100.0 - dwell_dev * 4.0 - max(0.0, (power_kw - 3.5) * 15.0))
            if ambient_noise_rolling_std > 2.0:
                confidence = "Low"
            elif abs(optical_estimated_cycle_time - dwell_time) < 3.0:
                confidence = "High"
            else:
                confidence = "Medium"
            return {
                "estimated_health_pct": round(float(np.clip(health, 0.0, 100.0)), 1),
                "data_confidence": confidence,
                "proxy_signals_used": ["RFID Scan-to-Scan Dwell", "Active Power Draw", "Ambient Noise", "Optical Cycle Time"]
            }

        X = self.extract_proxy_features(dwell_time, power_kw, nominal_ct, ambient_noise_db, optical_estimated_cycle_time)
        est_health = float(self.regressor.predict(X)[0])
        est_health = float(np.clip(est_health, 0.0, 100.0))

        # Confidence level based on stability of proxy measurements
        if ambient_noise_rolling_std > 2.0:
            confidence = "Low"
        elif abs(optical_estimated_cycle_time - dwell_time) < 3.0:
            confidence = "High"
        else:
            confidence = "Medium"

        return {
            "estimated_health_pct": round(est_health, 1),
            "data_confidence": confidence,
            "proxy_signals_used": ["RFID Scan-to-Scan Dwell", "Active Power Draw", "Ambient Noise", "Optical Cycle Time"]
        }

    def save(self, filepath: str) -> None:
        joblib.dump({"regressor": self.regressor, "is_fitted": self.is_fitted}, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.regressor = data["regressor"]
        self.is_fitted = data["is_fitted"]
