"""
Unit tests for Machine Learning Models:
Isolation Forest, PyTorch LSTM Forecaster, Random Forest Defect & Dark-Station models.
"""

import unittest
import numpy as np
import torch
from models.isolation_forest_model import StationIsolationForestDetector
from models.lstm_bottleneck_model import BottleneckForecaster
from models.random_forest_defect_model import DefectRiskRandomForest, DarkStationInferenceModel
from models.validation_metrics import TrustValidationTracker


class TestMLModels(unittest.TestCase):

    def test_isolation_forest_scoring(self):
        detector = StationIsolationForestDetector(contamination=0.05, random_state=42)
        # Create normal baseline data
        normal_data = [
            {"station_id": 8, "cycle_time": 59.0 + np.random.normal(0, 0.5), "torque_nm": 50.0 + np.random.normal(0, 0.5), "vibration_rms": 1.2, "temperature_c": 38.0}
            for _ in range(100)
        ]
        detector.fit(normal_data)
        
        # Test normal record
        normal_score = detector.score_record({"station_id": 8, "cycle_time": 59.1, "torque_nm": 50.2, "vibration_rms": 1.21, "temperature_c": 38.1})
        self.assertLess(normal_score, 45.0)

        # Test extreme anomaly record
        anom_score = detector.score_record({"station_id": 8, "cycle_time": 75.0, "torque_nm": 68.0, "vibration_rms": 4.5, "temperature_c": 52.0})
        self.assertGreater(anom_score, 60.0)

    def test_lstm_forecaster(self):
        forecaster = BottleneckForecaster(seq_len=10, horizon=3, hidden_dim=16)
        
        # Create synthetic trend data
        sample_series = [
            {"cycle_time": 58.0 + i * 0.1, "queue_len": 1 + i // 10, "throughput_uph": 55.0, "power_kw": 3.0}
            for i in range(50)
        ]
        X, y = forecaster.prepare_sequences(sample_series)
        self.assertEqual(len(X.shape), 3)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(y.shape[1], 3)

        # Predict with mock series
        pred = forecaster.predict_forecast(sample_series[:10])
        self.assertEqual(len(pred["forecast_cycle_times"]), 3)
        self.assertIn("is_bottleneck_predicted", pred)

    def test_random_forest_defect_and_explainability(self):
        rf = DefectRiskRandomForest(random_state=42)
        X = np.random.randn(50, 7).astype(np.float32)
        y = np.random.choice([0, 1], size=50).astype(np.int32)
        rf.fit(X, y)
        
        prob, importances = rf.predict_risk(X[0])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertEqual(len(importances), 7)
        self.assertAlmostEqual(sum(importances.values()), 100.0, delta=1.0)

    def test_dark_station_inference(self):
        dark_model = DarkStationInferenceModel(random_state=42)
        # Infer on nominal proxy signals (High confidence match)
        result_high = dark_model.infer_health_and_confidence(
            dwell_time=60.0, power_kw=3.1, nominal_ct=60.0,
            ambient_noise_db=75.0, optical_estimated_cycle_time=60.5, ambient_noise_rolling_std=0.5
        )
        self.assertIn("estimated_health_pct", result_high)
        self.assertEqual(result_high["data_confidence"], "High")

        # Infer on erratic noise (Low confidence)
        result_low = dark_model.infer_health_and_confidence(
            dwell_time=60.0, power_kw=3.1, nominal_ct=60.0,
            ambient_noise_db=85.0, optical_estimated_cycle_time=60.5, ambient_noise_rolling_std=2.5
        )
        self.assertEqual(result_low["data_confidence"], "Low")

    def test_validation_tracker_metrics(self):
        tracker = TrustValidationTracker(window_size=100, default_alert_threshold=0.5)
        # Log 10 TP, 2 FP, 20 TN, 1 FN
        for _ in range(10): tracker.log_defect_prediction(0.8, True)
        for _ in range(2):  tracker.log_defect_prediction(0.7, False)
        for _ in range(20): tracker.log_defect_prediction(0.2, False)
        for _ in range(1):  tracker.log_defect_prediction(0.3, True)

        metrics = tracker.get_current_metrics()
        self.assertAlmostEqual(metrics["defect_classifier"]["precision"], 10 / 12, places=2)
        self.assertAlmostEqual(metrics["defect_classifier"]["recall"], 10 / 11, places=2)


if __name__ == "__main__":
    unittest.main()
