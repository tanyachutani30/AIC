"""
Unit and integration tests for FastAPI backend and endpoints.
"""

import unittest
from fastapi.testclient import TestClient
from backend.main import app, engine, sim


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_config(self):
        resp = self.client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("station_count", data)
        self.assertIn("stations", data)

    def test_get_live_stations(self):
        resp = self.client.get("/api/stations/live")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("stations", data)
        self.assertIn("plant_kpis", data)
        self.assertIn("active_alerts", data)

    def test_get_validation_metrics(self):
        resp = self.client.get("/api/metrics/validation")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("metrics", data)
        self.assertIn("tradeoff_curve", data)

    def test_get_roi_metrics(self):
        resp = self.client.get("/api/metrics/roi")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("rule_of_ten", data)
        self.assertIn("realized_savings", data)

    def test_execute_action(self):
        payload = {
            "station_id": 8,
            "station_name": "Door Hinge Multi-Spindle",
            "action_type": "CALIBRATION",
            "description": "Recalibrated torque target"
        }
        resp = self.client.post("/api/action/execute", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["action"]["station_id"], 8)

    def test_update_threshold(self):
        resp = self.client.post("/api/threshold", json={"threshold": 0.65})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["active_threshold"], 0.65)

    def test_sim_control(self):
        resp = self.client.post("/api/sim/control", json={"speed": 2.0, "pause": False})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["speed"], 2.0)


if __name__ == "__main__":
    unittest.main()
