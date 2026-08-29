"""
Unit tests for SyntheticLineSimulator, Noise, Proxies, and Anomaly Injection.
Compatible with standard library unittest and pytest.
"""

import os
import unittest
import sqlite3
import tempfile
from data_sim.simulator import SyntheticLineSimulator
from data_sim.noise_models import IndustrialNoiseGenerator


class TestSimulator(unittest.TestCase):

    def test_configs_load(self):
        for cfg in ["config/line_config_default.json", "config/line_config_sparse.json", "config/line_config_dense.json"]:
            sim = SyntheticLineSimulator(cfg, seed=123)
            topo = sim.get_topology_config()
            self.assertIn("station_count", topo)
            self.assertEqual(len(topo["stations"]), topo["station_count"])

    def test_stream_tick_structure(self):
        sim = SyntheticLineSimulator("config/line_config_default.json", seed=42)
        tick = sim.stream_next_tick()
        self.assertEqual(tick["tick"], 0)
        self.assertIn("timestamp", tick)
        self.assertEqual(len(tick["stations"]), 36)
        
        # Check dark station vs sensor-rich station fields
        st5 = next(s for s in tick["stations"] if s["station_id"] == 5)
        self.assertFalse(st5["sensor_rich"])
        self.assertIsNone(st5["torque_nm"])
        self.assertGreater(st5["rfid_dwell_time_sec"], 0)
        self.assertGreater(st5["power_kw"], 0)
        self.assertIn("ambient_noise_db", st5)
        self.assertGreater(st5["ambient_noise_db"], 50.0)
        self.assertIn("optical_part_count", st5)
        self.assertIn("optical_estimated_cycle_time", st5)
        self.assertGreater(st5["optical_estimated_cycle_time"], 0)

        st8 = next(s for s in tick["stations"] if s["station_id"] == 8)
        self.assertTrue(st8["sensor_rich"])
        self.assertIsNotNone(st8["torque_nm"])
        self.assertIsNotNone(st8["vibration_rms"])

    def test_anomaly_injection_scenarios(self):
        sim = SyntheticLineSimulator("config/line_config_default.json", seed=100)
        
        # Run through 600 ticks to observe all 4 scenarios
        ticks = [sim.stream_next_tick() for _ in range(600)]
        
        # Check Scenario 1: Gradual drift at Station 8 around tick 150
        st8_drift = [t["stations"][7] for t in ticks if 80 <= t["tick"] < 220]
        self.assertTrue(all(s["ground_truth"]["scenario_type"] == "gradual_drift" for s in st8_drift))
        self.assertGreater(st8_drift[-1]["torque_nm"], st8_drift[0]["torque_nm"])

        # Check Scenario 2: Bearing wear at Station 28 around tick 300
        st28_wear = [t["stations"][27] for t in ticks if 270 <= t["tick"] < 330]
        self.assertTrue(any(s["ground_truth"]["scenario_type"] == "bearing_wear" for s in st28_wear))
        self.assertTrue(any(s["vibration_rms"] > 3.0 for s in st28_wear))

        # Check Scenario 3: Downstream defect manifest at Station 22 around tick 420
        st22_manifest = [t["stations"][21] for t in ticks if 410 <= t["tick"] < 460]
        self.assertTrue(any(s["ground_truth"]["scenario_type"] == "genealogy_defect_manifest" for s in st22_manifest))
        self.assertTrue(any(s["ground_truth"]["is_defect"] for s in st22_manifest))

        # Check Scenario 4: Bottleneck-only at Station 14 around tick 520
        st14_bottleneck = [t["stations"][13] for t in ticks if 500 <= t["tick"] < 550]
        self.assertTrue(any(s["ground_truth"]["is_bottleneck"] for s in st14_bottleneck))
        self.assertTrue(all(not s["ground_truth"]["is_defect"] for s in st14_bottleneck))

    def test_sqlite_dataset_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = os.path.join(tmp_dir, "test_digitaltwin.db")
            sim = SyntheticLineSimulator("config/line_config_default.json", seed=42)
            sim.generate_and_save_dataset(ticks_count=50, db_path=db_file)
            
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM telemetry")
            count = cur.fetchone()[0]
            conn.close()
            
            self.assertEqual(count, 50 * 36)

    def test_benchmark_vectorization(self):
        import time
        sim = SyntheticLineSimulator("config/line_config_default.json", seed=42)
        
        # Warmup
        sim.stream_next_tick()
        
        start_t = time.perf_counter()
        for _ in range(1000):
            sim.stream_next_tick()
        end_t = time.perf_counter()
        
        elapsed = end_t - start_t
        print(f"\nVectorized simulator throughput: 1000 ticks in {elapsed:.4f} seconds.")
        # Ensure it's reasonably fast
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
