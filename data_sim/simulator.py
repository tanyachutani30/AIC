"""
DigitalTwin.ai - Synthetic Assembly Line Simulator
Deterministic, seeded time-series simulator implementing LineDataSource.
Generates multi-station telemetry, proxy signals for dark stations, and ground-truth anomaly events.
Supports SQLite export and live streaming.
"""

import json
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from .adapter import LineDataSource
from .noise_models import IndustrialNoiseGenerator
from .proxy_generator import ProxySignalGenerator
from .anomaly_injector import AnomalyScenarioManager


class SyntheticLineSimulator(LineDataSource):
    """
    High-fidelity synthetic digital twin simulator for vehicle assembly lines.
    """

    def __init__(self, config_path: str, seed: int = 42):
        self.config_path = config_path
        self.seed = seed
        self.config = self._load_config(config_path)
        
        self.rng = np.random.RandomState(seed)
        self.noise_gen = IndustrialNoiseGenerator(seed=seed)
        self.proxy_gen = ProxySignalGenerator(self.rng, self.noise_gen)
        self.anomaly_mgr = AnomalyScenarioManager(self.config, self.rng)
        
        self.current_tick = 0
        self.station_queues: Dict[int, int] = {s["id"]: 1 for s in self.config["stations"]}
        self.base_time = datetime.datetime(2026, 8, 26, 8, 0, 0)
        self._init_batch_arrays()

    def _init_batch_arrays(self):
        n = len(self.config["stations"])
        self.n_stations = n
        self.station_ids = np.array([s["id"] for s in self.config["stations"]])
        takt_nominal = float(self.config.get("nominal_takt_time_sec", 60.0))
        self.nominal_cts = np.array([float(s.get("nominal_cycle_time", takt_nominal)) for s in self.config["stations"]])
        self.is_sensor_rich = np.array([bool(s.get("sensor_rich", False)) for s in self.config["stations"]])
        self.has_torque = np.array([bool(s.get("has_torque", False)) for s in self.config["stations"]])
        self.has_vibration = np.array([bool(s.get("has_vibration", False)) for s in self.config["stations"]])
        self.has_temp = np.array([bool(s.get("has_temp", False)) for s in self.config["stations"]])
        self.station_queues_batch = np.ones(n, dtype=int)

    def _load_config(self, path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent / path
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_topology_config(self) -> Dict[str, Any]:
        return self.config

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.noise_gen.reseed(self.seed)
        self.proxy_gen = ProxySignalGenerator(self.rng, self.noise_gen)
        self.anomaly_mgr = AnomalyScenarioManager(self.config, self.rng)
        self.current_tick = 0
        self.station_queues = {s["id"]: 1 for s in self.config["stations"]}
        self._init_batch_arrays()

    def stream_next_tick(self) -> Dict[str, Any]:
        tick = self.current_tick
        self.current_tick += 1
        tick_time = self.base_time + datetime.timedelta(minutes=tick)
        
        takt_nominal = float(self.config.get("nominal_takt_time_sec", 60.0))
        shift_drift = self.noise_gen.get_shift_drift(tick)
        diurnal_temp = self.noise_gen.get_diurnal_temp_drift(tick)

        station_records = []
        
        n = self.n_stations
        gt_list = []
        actual_cts = np.zeros(n)
        hidden_wear_factors = np.zeros(n)
        is_jammed = np.zeros(n, dtype=bool)

        torque_vals = np.full(n, np.nan)
        vib_vals = np.full(n, np.nan)
        temp_vals = np.full(n, np.nan)
        
        torque_nominal = 50.0
        vib_nominal = 1.25
        temp_nominal = 38.0

        for i, st in enumerate(self.config["stations"]):
            st_id = st["id"]
            gt = self.anomaly_mgr.get_ground_truth_for_tick(tick, st_id)
            gt_list.append(gt)

            if self.is_sensor_rich[i]:
                if self.has_torque[i]:
                    torque_vals[i] = torque_nominal
                if self.has_vibration[i]:
                    vib_vals[i] = vib_nominal
                if self.has_temp[i]:
                    temp_vals[i] = temp_nominal + diurnal_temp

        # Base Autocorrelated noises (vectorized)
        ct_noise = self.noise_gen.get_ar1_noise_batch("ct", n, np.full(n, 0.55), np.full(n, 1.4))
        actual_cts = self.nominal_cts + shift_drift + ct_noise
        
        t_noise = self.noise_gen.get_ar1_noise_batch("tq", n, np.full(n, 0.7), np.full(n, 0.8))
        v_noise = np.abs(self.noise_gen.get_ar1_noise_batch("vib", n, np.full(n, 0.6), np.full(n, 0.15)))
        t_deg_noise = self.noise_gen.get_ar1_noise_batch("tmp", n, np.full(n, 0.85), np.full(n, 0.4))
        
        for i, gt in enumerate(gt_list):
            if gt["scenario_type"] == "gradual_drift":
                dp = gt["severity"]
                if self.is_sensor_rich[i] and self.has_torque[i]:
                    torque_vals[i] += dp * 14.0
                actual_cts[i] += dp * 4.5
                hidden_wear_factors[i] = dp * 0.7
            elif gt["scenario_type"] == "bearing_wear":
                ws = gt["severity"]
                if self.is_sensor_rich[i] and self.has_vibration[i]:
                    vib_vals[i] += ws * 4.2 + self.rng.normal(0, 0.4)
                if self.is_sensor_rich[i] and self.has_temp[i]:
                    temp_vals[i] += ws * 12.0
                actual_cts[i] += ws * 6.0
                hidden_wear_factors[i] = ws
            elif gt["scenario_type"] == "upstream_drift_source":
                if self.is_sensor_rich[i] and self.has_torque[i]:
                    torque_vals[i] += 3.5 + self.rng.normal(0, 0.3)
                actual_cts[i] += 1.5
            elif gt["scenario_type"] == "genealogy_defect_manifest":
                actual_cts[i] += 5.0 + self.rng.normal(0, 0.8)
            elif gt["scenario_type"] == "bottleneck_only":
                actual_cts[i] += 22.0 + self.rng.normal(0, 2.0)
                hidden_wear_factors[i] = 0.0

        # Add noise to sensor values
        mask_tq = self.is_sensor_rich & self.has_torque
        torque_vals[mask_tq] += t_noise[mask_tq]
        
        mask_vib = self.is_sensor_rich & self.has_vibration
        vib_vals[mask_vib] += v_noise[mask_vib]
        
        mask_tmp = self.is_sensor_rich & self.has_temp
        temp_vals[mask_tmp] += t_deg_noise[mask_tmp]

        # Queue dynamics
        prev_queues = self.station_queues_batch
        queue_deltas = np.zeros(n, dtype=int)
        
        inc_mask = actual_cts > (takt_nominal + 3.0)
        dec_mask = (actual_cts < (takt_nominal - 2.0)) & (prev_queues > 0)
        
        queue_deltas[inc_mask] = 1
        queue_deltas[dec_mask] = -1
        
        curr_queues = np.clip(prev_queues + queue_deltas, 0, 15)
        self.station_queues_batch = curr_queues
        
        is_jammed = curr_queues >= 10
        
        # UPH
        uphs = np.clip(3600.0 / np.maximum(actual_cts, 1.0), 0.0, 80.0)

        # Proxies
        proxies_batch = self.proxy_gen.generate_proxies_batch(
            self.station_ids,
            self.nominal_cts,
            actual_cts,
            curr_queues,
            hidden_wear_factors,
            is_jammed
        )

        station_records = []
        for i, st in enumerate(self.config["stations"]):
            record = {
                "station_id": int(self.station_ids[i]),
                "name": st["name"],
                "zone": st["zone"],
                "sensor_rich": bool(self.is_sensor_rich[i]),
                "cycle_time": round(float(actual_cts[i]), 2),
                "throughput_uph": round(float(uphs[i]), 1),
                "queue_len": int(curr_queues[i]),
                "torque_nm": round(float(torque_vals[i]), 2) if not np.isnan(torque_vals[i]) else None,
                "vibration_rms": round(float(vib_vals[i]), 2) if not np.isnan(vib_vals[i]) else None,
                "temperature_c": round(float(temp_vals[i]), 2) if not np.isnan(temp_vals[i]) else None,
                "rfid_dwell_time_sec": float(proxies_batch["rfid_dwell_time_sec"][i]),
                "power_kw": float(proxies_batch["power_kw"][i]),
                "ambient_noise_db": float(proxies_batch["ambient_noise_db"][i]),
                "optical_part_count": int(proxies_batch["optical_part_count"][i]),
                "optical_estimated_cycle_time": float(proxies_batch["optical_estimated_cycle_time"][i]),
                "ground_truth": gt_list[i]
            }
            station_records.append(record)
            self.station_queues[st["id"]] = int(curr_queues[i])

        return {
            "tick": tick,
            "timestamp": tick_time.isoformat(),
            "line_id": self.config["line_id"],
            "line_name": self.config["line_name"],
            "station_count": len(self.config["stations"]),
            "nominal_takt_time_sec": takt_nominal,
            "stations": station_records
        }

    def fetch_recent_ticks(self, count: int = 60) -> List[Dict[str, Any]]:
        # Fast replay up to count ticks
        ticks = []
        for _ in range(count):
            ticks.append(self.stream_next_tick())
        return ticks

    def generate_and_save_dataset(self, ticks_count: int, db_path: str) -> None:
        """
        Runs the simulation for ticks_count ticks and writes to SQLite database.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                tick INTEGER,
                timestamp TEXT,
                line_id TEXT,
                station_id INTEGER,
                station_name TEXT,
                zone TEXT,
                sensor_rich INTEGER,
                cycle_time REAL,
                throughput_uph REAL,
                queue_len INTEGER,
                torque_nm REAL,
                vibration_rms REAL,
                temperature_c REAL,
                rfid_dwell_time_sec REAL,
                power_kw REAL,
                ambient_noise_db REAL,
                optical_part_count INTEGER,
                optical_estimated_cycle_time REAL,
                is_anomaly INTEGER,
                scenario_type TEXT,
                severity REAL,
                is_defect INTEGER,
                is_bottleneck INTEGER,
                description TEXT,
                PRIMARY KEY (tick, line_id, station_id)
            )
        """)
        
        self.reset()
        records_to_insert = []
        
        for _ in range(ticks_count):
            tick_data = self.stream_next_tick()
            t = tick_data["tick"]
            ts = tick_data["timestamp"]
            lid = tick_data["line_id"]
            
            for st in tick_data["stations"]:
                gt = st["ground_truth"]
                records_to_insert.append((
                    t,
                    ts,
                    lid,
                    st["station_id"],
                    st["name"],
                    st["zone"],
                    1 if st["sensor_rich"] else 0,
                    st["cycle_time"],
                    st["throughput_uph"],
                    st["queue_len"],
                    st["torque_nm"],
                    st["vibration_rms"],
                    st["temperature_c"],
                    st["rfid_dwell_time_sec"],
                    st["power_kw"],
                    st["ambient_noise_db"],
                    st["optical_part_count"],
                    st["optical_estimated_cycle_time"],
                    1 if gt["is_anomaly"] else 0,
                    gt["scenario_type"],
                    gt["severity"],
                    1 if gt["is_defect"] else 0,
                    1 if gt["is_bottleneck"] else 0,
                    gt["description"]
                ))
        
        cur.executemany("""
            INSERT OR REPLACE INTO telemetry VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, records_to_insert)
        
        conn.commit()
        conn.close()
