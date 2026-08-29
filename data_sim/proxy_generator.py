"""
DigitalTwin.ai - Dark Station Proxy Signal Generator
Generates realistic proxy telemetry for sensor-poor stations:
- RFID / Barcode scan dwell time (entry to exit timestamp delta)
- Active electrical power draw (kW / Amperes)
Ensures internal physical health remains hidden and must be inferred from proxies.
"""

import numpy as np
from typing import Dict, Any


class ProxySignalGenerator:
    """
    Simulates cheap, non-intrusive proxy sensors for legacy or 'dark' assembly stations.
    """

    def __init__(self, rng: np.random.RandomState, noise_gen=None):
        self.rng = rng
        self.noise_gen = noise_gen
        self.part_counts: Dict[int, int] = {}
        self.part_counts_batch: np.ndarray = None

    def generate_proxies(
        self,
        station_id: int,
        nominal_cycle_time: float,
        actual_cycle_time: float,
        queue_len: int,
        hidden_wear_factor: float,
        is_jammed: bool = False
    ) -> Dict[str, float]:
        """
        Generates proxy signals based on physical events without leaking ground-truth internal labels.

        - rfid_dwell_time: Cycle time plus transit latency and conveyor queuing delay.
        - power_kw: Idle electrical baseline + active motor load + wear friction multiplier.
        """
        # Conveyor transfer scan latency (typically 1.5 - 3.5 seconds)
        scan_overhead = self.rng.uniform(1.2, 3.8)
        queue_buffer_delay = max(0.0, queue_len * self.rng.uniform(0.8, 1.8))
        rfid_dwell = actual_cycle_time + scan_overhead + queue_buffer_delay

        # Base electric motor power: 3.5 kW nominal + load proportional to cycle time + wear friction
        idle_kw = 1.2 + self.rng.normal(0, 0.05)
        active_kw = 2.8 * (actual_cycle_time / max(nominal_cycle_time, 1.0))
        wear_power_penalty = hidden_wear_factor * 1.5  # Degraded bearings/clamps draw more current

        if is_jammed:
            # Overload spike when jammed
            power_kw = idle_kw + active_kw + 4.0 + self.rng.normal(0, 0.2)
        else:
            power_kw = idle_kw + active_kw + wear_power_penalty + self.rng.normal(0, 0.12)

        # Ambient noise (AR1 + spikes)
        ambient_noise_db = 75.0  # baseline
        if self.noise_gen:
            ambient_noise_db += self.noise_gen.get_ar1_noise(f"st_{station_id}_noise", phi=0.7, sigma=1.0)
        
        if is_jammed:
            ambient_noise_db += 15.0 + self.rng.normal(0, 2.0)
        elif hidden_wear_factor > 0.5:
            ambient_noise_db += hidden_wear_factor * 12.0 + self.rng.normal(0, 1.0)

        # Optical part counter and estimated cycle time (2-5% missed count)
        if station_id not in self.part_counts:
            self.part_counts[station_id] = 0
            
        if self.rng.rand() < 0.035:
            # Missed count due to occlusion
            optical_ct = actual_cycle_time * 2.0 + self.rng.normal(0, 0.5)
        else:
            self.part_counts[station_id] += 1
            optical_ct = actual_cycle_time + self.rng.normal(0, 0.2)

        return {
            "rfid_dwell_time_sec": round(float(max(10.0, rfid_dwell)), 2),
            "power_kw": round(float(max(0.5, power_kw)), 2),
            "ambient_noise_db": round(float(ambient_noise_db), 2),
            "optical_part_count": self.part_counts[station_id],
            "optical_estimated_cycle_time": round(float(max(1.0, optical_ct)), 2)
        }

    def generate_proxies_batch(
        self,
        station_ids: np.ndarray,
        nominal_cycle_time: np.ndarray,
        actual_cycle_time: np.ndarray,
        queue_len: np.ndarray,
        hidden_wear_factor: np.ndarray,
        is_jammed: np.ndarray
    ) -> Dict[str, np.ndarray]:
        n = len(station_ids)
        
        # Conveyor transfer scan latency (typically 1.5 - 3.5 seconds)
        scan_overhead = self.rng.uniform(1.2, 3.8, size=n)
        queue_buffer_delay = np.maximum(0.0, queue_len * self.rng.uniform(0.8, 1.8, size=n))
        rfid_dwell = actual_cycle_time + scan_overhead + queue_buffer_delay

        # Base electric motor power
        idle_kw = 1.2 + self.rng.normal(0, 0.05, size=n)
        active_kw = 2.8 * (actual_cycle_time / np.maximum(nominal_cycle_time, 1.0))
        wear_power_penalty = hidden_wear_factor * 1.5

        power_kw = idle_kw + active_kw + wear_power_penalty + self.rng.normal(0, 0.12, size=n)
        
        # jammed overrides
        jammed_mask = is_jammed > 0
        if np.any(jammed_mask):
            power_kw[jammed_mask] = idle_kw[jammed_mask] + active_kw[jammed_mask] + 4.0 + self.rng.normal(0, 0.2, size=np.sum(jammed_mask))

        # Ambient noise
        ambient_noise_db = np.full(n, 75.0)
        if self.noise_gen:
            phis = np.full(n, 0.7)
            sigmas = np.full(n, 1.0)
            ambient_noise_db += self.noise_gen.get_ar1_noise_batch("noise", n, phis, sigmas)
            
        if np.any(jammed_mask):
            ambient_noise_db[jammed_mask] += 15.0 + self.rng.normal(0, 2.0, size=np.sum(jammed_mask))
            
        wear_mask = (~jammed_mask) & (hidden_wear_factor > 0.5)
        if np.any(wear_mask):
            ambient_noise_db[wear_mask] += hidden_wear_factor[wear_mask] * 12.0 + self.rng.normal(0, 1.0, size=np.sum(wear_mask))

        # Optical part counter
        if self.part_counts_batch is None:
            self.part_counts_batch = np.zeros(n, dtype=int)
            
        miss_mask = self.rng.rand(n) < 0.035
        hit_mask = ~miss_mask
        
        self.part_counts_batch[hit_mask] += 1
        
        optical_ct = np.zeros(n)
        if np.any(miss_mask):
            optical_ct[miss_mask] = actual_cycle_time[miss_mask] * 2.0 + self.rng.normal(0, 0.5, size=np.sum(miss_mask))
        if np.any(hit_mask):
            optical_ct[hit_mask] = actual_cycle_time[hit_mask] + self.rng.normal(0, 0.2, size=np.sum(hit_mask))

        return {
            "rfid_dwell_time_sec": np.round(np.maximum(10.0, rfid_dwell), 2),
            "power_kw": np.round(np.maximum(0.5, power_kw), 2),
            "ambient_noise_db": np.round(ambient_noise_db, 2),
            "optical_part_count": self.part_counts_batch.copy(),
            "optical_estimated_cycle_time": np.round(np.maximum(1.0, optical_ct), 2)
        }
