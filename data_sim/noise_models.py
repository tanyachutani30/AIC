"""
DigitalTwin.ai - Realistic Noise and Drift Models
Implements realistic industrial time-series dynamics:
- AR(1) autoregressive noise (correlated sensor noise)
- Shift-to-shift operator and ambient drift
- Diurnal temperature cycles
- Heavy-tailed / micro-burst physical process variations
"""

import numpy as np
from typing import Dict, Any


class IndustrialNoiseGenerator:
    """
    Simulates non-Gaussian, autocorrelated physical noise matching real plant telemetry.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.ar_state: Dict[str, float] = {}
        self.ar_state_batch: Dict[str, np.ndarray] = {}

    def reseed(self, seed: int) -> None:
        self.rng = np.random.RandomState(seed)
        self.ar_state.clear()
        self.ar_state_batch.clear()

    def get_ar1_noise(self, key: str, phi: float = 0.65, sigma: float = 1.0) -> float:
        """
        Generates AR(1) autocorrelated noise: x_t = phi * x_{t-1} + e_t
        """
        prev = self.ar_state.get(key, 0.0)
        # White noise innovation with a small Student-t fat tail component
        innovation = self.rng.normal(0, sigma * np.sqrt(1 - phi ** 2))
        if self.rng.rand() < 0.03:  # 3% chance of minor physical micro-spike
            innovation += self.rng.choice([-1, 1]) * self.rng.exponential(sigma * 1.5)

        val = phi * prev + innovation
        self.ar_state[key] = val
        return val

    def get_ar1_noise_batch(self, channel: str, n_stations: int, phis: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """
        Generates AR(1) autocorrelated noise for all stations simultaneously.
        """
        if channel not in self.ar_state_batch:
            self.ar_state_batch[channel] = np.zeros(n_stations, dtype=np.float64)
            
        prev = self.ar_state_batch[channel]
        
        # White noise innovation with a small Student-t fat tail component
        std_devs = sigmas * np.sqrt(1 - phis ** 2)
        innovation = self.rng.normal(0, std_devs, size=n_stations)
        
        # 3% chance of minor physical micro-spike per station
        spike_mask = self.rng.rand(n_stations) < 0.03
        if np.any(spike_mask):
            spike_counts = np.sum(spike_mask)
            signs = self.rng.choice([-1, 1], size=spike_counts)
            spikes = signs * self.rng.exponential(sigmas[spike_mask] * 1.5)
            innovation[spike_mask] += spikes

        val = phis * prev + innovation
        self.ar_state_batch[channel] = val
        return val

    def get_shift_drift(self, tick: int, shift_length_ticks: int = 480) -> float:
        """
        Simulates operator fatigue and shift warmup variations over an 8-hour shift.
        """
        shift_progress = (tick % shift_length_ticks) / shift_length_ticks
        # Subtle bathtub / curve: slightly slower at shift start, optimal mid-shift, slight fatigue at end
        drift = 0.5 * np.sin(2 * np.pi * shift_progress - np.pi / 2) + 0.3 * (shift_progress ** 2)
        return float(drift)

    def get_diurnal_temp_drift(self, tick: int, day_length_ticks: int = 1440) -> float:
        """
        Ambient factory temperature drift (colder morning/night, warmer mid-day).
        """
        day_phase = (tick % day_length_ticks) / day_length_ticks
        # Peak temperature around tick 720 (mid-day)
        return float(2.5 * np.sin(2 * np.pi * day_phase - np.pi / 2))
