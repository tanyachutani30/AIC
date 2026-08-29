"""
DigitalTwin.ai - Expanded Feature Pipeline & Feature Engineering
Computes rich industrial engineering features:
- Rolling slopes and trends per channel (linear regression over rolling window)
- Time-since-last-maintenance and simulated tool wear
- Shift & Operator identifiers
- Upstream part-quality scores and genealogy propagation
- Environmental conditions (ambient temperature, relative humidity for paint curing)
- Micro-stop and short-downtime event counts in rolling window
- Station-level historical defect-rate priors
- Universal dwell times (entry-to-exit) for all stations
- Cross-station lag features (upstream queue and exit rates at t-1, t-2)
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np


class FeaturePipeline:
    """
    Online & offline feature engineering engine for DigitalTwin.ai.
    Maintains rolling buffers to compute slopes, cross-station lags, and environmental interactions.
    """

    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        # station_id -> deque of recent telemetry records
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        # station_id -> historical defect count / total count prior
        self.defect_priors: Dict[int, float] = {}

    def init_station_priors(self, station_config: List[Dict[str, Any]]) -> None:
        """Initializes baseline historical defect rate priors per station based on complexity."""
        for st in station_config:
            st_id = st["id"]
            zone = st.get("zone", "Body Construction")
            # Paint & Marriage stations have slightly higher historical complexity
            if "Paint" in zone:
                prior = 0.035
            elif "Final" in zone and ("Marriage" in st["name"] or "Torque" in st["name"]):
                prior = 0.040
            else:
                prior = 0.015
            self.defect_priors[st_id] = prior

    def compute_slope(self, values: List[float]) -> float:
        """Computes linear trend slope: slope = Cov(t, y) / Var(t)"""
        n = len(values)
        if n < 3:
            return 0.0
        x = np.arange(n, dtype=np.float32)
        y = np.array(values, dtype=np.float32)
        x_mean = (n - 1) / 2.0
        y_mean = float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2)) + 1e-6
        numer = float(np.sum((x - x_mean) * (y - y_mean)))
        return float(numer / denom)

    def extract_enriched_features(
        self,
        current_tick_records: List[Dict[str, Any]],
        tick_num: int,
        ambient_temp: float,
        ambient_humidity: float,
        shift_id: str
    ) -> List[Dict[str, Any]]:
        """
        Enriches raw station telemetry with rolling slope, lag, environmental, and upstream features.
        """
        # Sort stations by id for sequential cross-station processing
        sorted_records = sorted(current_tick_records, key=lambda r: r["station_id"])
        enriched_records = []
        
        # Upstream quality tracker along the line
        running_part_quality = 0.98

        for idx, r in enumerate(sorted_records):
            st_id = r["station_id"]
            nominal_ct = float(r.get("nominal_cycle_time", 60.0))
            
            # Store in history
            self.history[st_id].append(r)
            st_hist = list(self.history[st_id])
            
            # 1. Rolling Slopes & Trends
            ct_series = [float(h.get("cycle_time", nominal_ct)) for h in st_hist]
            tq_series = [float(h.get("torque_nm", 50.0)) for h in st_hist if h.get("torque_nm") is not None]
            vib_series = [float(h.get("vibration_rms", 1.25)) for h in st_hist if h.get("vibration_rms") is not None]
            noise_series = [float(h.get("ambient_noise_db", 75.0)) for h in st_hist if h.get("ambient_noise_db") is not None]
            optical_series = [float(h.get("optical_estimated_cycle_time", nominal_ct)) for h in st_hist if h.get("optical_estimated_cycle_time") is not None]

            slope_ct = self.compute_slope(ct_series)
            slope_tq = self.compute_slope(tq_series) if len(tq_series) >= 3 else 0.0
            slope_vib = self.compute_slope(vib_series) if len(vib_series) >= 3 else 0.0
            slope_noise = self.compute_slope(noise_series) if len(noise_series) >= 3 else 0.0
            slope_optical = self.compute_slope(optical_series) if len(optical_series) >= 3 else 0.0

            # Rolling statistics
            ct_mean = float(np.mean(ct_series))
            ct_std = float(np.std(ct_series))
            noise_mean = float(np.mean(noise_series)) if noise_series else 75.0
            noise_std = float(np.std(noise_series)) if noise_series else 0.0
            optical_mean = float(np.mean(optical_series)) if optical_series else nominal_ct
            optical_std = float(np.std(optical_series)) if optical_series else 0.0

            # 2. Equipment Age & Maintenance Schedule (cycles every 720 ticks / 12 hours)
            time_since_maint_hours = round(float((tick_num % 720) * (nominal_ct / 3600.0)), 2)

            # 3. Micro-stop count in rolling window (count of cycle times > takt + 8s)
            micro_stops_15m = sum(1 for ct in ct_series if ct > nominal_ct + 8.0)

            # 4. Upstream Part-Quality Propagation
            # Degrades if current station is anomalous, otherwise recovers slightly via rework/buffer
            gt = r.get("ground_truth", {})
            if gt.get("is_anomaly", False):
                running_part_quality = max(0.20, running_part_quality - gt.get("severity", 0.3) * 0.4)
            else:
                running_part_quality = min(0.99, running_part_quality + 0.02)
            
            upstream_quality_score = round(float(running_part_quality), 3)

            # 5. Cross-Station Lag Features (t-1, t-2 from upstream station idx-1)
            prev_st_queue_t1 = 0
            prev_st_exit_rate_t1 = 60.0
            prev_st_queue_t2 = 0
            
            if idx > 0:
                prev_st_id = sorted_records[idx - 1]["station_id"]
                prev_hist = list(self.history[prev_st_id])
                if len(prev_hist) >= 2:
                    prev_st_queue_t1 = prev_hist[-2].get("queue_len", 1)
                    prev_st_exit_rate_t1 = prev_hist[-2].get("throughput_uph", 60.0)
                if len(prev_hist) >= 3:
                    prev_st_queue_t2 = prev_hist[-3].get("queue_len", 1)

            # 6. Universal Dwell Time (entry-to-exit for all stations)
            universal_dwell = float(r.get("rfid_dwell_time_sec", r.get("cycle_time", nominal_ct) + 2.0))

            # 7. Station Defect Rate Prior
            defect_prior = self.defect_priors.get(st_id, 0.02)

            enriched = {
                **r,
                "shift_id": shift_id,
                "ambient_temp_c": round(ambient_temp, 2),
                "ambient_humidity_pct": round(ambient_humidity, 1),
                "time_since_maint_hours": time_since_maint_hours,
                "micro_stops_15m": micro_stops_15m,
                "upstream_quality_score": upstream_quality_score,
                "universal_dwell_sec": universal_dwell,
                "historical_defect_prior": defect_prior,
                "rolling_slope_ct": round(slope_ct, 4),
                "rolling_slope_tq": round(slope_tq, 4),
                "rolling_slope_vib": round(slope_vib, 4),
                "rolling_slope_noise": round(slope_noise, 4),
                "rolling_slope_optical": round(slope_optical, 4),
                "ct_rolling_mean": round(ct_mean, 2),
                "ct_rolling_std": round(ct_std, 2),
                "noise_rolling_mean": round(noise_mean, 2),
                "noise_rolling_std": round(noise_std, 2),
                "optical_rolling_mean": round(optical_mean, 2),
                "optical_rolling_std": round(optical_std, 2),
                "prev_st_queue_t1": prev_st_queue_t1,
                "prev_st_exit_rate_t1": round(prev_st_exit_rate_t1, 1),
                "prev_st_queue_t2": prev_st_queue_t2
            }
            enriched_records.append(enriched)

        return enriched_records
