"""
DigitalTwin.ai - Realistic Anomaly & Ground-Truth Injector
Encodes 4 distinct production anomaly scenarios with precise ground-truth labels:
1. Gradual Drift (creeping torque out-of-spec, catchable early by ML before SPC threshold)
2. Sudden Mechanical Wear (bearing cage degradation vibration harmonics)
3. Upstream Genealogy Propagation (upstream clamping drift causing downstream fitment defect)
4. Bottleneck-Only Non-Defective Event (slow cycle queue buildup without part defect)
"""

from typing import Dict, Any, List, Optional
import numpy as np


class AnomalyScenarioManager:
    """
    Manages deterministic injection of production anomalies and records ground-truth states.
    """

    def __init__(self, config: Dict[str, Any], rng: np.random.RandomState):
        self.config = config
        self.rng = rng
        self.scenarios_cfg = config.get("anomaly_scenarios_config", {})
        
        # Schedule windows: (start_tick, end_tick, scenario_type, target_station)
        # We define a cyclic schedule so that during simulation runs of any length (e.g. 1000 ticks),
        # scenarios occur predictably for verification.
        self.active_events: List[Dict[str, Any]] = []

    def get_ground_truth_for_tick(self, tick: int, station_id: int) -> Dict[str, Any]:
        """
        Determines if any anomaly is active for the given station at tick.
        Returns ground-truth metadata:
        {
            "is_anomaly": bool,
            "scenario_type": str or None,
            "severity": float (0.0 to 1.0),
            "is_defect": bool,
            "is_bottleneck": bool,
            "description": str
        }
        """
        # Periodic anomaly windows based on tick modulo 600 (approx 10-hour cycle)
        cycle_tick = tick % 600

        grad_station = self.scenarios_cfg.get("gradual_drift_station_id", 8)
        bearing_station = self.scenarios_cfg.get("bearing_wear_station_id", 28)
        upstream_src = self.scenarios_cfg.get("upstream_source_station_id", 4)
        downstream_dst = self.scenarios_cfg.get("downstream_manifest_station_id", 22)
        bottleneck_station = self.scenarios_cfg.get("bottleneck_only_station_id", 14)

        # Scenario 1: Gradual Drift (ticks 80 -> 220 at grad_station)
        if station_id == grad_station and 80 <= cycle_tick < 220:
            progress = (cycle_tick - 80) / (220 - 80)  # 0.0 to 1.0
            return {
                "is_anomaly": True,
                "scenario_type": "gradual_drift",
                "severity": float(progress),
                "is_defect": progress > 0.65,  # Becomes a hard defect late, but anomalous early
                "is_bottleneck": False,
                "description": f"Multi-spindle torque creeping up ({round(progress * 100)}% drift)"
            }

        # Scenario 2: Sudden Bearing Wear (ticks 260 -> 340 at bearing_station)
        if station_id == bearing_station and 260 <= cycle_tick < 340:
            severity = 0.85 if cycle_tick >= 270 else (cycle_tick - 260) / 10.0 * 0.85
            return {
                "is_anomaly": True,
                "scenario_type": "bearing_wear",
                "severity": float(severity),
                "is_defect": True,
                "is_bottleneck": True,
                "description": "Drive motor spindle bearing race micro-fracture vibration harmonics"
            }

        # Scenario 3: Upstream Stamping to Downstream Defect Propagation
        # Upstream source runs at ticks 380 -> 440
        if station_id == upstream_src and 380 <= cycle_tick < 440:
            return {
                "is_anomaly": True,
                "scenario_type": "upstream_drift_source",
                "severity": 0.45,
                "is_defect": False,  # Passes upstream station QC
                "is_bottleneck": False,
                "description": "Stamping clamp pressure variance (subtle root cause)"
            }

        # Downstream manifest runs 20 ticks later (travel time) ticks 400 -> 470
        if station_id == downstream_dst and 400 <= cycle_tick < 470:
            return {
                "is_anomaly": True,
                "scenario_type": "genealogy_defect_manifest",
                "severity": 0.90,
                "is_defect": True,  # Fails downstream inspection!
                "is_bottleneck": False,
                "description": f"Panel surface defect caused by upstream Station {upstream_src} clamp variance"
            }

        # Scenario 4: Bottleneck-Only (ticks 490 -> 560 at bottleneck_station)
        if station_id == bottleneck_station and 490 <= cycle_tick < 560:
            return {
                "is_anomaly": False,  # No equipment defect
                "scenario_type": "bottleneck_only",
                "severity": 0.0,
                "is_defect": False,   # Zero defects!
                "is_bottleneck": True, # Queue buildup only
                "description": "Curing conveyor thermal regulator pause (queue buildup, equipment nominal)"
            }

        # Default: Normal operation
        return {
            "is_anomaly": False,
            "scenario_type": None,
            "severity": 0.0,
            "is_defect": False,
            "is_bottleneck": False,
            "description": "Normal operating parameters"
        }
