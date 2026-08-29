"""
DigitalTwin.ai - Propagation Graph Model
Directed assembly line graph with empirically-learned edge weights.
Provides:
1. Forward Ripple Simulation: Projects downstream queue buildup and starvation/blocking from a slowdown at Station N.
2. Backward Genealogy Tracing: Traverses upstream correlation paths from a defect to identify root-cause origin station(s).
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import defaultdict
import joblib


class AssemblyLinePropagationGraph:
    """
    Directed Graph Model of the vehicle assembly line.
    Nodes = Assembly Stations.
    Edges = Conveyors / Buffers with empirical correlation & transit lag weights.
    """

    def __init__(self, station_count: int = 36):
        self.station_count = station_count
        self.adj_list: Dict[int, List[int]] = defaultdict(list)
        self.rev_adj_list: Dict[int, List[int]] = defaultdict(list)
        # Edge weights: (u, v) -> {"correlation": float, "lag_ticks": int, "transfer_efficiency": float}
        self.edge_weights: Dict[Tuple[int, int], Dict[str, float]] = {}
        self._build_topology()

    def _build_topology(self) -> None:
        """Constructs sequential directed linear topology with local buffer routing."""
        for i in range(1, self.station_count):
            self.adj_list[i].append(i + 1)
            self.rev_adj_list[i + 1].append(i)
            # Default empirical baseline weights
            self.edge_weights[(i, i + 1)] = {
                "correlation": 0.78,
                "lag_ticks": 1,
                "transfer_efficiency": 0.95
            }

    def calibrate_weights_from_telemetry(self, station_time_series_map: Dict[int, List[Dict[str, Any]]]) -> None:
        """
        Learns empirical edge weights from cross-station correlation of queue & cycle time time-series.
        """
        for u in range(1, self.station_count):
            v = u + 1
            u_records = station_time_series_map.get(u, [])
            v_records = station_time_series_map.get(v, [])
            
            if len(u_records) >= 30 and len(v_records) >= 30:
                u_queues = np.array([r.get("queue_len", 1) for r in u_records], dtype=np.float32)
                v_queues = np.array([r.get("queue_len", 1) for r in v_records], dtype=np.float32)
                
                # Cross correlation with lag=1
                u_std = float(np.std(u_queues))
                v_std = float(np.std(v_queues))
                if u_std > 1e-4 and v_std > 1e-4:
                    corr = float(np.corrcoef(u_queues[:-1], v_queues[1:])[0, 1])
                    corr = max(0.1, min(0.99, abs(corr)))
                else:
                    corr = 0.75

                self.edge_weights[(u, v)] = {
                    "correlation": round(corr, 3),
                    "lag_ticks": 1,
                    "transfer_efficiency": round(float(np.mean([r.get("throughput_uph", 60.0) for r in v_records]) / 60.0), 2)
                }

    def simulate_forward_ripple(self, source_station_id: int, slowdown_sec: float, horizon_minutes: int = 15) -> Dict[str, Any]:
        """
        Forward Simulation: Given a slowdown at source station, projects expected downstream queue buildup
        and starvation / blocking ripple over next M minutes.
        """
        projected_impacts = []
        accumulated_delay = slowdown_sec
        curr_station = source_station_id

        step = 1
        while curr_station < self.station_count and step <= horizon_minutes:
            next_station = curr_station + 1
            edge_info = self.edge_weights.get((curr_station, next_station), {"correlation": 0.75, "transfer_efficiency": 0.95})
            
            # Attenuate delay along downstream path based on buffer absorption
            accumulated_delay *= edge_info["correlation"] * edge_info["transfer_efficiency"]
            expected_queue_delta = max(0, int(round(accumulated_delay / 15.0)))
            
            projected_impacts.append({
                "station_id": next_station,
                "minute_offset": step,
                "projected_queue_delta": expected_queue_delta,
                "expected_cycle_delay_sec": round(float(accumulated_delay), 1),
                "risk_level": "High" if accumulated_delay > 15.0 else ("Medium" if accumulated_delay > 6.0 else "Low")
            })
            
            curr_station = next_station
            step += 1

        return {
            "source_station_id": source_station_id,
            "initial_slowdown_sec": slowdown_sec,
            "projected_downstream_impacts": projected_impacts,
            "total_downstream_stations_affected": len(projected_impacts)
        }

    def trace_backward_genealogy(self, defect_station_id: int, current_station_anomalies: Dict[int, float]) -> Dict[str, Any]:
        """
        Backward Tracing: Given a defect at a late inspection station, walks back through the highest
        upstream correlation path to surface the most probable origin station(s).
        """
        genealogy_path = []
        curr_station = defect_station_id
        
        # Traverse upstream
        while curr_station > 1:
            upstream_nodes = self.rev_adj_list.get(curr_station, [])
            if not upstream_nodes:
                break
            
            # Score each upstream candidate based on empirical edge correlation * anomaly score
            best_upstream = None
            best_score = -1.0
            
            for up in upstream_nodes:
                edge = self.edge_weights.get((up, curr_station), {"correlation": 0.75})
                anom_score = current_station_anomalies.get(up, 10.0)
                # Combined path affinity
                affinity = edge["correlation"] * (anom_score / 100.0 + 0.1)
                if affinity > best_score:
                    best_score = affinity
                    best_upstream = up

            if best_upstream is not None:
                genealogy_path.append({
                    "station_id": best_upstream,
                    "anomaly_score": current_station_anomalies.get(best_upstream, 10.0),
                    "path_correlation": self.edge_weights.get((best_upstream, curr_station), {}).get("correlation", 0.75),
                    "confidence": round(float(min(0.99, best_score * 1.5)), 2)
                })
                curr_station = best_upstream
            else:
                break

        # Identify most probable root cause station
        root_cause_station = max(genealogy_path, key=lambda x: x["anomaly_score"]) if genealogy_path else None

        return {
            "defect_manifest_station_id": defect_station_id,
            "most_likely_root_cause_station_id": root_cause_station["station_id"] if root_cause_station else defect_station_id,
            "root_cause_confidence": root_cause_station["confidence"] if root_cause_station else 0.85,
            "genealogy_trace_path": genealogy_path
        }

    def save(self, filepath: str) -> None:
        joblib.dump({
            "station_count": self.station_count,
            "adj_list": self.adj_list,
            "rev_adj_list": self.rev_adj_list,
            "edge_weights": self.edge_weights
        }, filepath)

    def load(self, filepath: str) -> None:
        data = joblib.load(filepath)
        self.station_count = data["station_count"]
        self.adj_list = data["adj_list"]
        self.rev_adj_list = data["rev_adj_list"]
        self.edge_weights = data["edge_weights"]
