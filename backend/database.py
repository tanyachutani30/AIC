"""
DigitalTwin.ai - Database Layer (SQLite)
Manages persistent storage for telemetry time-series, scored predictions,
supervisor action logs, and model evaluation metrics.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

DB_FILE = Path("digitaltwin.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    # Ensure tables exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS supervisor_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            station_id INTEGER,
            station_name TEXT,
            action_type TEXT,
            description TEXT,
            status TEXT,
            executed_by TEXT
        )
    """)
    conn.commit()
    return conn


def init_db() -> None:
    """Creates database schema if not already present."""
    conn = get_db_connection()
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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tick INTEGER,
            timestamp TEXT,
            line_id TEXT,
            station_id INTEGER,
            anomaly_score REAL,
            defect_risk_prob REAL,
            is_bottleneck_forecast INTEGER,
            max_forecast_ct REAL,
            max_forecast_queue INTEGER,
            lead_time_ticks INTEGER,
            estimated_health_pct REAL,
            data_confidence TEXT,
            feature_importances_json TEXT,
            is_alert_triggered INTEGER,
            recommended_action TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS supervisor_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            station_id INTEGER,
            station_name TEXT,
            action_type TEXT,
            description TEXT,
            status TEXT,
            executed_by TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS line_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_supervisor_action(station_id: int, station_name: str, action_type: str, description: str) -> Dict[str, Any]:
    conn = get_db_connection()
    cur = conn.cursor()
    import datetime
    now_iso = datetime.datetime.utcnow().isoformat()
    
    cur.execute("""
        INSERT INTO supervisor_actions (timestamp, station_id, station_name, action_type, description, status, executed_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now_iso, station_id, station_name, action_type, description, "EXECUTED_TO_PLC", "Floor Supervisor"))
    
    action_id = cur.lastrowid
    conn.commit()
    conn.close()
    
    return {
        "action_id": action_id,
        "timestamp": now_iso,
        "station_id": station_id,
        "station_name": station_name,
        "action_type": action_type,
        "description": description,
        "status": "EXECUTED_TO_PLC"
    }


def get_recent_actions(limit: int = 10) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM supervisor_actions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
