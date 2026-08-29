import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def run_eda(db_path="digitaltwin.db", output_dir="models/evaluation"):
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return
        
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM telemetry WHERE sensor_rich = 0", conn)
    conn.close()

    if df.empty:
        print("No dark station telemetry found.")
        return

    # The telemetry table stores ground truth fields as native columns
    # is_anomaly and is_jammed can be constructed directly from the DB columns
    df["is_jammed"] = df["scenario_type"].apply(lambda x: "jam" in str(x)) | (df["is_bottleneck"] == 1)
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    
    plt.figure(figsize=(12, 10))
    
    # Scatter 1: Noise vs Optical Cycle Time, colored by Anomaly
    plt.subplot(2, 1, 1)
    anom = df[df["is_anomaly"]]
    norm = df[~df["is_anomaly"]]
    
    plt.scatter(norm["ambient_noise_db"], norm["optical_estimated_cycle_time"], label="Normal", alpha=0.5, color='blue')
    plt.scatter(anom["ambient_noise_db"], anom["optical_estimated_cycle_time"], label="Anomaly", alpha=0.7, color='red')
    
    plt.title("Dark Station Proxies: Noise vs Optical Cycle Time")
    plt.xlabel("Ambient Noise (dB)")
    plt.ylabel("Optical Est. Cycle Time (s)")
    plt.legend()
    
    # Scatter 2: Noise over Time for a specific station (e.g. station_id 5)
    plt.subplot(2, 1, 2)
    st5 = df[df["station_id"] == 5].sort_values("tick")
    if not st5.empty:
        plt.plot(st5["tick"], st5["ambient_noise_db"], label="Ambient Noise dB", color='purple')
        
        jammed = st5[st5["is_jammed"]]
        plt.scatter(jammed["tick"], jammed["ambient_noise_db"], color='red', label="Jammed/Bottleneck", zorder=5)
        
        plt.title("Time Series of Ambient Noise (Station 5)")
        plt.xlabel("Tick")
        plt.ylabel("Noise (dB)")
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/eda_dark_station_proxies.png")
    print(f"EDA plot saved to {output_dir}/eda_dark_station_proxies.png")

if __name__ == "__main__":
    run_eda()
