import json
import glob
import os
import sys

def get_latest_reports(artifacts_dir="models/artifacts"):
    # Find all versioned evaluation reports
    files = glob.glob(os.path.join(artifacts_dir, "evaluation_report_*_*.json"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def compare_reports(current_path, previous_path):
    print(f"Comparing Current: {os.path.basename(current_path)}")
    print(f"       to Previous: {os.path.basename(previous_path)}")
    print("-" * 60)
    
    with open(current_path, "r") as f:
        curr = json.load(f)
    with open(previous_path, "r") as f:
        prev = json.load(f)

    # Defect Classifier
    try:
        curr_rf = curr["models_summary"]["random_forest_defect_classifier"]["held_out_metrics"]
        prev_rf = prev["models_summary"]["random_forest_defect_classifier"]["held_out_metrics"]
        
        print("Defect Classifier Metrics:")
        print(f"  Precision: {prev_rf['precision']:.3f} -> {curr_rf['precision']:.3f} (Delta: {curr_rf['precision'] - prev_rf['precision']:.3f})")
        print(f"  Recall:    {prev_rf['recall']:.3f} -> {curr_rf['recall']:.3f} (Delta: {curr_rf['recall'] - prev_rf['recall']:.3f})")
    except KeyError:
        print("Could not compare Defect Classifier metrics.")

    print("-" * 60)
    
    # LSTM Forecaster
    try:
        curr_lstm = curr["models_summary"]["lstm_forecaster"]["held_out_benchmark"]["lstm"]
        prev_lstm = prev["models_summary"]["lstm_forecaster"]["held_out_benchmark"]["lstm"]
        
        print("LSTM Forecaster Metrics:")
        print(f"  Cycle Time MAE: {prev_lstm['mae']:.3f} -> {curr_lstm['mae']:.3f} (Delta: {curr_lstm['mae'] - prev_lstm['mae']:.3f})")
        if "queue_mae" in curr_lstm and "queue_mae" in prev_lstm:
            print(f"  Queue MAE:      {prev_lstm['queue_mae']:.3f} -> {curr_lstm['queue_mae']:.3f} (Delta: {curr_lstm['queue_mae'] - prev_lstm['queue_mae']:.3f})")
    except KeyError:
        print("Could not compare LSTM Forecaster metrics.")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare_reports(sys.argv[1], sys.argv[2])
    else:
        reports = get_latest_reports()
        if len(reports) < 2:
            print("Not enough reports to compare. Need at least 2.")
            sys.exit(1)
        compare_reports(reports[0], reports[1])
