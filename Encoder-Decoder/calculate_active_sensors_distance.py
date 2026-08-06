"""
calculate_active_sensors_distance.py

Calculates the average distance from active sensors (sensors that detected the UAV) 
to the UAV for each experiment in the dataset.

Outputs:
1. A Python dictionary / map: {exp_id: avg_distance_in_meters}
2. A detailed CSV summary file: experiment_active_sensor_distances.csv
3. A JSON map file: experiment_active_sensor_distances.json

Usage:
    python calculate_active_sensors_distance.py
    python calculate_active_sensors_distance.py --data_dir ../Data --max_exps 50
    python calculate_active_sensors_distance.py --output_csv experiment_distances.csv
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from tqdm import tqdm


def resolve_dataset_paths(data_dir: str) -> Dict[str, str]:
    """
    Auto-detects and resolves the paths to the dataset files in the Data directory.
    Prefers 'no_anomalies_' files if available, otherwise falls back to standard patterns.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = os.listdir(data_dir)
    prefixes = ["no_anomalies_10sensors_1drone_25000_1exps_", "rssi_1000_exps_"]

    selected_prefix = None
    for prefix in prefixes:
        has_det = any(f.startswith(prefix) and f.endswith("_detections.csv") for f in files)
        has_dist = any(f.startswith(prefix) and f.endswith("_sensor_distances.csv") for f in files)
        if has_det and has_dist:
            selected_prefix = prefix
            break

    def find_file(suffix: str) -> str:
        if selected_prefix:
            target = f"{selected_prefix}{suffix}"
            full_path = os.path.join(data_dir, target)
            if os.path.exists(full_path):
                return full_path
        for f in files:
            if f.endswith(f"_{suffix}"):
                return os.path.join(data_dir, f)
        raise FileNotFoundError(f"Could not find required dataset file with suffix '_{suffix}' in {data_dir}")

    return {
        "det": find_file("detections.csv"),
        "dist": find_file("sensor_distances.csv"),
        "info": find_file("sensor_information.csv"),
        "traj": find_file("drone_trajectories.csv") if any(f.endswith("_drone_trajectories.csv") for f in files) else None
    }


def load_sensor_positions(dist_csv_path: str) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Extracts sensor coordinates (x, y, z) in meters for each (exp_id, sensor_id).
    Returns mapping: (exp_id, sensor_id) -> np.array([x, y, z], dtype=np.float64)
    """
    print(f"[1/3] Loading sensor coordinates from: {os.path.basename(dist_csv_path)}...")
    df_dist = pd.read_csv(dist_csv_path)
    col_exp = "exp_id" if "exp_id" in df_dist.columns else "experiment_id"

    pos_map = {}
    for _, r in df_dist.iterrows():
        eid = int(r[col_exp])
        s1 = int(r["sensor1_id"])
        s2 = int(r["sensor2_id"])
        if (eid, s1) not in pos_map:
            pos_map[(eid, s1)] = np.array([r["x1"], r["y1"], r["z1"]], dtype=np.float64)
        if (eid, s2) not in pos_map:
            pos_map[(eid, s2)] = np.array([r["x2"], r["y2"], r["z2"]], dtype=np.float64)

    print(f"      Mapped {len(pos_map):,} sensor coordinate entries.")
    return pos_map


def parse_coords_fast(val) -> Optional[Tuple[float, float, float]]:
    """
    Fast regex / string parser for drone_positions format like '[(x, y, z)]' or '(x, y, z)'.
    Returns (x, y, z) in meters, or None if parsing fails.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (tuple, list)) and len(val) == 3:
        return float(val[0]), float(val[1]), float(val[2])
    
    val_str = str(val).strip()
    # Strip brackets/parentheses: '[(2839.23, 1928.11, 459.22)]' -> '2839.23, 1928.11, 459.22'
    val_clean = val_str.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = val_clean.split(",")
    if len(parts) >= 3:
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            pass
    return None


def calculate_active_sensors_distance_map(
    data_dir: Optional[str] = None,
    max_exps: Optional[int] = None,
    chunk_size: int = 100000
) -> Tuple[Dict[int, float], pd.DataFrame]:
    """
    Processes all active detections to calculate the average distance from active sensors to UAV
    for each experiment.

    Returns:
        exp_to_avg_distance (Dict[int, float]): Dictionary mapping exp_id -> average 3D distance (meters).
        summary_df (pd.DataFrame): Detailed DataFrame with comprehensive statistics per experiment.
    """
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, "..", "Data"))
        if not os.path.exists(data_dir):
            data_dir = os.path.abspath(os.path.join(script_dir, "Data"))

    paths = resolve_dataset_paths(data_dir)
    pos_map = load_sensor_positions(paths["dist"])

    det_path = paths["det"]
    print(f"[2/3] Processing active detections from: {os.path.basename(det_path)} (streaming chunksize={chunk_size:,})...")

    # Accumulators per experiment: exp_id -> list of metrics
    exp_distances_3d = {}
    exp_distances_2d = {}
    exp_altitudes = {}
    exp_timestamps = {}
    exp_sensor_counts_per_ts = {}

    col_exp = None
    total_processed_rows = 0

    chunk_iter = pd.read_csv(
        det_path,
        chunksize=chunk_size,
        usecols=lambda c: c in ["exp_id", "experiment_id", "sensor_id", "timestamp", "drone_positions", "total_power_dBm"]
    )

    for chunk in chunk_iter:
        if col_exp is None:
            col_exp = "exp_id" if "exp_id" in chunk.columns else "experiment_id"

        total_processed_rows += len(chunk)

        for _, row in chunk.iterrows():
            eid = int(row[col_exp])
            if max_exps is not None and eid > max_exps and eid not in exp_distances_3d:
                continue

            sid = int(row["sensor_id"])
            s_pos = pos_map.get((eid, sid))
            if s_pos is None:
                continue

            uav_coords = parse_coords_fast(row["drone_positions"])
            if uav_coords is None:
                continue

            ux, uy, uz = uav_coords
            uav_pos = np.array([ux, uy, uz], dtype=np.float64)

            # 3D Euclidean Distance
            dist_3d = float(np.linalg.norm(s_pos - uav_pos))
            # 2D Horizontal Distance
            dist_2d = float(np.linalg.norm(s_pos[:2] - uav_pos[:2]))

            if eid not in exp_distances_3d:
                exp_distances_3d[eid] = []
                exp_distances_2d[eid] = []
                exp_altitudes[eid] = []
                exp_timestamps[eid] = set()
                exp_sensor_counts_per_ts[eid] = {}

            exp_distances_3d[eid].append(dist_3d)
            exp_distances_2d[eid].append(dist_2d)
            exp_altitudes[eid].append(uz)

            ts = row["timestamp"]
            exp_timestamps[eid].add(ts)
            exp_sensor_counts_per_ts[eid][ts] = exp_sensor_counts_per_ts[eid].get(ts, 0) + 1

        if max_exps is not None and len(exp_distances_3d) >= max_exps:
            # Check if all targeted exps have enough data
            pass

    print(f"      Processed {total_processed_rows:,} raw detection records across {len(exp_distances_3d)} experiments.")
    print(f"[3/3] Compiling experiment distance summary...")

    summary_records = []
    exp_to_avg_distance = {}

    for eid in sorted(exp_distances_3d.keys()):
        d3_list = np.array(exp_distances_3d[eid], dtype=np.float64)
        d2_list = np.array(exp_distances_2d[eid], dtype=np.float64)
        alt_list = np.array(exp_altitudes[eid], dtype=np.float64)

        avg_3d = float(np.mean(d3_list))
        avg_2d = float(np.mean(d2_list))
        min_3d = float(np.min(d3_list))
        max_3d = float(np.max(d3_list))
        median_3d = float(np.median(d3_list))
        std_3d = float(np.std(d3_list))

        ts_counts = list(exp_sensor_counts_per_ts[eid].values())
        avg_active_sensors = float(np.mean(ts_counts)) if ts_counts else 0.0

        exp_to_avg_distance[eid] = avg_3d

        summary_records.append({
            "exp_id": eid,
            "avg_active_sensor_distance_3d_m": avg_3d,
            "avg_active_sensor_distance_2d_m": avg_2d,
            "median_active_sensor_distance_3d_m": median_3d,
            "std_active_sensor_distance_3d_m": std_3d,
            "min_active_sensor_distance_3d_m": min_3d,
            "max_active_sensor_distance_3d_m": max_3d,
            "avg_active_sensors_per_timestamp": avg_active_sensors,
            "total_active_detections": len(d3_list),
            "unique_timestamps_with_detections": len(exp_timestamps[eid]),
            "avg_uav_altitude_m": float(np.mean(alt_list))
        })

    summary_df = pd.DataFrame(summary_records)
    return exp_to_avg_distance, summary_df


def get_experiment_active_sensor_distance_map(data_dir: Optional[str] = None) -> Dict[int, float]:
    """
    Public API helper function to obtain the {exp_id: avg_distance_m} map directly.
    """
    exp_map, _ = calculate_active_sensors_distance_map(data_dir=data_dir)
    return exp_map


def main():
    parser = argparse.ArgumentParser(description="Calculate average distance from active sensors to UAV per experiment.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to Data directory (default: ../Data)")
    parser.add_argument("--max_exps", type=int, default=None, help="Maximum number of experiments to process")
    parser.add_argument("--output_csv", type=str, default="experiment_active_sensor_distances.csv", help="Path for output CSV")
    parser.add_argument("--output_json", type=str, default="experiment_active_sensor_distances.json", help="Path for output JSON")
    args = parser.parse_args()

    print("=" * 75)
    print(" Average Distance from Active Sensors to UAV per Experiment")
    print("=" * 75)

    exp_map, summary_df = calculate_active_sensors_distance_map(
        data_dir=args.data_dir,
        max_exps=args.max_exps
    )

    # Save CSV
    summary_df.to_csv(args.output_csv, index=False)
    print(f"\n[OK] Detailed experiment statistics saved to: {args.output_csv}")

    # Save JSON map
    with open(args.output_json, "w") as f:
        json.dump(exp_map, f, indent=2)
    print(f"[OK] JSON dictionary map saved to           : {args.output_json}")

    # Print summary table
    print("\n" + "=" * 75)
    print(f" SUMMARY PREVIEW (Total Experiments: {len(summary_df)})")
    print("=" * 75)
    print(f"{'Exp ID':<8} | {'Avg 3D Dist (m)':<16} | {'Avg 2D Dist (m)':<16} | {'Min 3D (m)':<12} | {'Max 3D (m)':<12} | {'Active Sens/TS':<14}")
    print("-" * 85)
    for _, r in summary_df.head(20).iterrows():
        print(f"{int(r['exp_id']):<8} | {r['avg_active_sensor_distance_3d_m']:<16.2f} | {r['avg_active_sensor_distance_2d_m']:<16.2f} | {r['min_active_sensor_distance_3d_m']:<12.2f} | {r['max_active_sensor_distance_3d_m']:<12.2f} | {r['avg_active_sensors_per_timestamp']:<14.2f}")

    if len(summary_df) > 20:
        print(f"... and {len(summary_df) - 20} more experiments in CSV/JSON.")

    print("\n" + "-" * 75)
    print(" GLOBAL DATASET METRICS (Across All Active Detections)")
    print("-" * 75)
    print(f"  Mean Active Sensor Distance (3D)   : {summary_df['avg_active_sensor_distance_3d_m'].mean():.2f} m ({summary_df['avg_active_sensor_distance_3d_m'].mean()/1000:.3f} km)")
    print(f"  Median Active Sensor Distance (3D) : {summary_df['avg_active_sensor_distance_3d_m'].median():.2f} m")
    print(f"  Min Active Sensor Distance Observed: {summary_df['min_active_sensor_distance_3d_m'].min():.2f} m")
    print(f"  Max Active Sensor Distance Observed: {summary_df['max_active_sensor_distance_3d_m'].max():.2f} m")
    print(f"  Avg Active Sensors per Timestep    : {summary_df['avg_active_sensors_per_timestamp'].mean():.2f} sensors")
    print("=" * 75)

    return exp_map, summary_df


if __name__ == "__main__":
    main()
