"""
data_analysis.py

Statistical Analysis Script for UAV Localization Test Dataset (5K_4).
Analyzes:
1. Sensor-to-UAV 3D distance distribution
2. Active sensors count per timestamp distribution
3. RSSI power distribution and empirical path-loss propagation curve
4. Active sectors per sensor distribution
5. UAV 3D spatial trajectory bounds and altitude distribution
6. UAV Transmitter Power (P_tx) distribution

Saves CSV summary tables and high-resolution PNG plots to outputs/data_analysis/.
"""

import os
import sys
import ast
import re
import argparse
import time
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def resolve_5k4_data_dir() -> str:
    """
    Locates the 5th test set folder (5K_4) within data directory.
    """
    candidate_paths = [
        os.path.join(ROOT_DIR, "data", "test_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_4"),
        os.path.join(ROOT_DIR, "Data", "test_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_4"),
        os.path.join(ROOT_DIR, "test_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_4"),
        os.path.join(ROOT_DIR, "data", "5K_4"),
        os.path.join(ROOT_DIR, "Data", "5K_4"),
    ]

    for p in candidate_paths:
        if os.path.exists(p) and os.path.isdir(p):
            return os.path.abspath(p)

    import glob
    search_roots = [os.path.join(ROOT_DIR, "data"), os.path.join(ROOT_DIR, "Data"), ROOT_DIR]
    for s_root in search_roots:
        if os.path.exists(s_root):
            matches = glob.glob(os.path.join(s_root, "**", "*sensor_detection_summary.csv"), recursive=True)
            if matches:
                return os.path.abspath(os.path.dirname(matches[0]))

    raise FileNotFoundError(
        f"Could not locate raw test dataset folder containing 'sensor_detection_summary.csv'."
    )


def compute_quantile_stats(series: pd.Series, name: str) -> Dict[str, float]:
    """
    Computes standard descriptive and quantile statistics for a series.
    """
    s = series.dropna()
    if len(s) == 0:
        return {"metric": name, "count": 0, "mean": np.nan, "std": np.nan, "min": np.nan,
                "p10": np.nan, "p25": np.nan, "p50_median": np.nan, "p75": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}

    return {
        "metric": name,
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50_median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def parse_drone_position_str(pos_str: str) -> Optional[np.ndarray]:
    """
    Parses drone position tuple string e.g. "[(690.10, 817.59, 1250.84)]" into np.ndarray([x, y, z]).
    """
    if pd.isna(pos_str) or not isinstance(pos_str, str):
        return None
    try:
        match = re.search(r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)", pos_str)
        if match:
            return np.array([float(match.group(1)), float(match.group(2)), float(match.group(3))], dtype=float)
        parsed = ast.literal_eval(pos_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            elem = parsed[0]
            if isinstance(elem, (list, tuple)) and len(elem) >= 3:
                return np.array([float(elem[0]), float(elem[1]), float(elem[2])], dtype=float)
    except Exception:
        pass
    return None


class DatasetAnalyzer:
    """
    Analyzer for UAV localization dataset (5K_4).
    """

    def __init__(self, data_dir: str, output_dir: str, max_exps: Optional[int] = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_exps = max_exps
        os.makedirs(self.output_dir, exist_ok=True)

    def run_analysis(self):
        print("=" * 80)
        print(f" STATISTICAL DATASET ANALYSIS FOR TEST SET (5K_4)")
        print(f" Data Directory  : {self.data_dir}")
        print(f" Output Directory: {self.output_dir}")
        print("=" * 80)

        # 1. Load data
        print("\n[INFO] Reading dataset CSV files...")
        det_path = os.path.join(self.data_dir, "sensor_detection_summary.csv")
        dist_path = os.path.join(self.data_dir, "sensor_distances.csv")
        traj_path = os.path.join(self.data_dir, "drone_trajectories.csv")
        drone_info_path = os.path.join(self.data_dir, "drone_information.csv")

        det_df = pd.read_csv(det_path)
        dist_df = pd.read_csv(dist_path) if os.path.exists(dist_path) else pd.DataFrame()
        traj_df = pd.read_csv(traj_path) if os.path.exists(traj_path) else pd.DataFrame()
        drone_info_df = pd.read_csv(drone_info_path) if os.path.exists(drone_info_path) else pd.DataFrame()

        exp_col = "experiment_id" if "experiment_id" in det_df.columns else "exp_id"

        if self.max_exps is not None:
            unique_exps = np.sort(det_df[exp_col].unique())[:self.max_exps]
            det_df = det_df[det_df[exp_col].isin(unique_exps)].copy()
            if not traj_df.empty and exp_col in traj_df.columns:
                traj_df = traj_df[traj_df[exp_col].isin(unique_exps)].copy()
            if not dist_df.empty:
                dist_exp_col = "exp_id" if "exp_id" in dist_df.columns else ("experiment_id" if "experiment_id" in dist_df.columns else None)
                if dist_exp_col and dist_exp_col in dist_df.columns:
                    dist_df = dist_df[dist_df[dist_exp_col].isin(unique_exps)].copy()
            if not drone_info_df.empty and exp_col in drone_info_df.columns:
                drone_info_df = drone_info_df[drone_info_df[exp_col].isin(unique_exps)].copy()

        print(f"  - Total Detections Analyzed: {len(det_df):,}")
        print(f"  - Unique Experiments       : {det_df[exp_col].nunique():,}")

        # 2. Extract Sensor 3D Coordinates per Experiment
        print("\n[INFO] Extracting sensor 3D positions...")
        sensor_pos_map: Dict[Any, Dict[int, np.ndarray]] = {}
        if not dist_df.empty:
            dist_exp_col = "exp_id" if "exp_id" in dist_df.columns else ("experiment_id" if "experiment_id" in dist_df.columns else (dist_df.columns[0]))
            for exp_id, grp in dist_df.groupby(dist_exp_col):
                exp_map = {}
                for _, r in grp.iterrows():
                    s1, s2 = int(r["sensor1_id"]), int(r["sensor2_id"])
                    if s1 not in exp_map and "x1" in r:
                        exp_map[s1] = np.array([r["x1"], r["y1"], r["z1"]], dtype=float)
                    if s2 not in exp_map and "x2" in r:
                        exp_map[s2] = np.array([r["x2"], r["y2"], r["z2"]], dtype=float)
                sensor_pos_map[exp_id] = exp_map

        # Fallback if sensor_distances.csv is fixed across experiments
        first_exp_map = list(sensor_pos_map.values())[0] if sensor_pos_map else {}

        # 3. Analyze Active Sensors per Timestamp
        print("\n[INFO] Computing Active Sensors Distribution per Timestamp...")
        active_sensors_series = det_df.groupby([exp_col, "timestamp"])["sensor_id"].nunique()
        active_sensors_stats = compute_quantile_stats(active_sensors_series, "active_sensors_per_timestamp")

        active_sensors_df = pd.DataFrame([active_sensors_stats])
        active_sensors_df.to_csv(os.path.join(self.output_dir, "active_sensors_per_timestamp_stats.csv"), index=False)

        # 4. Analyze Sensor-UAV 3D Distances & RSSI
        print("\n[INFO] Computing Sensor-to-UAV 3D Distances & Power Propagation...")
        pow_col = "total_power_dBm" if "total_power_dBm" in det_df.columns else "received_power"

        distances_3d = []
        distances_2d = []
        altitude_diffs = []
        rssi_powers = []
        sensor_ids_list = []

        for idx, row in det_df.iterrows():
            e_id = row[exp_col]
            s_id = int(row["sensor_id"])
            p_rssi = float(row[pow_col])

            # Get sensor pos
            exp_sensors = sensor_pos_map.get(e_id, first_exp_map)
            s_pos = exp_sensors.get(s_id, None)

            # Get UAV pos
            u_pos = parse_drone_position_str(row.get("drone_positions", None))

            if s_pos is not None and u_pos is not None:
                d_3d = np.linalg.norm(u_pos - s_pos)
                d_2d = np.linalg.norm(u_pos[:2] - s_pos[:2])
                dz = abs(u_pos[2] - s_pos[2])

                distances_3d.append(d_3d)
                distances_2d.append(d_2d)
                altitude_diffs.append(dz)
                rssi_powers.append(p_rssi)
                sensor_ids_list.append(s_id)

        s_dist_3d = pd.Series(distances_3d)
        s_dist_2d = pd.Series(distances_2d)
        s_alt_diff = pd.Series(altitude_diffs)
        s_rssi = pd.Series(rssi_powers)

        distance_stats_rows = [
            compute_quantile_stats(s_dist_3d, "sensor_uav_3d_distance_m"),
            compute_quantile_stats(s_dist_2d, "sensor_uav_2d_distance_m"),
            compute_quantile_stats(s_alt_diff, "sensor_uav_z_altitude_diff_m"),
        ]
        distance_stats_df = pd.DataFrame(distance_stats_rows)
        distance_stats_df.to_csv(os.path.join(self.output_dir, "sensor_uav_distance_stats.csv"), index=False)

        rssi_stats = compute_quantile_stats(s_rssi, "rssi_power_dBm")
        rssi_stats_df = pd.DataFrame([rssi_stats])
        rssi_stats_df.to_csv(os.path.join(self.output_dir, "rssi_power_stats.csv"), index=False)

        # 5. UAV Transmitter Power Distribution (P_tx)
        print("\n[INFO] Computing UAV Transmitter Power (P_tx) Distribution...")
        if not drone_info_df.empty and "drone_transmitter_power" in drone_info_df.columns:
            ptx_series = drone_info_df["drone_transmitter_power"].dropna()
        else:
            ptx_series = pd.Series(dtype=float)

        ptx_stats = compute_quantile_stats(ptx_series, "uav_transmitter_power_dBm")
        ptx_stats_df = pd.DataFrame([ptx_stats])
        ptx_stats_df.to_csv(os.path.join(self.output_dir, "uav_transmitter_power_stats.csv"), index=False)

        # 6. UAV Trajectory & Spatial Distribution
        print("\n[INFO] Computing UAV Trajectory Spatial Distribution...")
        if not traj_df.empty:
            x_vals = pd.concat([traj_df["start_x"], traj_df["end_x"]])
            y_vals = pd.concat([traj_df["start_y"], traj_df["end_y"]])
            z_vals = pd.concat([traj_df["start_z"], traj_df["end_z"]])
            speed_vals = traj_df["speed"]
        else:
            x_vals = pd.Series([u[0] for u in [parse_drone_position_str(p) for p in det_df["drone_positions"]] if u is not None])
            y_vals = pd.Series([u[1] for u in [parse_drone_position_str(p) for p in det_df["drone_positions"]] if u is not None])
            z_vals = pd.Series([u[2] for u in [parse_drone_position_str(p) for p in det_df["drone_positions"]] if u is not None])
            speed_vals = pd.Series(dtype=float)

        spatial_stats_rows = [
            compute_quantile_stats(x_vals, "uav_position_x_m"),
            compute_quantile_stats(y_vals, "uav_position_y_m"),
            compute_quantile_stats(z_vals, "uav_altitude_z_m"),
        ]
        if not speed_vals.empty:
            spatial_stats_rows.append(compute_quantile_stats(speed_vals, "uav_speed_m_s"))

        spatial_stats_df = pd.DataFrame(spatial_stats_rows)
        spatial_stats_df.to_csv(os.path.join(self.output_dir, "uav_spatial_trajectory_stats.csv"), index=False)

        # 7. Generate Dataset Summary Overview CSV
        overall_summary = {
            "total_experiments": int(det_df[exp_col].nunique()),
            "total_detections": int(len(det_df)),
            "total_unique_timestamps": int(det_df.groupby(exp_col)["timestamp"].nunique().sum()),
            "mean_active_sensors_per_ts": float(active_sensors_series.mean()),
            "min_active_sensors_per_ts": int(active_sensors_series.min()),
            "max_active_sensors_per_ts": int(active_sensors_series.max()),
            "mean_3d_distance_m": float(s_dist_3d.mean()),
            "median_3d_distance_m": float(s_dist_3d.median()),
            "p95_3d_distance_m": float(s_dist_3d.quantile(0.95)),
            "mean_uav_altitude_m": float(z_vals.mean()),
            "min_uav_altitude_m": float(z_vals.min()),
            "max_uav_altitude_m": float(z_vals.max()),
            "mean_rssi_dBm": float(s_rssi.mean()),
            "min_rssi_dBm": float(s_rssi.min()),
            "max_rssi_dBm": float(s_rssi.max()),
            "mean_ptx_power_dBm": float(ptx_series.mean()) if not ptx_series.empty else np.nan,
            "min_ptx_power_dBm": float(ptx_series.min()) if not ptx_series.empty else np.nan,
            "max_ptx_power_dBm": float(ptx_series.max()) if not ptx_series.empty else np.nan,
        }
        pd.DataFrame([overall_summary]).to_csv(os.path.join(self.output_dir, "overall_dataset_summary.csv"), index=False)

        # 8. Generate Image Plots
        print("\n[INFO] Generating Statistical Visualization Plots...")

        # Plot 1: Sensor-UAV 3D Distance Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(s_dist_3d, bins=50, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
        ax.axvline(s_dist_3d.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {s_dist_3d.mean():.1f} m")
        ax.axvline(s_dist_3d.median(), color="green", linestyle="-.", linewidth=2, label=f"Median: {s_dist_3d.median():.1f} m")
        ax.set_title("Sensor-to-UAV 3D Distance Distribution (5K_4)", fontsize=13, fontweight="bold")
        ax.set_xlabel("3D Distance (meters)", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "sensor_uav_distance_distribution.png"), dpi=300)
        plt.close()

        # Plot 2: Active Sensors Distribution per Timestamp
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = active_sensors_series.value_counts().sort_index()
        bars = ax.bar(counts.index, counts.values, color="#2ca02c", edgecolor="black", alpha=0.8)
        ax.set_title("Active Sensors Count per Timestamp Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Active Detecting Sensors", fontsize=11)
        ax.set_ylabel("Timestamp Count", fontsize=11)
        ax.set_xticks(np.arange(1, 11))
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "active_sensors_distribution.png"), dpi=300)
        plt.close()

        # Plot 3: Empirical RSSI vs 3D Distance with Log-Distance Path Loss Fit
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(s_dist_3d) > 20000:
            sample_idx = np.random.choice(len(s_dist_3d), size=20000, replace=False)
            d_sample = s_dist_3d.iloc[sample_idx].values
            p_sample = s_rssi.iloc[sample_idx].values
        else:
            d_sample = s_dist_3d.values
            p_sample = s_rssi.values

        ax.scatter(d_sample, p_sample, alpha=0.15, s=8, color="#9467bd", label="Detections (Raw RSSI)")

        d_fit = np.linspace(max(10, d_sample.min()), d_sample.max(), 200)
        valid = (d_sample > 0) & (~np.isnan(p_sample))
        log_d = np.log10(d_sample[valid])
        p_val = p_sample[valid]
        slope, intercept = np.polyfit(log_d, p_val, 1)
        est_n = -slope / 10.0

        p_fit = intercept + slope * np.log10(d_fit)
        ax.plot(d_fit, p_fit, color="red", linewidth=2.5, label=f"Fitted Path Loss (n={est_n:.2f})")

        ax.set_title("Empirical RSSI Power vs. 3D Distance", fontsize=13, fontweight="bold")
        ax.set_xlabel("3D Sensor-to-UAV Distance (meters)", fontsize=11)
        ax.set_ylabel("Received Power (dBm)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "rssi_vs_3d_distance_pathloss.png"), dpi=300)
        plt.close()

        # Plot 4: UAV Altitude & Spatial Boundaries
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(z_vals, bins=40, color="#ff7f0e", edgecolor="black", alpha=0.75)
        ax1.axvline(z_vals.mean(), color="red", linestyle="--", label=f"Mean Z: {z_vals.mean():.1f} m")
        ax1.set_title("UAV Flight Altitude (Z) Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Altitude Z (meters)", fontsize=10)
        ax1.set_ylabel("Frequency", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2.scatter(x_vals[::10], y_vals[::10], alpha=0.2, s=5, color="#1f77b4")
        ax2.set_title("UAV 2D Flight Trajectory Coverage (X, Y)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("X Position (meters)", fontsize=10)
        ax2.set_ylabel("Y Position (meters)", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "uav_altitude_and_spatial_distribution.png"), dpi=300)
        plt.close()

        # Plot 5: UAV Transmitter Power (P_tx) Distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        if not ptx_series.empty:
            ax.hist(ptx_series, bins=30, color="#17becf", edgecolor="black", alpha=0.8)
            ax.axvline(ptx_series.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean P_tx: {ptx_series.mean():.2f} dBm")
            ax.axvline(ptx_series.median(), color="green", linestyle="-.", linewidth=2, label=f"Median P_tx: {ptx_series.median():.2f} dBm")
            ax.set_title("UAV Transmitter Power (P_tx) Distribution", fontsize=13, fontweight="bold")
            ax.set_xlabel("Transmitter Power P_tx (dBm)", fontsize=11)
            ax.set_ylabel("Frequency (Experiments)", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
        else:
            ax.text(0.5, 0.5, "No Drone Transmitter Power Data", ha="center", va="center", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "uav_transmitter_power_distribution.png"), dpi=300)
        plt.close()

        # Plot 6: Combined Analysis Dashboard
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("UAV Multi-Sensor Localization Test Dataset (5K_4) Statistical Dashboard", fontsize=15, fontweight="bold")

        # 1. 3D Distance
        axs[0, 0].hist(s_dist_3d, bins=40, color="#1f77b4", edgecolor="black", alpha=0.75)
        axs[0, 0].set_title("3D Distance Distribution (m)", fontsize=11, fontweight="bold")
        axs[0, 0].set_xlabel("Meters")

        # 2. Active Sensors
        axs[0, 1].bar(counts.index, counts.values, color="#2ca02c", edgecolor="black", alpha=0.8)
        axs[0, 1].set_title("Active Sensors / Timestamp", fontsize=11, fontweight="bold")
        axs[0, 1].set_xlabel("Active Sensor Count")

        # 3. RSSI vs Distance
        axs[0, 2].scatter(d_sample, p_sample, alpha=0.1, s=4, color="#9467bd")
        axs[0, 2].plot(d_fit, p_fit, color="red", linewidth=2)
        axs[0, 2].set_title(f"RSSI vs Distance (n={est_n:.2f})", fontsize=11, fontweight="bold")
        axs[0, 2].set_xlabel("Distance (m)")
        axs[0, 2].set_ylabel("Power (dBm)")

        # 4. Altitude Z
        axs[1, 0].hist(z_vals, bins=40, color="#ff7f0e", edgecolor="black", alpha=0.75)
        axs[1, 0].set_title("UAV Altitude Z (m)", fontsize=11, fontweight="bold")
        axs[1, 0].set_xlabel("Altitude (m)")

        # 5. 2D Coverage
        axs[1, 1].scatter(x_vals[::20], y_vals[::20], alpha=0.2, s=3, color="#1f77b4")
        axs[1, 1].set_title("2D Trajectory Coverage (X, Y)", fontsize=11, fontweight="bold")
        axs[1, 1].set_xlabel("X (m)")
        axs[1, 1].set_ylabel("Y (m)")

        # 6. UAV Transmitter Power (P_tx)
        if not ptx_series.empty:
            axs[1, 2].hist(ptx_series, bins=30, color="#17becf", edgecolor="black", alpha=0.8)
            axs[1, 2].axvline(ptx_series.mean(), color="red", linestyle="--", linewidth=1.5)
            axs[1, 2].set_title(f"UAV P_tx Dist (Mean={ptx_series.mean():.1f} dBm)", fontsize=11, fontweight="bold")
            axs[1, 2].set_xlabel("P_tx Power (dBm)")
        else:
            axs[1, 2].text(0.5, 0.5, "No P_tx Data", ha="center", va="center")

        for ax_item in axs.flat:
            ax_item.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(self.output_dir, "dataset_analysis_dashboard.png"), dpi=300)
        plt.close()

        print("\n" + "=" * 80)
        print(" [OK] DATASET ANALYSIS COMPLETE!")
        print(f"  - CSV Summaries Saved to : {self.output_dir}")
        print(f"  - Plots & Dashboard to   : {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze UAV Localization Test Dataset (5K_4).")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to 5K_4 dataset directory.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save CSVs and PNG plots.")
    parser.add_argument("--max_exps", type=int, default=None, help="Maximum number of experiments to process.")

    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir else resolve_5k4_data_dir()
    output_dir = args.output_dir if args.output_dir else os.path.join(ROOT_DIR, "outputs", "data_analysis")

    analyzer = DatasetAnalyzer(data_dir=data_dir, output_dir=output_dir, max_exps=args.max_exps)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
