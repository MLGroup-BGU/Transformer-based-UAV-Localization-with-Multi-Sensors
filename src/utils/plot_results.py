"""
plot_results.py

Visualization and Analysis Script for UAV Localization Predictors.
Scans experiment result directories, aggregates evaluation metrics across noise levels,
and plots performance comparison figures.
"""

import os
import sys
import glob
import re
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use("ggplot")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_noise_level(folder_name: str, file_name: str) -> float:
    match_file = re.search(r"pwr([0-9.]+)", file_name)
    if match_file:
        return float(match_file.group(1))

    match_folder_idx = re.search(r"results_rssi_(\d+)", folder_name)
    if match_folder_idx:
        idx = int(match_folder_idx.group(1))
        return idx * 0.01

    if "comparison_summary" in file_name:
        return 0.0

    return 0.0


def discover_and_load_summaries(base_dir: str) -> pd.DataFrame:
    summary_files = glob.glob(os.path.join(base_dir, "results_*", "*summary*.csv"))
    summary_files.extend(glob.glob(os.path.join(base_dir, "outputs", "*summary*.csv")))
    summary_files.extend(glob.glob(os.path.join(base_dir, "*summary*.csv")))

    if not summary_files:
        print(f"[WARNING] No summary CSV files found in {base_dir}")
        return pd.DataFrame()

    all_rows = []
    print(f"[INFO] Discovered {len(summary_files)} summary CSV files:")

    for file_path in summary_files:
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        noise_std = parse_noise_level(folder_name, file_name)

        print(f"  - {folder_name}/{file_name} -> Noise Std: {noise_std:.2f} ({noise_std*100:.1f}%)")

        df = pd.read_csv(file_path)
        df["Noise_Std"] = noise_std
        df["Noise_Pct"] = noise_std * 100.0
        df["Source_Folder"] = folder_name
        all_rows.append(df)

    combined_df = pd.concat(all_rows, ignore_index=True)
    combined_df.sort_values(by=["Noise_Std", "Predictor_Name"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


def load_predictions_for_noise_levels(base_dir: str) -> Dict[float, pd.DataFrame]:
    pred_files = glob.glob(os.path.join(base_dir, "results_*", "*predictions*.csv"))
    pred_files.extend(glob.glob(os.path.join(base_dir, "outputs", "*predictions*.csv")))
    preds_map = {}

    for file_path in pred_files:
        folder_name = os.path.basename(os.path.dirname(file_path))
        file_name = os.path.basename(file_path)
        noise_std = parse_noise_level(folder_name, file_name)

        df = pd.read_csv(file_path)
        preds_map[noise_std] = df

    return preds_map


def plot_noise_vs_3d_metrics(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("UAV Localization Accuracy vs Power Noise Level (%)", fontsize=16, fontweight="bold")

    predictors = df["Predictor_Name"].unique()
    colors = {"Encoder-Decoder": "#1f77b4", "RSSI": "#ff7f0e", "Other": "#2ca02c"}

    def get_color(name: str):
        for k, v in colors.items():
            if k.lower() in name.lower():
                return v
        return colors["Other"]

    metrics = [
        ("3D_Mean_Error (m)", "3D Mean Error (meters)", axes[0, 0]),
        ("3D_Median_Error (m)", "3D Median Error (meters)", axes[0, 1]),
        ("3D_RMSE (m)", "3D RMSE (meters)", axes[1, 0]),
        ("3D_P95_Error (m)", "3D 95th Percentile Error (meters)", axes[1, 1]),
    ]

    for metric_col, title, ax in metrics:
        if metric_col not in df.columns:
            continue

        for pred in predictors:
            sub = df[df["Predictor_Name"] == pred].sort_values("Noise_Pct")
            c = get_color(pred)
            ax.plot(
                sub["Noise_Pct"],
                sub[metric_col],
                marker="o",
                linewidth=2.5,
                markersize=6,
                label=pred,
                color=c,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Linear Power Noise Std (%)", fontsize=11)
        ax.set_ylabel("Error (m)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(output_dir, "01_noise_vs_3d_error_metrics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_noise_vs_2d_and_z_error(df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("2D Spatial (XY) vs Z-Axis (Altitude) Mean Errors", fontsize=15, fontweight="bold")

    predictors = df["Predictor_Name"].unique()

    metrics = [
        ("2D_Mean_Error (m)", "2D (XY) Mean Error (meters)", axes[0]),
        ("Z_Mean_Error (m)", "Z-Axis (Altitude) Mean Error (meters)", axes[1]),
    ]

    for metric_col, title, ax in metrics:
        if metric_col not in df.columns:
            continue

        for pred in predictors:
            sub = df[df["Predictor_Name"] == pred].sort_values("Noise_Pct")
            ax.plot(
                sub["Noise_Pct"],
                sub[metric_col],
                marker="s",
                linewidth=2.5,
                markersize=6,
                label=pred,
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Linear Power Noise Std (%)", fontsize=11)
        ax.set_ylabel("Error (m)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path = os.path.join(output_dir, "02_noise_vs_2d_and_z_error.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_noise_vs_mse_loss(df: pd.DataFrame, output_dir: str):
    if "PyTorch_MSELoss (km^2)" not in df.columns:
        return

    plt.figure(figsize=(9, 6))
    predictors = df["Predictor_Name"].unique()

    for pred in predictors:
        sub = df[df["Predictor_Name"] == pred].sort_values("Noise_Pct")
        plt.plot(
            sub["Noise_Pct"],
            sub["PyTorch_MSELoss (km^2)"],
            marker="^",
            linewidth=2.5,
            markersize=7,
            label=pred,
        )

    plt.title("PyTorch MSE Loss ($km^2$) vs Noise Level (%)", fontsize=14, fontweight="bold")
    plt.xlabel("Linear Power Noise Std (%)", fontsize=11)
    plt.ylabel("PyTorch MSE Loss ($km^2$)", fontsize=11)
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "03_noise_vs_mse_loss.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_error_cdfs(preds_map: Dict[float, pd.DataFrame], output_dir: str):
    if not preds_map:
        return

    sorted_noises = sorted(preds_map.keys())
    target_noises = [n for n in [0.01, 0.05, 0.10] if n in sorted_noises]
    if not target_noises and sorted_noises:
        target_noises = [sorted_noises[0], sorted_noises[len(sorted_noises)//2], sorted_noises[-1]]

    fig, axes = plt.subplots(1, len(target_noises), figsize=(5 * len(target_noises), 5), sharey=True)
    if len(target_noises) == 1:
        axes = [axes]

    fig.suptitle("Cumulative Distribution Function (CDF) of 3D Spatial Errors", fontsize=15, fontweight="bold")

    for ax, noise in zip(axes, target_noises):
        df_pred = preds_map[noise]
        predictors = df_pred["predictor_name"].unique()

        for pred in predictors:
            sub = df_pred[df_pred["predictor_name"] == pred]
            errors = np.sort(sub["error_3d_m"].dropna().values)
            if len(errors) == 0:
                continue

            cdf = np.arange(1, len(errors) + 1) / len(errors)
            ax.plot(errors, cdf, label=pred, linewidth=2.0)

        ax.set_title(f"Noise Std = {noise*100:.0f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("3D Error (meters)", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Cumulative Probability", fontsize=11)
        ax.set_xlim(0, 300)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    out_path = os.path.join(output_dir, "04_error_cdfs_across_noise.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_trajectory_comparison(preds_map: Dict[float, pd.DataFrame], output_dir: str):
    if not preds_map:
        return

    noise_level = min(preds_map.keys())
    df_pred = preds_map[noise_level]

    if df_pred.empty:
        return

    predictors = df_pred["predictor_name"].unique()
    first_pred = predictors[0]
    sub_gt = df_pred[df_pred["predictor_name"] == first_pred].sort_values("timestamp")

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sub_gt["true_x"], sub_gt["true_y"], "k--", label="Ground Truth", linewidth=2.5)

    for pred in predictors:
        sub_p = df_pred[df_pred["predictor_name"] == pred].sort_values("timestamp")
        ax1.plot(sub_p["pred_x"], sub_p["pred_y"], label=f"Pred: {pred}", linewidth=1.5, alpha=0.85)

    ax1.set_title(f"2D Flight Trajectory (XY Plane) [Noise {noise_level*100:.0f}%]", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X Coordinate (m)", fontsize=11)
    ax1.set_ylabel("Y Coordinate (m)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot(sub_gt["true_x"], sub_gt["true_y"], sub_gt["true_z"], "k--", label="Ground Truth", linewidth=2.5)

    for pred in predictors:
        sub_p = df_pred[df_pred["predictor_name"] == pred].sort_values("timestamp")
        ax2.plot(sub_p["pred_x"], sub_p["pred_y"], sub_p["pred_z"], label=f"Pred: {pred}", linewidth=1.5, alpha=0.85)

    ax2.set_title(f"3D UAV Trajectory Flight Path [Noise {noise_level*100:.0f}%]", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X (m)", fontsize=10)
    ax2.set_ylabel("Y (m)", fontsize=10)
    ax2.set_zlabel("Z (m)", fontsize=10)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "05_trajectory_3d_and_2d_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_error_over_time(preds_map: Dict[float, pd.DataFrame], output_dir: str):
    if not preds_map:
        return

    noise_level = min(preds_map.keys())
    df_pred = preds_map[noise_level]
    predictors = df_pred["predictor_name"].unique()

    plt.figure(figsize=(12, 5))
    for pred in predictors:
        sub = df_pred[df_pred["predictor_name"] == pred].sort_values("timestamp")
        plt.plot(sub["timestamp"], sub["error_3d_m"], label=pred, linewidth=1.5, alpha=0.85)

    plt.title(f"3D Localization Error over Timestamps [Noise {noise_level*100:.0f}%]", fontsize=14, fontweight="bold")
    plt.xlabel("Timestamp Step", fontsize=11)
    plt.ylabel("3D Error (meters)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "06_error_over_time.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_latency_and_throughput(df: pd.DataFrame, output_dir: str):
    if "Latency_Per_Timestamp (ms)" not in df.columns:
        return

    avg_df = df.groupby("Predictor_Name")[["Latency_Per_Timestamp (ms)", "Throughput (steps/s)"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Predictor Computational Efficiency Comparison", fontsize=15, fontweight="bold")

    ax1 = axes[0]
    bars1 = ax1.bar(avg_df["Predictor_Name"], avg_df["Latency_Per_Timestamp (ms)"], color=["#1f77b4", "#ff7f0e"])
    ax1.set_title("Inference Latency (ms per timestamp)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10)

    ax2 = axes[1]
    bars2 = ax2.bar(avg_df["Predictor_Name"], avg_df["Throughput (steps/s)"], color=["#1f77b4", "#ff7f0e"])
    ax2.set_title("Throughput (timestamps processed / second)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Steps / sec", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"{yval:.1f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_path = os.path.join(output_dir, "07_latency_and_throughput.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot all evaluation results across previous experiments.")
    parser.add_argument("--base_dir", type=str, default=SCRIPT_DIR, help="Base directory containing results_rssi_* or outputs folders.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated plots.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")

    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(base_dir, "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)

    combined_summary_df = discover_and_load_summaries(base_dir)

    if combined_summary_df.empty:
        print("[ERROR] No experiment summary data found. Exiting.")
        sys.exit(1)

    print(f"\n[INFO] Successfully loaded {len(combined_summary_df)} summary records across noise levels.")

    cols_to_print = [
        "Noise_Pct", "Predictor_Name", "3D_Mean_Error (m)", "3D_Median_Error (m)",
        "3D_RMSE (m)", "3D_P95_Error (m)", "2D_Mean_Error (m)", "Z_Mean_Error (m)"
    ]
    valid_cols = [c for c in cols_to_print if c in combined_summary_df.columns]
    print("\n--- AGGREGATED METRICS TABLE ---")
    print(combined_summary_df[valid_cols].to_string(index=False))
    print("--------------------------------\n")

    preds_map = load_predictions_for_noise_levels(base_dir)
    print(f"[INFO] Loaded detailed timestamp predictions for {len(preds_map)} noise levels.")

    print("\n[INFO] Generating plots...")
    plot_noise_vs_3d_metrics(combined_summary_df, output_dir)
    plot_noise_vs_2d_and_z_error(combined_summary_df, output_dir)
    plot_noise_vs_mse_loss(combined_summary_df, output_dir)
    plot_error_cdfs(preds_map, output_dir)
    plot_trajectory_comparison(preds_map, output_dir)
    plot_error_over_time(preds_map, output_dir)
    plot_latency_and_throughput(combined_summary_df, output_dir)

    print(f"\n[SUCCESS] All plots generated and saved to: {output_dir}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
