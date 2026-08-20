"""
main.py

Main Evaluation Script for UAV Localization Predictors on Raw Test Data.
Extracts raw data from test dataset folders, applies noise in linear power domain if requested,
runs predictors, computes side-by-side spatial & computational metrics, and saves outputs.
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.predictors.base_predictor import BaseUAVPredictor
from src.predictors.encoder_decoder_predictor import EncoderDecoderPredictor
from src.predictors.rssi_wcl_predictor import RSSIWCLPredictor
from src.predictors.rssi_trilateration_predictor import RSSITrilaterationPredictor


def resolve_5k4_data_dir() -> str:
    """
    Locates the 5th test set folder (5K_4) or any valid dataset folder within data directory.
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
        f"Could not locate raw test dataset folder containing 'sensor_detection_summary.csv'. Checked locations:\n"
        + "\n".join(f" - {p}" for p in candidate_paths)
        + "\n\nPlease run 'python data/download_drive_data.py' or verify that raw CSV datasets exist in 'data/'."
    )


def load_raw_dataset_files(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Loads raw CSV data files directly from the specified experiment folder into memory.
    """
    print(f"[INFO] Loading raw dataset files from: {data_dir}")
    files = os.listdir(data_dir)

    def find_file(suffix: str) -> str:
        for f in files:
            if f.endswith(suffix):
                return os.path.join(data_dir, f)
        raise FileNotFoundError(f"Could not find required dataset file with suffix '{suffix}' in {data_dir}")

    det_path = find_file("sensor_detection_summary.csv")
    info_path = find_file("sensor_information.csv")
    dist_path = find_file("sensor_distances.csv")
    drone_info_path = find_file("drone_information.csv")
    traj_path = find_file("drone_trajectories.csv")

    det_df = pd.read_csv(det_path)
    info_df = pd.read_csv(info_path)
    dist_df = pd.read_csv(dist_path)
    drone_info_df = pd.read_csv(drone_info_path)
    traj_df = pd.read_csv(traj_path)

    print(f"  - Raw Detections rows  : {len(det_df):,}")
    print(f"  - Sensor info rows     : {len(info_df):,}")
    print(f"  - Sensor distances rows: {len(dist_df):,}")
    print(f"  - Drone info rows      : {len(drone_info_df):,}")
    print(f"  - Drone trajectory rows: {len(traj_df):,}")

    return {
        "detections": det_df,
        "sensor_info": info_df,
        "sensor_dist": dist_df,
        "drone_info": drone_info_df,
        "drone_traj": traj_df,
    }


def apply_noise_to_raw_data(
    raw_data: Dict[str, pd.DataFrame],
    power_noise_std: float = 0.0,
    ptx_noise_std: float = 0.0,
    seed: int = 42,
    noise_mode: str = "relative",
) -> Dict[str, pd.DataFrame]:
    """
    Applies Gaussian normal distribution noise in the LINEAR power domain.
    """
    if power_noise_std <= 0.0 and ptx_noise_std <= 0.0:
        return raw_data

    print(f"\n[INFO] Injecting Gaussian Normal Noise in LINEAR Power Domain (mode={noise_mode}, seed={seed}):")
    if power_noise_std > 0.0:
        print(f"  - Received Power Linear Noise Std Dev : {power_noise_std}")
    if ptx_noise_std > 0.0:
        print(f"  - Transmit Power (P_tx) Linear Noise Std: {ptx_noise_std}")

    noisy_data = {k: v.copy() for k, v in raw_data.items()}

    if seed is not None:
        np.random.seed(seed)

    if power_noise_std > 0.0 and "detections" in noisy_data:
        det_df = noisy_data["detections"]
        pow_col = "total_power_dBm" if "total_power_dBm" in det_df.columns else "received_power"
        if pow_col in det_df.columns:
            active_mask = (det_df[pow_col] != 0.0) & (det_df[pow_col] > -120.0)
            num_active = active_mask.sum()
            if num_active > 0:
                p_dbm = det_df.loc[active_mask, pow_col].values.astype(float)
                p_linear = 10.0 ** (p_dbm / 10.0)

                if noise_mode.lower() == "relative":
                    noise = np.random.normal(loc=0.0, scale=power_noise_std, size=num_active)
                    p_linear_noisy = p_linear * (1.0 + noise)
                else:
                    noise = np.random.normal(loc=0.0, scale=power_noise_std, size=num_active)
                    p_linear_noisy = p_linear + noise

                p_linear_noisy = np.maximum(p_linear_noisy, 1e-15)
                p_dbm_noisy = 10.0 * np.log10(p_linear_noisy)
                det_df.loc[active_mask, pow_col] = p_dbm_noisy

    if ptx_noise_std > 0.0 and "drone_info" in noisy_data:
        drone_info_df = noisy_data["drone_info"]
        ptx_col = "drone_transmitter_power" if "drone_transmitter_power" in drone_info_df.columns else "p_tx"
        if ptx_col in drone_info_df.columns:
            num_rows = len(drone_info_df)
            ptx_dbm = drone_info_df[ptx_col].values.astype(float)
            ptx_linear = 10.0 ** (ptx_dbm / 10.0)

            if noise_mode.lower() == "relative":
                noise = np.random.normal(loc=0.0, scale=ptx_noise_std, size=num_rows)
                ptx_linear_noisy = ptx_linear * (1.0 + noise)
            else:
                noise = np.random.normal(loc=0.0, scale=ptx_noise_std, size=num_rows)
                ptx_linear_noisy = ptx_linear + noise

            ptx_linear_noisy = np.maximum(ptx_linear_noisy, 1e-15)
            ptx_dbm_noisy = 10.0 * np.log10(ptx_linear_noisy)
            drone_info_df[ptx_col] = ptx_dbm_noisy

    return noisy_data


def extract_ground_truth_for_timestamps(exp_id: int, timestamps: np.ndarray, traj_df: pd.DataFrame) -> np.ndarray:
    """
    Extracts ground truth 3D UAV coordinates (start_x, start_y, start_z) in meters
    from drone_trajectories.csv matching timestamp indices.
    """
    traj_exp_col = "experiment_id" if "experiment_id" in traj_df.columns else ("exp_id" if "exp_id" in traj_df.columns else traj_df.columns[0])
    exp_traj = traj_df[traj_df[traj_exp_col] == exp_id].sort_values("timestamp").reset_index(drop=True)

    ts_map = {idx: (r["start_x"], r["start_y"], r["start_z"]) for idx, r in exp_traj.iterrows()}

    true_list = []
    for ts_idx in timestamps:
        if ts_idx in ts_map:
            true_list.append(ts_map[ts_idx])
        elif ts_idx < len(exp_traj):
            r = exp_traj.iloc[ts_idx]
            true_list.append((r["start_x"], r["start_y"], r["start_z"]))
        else:
            true_list.append((0.0, 0.0, 0.0))

    return np.array(true_list, dtype=float)


def evaluate_predictors_on_test_data(
    raw_data: Dict[str, pd.DataFrame],
    max_exps: Optional[int] = None,
    output_dir: Optional[str] = None,
    power_noise_std: float = 0.0,
    ptx_noise_std: float = 0.0,
    noise_mode: str = "relative",
    seed: int = 42,
) -> pd.DataFrame:
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    raw_data_input = apply_noise_to_raw_data(
        raw_data=raw_data,
        power_noise_std=power_noise_std,
        ptx_noise_std=ptx_noise_std,
        noise_mode=noise_mode,
        seed=seed,
    )

    det_df = raw_data_input["detections"]
    exp_col = "experiment_id" if "experiment_id" in det_df.columns else "exp_id"
    unique_exps = np.sort(det_df[exp_col].unique())

    if max_exps is not None:
        unique_exps = unique_exps[:max_exps]

    print("\n" + "=" * 75)
    print(f" EVALUATING UAV LOCALIZATION PREDICTORS ON RAW TEST DATA: 5K_4")
    if power_noise_std > 0.0 or ptx_noise_std > 0.0:
        print(f" LINEAR NOISE ({noise_mode}): Received Power Std={power_noise_std} | P_tx Std={ptx_noise_std}")
    print("=" * 75)
    print(f"Total experiments to evaluate: {len(unique_exps)}")

    print("\n[INFO] Initializing predictors...")
    predictors: List[BaseUAVPredictor] = []

    try:
        enc_dec_predictor = EncoderDecoderPredictor()
        predictors.append(enc_dec_predictor)
    except FileNotFoundError as e:
        print(f"[WARNING] EncoderDecoderPredictor disabled:\n{e}\n")

    rssi_wcl = RSSIWCLPredictor()
    predictors.append(rssi_wcl)

    rssi_trilateration = RSSITrilaterationPredictor()
    predictors.append(rssi_trilateration)

    summary_rows = []
    all_exp_predictions = []

    for pred in predictors:
        print(f"\n---> Running evaluation for: {pred.name}...")

        all_pred_list, all_true_list, all_ts_list = [], [], []
        total_time_accum = 0.0

        for exp_id in tqdm(unique_exps, desc=f"Evaluating {pred.name}"):
            exp_det = det_df[det_df[exp_col] == exp_id].copy()

            raw_exp_input = {
                "detections": exp_det,
                "sensor_info": raw_data_input["sensor_info"],
                "sensor_dist": raw_data_input["sensor_dist"],
                "drone_info": raw_data_input["drone_info"],
            }

            pred_xyz_m, timestamps, proc_time_sec = pred.predict_experiment(raw_exp_input)
            true_xyz_m = extract_ground_truth_for_timestamps(exp_id, timestamps, raw_data["drone_traj"])

            total_time_accum += proc_time_sec

            all_pred_list.append(pred_xyz_m)
            all_true_list.append(true_xyz_m)
            all_ts_list.append(timestamps)

        pred_concat = np.concatenate(all_pred_list, axis=0)
        true_concat = np.concatenate(all_true_list, axis=0)
        ts_concat = np.concatenate(all_ts_list, axis=0)

        metrics, detailed_df = pred.evaluate(pred_concat, true_concat, ts_concat, total_time_accum)
        summary_rows.append(metrics)

        detailed_df["predictor_name"] = pred.name
        all_exp_predictions.append(detailed_df)

        print(pred.format_metrics_summary(metrics))

    summary_df = pd.DataFrame(summary_rows)

    if power_noise_std > 0.0 or ptx_noise_std > 0.0:
        summary_csv_path = os.path.join(output_dir, f"5K_4_summary_linear_noise_pwr{power_noise_std}_ptx{ptx_noise_std}.csv")
        predictions_csv_path = os.path.join(output_dir, f"5K_4_predictions_linear_noise_pwr{power_noise_std}_ptx{ptx_noise_std}.csv")
    else:
        summary_csv_path = os.path.join(output_dir, "5K_4_evaluation_comparison_summary.csv")
        predictions_csv_path = os.path.join(output_dir, "5K_4_detailed_predictions.csv")

    summary_df.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 85)
    print(" SIDE-BY-SIDE MODEL COMPARISON SUMMARY (RAW TEST SET: 5K_4)")
    print("=" * 85)
    cols_to_display = [
        "Predictor_Name", "3D_Mean_Error (m)", "3D_Median_Error (m)", "3D_RMSE (m)",
        "Trajectory_Step_Loss (m)", "PyTorch_MSELoss (km^2)", "3D_P95_Error (m)",
        "2D_Mean_Error (m)", "Z_Mean_Error (m)", "Total_Processing_Time (s)",
        "Latency_Per_Timestamp (ms)"
    ]
    disp_cols = [c for c in cols_to_display if c in summary_df.columns]
    print(summary_df[disp_cols].to_string(index=False))
    print("=" * 85)

    if all_exp_predictions:
        combined_preds_df = pd.concat(all_exp_predictions, ignore_index=True)
        combined_preds_df.to_csv(predictions_csv_path, index=False)

    print(f"\n[OK] Evaluation complete!")
    print(f"     - Summary CSV saved to    : {summary_csv_path}")
    print(f"     - Predictions CSV saved to: {predictions_csv_path}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UAV Localization Predictors on raw 5K_4 test dataset.")
    parser.add_argument("--max_exps", type=int, default=None, help="Maximum number of experiments to evaluate (default: all).")
    parser.add_argument("--power_noise_std", type=float, default=0.0, help="Std dev of Gaussian normal noise applied in LINEAR power domain (mW).")
    parser.add_argument("--ptx_noise_std", type=float, default=0.0, help="Std dev of Gaussian normal noise applied in LINEAR transmit power domain (mW).")
    parser.add_argument("--noise_mode", type=str, default="relative", choices=["relative", "absolute"], help="Linear noise mode: 'relative' or 'absolute'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible noise generation (default: 42).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation summary CSVs.")

    args = parser.parse_args()

    data_5k4_dir = resolve_5k4_data_dir()
    raw_dataset = load_raw_dataset_files(data_5k4_dir)

    evaluate_predictors_on_test_data(
        raw_data=raw_dataset,
        max_exps=args.max_exps,
        output_dir=args.output_dir,
        power_noise_std=args.power_noise_std,
        ptx_noise_std=args.ptx_noise_std,
        noise_mode=args.noise_mode,
        seed=args.seed,
    )
