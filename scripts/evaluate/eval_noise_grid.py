"""
main.py

Main Evaluation Script for UAV Localization Predictors on Raw Test Data.
Extracts raw data from test dataset folders, applies noise in linear power domain if requested,
runs predictors, computes side-by-side spatial & computational metrics, and saves outputs.
"""

#from _typeshed import _type_checker_internals
from pandas._libs.tslibs import timestamps
from matplotlib import backend_bases
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
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
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
        + "\n\nPlease run 'python scripts/data/download_drive_data.py' or verify that raw CSV datasets exist in 'data/'."
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

def sigma_from_noise_dbm(noise_floor_dbm: float) -> float:
    """Voltage-domain noise std (sqrt(mW)) for a given noise floor in dBm."""
    return 10.0 ** (noise_floor_dbm / 20.0)

def apply_noise_to_raw_data(
    raw_data: Dict[str, pd.DataFrame],
    samples_per_timestamp: int,
    background_noise_dbm: float,
    detection_threshold_dBm: float = -120.0,
    min_linear_power: float = 1e-15,
) -> Dict[str, pd.DataFrame]:
    """Apply AWGN in the voltage domain; report averaged power."""
    if samples_per_timestamp < 1:
        raise ValueError("samples_per_timestamp must be >= 1")
    if "detections" not in raw_data:
        raise KeyError("no detections table found")

    det_df = raw_data["detections"].copy()
    pow_col = ("total_power_dBm" if "total_power_dBm" in det_df.columns
               else "received_power")
    if pow_col not in det_df.columns:
        raise KeyError("no power column found in detections")
    
    background_noise_std = sigma_from_noise_dbm(background_noise_dbm)
    
    active = (det_df[pow_col] != 0.0) & (det_df[pow_col] > detection_threshold_dBm)
    n_active = int(active.sum())
    if n_active == 0:
        return {**raw_data, "detections": det_df}

    p_lin = 10.0 ** (det_df.loc[active, pow_col].to_numpy(float) / 10.0)
    v_sig = np.sqrt(p_lin)

    noise = np.random.normal(0.0, background_noise_std,
                             size=(samples_per_timestamp, n_active))
    p_noisy = np.mean((v_sig[None, :] + noise) ** 2, axis=0)
    p_noisy = np.maximum(p_noisy, min_linear_power)

    det_df.loc[active, pow_col] = 10.0 * np.log10(p_noisy)
    return {**raw_data, "detections": det_df}


def apply_ptx_uncertainty_to_raw_data(
    raw_data: Dict[str, pd.DataFrame],
    ptx_uncertainty_dbm: float,
) -> Dict[str, pd.DataFrame]:
    """
    Applies transmit power (P_tx) uncertainty in dBm directly to the drone_info table.
    """
    if ptx_uncertainty_dbm == 0.0 or "drone_info" not in raw_data:
        return raw_data

    noisy_data = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in raw_data.items()}
    drone_info_df = noisy_data["drone_info"].copy()
    ptx_col = "drone_transmitter_power" if "drone_transmitter_power" in drone_info_df.columns else "p_tx"

    if ptx_col in drone_info_df.columns:
        drone_info_df[ptx_col] = drone_info_df[ptx_col].astype(float) + ptx_uncertainty_dbm

    noisy_data["drone_info"] = drone_info_df
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
    background_noise_dbm: float = -110,
    samples_per_timestamp: int = 1,
    seed: int = 42,
    num_sensors: int = 10,
    sequence_size: int = 40,
) -> pd.DataFrame:
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    raw_data_input = apply_noise_to_raw_data(
        raw_data=raw_data,
        background_noise_dbm=background_noise_dbm,
        samples_per_timestamp=samples_per_timestamp,
    )

    det_df = raw_data_input["detections"]
    exp_col = "experiment_id" if "experiment_id" in det_df.columns else "exp_id"
    unique_exps = np.sort(det_df[exp_col].unique())

    if max_exps is not None:
        unique_exps = unique_exps[:max_exps]

    print("\n" + "=" * 75)
    print(f" EVALUATING UAV LOCALIZATION PREDICTORS ON RAW TEST DATA: 5K_4")
    print(f" Background noise: {background_noise_dbm} dBm, Samples per timestamp: {samples_per_timestamp}")
    print("=" * 75)
    print(f"Total experiments to evaluate: {len(unique_exps)}")

    print("\n[INFO] Initializing predictors...")
    predictor_configs = []

    try:
        enc_dec_predictor = EncoderDecoderPredictor(num_sensors=num_sensors, window_size=sequence_size)
        predictor_configs.append(("Encoder-Decoder", enc_dec_predictor, 0.0))
    except FileNotFoundError as e:
        print(f"[WARNING] EncoderDecoderPredictor disabled:\n{e}\n")

    rssi_uncertainties = [0.0, 1.0]
    for unc in rssi_uncertainties:
        label = "RSSI Trilateration (Correct P_tx)" if unc == 0.0 else f"RSSI Trilateration ({int(unc)}dBm Uncertainty)"
        rssi_pred = RSSITrilaterationPredictor(name=label)
        predictor_configs.append((label, rssi_pred, unc))

    summary_rows = []
    all_exp_predictions = []

    for pred_name, pred, ptx_unc in predictor_configs:
        print(f"\n---> Running evaluation for: {pred_name}...")

        if ptx_unc != 0.0:
            exp_raw_data = apply_ptx_uncertainty_to_raw_data(raw_data_input, ptx_unc)
        else:
            exp_raw_data = raw_data_input

        exp_det_df = exp_raw_data["detections"]

        all_pred_list, all_true_list, all_ts_list = [], [], []
        total_time_accum = 0.0

        for exp_id in tqdm(unique_exps, desc=f"Evaluating {pred_name}"):
            exp_det = exp_det_df[exp_det_df[exp_col] == exp_id].copy()

            raw_exp_input = {
                "detections": exp_det,
                "sensor_info": exp_raw_data["sensor_info"],
                "sensor_dist": exp_raw_data["sensor_dist"],
                "drone_info": exp_raw_data["drone_info"],
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
        metrics["Predictor_Name"] = pred_name
        summary_rows.append(metrics)

        detailed_df["predictor_name"] = pred_name
        all_exp_predictions.append(detailed_df)

        print(pred.format_metrics_summary(metrics))

    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = os.path.join(output_dir, f"5K_4_summary_background_noise_{background_noise_dbm}_samples_{samples_per_timestamp}.csv")
    predictions_csv_path = os.path.join(output_dir, f"5K_4_predictions_background_noise_{background_noise_dbm}_samples_{samples_per_timestamp}.csv")

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
    parser = argparse.ArgumentParser(description="Evaluate UAV Localization Predictors.")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to raw dataset folder")
    parser.add_argument("--num_sensors", type=int, default=10, help="Number of sensors")
    parser.add_argument("--sequence_size", type=int, default=40, help="Sequence size (window size) for Encoder-Decoder")
    parser.add_argument("--max_exps", type=int, default=100, help="Maximum number of experiments to evaluate (or None for all)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation outputs")
    parser.add_argument("--noise_floors", type=float, nargs="+", default=[-90.0, -100.0, -110.0], help="List of background noise floors in dBm")
    parser.add_argument("--samples_per_ts", type=int, nargs="+", default=[1, 10, 100], help="List of samples per timestamp")
    args = parser.parse_args()

    MAX_EXPS = args.max_exps
    noise_floors_dbm = args.noise_floors
    samples_per_ts_list = args.samples_per_ts

    data_dir = args.data_dir if args.data_dir else resolve_5k4_data_dir()
    raw_dataset = load_raw_dataset_files(data_dir)

    output_dir = args.output_dir if args.output_dir else os.path.join(ROOT_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    all_grid_summaries = []

    for noise_dbm in noise_floors_dbm:
        for samples_ts in samples_per_ts_list:
            print("\n" + "#" * 80)
            print(f" RUNNING EXPERIMENT: Noise = {noise_dbm} dBm | Samples/Timestamp = {samples_ts}")
            print("#" * 80)

            summary_df = evaluate_predictors_on_test_data(
                raw_data=raw_dataset,
                max_exps=MAX_EXPS,
                output_dir=output_dir,
                background_noise_dbm=noise_dbm,
                samples_per_timestamp=samples_ts,
                num_sensors=args.num_sensors,
                sequence_size=args.sequence_size,
            )

            summary_df["Background_Noise_dBm"] = noise_dbm
            summary_df["Samples_Per_Timestamp"] = samples_ts
            all_grid_summaries.append(summary_df)

    if all_grid_summaries:
        full_grid_df = pd.concat(all_grid_summaries, ignore_index=True)
        full_grid_csv_path = os.path.join(output_dir, "5K_4_full_grid_summary.csv")
        full_grid_df.to_csv(full_grid_csv_path, index=False)
        print("\n" + "=" * 85)
        print(" MASTER EXPERIMENT GRID SUMMARY SAVED")
        print("=" * 85)
        print(f" Combined Summary CSV saved to: {full_grid_csv_path}")


