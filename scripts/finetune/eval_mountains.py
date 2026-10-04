"""
main.py (fine-tuned models)

Evaluation of the Mountains fine-tuned Encoder-Decoder against the original (pre-trained) Encoder-Decoder
and RSSI trilateration on the raw, noise-free Mountains test data (Data/MountainsData/10_sensors_mountains,
test_25K_0).

Mirrors scripts/evaluate/eval_noise_grid.py without the noise grid: raw test files are loaded, every predictor runs on each
experiment, and side-by-side metrics are saved.

Mountains-specific differences from scripts/evaluate/eval_noise_grid.py:
  * Ground truth is the position where the RSSI was measured (end_xyz for Mountains, see
    preprocess_mountains.detect_uav_position_side), matched to each prediction by its actual timestamp.
  * RSSI trilateration needs the drone transmit power (P_tx) from <prefix>_drone_information.csv;
    when that file is missing (as in the current Mountains data), RSSI trilateration is skipped.
  * Note: test_25K_0 is also the validation file used to pick decoder_mountains_finetuned_best.pth.

Usage:
    python scripts/finetune/eval_mountains.py
    python scripts/finetune/eval_mountains.py --max_exps 0      # all test experiments
"""

import os
import sys
import argparse
import importlib.util
from typing import Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in (ROOT_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.append(p)

from src.predictors.encoder_decoder_predictor import EncoderDecoderPredictor
from src.predictors.rssi_trilateration_predictor import RSSITrilaterationPredictor
from preprocess_mountains import RAW_DIR, find_raw_prefixes, raw_files, detect_uav_position_side

# scripts/evaluate/eval_noise_grid.py shares this module's name, so it is loaded from its path
_spec = importlib.util.spec_from_file_location("base_main", os.path.join(ROOT_DIR, "scripts", "evaluate", "eval_noise_grid.py"))
base_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base_main)
apply_ptx_uncertainty_to_raw_data = base_main.apply_ptx_uncertainty_to_raw_data

DEFAULT_FINETUNED_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "decoder_mountains_finetuned_best.pth")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "evaluation")


def resolve_test_prefix(raw_dir: str, test_tag: Optional[str] = None) -> str:
    """Returns the raw file prefix of the (non-anomaly) Mountains test file."""
    val = {tag: prefix for tag, (prefix, split) in find_raw_prefixes(raw_dir).items() if split == "val"}
    if not val:
        raise FileNotFoundError(f"No non-anomaly Mountains test files found in {raw_dir}")
    if test_tag is not None:
        if test_tag not in val:
            raise KeyError(f"Test file '{test_tag}' not found. Available: {sorted(val)}")
        return val[test_tag]
    return val[sorted(val)[0]]


def read_first_experiments(path: str, exp_col: str, exp_ids: Optional[set], chunksize: int = 1_000_000) -> pd.DataFrame:
    """Reads only the rows of the given experiments (whole file if exp_ids is None); files are sorted by experiment."""
    if exp_ids is None:
        return pd.read_csv(path)
    max_id = max(exp_ids)
    parts = []
    for chunk in pd.read_csv(path, chunksize=chunksize):
        parts.append(chunk[chunk[exp_col].isin(exp_ids)])
        if chunk[exp_col].min() > max_id:
            break
    return pd.concat(parts, ignore_index=True)


def load_mountains_test_files(prefix: str, max_exps: Optional[int]) -> Dict[str, pd.DataFrame]:
    """Loads the raw Mountains test tables (limited to the first max_exps experiments) into scripts/evaluate/eval_noise_grid.py's layout."""
    files = raw_files(prefix)
    print(f"[INFO] Loading Mountains test files: {os.path.basename(prefix)}_*.csv")

    traj_df = pd.read_csv(files["drone_trajectories"])
    exp_ids = None
    if max_exps is not None:
        exp_ids = set(np.sort(traj_df["experiment_id"].unique())[:max_exps].tolist())
        traj_df = traj_df[traj_df["experiment_id"].isin(exp_ids)].reset_index(drop=True)

    det_df = read_first_experiments(files["detections"], "experiment_id", exp_ids)
    dist_df = read_first_experiments(files["sensor_distances"], "exp_id", exp_ids)
    info_df = pd.read_csv(files["sensor_information"])

    drone_info_path = f"{prefix}_drone_information.csv"
    drone_info_df = pd.read_csv(drone_info_path) if os.path.exists(drone_info_path) else None

    print(f"  - Raw Detections rows  : {len(det_df):,}")
    print(f"  - Sensor info rows     : {len(info_df):,}")
    print(f"  - Sensor distances rows: {len(dist_df):,}")
    print(f"  - Drone trajectory rows: {len(traj_df):,}")
    print(f"  - Drone info rows      : {len(drone_info_df):,}" if drone_info_df is not None
          else "  - Drone info           : not available (no drone_information.csv)")

    side = detect_uav_position_side(files, traj_df)
    return {
        "detections": det_df,
        "sensor_info": info_df,
        "sensor_dist": dist_df,
        "drone_info": drone_info_df,
        "drone_traj": traj_df,
        "uav_side": side,
    }


def has_ptx(drone_info_df: Optional[pd.DataFrame]) -> bool:
    """True if the drone transmit power is available for RSSI trilateration."""
    if drone_info_df is None or drone_info_df.empty:
        return False
    ptx_col = next((c for c in ("drone_transmitter_power", "p_tx") if c in drone_info_df.columns), None)
    return ptx_col is not None and drone_info_df[ptx_col].notna().all()


def extract_ground_truth_for_timestamps(exp_det: pd.DataFrame, ts_indices: np.ndarray,
                                        exp_traj: pd.DataFrame, side: str) -> np.ndarray:
    """
    Ground truth (m) for the predictors' timestamp indices. Both predictors index the experiment's sorted
    unique detection timestamps, so each index is mapped back to its timestamp and looked up in the
    trajectory at the position where the RSSI was measured.
    """
    unique_ts = np.sort(exp_det["timestamp"].unique())
    pos = exp_traj.set_index("timestamp")[[f"{side}_x", f"{side}_y", f"{side}_z"]]
    true_xyz = pos.reindex(unique_ts[ts_indices]).to_numpy(dtype=float)
    return np.nan_to_num(true_xyz, nan=0.0)  # (0, 0, 0) rows are excluded by BaseUAVPredictor.evaluate


def build_predictors(args, ptx_available: bool) -> list:
    """Returns [(name, predictor, ptx_uncertainty_dbm)]."""
    configs = [
        ("Encoder-Decoder (Original)",
         EncoderDecoderPredictor(num_sensors=args.num_sensors, window_size=args.sequence_size,
                                 name="Encoder-Decoder (Original)"), 0.0),
        ("Encoder-Decoder (Fine-tuned Mountains)",
         EncoderDecoderPredictor(dec_ckpt_path=args.finetuned_ckpt, num_sensors=args.num_sensors,
                                 window_size=args.sequence_size, name="Encoder-Decoder (Fine-tuned Mountains)"), 0.0),
    ]
    print(f"[INFO] Original decoder  : {configs[0][1].dec_ckpt_path}")
    print(f"[INFO] Fine-tuned decoder: {configs[1][1].dec_ckpt_path}")

    if not ptx_available:
        print("[INFO] Drone transmit power (P_tx) not available - RSSI Trilateration is skipped.")
        return configs

    for unc in args.ptx_uncertainties:
        label = "RSSI Trilateration (Correct P_tx)" if unc == 0.0 else f"RSSI Trilateration ({unc:g}dBm Uncertainty)"
        configs.append((label, RSSITrilaterationPredictor(name=label), unc))
    return configs


def evaluate_predictors_on_mountains(
    raw_data: Dict[str, pd.DataFrame],
    predictor_configs: list,
    output_dir: str,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    det_df = raw_data["detections"]
    traj_df = raw_data["drone_traj"]
    side = raw_data["uav_side"]
    unique_exps = np.sort(det_df["experiment_id"].unique())
    det_groups = {exp_id: g for exp_id, g in det_df.groupby("experiment_id")}
    traj_groups = {exp_id: g for exp_id, g in traj_df.groupby("experiment_id")}

    print("\n" + "=" * 75)
    print(f" EVALUATING PREDICTORS ON MOUNTAINS TEST DATA (NO NOISE)")
    print("=" * 75)
    print(f"Total experiments to evaluate: {len(unique_exps)}")

    summary_rows = []
    all_exp_predictions = []

    for pred_name, pred, ptx_unc in predictor_configs:
        print(f"\n---> Running evaluation for: {pred_name}...")
        exp_raw_data = apply_ptx_uncertainty_to_raw_data(raw_data, ptx_unc) if ptx_unc != 0.0 else raw_data
        drone_info_df = exp_raw_data["drone_info"] if exp_raw_data["drone_info"] is not None else pd.DataFrame()

        all_pred_list, all_true_list, all_ts_list, all_exp_list = [], [], [], []
        total_time_accum = 0.0

        for exp_id in tqdm(unique_exps, desc=f"Evaluating {pred_name}"):
            exp_det = det_groups[exp_id]
            raw_exp_input = {
                "detections": exp_det,
                "sensor_info": exp_raw_data["sensor_info"],
                "sensor_dist": exp_raw_data["sensor_dist"],
                "drone_info": drone_info_df,
            }

            pred_xyz_m, timestamps, proc_time_sec = pred.predict_experiment(raw_exp_input)
            true_xyz_m = extract_ground_truth_for_timestamps(exp_det, timestamps, traj_groups[exp_id], side)

            total_time_accum += proc_time_sec
            all_pred_list.append(pred_xyz_m)
            all_true_list.append(true_xyz_m)
            all_ts_list.append(timestamps)
            all_exp_list.append(np.full(len(timestamps), exp_id))

        pred_concat = np.concatenate(all_pred_list, axis=0)
        true_concat = np.concatenate(all_true_list, axis=0)
        ts_concat = np.concatenate(all_ts_list, axis=0)

        metrics, detailed_df = pred.evaluate(pred_concat, true_concat, ts_concat, total_time_accum)
        metrics["Predictor_Name"] = pred_name
        summary_rows.append(metrics)

        detailed_df.insert(0, "experiment_id", np.concatenate(all_exp_list))
        detailed_df["predictor_name"] = pred_name
        all_exp_predictions.append(detailed_df)

        print(pred.format_metrics_summary(metrics))

    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = os.path.join(output_dir, "mountains_summary_no_noise.csv")
    predictions_csv_path = os.path.join(output_dir, "mountains_predictions_no_noise.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    pd.concat(all_exp_predictions, ignore_index=True).to_csv(predictions_csv_path, index=False)

    print("\n" + "=" * 85)
    print(" SIDE-BY-SIDE MODEL COMPARISON SUMMARY (MOUNTAINS TEST SET, NO NOISE)")
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

    print(f"\n[OK] Evaluation complete!")
    print(f"     - Summary CSV saved to    : {summary_csv_path}")
    print(f"     - Predictions CSV saved to: {predictions_csv_path}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare the Mountains fine-tuned model against the original model and RSSI trilateration.")
    parser.add_argument("--raw_dir", type=str, default=RAW_DIR, help="Folder with the raw Mountains CSV files")
    parser.add_argument("--test_tag", type=str, default=None, help="Test file tag (default: first one, e.g. test_25K_0)")
    parser.add_argument("--finetuned_ckpt", type=str, default=DEFAULT_FINETUNED_CKPT, help="Fine-tuned decoder checkpoint")
    parser.add_argument("--num_sensors", type=int, default=10, help="Number of sensors")
    parser.add_argument("--sequence_size", type=int, default=40, help="Sequence size (window size) for Encoder-Decoder")
    parser.add_argument("--max_exps", type=int, default=100, help="Maximum number of experiments to evaluate (<= 0 for all)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation outputs")
    parser.add_argument("--ptx_uncertainties", type=float, nargs="+", default=[0.0, 1.0],
                        help="P_tx errors (dB) for RSSI trilateration (only used when P_tx is available)")
    args = parser.parse_args()

    max_exps = args.max_exps if args.max_exps > 0 else None

    test_prefix = resolve_test_prefix(args.raw_dir, args.test_tag)
    raw_dataset = load_mountains_test_files(test_prefix, max_exps)

    print("\n[INFO] Initializing predictors...")
    predictor_configs = build_predictors(args, has_ptx(raw_dataset["drone_info"]))

    evaluate_predictors_on_mountains(
        raw_data=raw_dataset,
        predictor_configs=predictor_configs,
        output_dir=args.output_dir,
    )
