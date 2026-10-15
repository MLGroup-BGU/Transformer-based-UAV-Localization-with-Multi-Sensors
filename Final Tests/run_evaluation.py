"""
run_evaluation.py

Final evaluation: runs 5 predictors on 5 validation datasets and records the results.

Predictors:
  * RSSI Trilateration  (correct P_tx from drone_information.csv)
  * RSSI Trilateration (1dBm Uncertainty)  (P_tx shifted by +1 dB, as in the source pipeline)
  * WCL                 (RSSI weighted centroid)
  * SeqED Basic         (encoder epoch 16 + decoder epoch 12, trained on clean data)
  * SeqED Fine-tuned    (decoder fine-tuned for the dataset's condition, lowest-val-loss epoch;
                         no fine-tuned model exists for the clean dataset, so it is skipped there)

Datasets (Datasets/<...>/5K_4):
  * clean, Mountains, noise/background110, noise/background100, noise/background90

Copied from Transformer-based-UAV-Localization-with-Multi-Sensors:
  * load_raw_dataset_files                 <- scripts/evaluate/eval_noise_grid.py
  * apply_ptx_uncertainty_to_raw_data      <- scripts/evaluate/eval_noise_grid.py
  * detect_uav_position_side               <- scripts/finetune_noise/preprocess_noise.py
  * read_first_experiments, extract_ground_truth_for_timestamps,
    evaluate_predictors (evaluation loop)  <- scripts/finetune_noise/eval_noise.py
  * src/                                   <- src/ (unchanged)

Timing: Latency_Per_Timestamp / Throughput / Total_Processing_Time cover model computation only
(the solver for RSSI/WCL, encoder+decoder forward pass for SeqED). Data preparation (pandas wrangling,
densification, windowing, device transfer, aggregation) is reported as Preprocessing_Time (s) /
Preprocessing_Per_Timestamp (ms); End_to_End_Per_Timestamp (ms) is their sum. Each predictor gets one
untimed warm-up call before timing.

Predictor groups (--predictors): trilateration (both P_tx cases), wcl  -> CPU only
                                 seqed_basic, seqed_finetuned          -> GPU

Outputs (results/):
  * <dataset>/summary_<group>.csv, <dataset>/predictions_<group>.csv   per dataset and predictor group
  * all_summary.csv              every metric, one row per (dataset, predictor), rebuilt from all
                                 summary_<group>.csv files on disk after every run
  * pivot_<metric>.csv           predictor x dataset tables of the main metrics

Usage:
    python run_evaluation.py                  # first 100 experiments of every dataset (source default)
    python run_evaluation.py --max_exps 0     # all experiments
    python run_evaluation.py --datasets clean mountains
    python run_evaluation.py --predictors trilateration wcl           # CPU predictors only
    python run_evaluation.py --predictors seqed_basic seqed_finetuned # GPU predictors only
"""

import os
import sys
import ast
import argparse
from typing import Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.predictors.encoder_decoder_predictor import EncoderDecoderPredictor
from src.predictors.rssi_trilateration_predictor import RSSITrilaterationPredictor
from src.predictors.rssi_wcl_predictor import RSSIWCLPredictor

DATASETS_DIR = os.path.join(ROOT_DIR, "Datasets")
FINETUNED_DIR = os.path.join(ROOT_DIR, "models", "finetuned")
FINETUNED_CKPT_NAME = "decoder_mountains_finetuned_best.pth"

# key -> (data folder, fine-tuned decoder checkpoint or None)
DATASETS = {
    "clean":      (os.path.join(DATASETS_DIR, "clean", "5K_4"), None),
    "mountains":  (os.path.join(DATASETS_DIR, "Mountains", "5K_4"),
                   os.path.join(FINETUNED_DIR, "mountains", FINETUNED_CKPT_NAME)),
    "noise_-110": (os.path.join(DATASETS_DIR, "noise", "background110", "5K_4"),
                   os.path.join(FINETUNED_DIR, "noise_f110", FINETUNED_CKPT_NAME)),
    "noise_-100": (os.path.join(DATASETS_DIR, "noise", "background100", "5K_4"),
                   os.path.join(FINETUNED_DIR, "noise_f100", FINETUNED_CKPT_NAME)),
    "noise_-90":  (os.path.join(DATASETS_DIR, "noise", "background90", "5K_4"),
                   os.path.join(FINETUNED_DIR, "noise_f90", FINETUNED_CKPT_NAME)),
}

# Predictor groups (--predictors). "trilateration" runs once per --ptx_uncertainties value.
PREDICTOR_GROUPS = ["trilateration", "wcl", "seqed_basic", "seqed_finetuned"]

PIVOT_METRICS = ["3D_Mean_Error (m)", "3D_Median_Error (m)", "3D_RMSE (m)", "3D_P95_Error (m)",
                 "2D_Mean_Error (m)", "Z_Mean_Error (m)",
                 "Latency_Per_Timestamp (ms)", "Preprocessing_Per_Timestamp (ms)"]


def find_file(data_dir: str, suffix: str) -> str:
    for f in os.listdir(data_dir):
        if f.endswith(suffix):
            return os.path.join(data_dir, f)
    raise FileNotFoundError(f"Could not find required dataset file with suffix '{suffix}' in {data_dir}")


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


def detect_uav_position_side(det_path, traj, n_rows=20000, tol_m=1e-3):
    """
    Returns "start" or "end": which trajectory position the detections at a timestamp were measured at.

    The original training data measures RSSI at the trajectory row's start_xyz (normalized_data.csv uses
    start_xyz as the label), while the Mountains simulator measures it at end_xyz. The label must be the
    position where the RSSI was actually measured, so it is taken from the detections' drone_positions.
    """
    det = pd.read_csv(det_path, nrows=n_rows, usecols=["experiment_id", "timestamp", "drone_positions"])
    det = det.merge(traj, on=["experiment_id", "timestamp"], how="inner")
    pos = np.array([ast.literal_eval(p)[0] for p in det["drone_positions"]], dtype=np.float64)

    match = {}
    for side in ("start", "end"):
        ref = det[[f"{side}_x", f"{side}_y", f"{side}_z"]].to_numpy(np.float64)
        match[side] = float((np.linalg.norm(pos - ref, axis=1) < tol_m).mean())
    side = max(match, key=match.get)
    if match[side] < 0.99:
        raise ValueError(f"Detection positions do not match trajectory start/end positions: {match}")
    print(f"    [*] Detections measured at trajectory '{side}' position (match rate {match})")
    return side


def load_raw_dataset_files(data_dir: str, max_exps: Optional[int]) -> Dict[str, pd.DataFrame]:
    """
    Loads raw CSV data files directly from the specified experiment folder into memory
    (limited to the first max_exps experiments).
    """
    print(f"[INFO] Loading raw dataset files from: {data_dir}")
    det_path = find_file(data_dir, "sensor_detection_summary.csv")
    info_path = find_file(data_dir, "sensor_information.csv")
    dist_path = find_file(data_dir, "sensor_distances.csv")
    drone_info_path = find_file(data_dir, "drone_information.csv")
    traj_path = find_file(data_dir, "drone_trajectories.csv")

    traj_df = pd.read_csv(traj_path)
    exp_ids = None
    if max_exps is not None:
        exp_ids = set(np.sort(traj_df["experiment_id"].unique())[:max_exps].tolist())
        traj_df = traj_df[traj_df["experiment_id"].isin(exp_ids)].reset_index(drop=True)

    det_df = read_first_experiments(det_path, "experiment_id", exp_ids)
    dist_df = read_first_experiments(dist_path, "exp_id", exp_ids)
    info_df = pd.read_csv(info_path)
    drone_info_df = pd.read_csv(drone_info_path)

    print(f"  - Raw Detections rows  : {len(det_df):,}")
    print(f"  - Sensor info rows     : {len(info_df):,}")
    print(f"  - Sensor distances rows: {len(dist_df):,}")
    print(f"  - Drone info rows      : {len(drone_info_df):,}")
    print(f"  - Drone trajectory rows: {len(traj_df):,}")

    side = detect_uav_position_side(det_path, traj_df)
    return {
        "detections": det_df,
        "sensor_info": info_df,
        "sensor_dist": dist_df,
        "drone_info": drone_info_df,
        "drone_traj": traj_df,
        "uav_side": side,
    }


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


def extract_ground_truth_for_timestamps(exp_det: pd.DataFrame, ts_indices: np.ndarray,
                                        exp_traj: pd.DataFrame, side: str) -> np.ndarray:
    """
    Ground truth (m) for the predictors' timestamp indices. All predictors index the experiment's sorted
    unique detection timestamps, so each index is mapped back to its timestamp and looked up in the
    trajectory at the position where the RSSI was measured.
    """
    unique_ts = np.sort(exp_det["timestamp"].unique())
    pos = exp_traj.set_index("timestamp")[[f"{side}_x", f"{side}_y", f"{side}_z"]]
    true_xyz = pos.reindex(unique_ts[ts_indices]).to_numpy(dtype=float)
    return np.nan_to_num(true_xyz, nan=0.0)  # (0, 0, 0) rows are excluded by BaseUAVPredictor.evaluate


_basic_predictor = None


def get_basic_predictor(args) -> EncoderDecoderPredictor:
    """SeqED Basic is loaded once, and only when it is requested."""
    global _basic_predictor
    if _basic_predictor is None:
        _basic_predictor = EncoderDecoderPredictor(num_sensors=args.num_sensors, window_size=args.sequence_size,
                                                   name="SeqED Basic")
        print(f"[INFO] Encoder: {_basic_predictor.enc_ckpt_path}")
        print(f"[INFO] Basic decoder: {_basic_predictor.dec_ckpt_path}")
    return _basic_predictor


def build_predictors(args, group: str, finetuned_ckpt: Optional[str]) -> list:
    """Returns [(name, predictor, ptx_uncertainty_dbm)] for one predictor group (see PREDICTOR_GROUPS)."""
    configs = []
    if group == "trilateration":
        for unc in args.ptx_uncertainties:
            label = "RSSI Trilateration" if unc == 0.0 else f"RSSI Trilateration ({unc:g}dBm Uncertainty)"
            configs.append((label, RSSITrilaterationPredictor(name=label), unc))
    elif group == "wcl":
        configs.append(("WCL", RSSIWCLPredictor(name="WCL"), 0.0))
    elif group == "seqed_basic":
        configs.append(("SeqED Basic", get_basic_predictor(args), 0.0))
    elif group == "seqed_finetuned":
        if finetuned_ckpt is None:
            print("[INFO] No fine-tuned decoder for this dataset - SeqED Fine-tuned is skipped.")
        else:
            finetuned = EncoderDecoderPredictor(dec_ckpt_path=finetuned_ckpt, num_sensors=args.num_sensors,
                                                window_size=args.sequence_size, name="SeqED Fine-tuned")
            print(f"[INFO] Fine-tuned decoder: {finetuned.dec_ckpt_path}")
            configs.append(("SeqED Fine-tuned", finetuned, 0.0))
    return configs


def evaluate_predictors(
    raw_data: Dict[str, pd.DataFrame],
    predictor_configs: list,
    output_dir: str,
    data_label: str,
    tag: str,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    det_df = raw_data["detections"]
    traj_df = raw_data["drone_traj"]
    side = raw_data["uav_side"]
    unique_exps = np.sort(det_df["experiment_id"].unique())
    det_groups = {exp_id: g for exp_id, g in det_df.groupby("experiment_id")}
    traj_groups = {exp_id: g for exp_id, g in traj_df.groupby("experiment_id")}

    print("\n" + "=" * 75)
    print(f" EVALUATING PREDICTORS ON TEST DATA ({data_label})")
    print("=" * 75)
    print(f"Total experiments to evaluate: {len(unique_exps)}")

    summary_rows = []
    all_exp_predictions = []

    for pred_name, pred, ptx_unc in predictor_configs:
        print(f"\n---> Running evaluation for: {pred_name}...")
        exp_raw_data = apply_ptx_uncertainty_to_raw_data(raw_data, ptx_unc) if ptx_unc != 0.0 else raw_data
        drone_info_df = exp_raw_data["drone_info"]

        all_pred_list, all_true_list, all_ts_list, all_exp_list = [], [], [], []
        total_time_accum = 0.0
        preprocessing_time_accum = 0.0

        def exp_input(exp_id):
            return {
                "detections": det_groups[exp_id],
                "sensor_info": exp_raw_data["sensor_info"],
                "sensor_dist": exp_raw_data["sensor_dist"],
                "drone_info": drone_info_df,
            }

        # Untimed warm-up so one-time costs (CUDA context / kernel init) are not counted as latency
        pred.predict_experiment(exp_input(unique_exps[0]))

        for exp_id in tqdm(unique_exps, desc=f"Evaluating {pred_name}"):
            exp_det = det_groups[exp_id]
            raw_exp_input = exp_input(exp_id)

            pred_xyz_m, timestamps, proc_time_sec = pred.predict_experiment(raw_exp_input)
            true_xyz_m = extract_ground_truth_for_timestamps(exp_det, timestamps, traj_groups[exp_id], side)

            total_time_accum += proc_time_sec
            preprocessing_time_accum += pred.last_preprocessing_time_sec
            all_pred_list.append(pred_xyz_m)
            all_true_list.append(true_xyz_m)
            all_ts_list.append(timestamps)
            all_exp_list.append(np.full(len(timestamps), exp_id))

        pred_concat = np.concatenate(all_pred_list, axis=0)
        true_concat = np.concatenate(all_true_list, axis=0)
        ts_concat = np.concatenate(all_ts_list, axis=0)

        metrics, detailed_df = pred.evaluate(pred_concat, true_concat, ts_concat, total_time_accum)
        metrics["Predictor_Name"] = pred_name
        # Latency/throughput above cover model computation only; pre/post-processing is reported separately
        n_steps = metrics["Total_Timestamps"]
        metrics["Preprocessing_Time (s)"] = preprocessing_time_accum
        metrics["Preprocessing_Per_Timestamp (ms)"] = preprocessing_time_accum * 1000.0 / n_steps if n_steps else 0.0
        metrics["End_to_End_Per_Timestamp (ms)"] = (metrics["Latency_Per_Timestamp (ms)"]
                                                    + metrics["Preprocessing_Per_Timestamp (ms)"])
        summary_rows.append(metrics)

        detailed_df.insert(0, "experiment_id", np.concatenate(all_exp_list))
        detailed_df["predictor_name"] = pred_name
        all_exp_predictions.append(detailed_df)

        print(pred.format_metrics_summary(metrics))

    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = os.path.join(output_dir, f"summary_{tag}.csv")
    predictions_csv_path = os.path.join(output_dir, f"predictions_{tag}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    pd.concat(all_exp_predictions, ignore_index=True).to_csv(predictions_csv_path, index=False)

    print("\n" + "=" * 85)
    print(f" SIDE-BY-SIDE MODEL COMPARISON SUMMARY (TEST SET, {data_label})")
    print("=" * 85)
    cols_to_display = [
        "Predictor_Name", "3D_Mean_Error (m)", "3D_Median_Error (m)", "3D_RMSE (m)",
        "Trajectory_Step_Loss (m)", "PyTorch_MSELoss (km^2)", "3D_P95_Error (m)",
        "2D_Mean_Error (m)", "Z_Mean_Error (m)", "Total_Processing_Time (s)",
        "Latency_Per_Timestamp (ms)", "Preprocessing_Per_Timestamp (ms)"
    ]
    disp_cols = [c for c in cols_to_display if c in summary_df.columns]
    print(summary_df[disp_cols].to_string(index=False))
    print("=" * 85)

    print(f"\n[OK] Evaluation complete!")
    print(f"     - Summary CSV saved to    : {summary_csv_path}")
    print(f"     - Predictions CSV saved to: {predictions_csv_path}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 4 predictors on the 5 Final Tests datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", default=list(DATASETS), choices=list(DATASETS),
                        help="Datasets to evaluate (default: all)")
    parser.add_argument("--predictors", type=str, nargs="+", default=PREDICTOR_GROUPS, choices=PREDICTOR_GROUPS,
                        help="Predictor groups to run (default: all). CPU: trilateration wcl; GPU: seqed_basic seqed_finetuned")
    parser.add_argument("--num_sensors", type=int, default=10, help="Number of sensors")
    parser.add_argument("--sequence_size", type=int, default=40, help="Sequence size (window size) for Encoder-Decoder")
    parser.add_argument("--max_exps", type=int, default=100, help="Maximum number of experiments to evaluate (<= 0 for all)")
    parser.add_argument("--output_dir", type=str, default=os.path.join(ROOT_DIR, "results"),
                        help="Directory to save evaluation outputs")
    parser.add_argument("--ptx_uncertainties", type=float, nargs="+", default=[0.0, 1.0],
                        help="P_tx errors (dB) for RSSI trilateration (default: correct P_tx and +1 dB)")
    args = parser.parse_args()

    max_exps = args.max_exps if args.max_exps > 0 else None
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Predictor groups: {args.predictors}")

    for ds_key in args.datasets:
        data_dir, finetuned_ckpt = DATASETS[ds_key]
        print("\n" + "#" * 80)
        print(f" DATASET: {ds_key}")
        print("#" * 80)

        raw_dataset = load_raw_dataset_files(data_dir, max_exps)
        num_exps = raw_dataset["detections"]["experiment_id"].nunique()

        for group in args.predictors:
            predictor_configs = build_predictors(args, group, finetuned_ckpt)
            if not predictor_configs:
                continue
            summary_df = evaluate_predictors(
                raw_data=raw_dataset,
                predictor_configs=predictor_configs,
                output_dir=os.path.join(args.output_dir, ds_key),
                data_label=f"{ds_key}, {group}",
                tag=group,
            )
            summary_df.insert(0, "Dataset", ds_key)
            summary_df.insert(2, "Num_Experiments", num_exps)
            summary_df.to_csv(os.path.join(args.output_dir, ds_key, f"summary_{group}.csv"), index=False)
        del raw_dataset

    # Combined tables are rebuilt from every <output_dir>/<dataset>/summary_<group>.csv on disk, so separate
    # jobs (per dataset, CPU/GPU groups) do not overwrite each other; the last job to finish has everything.
    summary_paths = [os.path.join(args.output_dir, k, f"summary_{g}.csv") for k in DATASETS for g in PREDICTOR_GROUPS]
    all_df = pd.concat([pd.read_csv(p) for p in summary_paths if os.path.exists(p)], ignore_index=True)
    done_datasets = [k for k in DATASETS if k in set(all_df["Dataset"])]
    all_csv_path = os.path.join(args.output_dir, "all_summary.csv")
    all_df.to_csv(all_csv_path, index=False)

    predictor_order = list(dict.fromkeys(all_df["Predictor_Name"]))
    for metric in PIVOT_METRICS:
        pivot = all_df.pivot(index="Predictor_Name", columns="Dataset", values=metric)
        pivot = pivot.reindex(index=predictor_order, columns=done_datasets)
        fname = "pivot_" + metric.split(" (")[0] + ".csv"
        pivot.to_csv(os.path.join(args.output_dir, fname))
        if metric == "3D_Mean_Error (m)":
            print("\n" + "=" * 85)
            print(f" {metric}: PREDICTOR x DATASET")
            print("=" * 85)
            print(pivot.round(2).to_string())
            print("=" * 85)

    print(f"\n[OK] Evaluated: {args.datasets}. Combined summary ({len(done_datasets)} datasets): {all_csv_path}")
