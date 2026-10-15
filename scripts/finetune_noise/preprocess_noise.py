"""
preprocess_noise.py

Converts the raw noisy simulator output (e.g. Data/noise_data/noise/awgn_f100) into the exact per-row
feature layout of the `normalized_data.csv` files the encoder/decoder were trained on.

Expected raw layout (--raw_dir points at one noise level):

    <raw_dir>/train_data_*/<NK_i>/{sensor_detection_summary,drone_trajectories,sensor_distances,sensor_information}.csv
    <raw_dir>/test_data_*/<NK_i>/...

Columns written per row:

    experiment_id, sensor_id, time_step,
    sector_1_power .. sector_6_power,          # dBm in the detected sector, 0.0 elsewhere / no detection
    sensor_x, sensor_y, sensor_z,              # km
    gain, sinr_threshold, sensitivity,         # gain raw, sinr / 10, sensitivity / 90
    uav_x, uav_y, uav_z                        # UAV position where the RSSI was measured, km
                                               # (start_xyz or end_xyz - detected automatically from
                                               #  the detections' drone_positions)

Instead of a CSV + stacked sliding-window cache, every raw experiment folder is stored as a compact
block of rows plus per-experiment offsets:

    <cache_dir>/<split>/<tag>_data.npy     float32 [num_rows, 18]
    <cache_dir>/<split>/<tag>_offsets.npy  int64   [num_experiments + 1]

Sliding windows (seq_len rows, step 1) are cut on the fly by finetune_decoder.py, exactly like
scripts/data/build_dataset_cache.py does (rows of an experiment sorted by time_step, window = rows[i:i+seq_len]).

Folders under test_data_* go to the validation cache, everything else to the training cache.
The cache defaults to <run_dir>/cache, with run_dir = scripts/finetune_noise/runs/<basename(raw_dir)>,
so every noise level gets its own cache.

Usage:
    python scripts/finetune_noise/preprocess_noise.py --raw_dir Data/noise_data/noise/awgn_f110
"""

import os
import ast
import glob
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

RAW_DIR = os.path.join(ROOT_DIR, "Data", "noise_data", "noise", "awgn_f100")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")

# Table name used in the code -> CSV file name inside an experiment folder
RAW_FILE_NAMES = {"drone_trajectories": "drone_trajectories", "sensor_distances": "sensor_distances",
                  "sensor_information": "sensor_information", "detections": "sensor_detection_summary"}

NUM_SECTORS = 6
COLUMNS = (["experiment_id", "sensor_id", "time_step"]
           + [f"sector_{k}_power" for k in range(1, NUM_SECTORS + 1)]
           + ["sensor_x", "sensor_y", "sensor_z", "gain", "sinr_threshold", "sensitivity",
              "uav_x", "uav_y", "uav_z"])


def default_run_dir(raw_dir):
    """Per-noise-level output folder: scripts/finetune_noise/runs/<basename(raw_dir)>."""
    return os.path.join(RUNS_DIR, os.path.basename(os.path.normpath(raw_dir)))


def find_raw_prefixes(raw_dir=RAW_DIR):
    """Returns {tag: (experiment_dir, split)} for every <split>_data_*/<NK_i>/ folder in raw_dir."""
    prefixes = {}
    for det_path in sorted(glob.glob(os.path.join(raw_dir, "*", "*", f"{RAW_FILE_NAMES['detections']}.csv"))):
        exp_dir = os.path.dirname(det_path)
        split = "val" if os.path.basename(os.path.dirname(exp_dir)).startswith("test") else "train"
        tag = ("test_" if split == "val" else "") + os.path.basename(exp_dir)   # e.g. 5K_3 / test_5K_0
        prefixes[tag] = (exp_dir, split)
    return prefixes


def raw_files(prefix):
    """Maps the raw table names to the CSV paths inside one experiment folder."""
    return {t: os.path.join(prefix, f"{name}.csv") for t, name in RAW_FILE_NAMES.items()}


def detect_uav_position_side(files, traj, n_rows=20000, tol_m=1e-3):
    """
    Returns "start" or "end": which trajectory position the detections at a timestamp were measured at.

    The original training data measures RSSI at the trajectory row's start_xyz (normalized_data.csv uses
    start_xyz as the label), while the Mountains simulator measures it at end_xyz. The label must be the
    position where the RSSI was actually measured, so it is taken from the detections' drone_positions.
    """
    det = pd.read_csv(files["detections"], nrows=n_rows, usecols=["experiment_id", "timestamp", "drone_positions"])
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


def build_normalized_rows(files):
    """Builds the normalized feature rows (same columns/scaling as normalized_data.csv) from raw tables."""
    # --- UAV trajectory: one timestamp per row, time_step = running row index across the file ---
    traj = pd.read_csv(files["drone_trajectories"],
                       usecols=["experiment_id", "timestamp", "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"])
    traj = traj.sort_values(["experiment_id", "timestamp"], kind="stable").reset_index(drop=True)
    traj["time_step"] = np.arange(len(traj), dtype=np.int64)
    side = detect_uav_position_side(files, traj)
    traj["uav_x"] = traj[f"{side}_x"] / 1000.0
    traj["uav_y"] = traj[f"{side}_y"] / 1000.0
    traj["uav_z"] = traj[f"{side}_z"] / 1000.0
    traj = traj.drop(columns=["start_x", "start_y", "start_z", "end_x", "end_y", "end_z"])

    # --- Sensor locations per experiment (union of both ends of the pairwise distance table) ---
    dist = pd.read_csv(files["sensor_distances"])
    loc = pd.concat([
        dist[["exp_id", "sensor1_id", "x1", "y1", "z1"]].set_axis(["experiment_id", "sensor_id", "x", "y", "z"], axis=1),
        dist[["exp_id", "sensor2_id", "x2", "y2", "z2"]].set_axis(["experiment_id", "sensor_id", "x", "y", "z"], axis=1),
    ]).drop_duplicates(["experiment_id", "sensor_id"])
    del dist
    loc["sensor_x"] = loc.pop("x") / 1000.0
    loc["sensor_y"] = loc.pop("y") / 1000.0
    loc["sensor_z"] = loc.pop("z") / 1000.0

    # --- Static sensor hardware parameters (sensor_id is globally unique) ---
    info = pd.read_csv(files["sensor_information"], usecols=["sensor_id", "gain", "sinr_threshold", "sensitivity"])
    info = info.drop_duplicates("sensor_id")
    info["sinr_threshold"] = info["sinr_threshold"] / 10.0
    info["sensitivity"] = info["sensitivity"] / 90.0

    # --- Dense grid: every sensor of the experiment at every trajectory timestamp ---
    grid = traj[["experiment_id", "time_step", "uav_x", "uav_y", "uav_z"]].merge(loc, on="experiment_id", how="inner")
    grid = grid.merge(info, on="sensor_id", how="left")
    del loc, info

    # --- Detections -> per-sector received power (dBm), 0.0 when not detected ---
    det = pd.read_csv(files["detections"],
                      usecols=["experiment_id", "sensor_id", "timestamp", "total_power_dBm", "sensor_sectors_detected"])
    det["sector"] = det.pop("sensor_sectors_detected").str.strip("[]").str.split(",").str[0].astype(np.int64)
    det = det.merge(traj[["experiment_id", "timestamp", "time_step"]], on=["experiment_id", "timestamp"], how="inner")
    det = det.drop(columns=["experiment_id", "timestamp"]).drop_duplicates(["time_step", "sensor_id"])
    del traj

    grid = grid.merge(det, on=["time_step", "sensor_id"], how="left")
    del det
    power = grid.pop("total_power_dBm").to_numpy()
    sector = grid.pop("sector").to_numpy()
    for k in range(1, NUM_SECTORS + 1):
        grid[f"sector_{k}_power"] = np.where(sector == k, power, 0.0)

    grid = grid.sort_values(["time_step", "sensor_id"], kind="stable").reset_index(drop=True)
    return grid[COLUMNS]


def to_experiment_blocks(df):
    """
    Groups rows per experiment exactly like build_dataset_cache.process_single_csv_to_cache:
    each experiment's rows are ordered with pandas' default `sort_values('time_step')` and
    experiments containing NaNs are dropped.
    """
    data = df.to_numpy(dtype=np.float64)
    exp_ids = data[:, 0]
    _, starts = np.unique(exp_ids, return_index=True)
    bounds = np.append(np.sort(starts), len(data))

    blocks, offsets = [], [0]
    dropped = 0
    for s, e in zip(bounds[:-1], bounds[1:]):
        block = data[s:e]
        # Same (default quicksort) ordering pandas applied when the original cache was built
        order = pd.Series(block[:, 2].astype(np.int64)).sort_values().index.to_numpy()
        block = block[order].astype(np.float32)
        if np.isnan(block).any():
            dropped += 1
            continue
        blocks.append(block)
        offsets.append(offsets[-1] + len(block))

    if dropped:
        print(f"    [!] Dropped {dropped} experiment(s) containing NaNs")
    return np.concatenate(blocks, axis=0), np.asarray(offsets, dtype=np.int64)


def cache_paths(cache_dir, tag):
    return os.path.join(cache_dir, f"{tag}_data.npy"), os.path.join(cache_dir, f"{tag}_offsets.npy")


def preprocess_all(raw_dir=RAW_DIR, cache_dir=None, overwrite=False, save_csv=False):
    """Preprocesses every raw experiment folder. Returns {'train': [(data, offsets), ...], 'val': [...]}"""
    if cache_dir is None:
        cache_dir = os.path.join(default_run_dir(raw_dir), "cache")
    prefixes = find_raw_prefixes(raw_dir)
    if not prefixes:
        raise FileNotFoundError(f"No */*/{RAW_FILE_NAMES['detections']}.csv files found in {raw_dir}")

    out = {"train": [], "val": []}
    for tag, (prefix, split) in prefixes.items():
        split_dir = os.path.join(cache_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        data_path, off_path = cache_paths(split_dir, tag)
        out[split].append((data_path, off_path))

        if os.path.exists(data_path) and os.path.exists(off_path) and not overwrite:
            print(f"[*] Cache exists: {split}/{tag} (skipping)")
            continue

        print(f"[*] Preprocessing {os.path.relpath(prefix, raw_dir)} -> {split}/{tag}")
        df = build_normalized_rows(raw_files(prefix))
        if save_csv:
            csv_dir = os.path.join(split_dir, tag)
            os.makedirs(csv_dir, exist_ok=True)
            df.to_csv(os.path.join(csv_dir, "normalized_data.csv"), index=False)
        data, offsets = to_experiment_blocks(df)
        del df
        np.save(data_path, data)
        np.save(off_path, offsets)
        print(f"  [+] {len(offsets) - 1:,} experiments, {len(data):,} rows -> {os.path.basename(data_path)}")
        del data, offsets

    return out


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw noise data into normalized experiment blocks.")
    parser.add_argument("--raw_dir", default=RAW_DIR, help="One noise level, e.g. Data/noise_data/noise/awgn_f100")
    parser.add_argument("--run_dir", default=None, help="Default: runs/<basename(raw_dir)>")
    parser.add_argument("--cache_dir", default=None, help="Default: <run_dir>/cache")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_csv", action="store_true", help="Also write normalized_data.csv per file (large).")
    args = parser.parse_args()
    run_dir = args.run_dir or default_run_dir(args.raw_dir)
    cache_dir = args.cache_dir or os.path.join(run_dir, "cache")
    preprocess_all(args.raw_dir, cache_dir, overwrite=args.overwrite, save_csv=args.save_csv)


if __name__ == "__main__":
    main()
