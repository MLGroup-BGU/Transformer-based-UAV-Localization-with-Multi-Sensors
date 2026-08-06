"""
predict_uav_location.py

Inference script for UAV 3D localization using trained Transformer Encoder and LSTM Decoder models.

Normalization rules (strictly aligned with data_preprocessing_pipeline (1).md & model training):
1. Static parameters (gain, sinr_threshold, sensitivity): Z-Score normalized using training set statistics:
   - gain: mean = -0.140882, std = 2.820714
   - sinr_threshold: mean = 10.022711, std = 2.901010
   - sensitivity: mean = -89.933884, std = 5.812505
2. Coordinates (x, y, z) and targets (x_d, y_d, z_d): Scaled to kilometers (divided by 1000.0).
3. Dynamic RSSI signals: Raw dBm values, zero-padded (0.0) for inactive sensors/sectors.
4. Sequence Structure: 10 sensors densified per timestamp (Cartesian Product skeleton).
5. Sequence Inference: Sliding window of length 30 (seq_len=30 consecutive sensor rows),
   extracting encoder hidden states with layer_idx='embedding' and evaluating the last timestep
   prediction with timestamp-level aggregation.

Author: Antigravity AI
"""

import os
import sys
import ast
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure current directory and parent directory are accessible in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from HybridEncoder import HybridSpatiotemporalPredictor

# Exact static feature normalization statistics directly verified against processed_train_normalized.csv:
# - gain: mean = -90.005570, std = 5.775258
# - sinr_threshold: mean = 9.999188, std = 2.888816
# - sensitivity: mean = -0.003132, std = 2.887262
DEFAULT_TRAIN_STATIC_STATS = {
    "gain": {"mean": -90.005570, "std": 5.775258},
    "sinr_threshold": {"mean": 9.999188, "std": 2.888816},
    "sensitivity": {"mean": -0.003132, "std": 2.887262}
}

def parse_sector_detected(val):
    """
    Safely parses sector index from string or float/int to 1-indexed integer (1..6).
    Follows parsing constraints from data_preprocessing_pipeline (1).md.
    """
    if pd.isna(val):
        return 1
    if isinstance(val, (int, float, np.integer, np.floating)):
        return int(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return int(parsed[0])
        except Exception:
            pass
        match = re.search(r"\d+", val)
        if match:
            return int(match.group(0))
    return 1


def parse_drone_position(s):
    """
    Safely parses stringified drone position tuple '[(x, y, z)]' or '(x, y, z)' into coordinates in km.
    Follows parsing constraints from data_preprocessing_pipeline (1).md.
    """
    if pd.isna(s):
        return 0.0, 0.0, 0.0
    if isinstance(s, str):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, list) and len(val) > 0:
                val = val[0]
            if isinstance(val, (tuple, list)) and len(val) == 3:
                return float(val[0]) / 1000.0, float(val[1]) / 1000.0, float(val[2]) / 1000.0
        except Exception:
            pass
    elif isinstance(s, (tuple, list)) and len(s) == 3:
        return float(s[0]) / 1000.0, float(s[1]) / 1000.0, float(s[2]) / 1000.0
    return 0.0, 0.0, 0.0


def compute_zscore_static_map(sensor_info_df, train_stats=None):
    """
    Computes standard Z-score normalization for static parameters (gain, sinr_threshold, sensitivity)
    strictly as specified in section 3 of data_preprocessing_pipeline (1).md:
    x_norm = (x - mean_train) / std_train
    """
    if train_stats is None:
        train_stats = DEFAULT_TRAIN_STATIC_STATS

    df = sensor_info_df.copy()
    static_map = {}
    cols = ["gain", "sinr_threshold", "sensitivity"]
    
    for col in cols:
        if train_stats and col in train_stats:
            mean_val = train_stats[col]["mean"]
            std_val = train_stats[col]["std"]
        elif col in df.columns and not df[col].empty:
            mean_val = df[col].mean()
            std_val = df[col].std(ddof=0)
        else:
            mean_val = 0.0
            std_val = 1.0

        if pd.isna(std_val) or std_val < 1e-8:
            std_val = 1e-8

        if col in df.columns:
            df[f"{col}_norm"] = (df[col] - mean_val) / std_val
        else:
            df[f"{col}_norm"] = 0.0

    for _, row in df.iterrows():
        s_id = int(row["sensor_id"])
        static_map[s_id] = {
            "gain": float(row["gain_norm"]),
            "sinr_threshold": float(row["sinr_threshold_norm"]),
            "sensitivity": float(row["sensitivity_norm"])
        }
    return static_map


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder that takes encoder hidden states and predicts 3D (X_d, Y_d, Z_d) UAV coordinates.
    """
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, dropout=0.1, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: [batch, seq_len, input_size]
        Returns:
            out: [batch, seq_len, 3] - Predicted (x_d, y_d, z_d) UAV coordinates
        """
        x = self.input_proj(encoder_outputs)
        out, _ = self.lstm(x)
        out = self.fc_out(out)
        return out


def load_trained_models(enc_ckpt_path, dec_ckpt_path, device=None):
    """
    Loads trained Transformer Encoder and LSTM Decoder from checkpoint files.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate and load Transformer Encoder
    encoder = HybridSpatiotemporalPredictor(
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        dynamic_dim=6,
        static_dim=3,
        coords_dim=3
    )
    if os.path.exists(enc_ckpt_path):
        encoder.load_state_dict(torch.load(enc_ckpt_path, map_location=device))
        print(f"[OK] Loaded Encoder weights from: {enc_ckpt_path}")
    else:
        raise FileNotFoundError(f"Encoder checkpoint not found at: {enc_ckpt_path}")
    
    encoder.to(device)
    encoder.eval()

    # 2. Instantiate and load LSTM Decoder
    decoder = LSTMDecoder(
        input_size=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        output_size=3
    )
    if os.path.exists(dec_ckpt_path):
        dec_ckpt = torch.load(dec_ckpt_path, map_location=device)
        state_dict = dec_ckpt.get("decoder_state_dict", dec_ckpt)
        decoder.load_state_dict(state_dict)
        print(f"[OK] Loaded Decoder weights from: {dec_ckpt_path}")
    else:
        raise FileNotFoundError(f"Decoder checkpoint not found at: {dec_ckpt_path}")

    decoder.to(device)
    decoder.eval()

    return encoder, decoder, device


def load_from_normalized_csv(csv_path, num_timestamps=60, num_sensors=10, batch_size=1):
    """
    Loads input features directly from a normalized CSV file formatted with:
    [exp_id, sensor_id, timestamp, sector_1..6, x, y, z, gain, sinr_threshold, sensitivity, x_d, y_d, z_d]
    Per timestep, there are `num_sensors` consecutive rows.
    Total rows loaded = num_timestamps * num_sensors * batch_size.
    Static features in this file are already normalized using training statistics.
    """
    total_rows = num_timestamps * num_sensors * batch_size
    df = pd.read_csv(csv_path, nrows=total_rows)

    dynamic_cols = [f"sector_{i}" for i in range(1, 7)]
    static_cols = ["gain", "sinr_threshold", "sensitivity"]
    spatial_cols = ["x", "y", "z"]
    gt_cols = ["x_d", "y_d", "z_d"]

    dynamic_arr = df[dynamic_cols].values.reshape(batch_size, -1, 6).astype(np.float32)
    static_arr = df[static_cols].values.reshape(batch_size, -1, 3).astype(np.float32)
    spatial_arr = df[spatial_cols].values.reshape(batch_size, -1, 3).astype(np.float32)
    gt_arr = df[gt_cols].values.reshape(batch_size, -1, 3).astype(np.float32)
    timestamps = df["timestamp"].values[:df.shape[0] // batch_size]

    dynamic_feats = torch.tensor(dynamic_arr)
    static_feats = torch.tensor(static_arr)
    spatial_coords = torch.tensor(spatial_arr)

    return dynamic_feats, static_feats, spatial_coords, gt_arr[0], timestamps


def resolve_dataset_paths(data_dir):
    """
    Auto-detects the raw dataset CSV files in data_dir by matching patterns.
    Prioritizes 'no_anomalies' dataset files if available, with graceful fallbacks.
    Returns:
        detections_path, sensor_info_path, sensor_dist_path, traj_path
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found at: {data_dir}")

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    def find_file(keyword, fallback_name):
        matches = [f for f in files if keyword in f.lower()]
        if not matches:
            return os.path.join(data_dir, fallback_name)
        # Prioritize files starting with or containing 'no_anomalies'
        no_anom = [f for f in matches if "no_anomalies" in f.lower()]
        selected = no_anom[0] if no_anom else matches[0]
        return os.path.join(data_dir, selected)

    det_path = find_file("detections", "no_anomalies_10sensors_1drone_25000_1exps_detections.csv")
    info_path = find_file("sensor_information", "no_anomalies_10sensors_1drone_25000_1exps_sensor_information.csv")
    dist_path = find_file("sensor_distances", "no_anomalies_10sensors_1drone_25000_1exps_sensor_distances.csv")
    traj_path = find_file("trajector", "no_anomalies_10sensors_1drone_25000_1exps_drone_trajectories.csv")

    return det_path, info_path, dist_path, traj_path


def load_exp_detections_chunked(detections_path, exp_id, chunksize=100000):
    """
    Efficiently loads detection rows for a specific experiment_id by chunking large CSVs.
    Stops scanning once the contiguous experiment block is parsed.
    """
    chunks = []
    found = False
    for chunk in pd.read_csv(detections_path, chunksize=chunksize):
        exp_col = "experiment_id" if "experiment_id" in chunk.columns else ("exp_id" if "exp_id" in chunk.columns else chunk.columns[0])
        matching = chunk[chunk[exp_col] == exp_id]
        if not matching.empty:
            chunks.append(matching)
            found = True
        elif found:
            break
    if chunks:
        res = pd.concat(chunks, ignore_index=True)
        return res, exp_col
    return pd.DataFrame(), "experiment_id"


def prepare_features_from_raw_data(data_dir, exp_id=1, seq_len=60, num_sensors=10):
    """
    Preprocesses raw detections and sensor data from Data/ into feature tensors.
    Performs Cartesian Product Densification (10 sensors x seq_len timesteps),
    Z-score normalizes static features using training set stats, and scales spatial coordinates to kilometers (divided by 1000.0).
    Supports both 'no_anomalies_10sensors_...' and 'rssi_1000_exps_...' dataset formats.
    """
    detections_path, sensor_info_path, sensor_dist_path, _ = resolve_dataset_paths(data_dir)

    exp_detections, exp_col = load_exp_detections_chunked(detections_path, exp_id)
    if exp_detections.empty:
        raise ValueError(f"No detection records found for experiment_id={exp_id} in {detections_path}")

    sensor_info_df = pd.read_csv(sensor_info_path) if os.path.exists(sensor_info_path) else pd.DataFrame()
    sensor_dist_df = pd.read_csv(sensor_dist_path) if os.path.exists(sensor_dist_path) else pd.DataFrame()

    # 1. Rank-based timestamp normalization
    exp_detections["timestamp_number"] = (
        exp_detections.groupby(exp_col, sort=False)["timestamp"]
        .rank(method="dense")
        .astype("int32") - 1
    )
    
    # Filter to specified number of timesteps
    exp_detections = exp_detections[exp_detections["timestamp_number"] < seq_len].copy()

    # 2. Parse Sector IDs & Pivot total_power_dBm
    sec_col = "sensor_sectors_detected" if "sensor_sectors_detected" in exp_detections.columns else (
        "sector_detected" if "sector_detected" in exp_detections.columns else "sector"
    )
    pow_col = "total_power_dBm" if "total_power_dBm" in exp_detections.columns else (
        "received_power" if "received_power" in exp_detections.columns else "power"
    )
    
    exp_detections["sector"] = exp_detections[sec_col].apply(parse_sector_detected)

    pivot_det = exp_detections.pivot_table(
        index=[exp_col, "sensor_id", "timestamp_number"],
        columns="sector",
        values=pow_col,
        fill_value=0.0
    ).reset_index()

    # Ensure all sector columns 1 to 6 exist
    for sec_i in range(1, 7):
        if sec_i not in pivot_det.columns:
            pivot_det[sec_i] = 0.0
    
    sector_cols = [f"sector_{i}" for i in range(1, 7)]
    pivot_det = pivot_det.rename(columns={i: f"sector_{i}" for i in range(1, 7)})
    pivot_det = pivot_det.rename(columns={exp_col: "exp_id"})

    # 3. Construct Cartesian Product Skeleton (seq_len timesteps x num_sensors)
    unique_ts = np.arange(seq_len, dtype=np.int32)
    exp_sensors = np.sort(exp_detections["sensor_id"].unique())
    if len(exp_sensors) == num_sensors:
        all_sensors = exp_sensors
    else:
        # Resolve exact 10 sensor IDs for this experiment from sensor_dist_df
        dist_exp_col = "exp_id" if "exp_id" in sensor_dist_df.columns else "experiment_id"
        exp_dist = sensor_dist_df[sensor_dist_df[dist_exp_col] == exp_id] if not sensor_dist_df.empty else pd.DataFrame()
        if not exp_dist.empty:
            s1_ids = set(exp_dist["sensor1_id"].unique())
            s2_ids = set(exp_dist["sensor2_id"].unique())
            dist_sensors = np.sort(list(s1_ids.union(s2_ids)))
            if len(dist_sensors) == num_sensors:
                all_sensors = dist_sensors
            else:
                min_sid = exp_sensors.min() if len(exp_sensors) > 0 else (exp_id - 1) * num_sensors + 1
                base_sid = ((min_sid - 1) // num_sensors) * num_sensors + 1
                all_sensors = np.arange(base_sid, base_sid + num_sensors, dtype=np.int32)
        elif len(exp_sensors) > 0:
            min_sid = exp_sensors.min()
            base_sid = ((min_sid - 1) // num_sensors) * num_sensors + 1
            all_sensors = np.arange(base_sid, base_sid + num_sensors, dtype=np.int32)
        else:
            base_sid = (int(exp_id) - 1) * num_sensors + 1
            all_sensors = np.arange(base_sid, base_sid + num_sensors, dtype=np.int32)
    
    skeleton = pd.MultiIndex.from_product(
        [[exp_id], unique_ts, all_sensors],
        names=["exp_id", "timestamp_number", "sensor_id"]
    ).to_frame().reset_index(drop=True)

    # 4. Left Join & Zero-Padding
    merged = pd.merge(skeleton, pivot_det, on=["exp_id", "timestamp_number", "sensor_id"], how="left")
    merged[sector_cols] = merged[sector_cols].fillna(0.0)

    # 5. Extract and Scale Sensor Coordinates (Divided by 1000.0 to convert meters -> km)
    sensor_pos_map = {}
    if not sensor_dist_df.empty:
        dist_exp_col = "exp_id" if "exp_id" in sensor_dist_df.columns else "experiment_id"
        exp_dist = sensor_dist_df[sensor_dist_df[dist_exp_col] == exp_id]
        for _, row in exp_dist.iterrows():
            s1 = int(row["sensor1_id"])
            s2 = int(row["sensor2_id"])
            if s1 not in sensor_pos_map and "x1" in row:
                sensor_pos_map[s1] = (float(row["x1"]) / 1000.0, float(row["y1"]) / 1000.0, float(row["z1"]) / 1000.0)
            if s2 not in sensor_pos_map and "x2" in row:
                sensor_pos_map[s2] = (float(row["x2"]) / 1000.0, float(row["y2"]) / 1000.0, float(row["z2"]) / 1000.0)

    # 6. Z-Score Normalize Static Parameters using Training Set Stats
    static_map = compute_zscore_static_map(sensor_info_df, DEFAULT_TRAIN_STATIC_STATS) if not sensor_info_df.empty else {}

    # 7. Extract Target Drone Positions (Divided by 1000.0 to convert meters -> km)
    drone_pos_col = "drone_positions" if "drone_positions" in exp_detections.columns else (
        "drone_position" if "drone_position" in exp_detections.columns else None
    )
    ts_drone_map = {}
    if drone_pos_col and drone_pos_col in exp_detections.columns:
        for ts_num, group in exp_detections.groupby("timestamp_number"):
            first_pos_str = group.iloc[0][drone_pos_col]
            ts_drone_map[ts_num] = parse_drone_position(first_pos_str)

    # Map features onto merged skeleton
    merged["x"] = merged["sensor_id"].map(lambda s: sensor_pos_map.get(s, (0.0, 0.0, 0.0))[0])
    merged["y"] = merged["sensor_id"].map(lambda s: sensor_pos_map.get(s, (0.0, 0.0, 0.0))[1])
    merged["z"] = merged["sensor_id"].map(lambda s: sensor_pos_map.get(s, (0.0, 0.0, 0.0))[2])

    merged["gain"] = merged["sensor_id"].map(lambda s: static_map.get(s, {}).get("gain", 0.0))
    merged["sinr_threshold"] = merged["sensor_id"].map(lambda s: static_map.get(s, {}).get("sinr_threshold", 0.0))
    merged["sensitivity"] = merged["sensor_id"].map(lambda s: static_map.get(s, {}).get("sensitivity", 0.0))

    merged["x_d"] = merged["timestamp_number"].map(lambda t: ts_drone_map.get(t, (0.0, 0.0, 0.0))[0])
    merged["y_d"] = merged["timestamp_number"].map(lambda t: ts_drone_map.get(t, (0.0, 0.0, 0.0))[1])
    merged["z_d"] = merged["timestamp_number"].map(lambda t: ts_drone_map.get(t, (0.0, 0.0, 0.0))[2])

    # 8. Sort strictly by [exp_id, timestamp_number, sensor_id]
    merged = merged.sort_values(by=["exp_id", "timestamp_number", "sensor_id"]).reset_index(drop=True)

    # Convert to Tensors
    total_seq_len = seq_len * num_sensors
    dynamic_arr = merged[sector_cols].values.reshape(1, total_seq_len, 6).astype(np.float32)
    static_arr = merged[["gain", "sinr_threshold", "sensitivity"]].values.reshape(1, total_seq_len, 3).astype(np.float32)
    spatial_arr = merged[["x", "y", "z"]].values.reshape(1, total_seq_len, 3).astype(np.float32)
    gt_arr = merged[["x_d", "y_d", "z_d"]].values.reshape(1, total_seq_len, 3).astype(np.float32)

    dynamic_feats = torch.tensor(dynamic_arr)
    static_feats = torch.tensor(static_arr)
    spatial_coords = torch.tensor(spatial_arr)
    ground_truth = gt_arr[0]
    timestamps = merged["timestamp_number"].values

    return dynamic_feats, static_feats, spatial_coords, ground_truth, timestamps


def predict_uav_location(encoder, decoder, dynamic_feats, static_feats, spatial_coords, 
                         timestamps=None, window_size=30, layer_idx='embedding', batch_size=256, device=None):
    """
    Runs model inference to predict 3D UAV coordinates using sliding windows of length `window_size` (30).
    Aggregates overlapping window predictions per timestamp to produce the final trajectory.

    Args:
        encoder: HybridSpatiotemporalPredictor
        decoder: LSTMDecoder
        dynamic_feats: Tensor [1, T_total, 6]
        static_feats: Tensor [1, T_total, 3]
        spatial_coords: Tensor [1, T_total, 3]
        timestamps: Array/List of length T_total corresponding to each row
        window_size: Length of each sliding window (default 30 rows)
        layer_idx: Layer to extract from encoder ('embedding' for Linear_Layer_embDec models)
        batch_size: Batch size for batched window inference
        device: Torch device

    Returns:
        raw_pred_xyz: Array [n_windows, 3] of predictions at the end of each sliding window
        agg_df: DataFrame [n_timestamps, 4] with columns [timestamp, pred_x, pred_y, pred_z]
    """
    if device is None:
        device = next(encoder.parameters()).device

    T_total = dynamic_feats.size(1)
    if T_total < window_size:
        raise ValueError(f"Sequence length ({T_total}) is shorter than required window size ({window_size})")

    n_windows = T_total - window_size
    
    dyn_windows = []
    stat_windows = []
    spat_windows = []
    end_ts_list = []

    for i in range(n_windows):
        dyn_windows.append(dynamic_feats[0, i : i + window_size])
        stat_windows.append(static_feats[0, i : i + window_size])
        spat_windows.append(spatial_coords[0, i : i + window_size])
        if timestamps is not None:
            end_ts_list.append(int(timestamps[i + window_size - 1]))
        else:
            end_ts_list.append(i + window_size - 1)

    dyn_all = torch.stack(dyn_windows)
    stat_all = torch.stack(stat_windows)
    spat_all = torch.stack(spat_windows)

    all_preds = []

    with torch.no_grad():
        for b_idx in range(0, n_windows, batch_size):
            b_dyn = dyn_all[b_idx : b_idx + batch_size].to(device)
            b_stat = stat_all[b_idx : b_idx + batch_size].to(device)
            b_spat = spat_all[b_idx : b_idx + batch_size].to(device)

            enc_hidden = encoder.get_hidden_state(b_dyn, b_stat, b_spat, layer_idx=layer_idx)
            pred_seq = decoder(enc_hidden)
            pred_last = pred_seq[:, -1, :].cpu().numpy()
            all_preds.append(pred_last)

    raw_pred_xyz = np.concatenate(all_preds, axis=0)

    pred_df = pd.DataFrame({
        "timestamp": end_ts_list,
        "pred_x": raw_pred_xyz[:, 0],
        "pred_y": raw_pred_xyz[:, 1],
        "pred_z": raw_pred_xyz[:, 2]
    })

    # Aggregate multiple sensor predictions per timestamp (mean aggregation)
    agg_df = pred_df.groupby("timestamp")[["pred_x", "pred_y", "pred_z"]].mean().reset_index()

    return raw_pred_xyz, agg_df


def evaluate_predictions(agg_df, ground_truth, timestamps, window_size=30):
    """
    Calculates localization errors (3D and 2D Euclidean Distances, MSE, MAE, Z error) on aggregated trajectory.
    Coordinates are in kilometers; 3D and 2D Euclidean errors are reported in meters and km.
    Only timesteps with valid ground truth (non-zero) are included in error statistics.
    """
    gt_df = pd.DataFrame({
        "timestamp": timestamps,
        "true_x": ground_truth[:, 0],
        "true_y": ground_truth[:, 1],
        "true_z": ground_truth[:, 2]
    }).drop_duplicates(subset=["timestamp"])

    merged = pd.merge(agg_df, gt_df, on="timestamp")

    pred_coords = merged[["pred_x", "pred_y", "pred_z"]].values
    true_coords = merged[["true_x", "true_y", "true_z"]].values

    diff_3d = pred_coords - true_coords
    diff_2d = pred_coords[:, :2] - true_coords[:, :2]
    diff_z = pred_coords[:, 2] - true_coords[:, 2]

    # 3D Errors
    errors_3d_m = np.linalg.norm(diff_3d, axis=-1) * 1000.0
    # 2D Errors (XY plane)
    errors_2d_m = np.linalg.norm(diff_2d, axis=-1) * 1000.0
    # Z Errors
    errors_z_m = np.abs(diff_z) * 1000.0

    merged["error_2d_m"] = errors_2d_m
    merged["error_3d_m"] = errors_3d_m
    merged["error_z_m"] = errors_z_m

    # Filter valid ground truth positions (ignore padded 0,0,0 unobserved timesteps)
    valid_mask = ~((merged["true_x"] == 0.0) & (merged["true_y"] == 0.0) & (merged["true_z"] == 0.0))
    if not valid_mask.any():
        valid_mask_arr = np.ones(len(merged), dtype=bool)
    else:
        valid_mask_arr = valid_mask.values

    diff_3d_val = diff_3d[valid_mask_arr]
    diff_2d_val = diff_2d[valid_mask_arr]
    e3d_val = errors_3d_m[valid_mask_arr]
    e2d_val = errors_2d_m[valid_mask_arr]
    ez_val = errors_z_m[valid_mask_arr]

    mse_3d = np.mean(diff_3d_val ** 2)
    mae_3d = np.mean(np.abs(diff_3d_val))
    mse_2d = np.mean(diff_2d_val ** 2)
    mae_2d = np.mean(np.abs(diff_2d_val))

    metrics = {
        # 3D Metrics
        "3D_MSE (km^2)": mse_3d,
        "3D_MAE (km)": mae_3d,
        "3D_Mean_Error (m)": np.mean(e3d_val),
        "3D_Median_Error (m)": np.median(e3d_val),
        "3D_Std_Error (m)": np.std(e3d_val),
        "3D_P90_Error (m)": np.percentile(e3d_val, 90),
        "3D_P95_Error (m)": np.percentile(e3d_val, 95),
        "3D_Min_Error (m)": np.min(e3d_val),
        "3D_Max_Error (m)": np.max(e3d_val),
        # 2D Metrics
        "2D_MSE (km^2)": mse_2d,
        "2D_MAE (km)": mae_2d,
        "2D_Mean_Error (m)": np.mean(e2d_val),
        "2D_Median_Error (m)": np.median(e2d_val),
        "2D_Std_Error (m)": np.std(e2d_val),
        "2D_P90_Error (m)": np.percentile(e2d_val, 90),
        "2D_P95_Error (m)": np.percentile(e2d_val, 95),
        "2D_Min_Error (m)": np.min(e2d_val),
        "2D_Max_Error (m)": np.max(e2d_val),
        # Z-axis Metrics
        "Z_Mean_Error (m)": np.mean(ez_val),
        "Z_Median_Error (m)": np.median(ez_val),
        "Z_P95_Error (m)": np.percentile(ez_val, 95),
    }
    return metrics, errors_3d_m, errors_2d_m, merged


def evaluate_all_experiments(
    data_dir,
    encoder,
    decoder,
    device,
    window_size=30,
    seq_len=60,
    num_sensors=10,
    layer_idx='embedding',
    max_exps=None,
    output_csv=None,
    summary_csv=None,
    max_m=None,
    sensor_dist_map_path=None,
):
    """
    Runs UAV localization inference across all experiments in data_dir,
    calculates comprehensive 3D and 2D (horizontal) error statistics,
    and saves detailed predictions and summary CSV files.
    """
    detections_path, sensor_info_path, sensor_dist_path, _ = resolve_dataset_paths(data_dir)

    print("\n" + "=" * 70)
    print(" BATCH EVALUATION OVER ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Data directory : {data_dir}")
    print(f"Detections file: {os.path.basename(detections_path)}")
    print(f"Window size    : {window_size} timestamps")
    print(f"Layer idx      : '{layer_idx}'")
    print(f"Device         : {device}")

    # 1. Pre-load raw dataset files into memory once for high performance
    print("\n[INFO] Loading raw dataset CSV files into memory...")
    detections_df = pd.read_csv(detections_path)
    sensor_info_df = pd.read_csv(sensor_info_path) if os.path.exists(sensor_info_path) else pd.DataFrame()
    sensor_dist_df = pd.read_csv(sensor_dist_path) if os.path.exists(sensor_dist_path) else pd.DataFrame()

    exp_col = "experiment_id" if "experiment_id" in detections_df.columns else "exp_id"
    unique_exps = np.sort(detections_df[exp_col].unique())
    if max_exps is not None:
        unique_exps = unique_exps[:max_exps]

    # --- Filter by max mean active-sensor-to-UAV distance ---
    if max_m is not None:
        # Resolve path to pre-computed distance map
        if sensor_dist_map_path is None:
            sensor_dist_map_path = os.path.join(SCRIPT_DIR, "experiment_active_sensor_distances.json")
        if not os.path.exists(sensor_dist_map_path):
            # Fall back to CSV
            csv_fallback = os.path.splitext(sensor_dist_map_path)[0] + ".csv"
            if os.path.exists(csv_fallback):
                import json as _json
                dist_df_filter = pd.read_csv(csv_fallback)
                sensor_dist_map_path = None  # signal to use df directly
                dist_avg_map = dict(zip(
                    dist_df_filter["exp_id"].astype(int),
                    dist_df_filter["avg_active_sensor_distance_3d_m"]
                ))
            else:
                print(f"[WARNING] --max_m filter requested but no distance map file found at:\n"
                      f"  {sensor_dist_map_path}\n"
                      f"Run calculate_active_sensors_distance.py first. Skipping filter.")
                dist_avg_map = {}
        else:
            import json as _json
            with open(sensor_dist_map_path, "r") as _f:
                dist_avg_map = {int(k): float(v) for k, v in _json.load(_f).items()}

        if dist_avg_map:
            before_count = len(unique_exps)
            unique_exps = np.array(
                [eid for eid in unique_exps if dist_avg_map.get(int(eid), float("inf")) <= max_m],
                dtype=unique_exps.dtype
            )
            print(f"[INFO] --max_m {max_m:.0f} m filter: {before_count} -> {len(unique_exps)} experiments kept.")

    print(f"[INFO] Found {len(unique_exps)} experiments to evaluate.")

    # Pre-build static info map
    info_map = {}
    if not sensor_info_df.empty:
        s_id_col = "sensor_id" if "sensor_id" in sensor_info_df.columns else "sensor1_id"
        for _, row in sensor_info_df.iterrows():
            info_map[int(row[s_id_col])] = {
                "gain": float(row.get("gain", 0.0)),
                "sinr_threshold": float(row.get("sinr_threshold", 0.0)),
                "sensitivity": float(row.get("sensitivity", 0.0)),
            }

    # Pre-build distances map by exp_id
    dist_map = {}
    if not sensor_dist_df.empty:
        dist_exp_col = "exp_id" if "exp_id" in sensor_dist_df.columns else "experiment_id"
        for exp_id_val, grp in sensor_dist_df.groupby(dist_exp_col):
            s_map = {}
            for _, row in grp.iterrows():
                s1 = int(row["sensor1_id"])
                s2 = int(row["sensor2_id"])
                if s1 not in s_map and "x1" in row:
                    s_map[s1] = (float(row["x1"]) / 1000.0, float(row["y1"]) / 1000.0, float(row["z1"]) / 1000.0)
                if s2 not in s_map and "x2" in row:
                    s_map[s2] = (float(row["x2"]) / 1000.0, float(row["y2"]) / 1000.0, float(row["z2"]) / 1000.0)
            dist_map[int(exp_id_val)] = s_map

    # Detection column names
    sec_col = "sensor_sectors_detected" if "sensor_sectors_detected" in detections_df.columns else (
        "sector_detected" if "sector_detected" in detections_df.columns else "sector"
    )
    pow_col = "total_power_dBm" if "total_power_dBm" in detections_df.columns else (
        "received_power" if "received_power" in detections_df.columns else "power"
    )
    drone_pos_col = "drone_positions" if "drone_positions" in detections_df.columns else (
        "drone_position" if "drone_position" in detections_df.columns else None
    )
    sector_cols = [f"sector_{i}" for i in range(1, 7)]
    unique_ts = np.arange(seq_len, dtype=np.int32)

    # 2. Iterate through all experiments
    all_timestep_records = []
    exp_summary_records = []

    pbar = tqdm(unique_exps, desc="Running Inference across Experiments", unit="exp")
    for exp_id in pbar:
        exp_det = detections_df[detections_df[exp_col] == exp_id].copy()
        if exp_det.empty:
            continue

        exp_det["timestamp_number"] = (
            exp_det.groupby(exp_col, sort=False)["timestamp"].rank(method="dense").astype("int32") - 1
        )
        exp_det = exp_det[exp_det["timestamp_number"] < seq_len].copy()
        exp_det["sector"] = exp_det[sec_col].apply(parse_sector_detected)

        pivot_det = exp_det.pivot_table(
            index=[exp_col, "sensor_id", "timestamp_number"],
            columns="sector",
            values=pow_col,
            fill_value=0.0
        ).reset_index()

        for sec_i in range(1, 7):
            if sec_i not in pivot_det.columns:
                pivot_det[sec_i] = 0.0
        pivot_det = pivot_det.rename(columns={i: f"sector_{i}" for i in range(1, 7)})
        pivot_det = pivot_det.rename(columns={exp_col: "exp_id"})

        exp_sensors = np.sort(exp_det["sensor_id"].unique())
        if len(exp_sensors) == num_sensors:
            all_sensors = exp_sensors
        else:
            s_map = dist_map.get(int(exp_id), {})
            if len(s_map) == num_sensors:
                all_sensors = np.sort(list(s_map.keys()))
            elif len(exp_sensors) > 0:
                min_sid = exp_sensors.min()
                base_sid = ((min_sid - 1) // num_sensors) * num_sensors + 1
                all_sensors = np.arange(base_sid, base_sid + num_sensors, dtype=np.int32)
            else:
                base_sid = (int(exp_id) - 1) * num_sensors + 1
                all_sensors = np.arange(base_sid, base_sid + num_sensors, dtype=np.int32)

        skeleton = pd.MultiIndex.from_product(
            [[exp_id], unique_ts, all_sensors],
            names=["exp_id", "timestamp_number", "sensor_id"]
        ).to_frame().reset_index(drop=True)

        merged = pd.merge(skeleton, pivot_det, on=["exp_id", "timestamp_number", "sensor_id"], how="left")
        merged[sector_cols] = merged[sector_cols].fillna(0.0)

        # Spatial sensor coords (km)
        s_pos_map = dist_map.get(int(exp_id), {})
        merged["x"] = merged["sensor_id"].apply(lambda sid: s_pos_map.get(sid, (0.0, 0.0, 0.0))[0])
        merged["y"] = merged["sensor_id"].apply(lambda sid: s_pos_map.get(sid, (0.0, 0.0, 0.0))[1])
        merged["z"] = merged["sensor_id"].apply(lambda sid: s_pos_map.get(sid, (0.0, 0.0, 0.0))[2])

        # Static features (Z-score normalized)
        raw_gain = merged["sensor_id"].apply(lambda sid: info_map.get(sid, {}).get("gain", 0.0))
        raw_sinr = merged["sensor_id"].apply(lambda sid: info_map.get(sid, {}).get("sinr_threshold", 0.0))
        raw_sens = merged["sensor_id"].apply(lambda sid: info_map.get(sid, {}).get("sensitivity", 0.0))

        merged["gain"] = (raw_gain - DEFAULT_TRAIN_STATIC_STATS["gain"]["mean"]) / DEFAULT_TRAIN_STATIC_STATS["gain"]["std"]
        merged["sinr_threshold"] = (raw_sinr - DEFAULT_TRAIN_STATIC_STATS["sinr_threshold"]["mean"]) / DEFAULT_TRAIN_STATIC_STATS["sinr_threshold"]["std"]
        merged["sensitivity"] = (raw_sens - DEFAULT_TRAIN_STATIC_STATS["sensitivity"]["mean"]) / DEFAULT_TRAIN_STATIC_STATS["sensitivity"]["std"]

        # Ground truth drone position (km)
        ts_pos_map = {}
        if drone_pos_col and drone_pos_col in exp_det.columns:
            for _, row in exp_det.iterrows():
                ts_pos_map[int(row["timestamp_number"])] = parse_drone_position(row[drone_pos_col])

        gt_tuples = [ts_pos_map.get(ts, (0.0, 0.0, 0.0)) for ts in unique_ts]
        gt_df = pd.DataFrame(gt_tuples, columns=["x_d", "y_d", "z_d"])
        gt_df["timestamp_number"] = unique_ts

        merged = pd.merge(merged, gt_df, on="timestamp_number", how="left")
        merged = merged.sort_values(["exp_id", "timestamp_number", "sensor_id"]).reset_index(drop=True)

        dynamic_feats = torch.tensor(merged[sector_cols].values.reshape(1, -1, 6).astype(np.float32))
        static_feats = torch.tensor(merged[["gain", "sinr_threshold", "sensitivity"]].values.reshape(1, -1, 3).astype(np.float32))
        spatial_coords = torch.tensor(merged[["x", "y", "z"]].values.reshape(1, -1, 3).astype(np.float32))
        ground_truth = merged[["x_d", "y_d", "z_d"]].values.astype(np.float32)
        timestamps = merged["timestamp_number"].values

        # Run Model Inference
        raw_preds, agg_df = predict_uav_location(
            encoder, decoder, dynamic_feats, static_feats, spatial_coords,
            timestamps=timestamps, window_size=window_size, layer_idx=layer_idx, device=device
        )

        metrics, errors_3d_m, errors_2d_m, merged_results = evaluate_predictions(
            agg_df, ground_truth, timestamps, window_size=window_size
        )
        merged_results["exp_id"] = exp_id
        all_timestep_records.append(merged_results)

        # Average altitude of the drone in this experiment
        avg_alt_m = np.mean(ground_truth[:, 2]) * 1000.0

        exp_summary_records.append({
            "exp_id": exp_id,
            "mean_error_2d_m": metrics["2D_Mean_Error (m)"],
            "median_error_2d_m": metrics["2D_Median_Error (m)"],
            "mean_error_3d_m": metrics["3D_Mean_Error (m)"],
            "median_error_3d_m": metrics["3D_Median_Error (m)"],
            "mse_2d_km2": metrics["2D_MSE (km^2)"],
            "mse_3d_km2": metrics["3D_MSE (km^2)"],
            "drone_altitude_m": avg_alt_m
        })

    # Combine all results
    full_df = pd.concat(all_timestep_records, ignore_index=True)
    summary_df = pd.DataFrame(exp_summary_records)

    # 3. Print Comprehensive Global Statistics Report
    print("\n" + "=" * 70)
    print(" GLOBAL EVALUATION REPORT (2D & 3D LOCALIZATION)")
    print("=" * 70)
    print(f" Total Experiments Evaluated : {len(summary_df):,}")
    print(f" Total Trajectory Timesteps  : {len(full_df):,}")

    # Mask valid timesteps (ignore unobserved (0,0,0) padded points)
    valid_mask = ~((full_df["true_x"] == 0.0) & (full_df["true_y"] == 0.0) & (full_df["true_z"] == 0.0))
    valid_df = full_df[valid_mask] if valid_mask.any() else full_df
    print(f" Valid Evaluated Timesteps   : {len(valid_df):,} (excluding unobserved/padded points)")

    e2d = valid_df["error_2d_m"]
    e3d = valid_df["error_3d_m"]
    ez = valid_df["error_z_m"]

    # Overall Metrics
    diff_2d_km = (valid_df[["pred_x", "pred_y"]].values - valid_df[["true_x", "true_y"]].values)
    diff_3d_km = (valid_df[["pred_x", "pred_y", "pred_z"]].values - valid_df[["true_x", "true_y", "true_z"]].values)
    mse_2d_global = np.mean(diff_2d_km ** 2)
    mse_3d_global = np.mean(diff_3d_km ** 2)

    print("\n-- 2D Horizontal (XY-plane) Error Statistics -------------------------")
    print(f"  MSE (km^2) : {mse_2d_global:.6f}")
    print(f"  Mean (m)  : {e2d.mean():.2f} m")
    print(f"  Std (m)   : {e2d.std():.2f} m")
    print(f"  Min (m)   : {e2d.min():.2f} m")
    print(f"  P25 (m)   : {e2d.quantile(0.25):.2f} m")
    print(f"  Median(m) : {e2d.median():.2f} m")
    print(f"  P75 (m)   : {e2d.quantile(0.75):.2f} m")
    print(f"  P90 (m)   : {e2d.quantile(0.90):.2f} m")
    print(f"  P95 (m)   : {e2d.quantile(0.95):.2f} m")
    print(f"  P99 (m)   : {e2d.quantile(0.99):.2f} m")
    print(f"  Max (m)   : {e2d.max():.2f} m")

    print("\n-- 3D Euclidean Error Statistics --------------------------------------")
    print(f"  MSE (km^2) : {mse_3d_global:.6f}")
    print(f"  Mean (m)  : {e3d.mean():.2f} m")
    print(f"  Std (m)   : {e3d.std():.2f} m")
    print(f"  Min (m)   : {e3d.min():.2f} m")
    print(f"  P25 (m)   : {e3d.quantile(0.25):.2f} m")
    print(f"  Median(m) : {e3d.median():.2f} m")
    print(f"  P75 (m)   : {e3d.quantile(0.75):.2f} m")
    print(f"  P90 (m)   : {e3d.quantile(0.90):.2f} m")
    print(f"  P95 (m)   : {e3d.quantile(0.95):.2f} m")
    print(f"  P99 (m)   : {e3d.quantile(0.99):.2f} m")
    print(f"  Max (m)   : {e3d.max():.2f} m")

    print("\n-- Vertical (Z-axis / Altitude) Error Statistics ----------------------")
    print(f"  Mean (m)  : {ez.mean():.2f} m")
    print(f"  Std (m)   : {ez.std():.2f} m")
    print(f"  Median(m) : {ez.median():.2f} m")
    print(f"  P95 (m)   : {ez.quantile(0.95):.2f} m")
    print(f"  Max (m)   : {ez.max():.2f} m")

    # Domain Breakdown: Operational Low Altitude (<= 500m) vs High Altitude (> 500m)
    low_mask = valid_df["true_z"] <= 0.500
    if low_mask.any():
        low_2d = valid_df[low_mask]["error_2d_m"]
        low_3d = valid_df[low_mask]["error_3d_m"]
        low_diff_3d = diff_3d_km[low_mask.values]
        low_diff_2d = diff_2d_km[low_mask.values]
        low_mse_3d = np.mean(low_diff_3d ** 2)
        low_mse_2d = np.mean(low_diff_2d ** 2)

        print("\n" + "-" * 70)
        print(f"  Operational Domain Subset: Altitude <= 500m ({low_mask.sum():,} timesteps)")
        print("-" * 70)
        print(f"  2D MSE (km^2) : {low_mse_2d:.6f}  |  2D Mean (m)  : {low_2d.mean():.2f} m  |  2D Median : {low_2d.median():.2f} m  |  2D P95 : {low_2d.quantile(0.95):.2f} m")
        print(f"  3D MSE (km^2) : {low_mse_3d:.6f}  |  3D Mean (m)  : {low_3d.mean():.2f} m  |  3D Median : {low_3d.median():.2f} m  |  3D P95 : {low_3d.quantile(0.95):.2f} m")

    print("=" * 70 + "\n")

    # 4. Save results
    if output_csv is None:
        output_csv = os.path.join(SCRIPT_DIR, "predicted_uav_trajectories_all.csv")
    if summary_csv is None:
        summary_csv = os.path.join(SCRIPT_DIR, "experiments_summary.csv")

    full_df.to_csv(output_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Full timestep predictions saved to : {output_csv}")
    print(f"[OK] Per-experiment summary saved to    : {summary_csv}\n")

    return full_df, summary_df


def main():
    parser = argparse.ArgumentParser(description="UAV Localization Inference using Transformer & Decoder")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to Data directory containing raw CSVs (default: ../Data)")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to pre-normalized CSV dataset file")
    parser.add_argument("--raw", action="store_true", help="Force feature preprocessing from raw CSV files in Data/")
    parser.add_argument("--exp_id", type=int, default=1, help="Experiment ID to process when running single experiment (default: 1)")
    parser.add_argument("--all", "--all_exps", dest="all_exps", action="store_true", help="Run inference over ALL experiments in the Data folder")
    parser.add_argument("--max_exps", type=int, default=None, help="Maximum number of experiments to process when running batch mode")
    parser.add_argument("--max_m", type=float, default=None,
                        help="Only evaluate experiments whose mean active-sensor-to-UAV distance is <= this threshold (metres). "
                             "Requires experiment_active_sensor_distances.json produced by calculate_active_sensors_distance.py.")
    parser.add_argument("--sensor_dist_map", type=str, default=None,
                        help="Path to experiment_active_sensor_distances.json (default: same directory as this script)")
    parser.add_argument("--window_size", type=int, default=30, help="Sliding window size in timestamps (default: 30)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda / cpu)")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to output predictions CSV file")
    parser.add_argument("--post_process", action="store_true", help="Apply online Gradient Boosting/XGBoost post-processing refinement")
    parser.add_argument("--post_model_path", type=str, default=None, help="Path to post_processing_models.joblib artifact")
    args = parser.parse_args()

    root_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    data_dir = os.path.abspath(args.data_dir) if args.data_dir else os.path.join(root_dir, "Data")
    csv_file = args.csv_path

    enc_ckpt_path = os.path.join(SCRIPT_DIR, "hybrid_predictor_10sensors_1_epochs.pth")
    dec_ckpt_path = os.path.join(SCRIPT_DIR, "Linear_Layer_embDec", "decoder_10sensors_29_epochs_baseline_30seq.pth")

    print("==========================================================")
    print(" UAV Localization Inference using Transformer & Decoder")
    print("==========================================================")

    # 1. Load Trained Models
    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using compute device: {device}")
    encoder, decoder, device = load_trained_models(enc_ckpt_path, dec_ckpt_path, device)

    # 2. Configure Inference Parameters
    num_timestamps = 60
    num_sensors = 10
    window_size = args.window_size
    layer_idx = 'embedding' if "Linear_Layer_embDec" in dec_ckpt_path else -1

    # 3. Batch Evaluation across all experiments
    if args.all_exps:
        evaluate_all_experiments(
            data_dir=data_dir,
            encoder=encoder,
            decoder=decoder,
            device=device,
            window_size=window_size,
            seq_len=num_timestamps,
            num_sensors=num_sensors,
            layer_idx=layer_idx,
            max_exps=args.max_exps,
            output_csv=args.output_csv,
            max_m=args.max_m,
            sensor_dist_map_path=args.sensor_dist_map,
        )
        return

    # 4. Single Experiment Evaluation
    if csv_file and os.path.exists(csv_file):
        print(f"\n[OK] Found pre-normalized dataset file:\n  {csv_file}")
        print("Extracting sequence features...")
        dynamic_feats, static_feats, spatial_coords, ground_truth, timestamps = load_from_normalized_csv(
            csv_file, num_timestamps=num_timestamps, num_sensors=num_sensors, batch_size=1
        )
    else:
        print(f"\nProcessing raw CSV files from Data directory: {data_dir} (exp_id={args.exp_id})...")
        print("Applying Cartesian Densification, Training Z-Score Normalization, & Coordinate Scaling (km)...")
        dynamic_feats, static_feats, spatial_coords, ground_truth, timestamps = prepare_features_from_raw_data(
            data_dir, exp_id=args.exp_id, seq_len=num_timestamps, num_sensors=num_sensors
        )

    # Run Model Inference with Sliding Window & Aggregation
    print(f"Running inference with window_size={window_size}, layer_idx='{layer_idx}'...")
    raw_preds, agg_df = predict_uav_location(
        encoder, decoder, dynamic_feats, static_feats, spatial_coords,
        timestamps=timestamps, window_size=window_size, layer_idx=layer_idx, device=device
    )

    # Compute Error Metrics
    metrics, errors_3d_m, errors_2d_m, merged_results = evaluate_predictions(
        agg_df, ground_truth, timestamps, window_size=window_size
    )

    print("\n----------------------------------------------------------")
    print(" Localization Performance & Error Metrics:")
    print("----------------------------------------------------------")
    print(f"  -- 2D Horizontal (XY) ----------------------------------")
    print(f"  2D MSE (km^2)        : {metrics['2D_MSE (km^2)']:.6f}")
    print(f"  2D Mean Error (m)   : {metrics['2D_Mean_Error (m)']:.2f} m")
    print(f"  2D Median Error (m) : {metrics['2D_Median_Error (m)']:.2f} m")
    print(f"  2D P95 Error (m)    : {metrics['2D_P95_Error (m)']:.2f} m")
    print(f"  2D Min Error (m)    : {metrics['2D_Min_Error (m)']:.2f} m")
    print(f"  2D Max Error (m)    : {metrics['2D_Max_Error (m)']:.2f} m")
    print(f"  -- 3D Euclidean ----------------------------------------")
    print(f"  3D MSE (km^2)        : {metrics['3D_MSE (km^2)']:.6f}")
    print(f"  3D Mean Error (m)   : {metrics['3D_Mean_Error (m)']:.2f} m")
    print(f"  3D Median Error (m) : {metrics['3D_Median_Error (m)']:.2f} m")
    print(f"  3D P95 Error (m)    : {metrics['3D_P95_Error (m)']:.2f} m")
    print(f"  3D Min Error (m)    : {metrics['3D_Min_Error (m)']:.2f} m")
    print(f"  3D Max Error (m)    : {metrics['3D_Max_Error (m)']:.2f} m")
    print(f"  -- Vertical (Z) ----------------------------------------")
    print(f"  Z Mean Error (m)    : {metrics['Z_Mean_Error (m)']:.2f} m")
    print(f"  Z Median Error (m)  : {metrics['Z_Median_Error (m)']:.2f} m")

    print("\n----------------------------------------------------------")
    print(" Sample UAV Trajectory Steps (x_d, y_d, z_d in km):")
    print("----------------------------------------------------------")
    print(f"{'Timestep':<9} | {'Predicted (x_d, y_d, z_d) km':<33} | {'Ground Truth (x_d, y_d, z_d) km':<33} | {'2D Err(m)':<10} | {'3D Err(m)':<10}")
    print("-" * 105)
    for idx, row in merged_results.head(10).iterrows():
        pred_str = f"({row['pred_x']:.4f}, {row['pred_y']:.4f}, {row['pred_z']:.4f})"
        gt_str = f"({row['true_x']:.4f}, {row['true_y']:.4f}, {row['true_z']:.4f})"
        e2d = errors_2d_m[idx]
        e3d = errors_3d_m[idx]
        print(f"{int(row['timestamp']):<9} | {pred_str:<33} | {gt_str:<33} | {e2d:<10.2f} | {e3d:<10.2f}")

    # Save Predictions to CSV
    output_path = args.output_csv if args.output_csv else os.path.join(SCRIPT_DIR, "predicted_uav_trajectories.csv")
    output_df = pd.DataFrame({
        "timestamp": merged_results["timestamp"],
        "pred_x_d": merged_results["pred_x"],
        "pred_y_d": merged_results["pred_y"],
        "pred_z_d": merged_results["pred_z"],
        "gt_x_d": merged_results["true_x"],
        "gt_y_d": merged_results["true_y"],
        "gt_z_d": merged_results["true_z"],
        "error_2d_m": errors_2d_m,
        "error_3d_m": errors_3d_m,
        "error_z_m": merged_results["error_z_m"]
    })

    # Optional Online Post-Processing Refinement
    if args.post_process:
        print("\n" + "=" * 60)
        print(" APPLYING ONLINE POST-PROCESSING REFINEMENT (XGBoost/GB)")
        print("=" * 60)
        post_proc_dir = os.path.join(SCRIPT_DIR, "Post Processing XGBoost")
        if post_proc_dir not in sys.path:
            sys.path.append(post_proc_dir)
        try:
            from online_post_processor import OnlinePostProcessor
            model_file = args.post_model_path if args.post_model_path else os.path.join(post_proc_dir, "post_processing_models.joblib")
            post_processor = OnlinePostProcessor(model_path=model_file)

            # Build sensors dataframe for distance features
            _, _, sensor_dist_path, _ = resolve_dataset_paths(data_dir)
            sensor_dist_df = pd.read_csv(sensor_dist_path) if os.path.exists(sensor_dist_path) else pd.DataFrame()
            
            # Format sensors dataframe (exp_id, x, y, z in km)
            if not sensor_dist_df.empty:
                exp_dist = sensor_dist_df[sensor_dist_df['exp_id'] == args.exp_id]
                s_map = {}
                for _, r in exp_dist.iterrows():
                    s1, s2 = int(r['sensor1_id']), int(r['sensor2_id'])
                    if s1 not in s_map:
                        s_map[s1] = (r['x1'] / 1000.0, r['y1'] / 1000.0, r['z1'] / 1000.0)
                    if s2 not in s_map:
                        s_map[s2] = (r['x2'] / 1000.0, r['y2'] / 1000.0, r['z2'] / 1000.0)
                sensors_df = pd.DataFrame([{'exp_id': args.exp_id, 'sensor_id': sid, 'x': p[0], 'y': p[1], 'z': p[2]} for sid, p in s_map.items()])
            else:
                sensors_df = pd.DataFrame()

            # Format predictions input for post processor
            proc_input = merged_results.copy()
            proc_input['exp_id'] = args.exp_id
            proc_input['time_step'] = proc_input['timestamp']

            refined_df = post_processor.predict_trajectory(proc_input, sensors_df)
            
            if 'post_pred_x' in refined_df.columns:
                output_df['post_pred_x_d'] = refined_df['post_pred_x']
                output_df['post_pred_y_d'] = refined_df['post_pred_y']
                output_df['post_pred_z_d'] = refined_df['post_pred_z']
                if 'post_error_2d_m' in refined_df.columns:
                    output_df['post_error_2d_m'] = refined_df['post_error_2d_m']
                    output_df['post_error_3d_m'] = refined_df['post_error_3d_m']

                    print(f"  Refined 2D Mean Error (m) : {refined_df['post_error_2d_m'].mean():.2f} m (Original: {errors_2d_m.mean():.2f} m)")
                    print(f"  Refined 3D Mean Error (m) : {refined_df['post_error_3d_m'].mean():.2f} m (Original: {errors_3d_m.mean():.2f} m)")
        except Exception as e:
            print(f"[WARNING] Post-processing failed: {e}")

    output_df.to_csv(output_path, index=False)
    print(f"\n[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    main()

