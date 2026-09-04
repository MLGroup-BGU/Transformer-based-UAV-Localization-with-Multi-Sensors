"""
train_xgboost.py

Script to train the XGBoost Trajectory Post-Processor on a single experiment subfolder or CSV file (normalized_data.csv).
Applies the exact post-processing feature extraction logic from src/postprocessing/online_post_processor.py.
Features experiment-chunked processing to scale smoothly to 1,000+ experiment trajectories without memory pressure.
"""

import os
import sys
import glob
import gc
import argparse
import time
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import torch

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    GradientBoostingRegressor = None

try:
    import joblib
except ImportError:
    joblib = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.postprocessing.online_post_processor import OnlinePostProcessor
from src.models.hybrid_encoder import HybridSpatiotemporalPredictor
from src.models.lstm_decoder import LSTMDecoder


def load_encoder_decoder_models(device: torch.device) -> Tuple[Optional[HybridSpatiotemporalPredictor], Optional[LSTMDecoder]]:
    """
    Attempts to locate and load trained Transformer Encoder and LSTM Decoder models.
    """
    enc_dir = os.path.join(ROOT_DIR, "models", "encoder")
    dec_dir = os.path.join(ROOT_DIR, "models", "decoder")

    enc_ckpt = os.path.join(enc_dir, "hybrid_predictor_10sensors_16_epochs.pth")
    dec_ckpt = os.path.join(dec_dir, "decoder_10sensors_12_epochs.pth")

    if not os.path.exists(enc_ckpt):
        enc_files = sorted(glob.glob(os.path.join(enc_dir, "*.pth")))
        enc_ckpt = enc_files[-1] if enc_files else None

    if not os.path.exists(dec_ckpt):
        dec_files = sorted(glob.glob(os.path.join(dec_dir, "*.pth")))
        dec_ckpt = dec_files[-1] if dec_files else None

    if not enc_ckpt or not dec_ckpt or not os.path.exists(enc_ckpt) or not os.path.exists(dec_ckpt):
        print("[WARNING] Encoder or Decoder checkpoints not found. Falling back to synthetic noisy predictions.")
        return None, None

    print(f"[*] Loading Encoder model checkpoint: {os.path.basename(enc_ckpt)}")
    encoder = HybridSpatiotemporalPredictor(
        d_model=128, nhead=8, num_layers=3, dim_feedforward=512,
        dropout=0.1, dynamic_dim=6, static_dim=3, coords_dim=3
    )
    encoder.load_state_dict(torch.load(enc_ckpt, map_location=device))
    encoder.to(device)
    encoder.eval()

    print(f"[*] Loading Decoder model checkpoint: {os.path.basename(dec_ckpt)}")
    decoder = LSTMDecoder(
        input_size=128, hidden_size=256, num_layers=2, dropout=0.1, output_size=3
    )
    dec_state = torch.load(dec_ckpt, map_location=device)
    if isinstance(dec_state, dict) and "decoder_state_dict" in dec_state:
        dec_state = dec_state["decoder_state_dict"]
    decoder.load_state_dict(dec_state)
    decoder.to(device)
    decoder.eval()

    return encoder, decoder


def resolve_csv_source_file(specified_path: Optional[str] = None) -> str:
    """
    Resolves path to normalized_data.csv file or subcache block file.
    """
    if specified_path and os.path.exists(specified_path):
        return os.path.abspath(specified_path)

    search_paths = [
        os.path.join(ROOT_DIR, "data", "train_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_0", "normalized_data.csv"),
        os.path.join(ROOT_DIR, "data", "test_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_0", "normalized_data.csv"),
        os.path.join(ROOT_DIR, "Data", "train_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_0", "normalized_data.csv"),
    ]

    for p in search_paths:
        if os.path.exists(p):
            return os.path.abspath(p)

    norm_matches = glob.glob(os.path.join(ROOT_DIR, "data", "**", "normalized_data.csv"), recursive=True)
    if norm_matches:
        return os.path.abspath(norm_matches[0])

    raise FileNotFoundError("Could not find any valid normalized_data.csv file in data directory.")


def process_single_exp_group(
    exp_id: int,
    exp_group: pd.DataFrame,
    encoder: Optional[HybridSpatiotemporalPredictor],
    decoder: Optional[LSTMDecoder],
    device: torch.device,
    time_col: str,
    seq_len: int = 40,
    batch_size: int = 2048,
    stride: int = 50,
    max_windows: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """Processes a single experiment trajectory dataframe into model predictions."""
    exp_group = exp_group.sort_values(by=[time_col, 'sensor_id']).reset_index(drop=True)
    feat_vals = exp_group.values.astype(np.float32)

    N_rows = len(feat_vals)
    if N_rows < seq_len:
        return None

    # Vectorized sliding window index extraction
    idxs = np.arange(seq_len)[None, :] + np.arange(0, N_rows - seq_len + 1, stride)[:, None]
    if max_windows is not None and len(idxs) > max_windows:
        idxs = idxs[:max_windows]
    X_arr = feat_vals[idxs]
    N_w = len(X_arr)

    ts_arr = X_arr[:, -1, 2]
    true_xyz = X_arr[:, -1, 15:18]

    px_arr = np.zeros(N_w, dtype=np.float32)
    py_arr = np.zeros(N_w, dtype=np.float32)
    pz_arr = np.zeros(N_w, dtype=np.float32)

    if encoder is not None and decoder is not None:
        for b in range(0, N_w, batch_size):
            xb = torch.from_numpy(X_arr[b : b + batch_size]).to(device)
            dyn_feat = xb[:, :, 3:9]
            spat_coords = xb[:, :, 9:12]
            stat_feat = xb[:, :, 12:15]

            with torch.no_grad():
                enc_out = encoder.get_hidden_state(dyn_feat, stat_feat, spat_coords, layer_idx="embedding")
                pred_seq = decoder(enc_out)
                pred_xyz = pred_seq[:, -1, :].cpu().numpy()

            px_arr[b : b + batch_size] = pred_xyz[:, 0]
            py_arr[b : b + batch_size] = pred_xyz[:, 1]
            pz_arr[b : b + batch_size] = pred_xyz[:, 2]
    else:
        px_arr = true_xyz[:, 0] + np.random.normal(0, 0.04, size=N_w)
        py_arr = true_xyz[:, 1] + np.random.normal(0, 0.04, size=N_w)
        pz_arr = true_xyz[:, 2] + np.random.normal(0, 0.02, size=N_w)

    exp_df = pd.DataFrame({
        'exp_id': int(exp_id),
        'time_step': ts_arr,
        'pred_x': px_arr,
        'pred_y': py_arr,
        'pred_z': pz_arr,
        'true_x': true_xyz[:, 0],
        'true_y': true_xyz[:, 1],
        'true_z': true_xyz[:, 2],
    })
    return exp_df


def process_csv_in_chunks(
    csv_path: str,
    encoder: Optional[HybridSpatiotemporalPredictor],
    decoder: Optional[LSTMDecoder],
    device: torch.device,
    post_processor: OnlinePostProcessor,
    exp_chunk_size: int = 50,
    stride: int = 50,
    max_exps: Optional[int] = None,
    max_samples: Optional[int] = 10000,
    max_windows_per_exp: Optional[int] = 200,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Processes normalized_data.csv in experiment chunks (e.g. 50 experiments per chunk),
    running model inference and feature extraction per chunk to keep RAM footprint low and fast.
    """
    print(f"[*] Loading normalized dataset CSV: {os.path.basename(csv_path)}")
    t0 = time.time()
    df_raw = pd.read_csv(csv_path)
    print(f"  [+] CSV loaded into memory in {time.time() - t0:.2f}s ({len(df_raw):,} rows).")

    exp_col = 'experiment_id' if 'experiment_id' in df_raw.columns else 'exp_id'
    time_col = 'time_step' if 'time_step' in df_raw.columns else 'timestamp'

    # Extract ground sensor locations for sensors_df
    sensor_rows = df_raw[['sensor_id', 'sensor_x', 'sensor_y', 'sensor_z']].drop_duplicates(subset=['sensor_id']).copy()
    sensors_df = pd.DataFrame({
        'exp_id': 1,
        'sensor_id': sensor_rows['sensor_id'],
        'x': sensor_rows['sensor_x'],
        'y': sensor_rows['sensor_y'],
        'z': sensor_rows['sensor_z']
    }).reset_index(drop=True)

    all_exp_ids = df_raw[exp_col].unique()
    if max_exps is not None and max_exps > 0:
        all_exp_ids = all_exp_ids[:max_exps]
        print(f"  [+] Filtered dataset to first {len(all_exp_ids)} experiment trajectories.")

    # Split experiment IDs into chunks of exp_chunk_size
    chunk_list = [all_exp_ids[i : i + exp_chunk_size] for i in range(0, len(all_exp_ids), exp_chunk_size)]
    print(f"[*] Split {len(all_exp_ids)} experiments into {len(chunk_list)} processing chunks (chunk_size={exp_chunk_size}).\n")

    feature_dfs = []

    for c_idx, chunk_exps in enumerate(tqdm(chunk_list, desc="Processing Trajectory Chunks", dynamic_ncols=True)):
        chunk_df_raw = df_raw[df_raw[exp_col].isin(chunk_exps)]
        pred_dfs = []

        exp_groups = list(chunk_df_raw.groupby(exp_col))
        for exp_id, exp_group in tqdm(exp_groups, desc=f"  Chunk {c_idx+1} Inference", leave=False, dynamic_ncols=True):
            exp_pred_df = process_single_exp_group(
                exp_id=exp_id,
                exp_group=exp_group,
                encoder=encoder,
                decoder=decoder,
                device=device,
                time_col=time_col,
                seq_len=40,
                batch_size=2048,
                stride=stride,
                max_windows=max_windows_per_exp
            )
            if exp_pred_df is not None:
                pred_dfs.append(exp_pred_df)

        if not pred_dfs:
            continue

        raw_preds_df = pd.concat(pred_dfs, ignore_index=True)

        # Aggregate predictions per (exp_id, time_step)
        predictions_df_chunk = raw_preds_df.groupby(['exp_id', 'time_step'], as_index=False).agg({
            'pred_x': 'mean',
            'pred_y': 'mean',
            'pred_z': 'mean',
            'true_x': 'first',
            'true_y': 'first',
            'true_z': 'first'
        })

        # Feature extraction for current chunk
        feat_df_chunk = post_processor.extract_features(predictions_df_chunk, sensors_df)
        feature_dfs.append(feat_df_chunk)

        # Clean memory after each chunk
        del chunk_df_raw, pred_dfs, raw_preds_df, predictions_df_chunk, feat_df_chunk
        gc.collect()

    if not feature_dfs:
        raise ValueError("No valid features extracted from dataset.")

    full_feature_df = pd.concat(feature_dfs, ignore_index=True)

    if max_samples and len(full_feature_df) > max_samples:
        sub_indices = np.linspace(0, len(full_feature_df) - 1, max_samples, dtype=int)
        full_feature_df = full_feature_df.iloc[sub_indices].reset_index(drop=True)
        print(f"  [+] Subsampled total dataset to {len(full_feature_df):,} feature rows for fast model training.")
    else:
        print(f"  [+] Extracted total feature dataset with {len(full_feature_df):,} feature rows.")

    return full_feature_df


def train_xgboost_post_processor(
    source_path: str,
    save_path: str,
    context_window: int = 5,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    test_size: float = 0.2,
    random_seed: int = 42,
    stride: int = 50,
    exp_chunk_size: int = 50,
    max_exps: Optional[int] = None,
    max_samples: Optional[int] = 10000,
    max_windows_per_exp: Optional[int] = 200,
):
    print("==========================================================================")
    print("      XGBoost Post-Processor Training (Normalized Data CSV)              ")
    print("==========================================================================")
    print(f"Source Data Path: {source_path}")
    print(f"Save Model Path : {save_path}")
    print(f"Context Window  : {context_window}")
    print(f"Estimators      : {n_estimators}")
    print(f"Max Depth       : {max_depth}")
    print(f"Learning Rate   : {learning_rate}")
    print(f"Stride          : {stride}")
    print(f"Chunk Size      : {exp_chunk_size} exps / chunk")
    print(f"Max Experiments : {max_exps if max_exps else 'All'}")
    print(f"Max Traj Steps  : {max_samples if max_samples else 'All'}")
    print(f"Max Windows/Exp : {max_windows_per_exp if max_windows_per_exp else 'All'}")
    print("==========================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Step 1: Load trained primary model (Encoder + Decoder) if available
    encoder, decoder = load_encoder_decoder_models(device)

    # Step 2: Initialize post-processor
    post_processor = OnlinePostProcessor(model_path="__no_load__", context_window=context_window)

    # Step 3: Extract predictions and features in experiment chunks
    t_feat = time.time()
    feature_df = process_csv_in_chunks(
        csv_path=source_path,
        encoder=encoder,
        decoder=decoder,
        device=device,
        post_processor=post_processor,
        exp_chunk_size=exp_chunk_size,
        stride=stride,
        max_exps=max_exps,
        max_samples=max_samples,
        max_windows_per_exp=max_windows_per_exp,
    )
    print(f"[*] Feature processing completed in {time.time() - t_feat:.2f}s! Shape: {feature_df.shape}")

    excluded_cols = {
        'exp_id', 'time_step', 'orig_x', 'orig_y', 'orig_z',
        'true_x', 'true_y', 'true_z', 'post_pred_x', 'post_pred_y', 'post_pred_z'
    }
    feature_cols = [c for c in feature_df.columns if c not in excluded_cols]
    print(f"  [+] Extracted {len(feature_cols)} feature columns for regressor input.")

    X = feature_df[feature_cols].values

    # Residual Targets (Delta in KM)
    y_x = feature_df['true_x'].values - feature_df['orig_x'].values
    y_y = feature_df['true_y'].values - feature_df['orig_y'].values
    y_z = feature_df['true_z'].values - feature_df['orig_z'].values

    true_x_all = feature_df['true_x'].values
    true_y_all = feature_df['true_y'].values
    true_z_all = feature_df['true_z'].values

    # Train / Validation Split by whole trajectories (or contiguous split)
    unique_exps = feature_df['exp_id'].unique()
    np.random.seed(random_seed)

    if len(unique_exps) > 1:
        np.random.shuffle(unique_exps)
        split_e = int(len(unique_exps) * (1.0 - test_size))
        train_exps, val_exps = unique_exps[:split_e], unique_exps[split_e:]
        train_mask = feature_df['exp_id'].isin(train_exps)
        val_mask = feature_df['exp_id'].isin(val_exps)
    else:
        n_samples = len(feature_df)
        split_i = int(n_samples * (1.0 - test_size))
        train_mask = np.zeros(n_samples, dtype=bool)
        val_mask = np.zeros(n_samples, dtype=bool)
        train_mask[:split_i] = True
        val_mask[split_i:] = True

    X_train, X_val = X[train_mask], X[val_mask]
    y_x_train, y_x_val = y_x[train_mask], y_x[val_mask]
    y_y_train, y_y_val = y_y[train_mask], y_y[val_mask]
    y_z_train, y_z_val = y_z[train_mask], y_z[val_mask]

    # Step 4: Train Regressors for X, Y, and Z residual targets
    print("\n[*] Training XGBoost Regressors for residual coordinate refinement...")
    if xgb is not None:
        print("  - Using XGBoost (xgb.XGBRegressor)")
        model_x = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, random_state=random_seed, n_jobs=-1
        )
        model_y = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, random_state=random_seed, n_jobs=-1
        )
        model_z = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, random_state=random_seed, n_jobs=-1
        )
    elif GradientBoostingRegressor is not None:
        print("  - [Fallback] XGBoost not installed. Using sklearn.ensemble.GradientBoostingRegressor")
        model_x = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_seed
        )
        model_y = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_seed
        )
        model_z = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            random_state=random_seed
        )
    else:
        raise ImportError("Neither xgboost nor scikit-learn is available for regressor training.")

    start_train_t = time.time()
    model_x.fit(X_train, y_x_train)
    model_y.fit(X_train, y_y_train)
    model_z.fit(X_train, y_z_train)
    train_duration = time.time() - start_train_t
    print(f"  [+] Regressor training completed in {train_duration:.2f} seconds.")

    # Step 5: Evaluate predictions on validation split
    val_orig_x = feature_df['orig_x'].values[val_mask]
    val_orig_y = feature_df['orig_y'].values[val_mask]
    val_orig_z = feature_df['orig_z'].values[val_mask]

    val_true_x = true_x_all[val_mask]
    val_true_y = true_y_all[val_mask]
    val_true_z = true_z_all[val_mask]

    # Clip residual corrections to max 150m (0.15 km)
    max_delta_km = 0.15
    pred_dx_val = np.clip(model_x.predict(X_val), -max_delta_km, max_delta_km)
    pred_dy_val = np.clip(model_y.predict(X_val), -max_delta_km, max_delta_km)
    pred_dz_val = np.clip(model_z.predict(X_val), -max_delta_km, max_delta_km)

    post_x_val = val_orig_x + pred_dx_val
    post_y_val = val_orig_y + pred_dy_val
    post_z_val = val_orig_z + pred_dz_val

    # Errors before post-processing (KM -> meters (* 1000.0))
    orig_diff_3d = np.column_stack([val_orig_x - val_true_x, val_orig_y - val_true_y, val_orig_z - val_true_z])
    orig_diff_2d = np.column_stack([val_orig_x - val_true_x, val_orig_y - val_true_y])
    orig_err_3d_m = np.linalg.norm(orig_diff_3d, axis=1) * 1000.0
    orig_err_2d_m = np.linalg.norm(orig_diff_2d, axis=1) * 1000.0

    # Errors after XGBoost post-processing (KM -> meters (* 1000.0))
    post_diff_3d = np.column_stack([post_x_val - val_true_x, post_y_val - val_true_y, post_z_val - val_true_z])
    post_diff_2d = np.column_stack([post_x_val - val_true_x, post_y_val - val_true_y])
    post_err_3d_m = np.linalg.norm(post_diff_3d, axis=1) * 1000.0
    post_err_2d_m = np.linalg.norm(post_diff_2d, axis=1) * 1000.0

    print("\n==========================================================================")
    print("                     Validation Results Summary                           ")
    print("==========================================================================")
    print(f"Baseline (Raw Model) 3D Mean Error : {np.mean(orig_err_3d_m):.2f} m")
    print(f"Baseline (Raw Model) 2D Mean Error : {np.mean(orig_err_2d_m):.2f} m")
    print(f"XGBoost Post-Processed 3D Mean Error: {np.mean(post_err_3d_m):.2f} m")
    print(f"XGBoost Post-Processed 2D Mean Error: {np.mean(post_err_2d_m):.2f} m")
    diff_3d = np.mean(orig_err_3d_m) - np.mean(post_err_3d_m)
    pct_imp = (diff_3d / np.mean(orig_err_3d_m)) * 100.0 if np.mean(orig_err_3d_m) > 0 else 0.0
    print(f"--> Improvement in 3D Error        : {diff_3d:.2f} m ({pct_imp:+.2f}%)")
    print("==========================================================================\n")

    # Step 6: Save model checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    post_processor.save(
        save_path=save_path,
        model_x=model_x,
        model_y=model_y,
        model_z=model_z,
        feature_cols=feature_cols
    )
    print(f"[OK] Trained XGBoost post-processor model saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Post-Processor on normalized_data.csv.")
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT_DIR, "data", "train_data_10_sensors_zsectors_ptx_model_preprocessing", "5K_0", "normalized_data.csv"), help="Path to normalized_data.csv file")
    parser.add_argument("--save_path", type=str, default=os.path.join(ROOT_DIR, "models", "xgboost", "post_processing_models.joblib"), help="Output joblib file path")
    parser.add_argument("--context_window", type=int, default=5, help="Context window length for temporal features")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of gradient boosted trees")
    parser.add_argument("--max_depth", type=int, default=4, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Boosting learning rate")
    parser.add_argument("--test_size", type=float, default=0.2, help="Validation set split ratio")
    parser.add_argument("--stride", type=int, default=50, help="Sequence window extraction stride (default: 50)")
    parser.add_argument("--exp_chunk_size", type=int, default=50, help="Experiments per processing chunk (default: 50)")
    parser.add_argument("--max_exps", type=int, default=50, help="Max experiment trajectories to load & process (e.g. 100 or 1000)")
    parser.add_argument("--max_samples", type=int, default=10000, help="Max trajectory steps for feature extraction (default: 10000)")
    parser.add_argument("--max_windows", type=int, default=200, help="Max sliding windows per experiment for model inference (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    resolved_csv = resolve_csv_source_file(args.data_path)

    train_xgboost_post_processor(
        source_path=resolved_csv,
        save_path=args.save_path,
        context_window=args.context_window,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        random_seed=args.seed,
        stride=args.stride,
        exp_chunk_size=args.exp_chunk_size,
        max_exps=args.max_exps,
        max_samples=args.max_samples,
        max_windows_per_exp=args.max_windows,
    )


if __name__ == "__main__":
    main()
