"""
build_dataset_cache.py

Data Pre-caching Utility Script.
Scans multi-source CSV subfolders, extracts sliding sequence windows (x, y) per subfolder,
stacks them into contiguous arrays, and saves per-subfolder binary PyTorch cache blocks (subcache_*.pt).
"""

import os
import glob
import gc
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def _resolve_csv_paths(source):
    """
    Converts a single string path, directory path, list of paths, or glob pattern
    into a sorted list of absolute file paths to CSV files.
    """
    if source is None:
        return []
    
    if isinstance(source, (list, tuple)):
        resolved = []
        for item in source:
            resolved.extend(_resolve_csv_paths(item))
        return sorted(list(set(resolved)))
    
    source_str = str(source)
    if os.path.isdir(source_str):
        norm_files = glob.glob(os.path.join(source_str, "**", "normalized_data.csv"), recursive=True)
        if norm_files:
            return sorted(norm_files)
        csv_files = glob.glob(os.path.join(source_str, "**", "*.csv"), recursive=True)
        return sorted(csv_files)
    
    if "*" in source_str or "?" in source_str:
        matched = glob.glob(source_str, recursive=True)
        return sorted([f for f in matched if os.path.isfile(f)])
    
    if os.path.isfile(source_str):
        return [os.path.abspath(source_str)]
    
    return [source_str]


def process_single_csv_to_cache(csv_path, output_pt_path, seq_len=40, step=1, chunk_size=100000, shuffle=True, overwrite=False):
    """
    Extracts sequence windows (x, y) from a single CSV file, stacks them into 
    two contiguous 3D NumPy float32 arrays, converts to PyTorch Tensors, and saves to output_pt_path.
    """
    if os.path.exists(output_pt_path) and not overwrite:
        print(f"[*] Sub-cache exists: {os.path.basename(output_pt_path)} (skipping)")
        return output_pt_path

    if not os.path.exists(csv_path):
        return None

    x_list = []
    y_list = []

    csv_iter = pd.read_csv(csv_path, chunksize=chunk_size)
    for chunk in csv_iter:
        id_col = 'exp_id' if 'exp_id' in chunk.columns else ('experiment_id' if 'experiment_id' in chunk.columns else None)
        if id_col is None:
            continue
        time_col = 'timestamp' if 'timestamp' in chunk.columns else ('time_step' if 'time_step' in chunk.columns else None)

        for exp_id, exp_group in chunk.groupby(id_col):
            if time_col:
                exp_group = exp_group.sort_values(time_col)

            features = exp_group.values.astype(np.float32)
            if np.isnan(features).any():
                continue

            if len(features) > seq_len:
                for i in range(0, len(features) - seq_len, step):
                    x_list.append(features[i : i + seq_len])
                    y_list.append(features[i + 1 : i + seq_len + 1])

        del chunk
        gc.collect()

    if not x_list:
        print(f"[Warning] No valid sequences found in {csv_path}")
        return None

    X_arr = np.stack(x_list, axis=0)
    Y_arr = np.stack(y_list, axis=0)
    del x_list, y_list
    gc.collect()

    X_tensor = torch.from_numpy(X_arr)
    Y_tensor = torch.from_numpy(Y_arr)
    del X_arr, Y_arr
    gc.collect()

    torch.save({'X': X_tensor, 'Y': Y_tensor}, output_pt_path)
    print(f"  [+] Saved sub-cache ({len(X_tensor):,} sequences) -> {os.path.basename(output_pt_path)}")
    return output_pt_path


def build_tensor_cache(csv_source, cache_dir, seq_len=40, step=1, overwrite=False):
    os.makedirs(cache_dir, exist_ok=True)
    csv_paths = _resolve_csv_paths(csv_source)
    if not csv_paths:
        raise ValueError(f"No CSV files found in source: {csv_source}")

    print(f"[*] Processing {len(csv_paths)} CSV file(s) into stacked tensor cache blocks in:\n    {cache_dir}")
    created_caches = []

    for p in tqdm(csv_paths, desc="Processing CSV subfolders"):
        parent_tag = os.path.basename(os.path.dirname(p))
        if not parent_tag or parent_tag in (".", ".."):
            parent_tag = os.path.splitext(os.path.basename(p))[0]
        
        out_name = f"subcache_{parent_tag}.pt"
        out_path = os.path.join(cache_dir, out_name)
        
        res = process_single_csv_to_cache(p, out_path, seq_len=seq_len, step=step, overwrite=overwrite)
        if res:
            created_caches.append(res)

    print(f"[OK] Total sub-cache blocks ready: {len(created_caches)} in {cache_dir}")
    return cache_dir


def main():
    train_source_dir = os.path.join(ROOT_DIR, "data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")
    if not os.path.exists(train_source_dir):
        train_source_dir = os.path.join(ROOT_DIR, "Data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")

    test_source_dir = os.path.join(ROOT_DIR, "data", "test_data_10_sensors_zsectors_ptx_model_preprocessing")
    if not os.path.exists(test_source_dir):
        test_source_dir = os.path.join(ROOT_DIR, "Data", "test_data_10_sensors_zsectors_ptx_model_preprocessing")

    train_cache_dir = os.path.join(ROOT_DIR, "data", "cache", "train")
    test_cache_dir  = os.path.join(ROOT_DIR, "data", "cache", "test")
    
    seq_len = 40
    
    print("==================================================")
    print("  Memory-Optimized Binary Tensor Cache Builder    ")
    print("==================================================")
    print(f"Sequence Length : {seq_len}")
    print(f"Train Source    : {train_source_dir}")
    print(f"Test Source     : {test_source_dir}")
    print("==================================================\n")
    
    print(">>> 1. Processing Training Dataset Subfolders...")
    build_tensor_cache(train_source_dir, train_cache_dir, seq_len=seq_len)
    
    print("\n>>> 2. Processing Test Dataset Subfolders...")
    build_tensor_cache(test_source_dir, test_cache_dir, seq_len=seq_len)
    
    print("\n[OK] All dataset tensor cache blocks built successfully!")


if __name__ == "__main__":
    main()
