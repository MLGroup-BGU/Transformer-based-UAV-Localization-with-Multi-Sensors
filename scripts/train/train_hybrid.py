"""
train_hybrid.py

Script to train the Transformer Encoder (HybridSpatiotemporalPredictor).
"""

import sys
import os
import glob
import gc
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

try:
    import comet_ml
except ImportError:
    comet_ml = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models.hybrid_encoder import HybridSpatiotemporalPredictor
from scripts.data.build_dataset_cache import build_tensor_cache, _resolve_csv_paths


class SubcacheSequenceDataset(Dataset):
    """
    Map-style PyTorch dataset loaded from a single subcache block file (subcache_*.pt).
    """
    def __init__(self, subcache_path):
        data = torch.load(subcache_path)
        self.X = data['X']
        self.Y = data['Y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_subcache_files(cache_dir, csv_source, seq_len=40):
    os.makedirs(cache_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(cache_dir, "subcache_*.pt")))
    if not files:
        print(f"[*] Sub-cache directory empty ({cache_dir}). Building sub-cache files...")
        build_tensor_cache(csv_source, cache_dir, seq_len=seq_len, overwrite=False)
        files = sorted(glob.glob(os.path.join(cache_dir, "subcache_*.pt")))
    return files


def plot_losses(train_losses, val_losses, output_path="hybrid_train_val_loss.png"):
    if plt is None:
        print("[Warning] matplotlib is not installed. Skipping plot generation.")
        return
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_model(model, train_files, test_files, device, estimated_train_batches, estimated_test_batches,
                batch_size=512, num_epochs=10, lr=1e-4, warmup_positions=10, start_epoch=0, run=None, use_amp=True,
                save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, "models", "encoder")
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == 'cuda'))

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        current_train_files = list(train_files)
        random.shuffle(current_train_files)

        pbar = tqdm(desc=f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} [Train]", 
                    total=estimated_train_batches, unit="batch", dynamic_ncols=True)  

        for subcache_path in current_train_files:
            try:
                sub_dataset = SubcacheSequenceDataset(subcache_path)
                N = len(sub_dataset.X)
                perm = torch.randperm(N)

                try:
                    X_data = sub_dataset.X.pin_memory()
                    Y_data = sub_dataset.Y.pin_memory()
                except Exception:
                    X_data = sub_dataset.X
                    Y_data = sub_dataset.Y

                for i in range(0, N, batch_size):
                    batch_idx = perm[i : i + batch_size]
                    x = X_data[batch_idx].to(device, non_blocking=True)
                    y = Y_data[batch_idx].to(device, non_blocking=True)

                    optimizer.zero_grad()
                    
                    dynamic_features = x[:, :, 3:9]   # sector_1 to sector_6
                    spatial_coords   = x[:, :, 9:12]  # x, y, z
                    static_features  = x[:, :, 12:15] # gain, sinr_threshold, sensitivity
                    
                    with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                        output = model(
                            dynamic_features=dynamic_features, 
                            static_features=static_features, 
                            spatial_coords=spatial_coords
                        )  
                        
                        loss = criterion(output[:, warmup_positions:, :], y[:, warmup_positions:, 3:9])  
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    pbar.update(1)

                del sub_dataset, X_data, Y_data
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[Warning] Error processing subcache {os.path.basename(subcache_path)}: {e}")

        pbar.close()

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        if run:
            run.log_metric("train/batch_loss", avg_train_loss)

        avg_val_loss = None
        if test_files:
            model.eval()
            total_val_loss = 0
            val_num_batches = 0

            pbar_val = tqdm(desc=f"Epoch {start_epoch+epoch+1}/{start_epoch+num_epochs} [Val]", 
                            total=estimated_test_batches, unit="batch", dynamic_ncols=True)

            with torch.no_grad():
                for subcache_path in test_files:
                    try:
                        sub_dataset = SubcacheSequenceDataset(subcache_path)
                        N = len(sub_dataset.X)

                        try:
                            X_data = sub_dataset.X.pin_memory()
                            Y_data = sub_dataset.Y.pin_memory()
                        except Exception:
                            X_data = sub_dataset.X
                            Y_data = sub_dataset.Y

                        for i in range(0, N, batch_size):
                            batch_idx = slice(i, i + batch_size)
                            x = X_data[batch_idx].to(device, non_blocking=True)
                            y = Y_data[batch_idx].to(device, non_blocking=True)
                            
                            dynamic_features = x[:, :, 3:9]
                            spatial_coords   = x[:, :, 9:12] 
                            static_features  = x[:, :, 12:15]
                            
                            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == 'cuda')):
                                output = model(
                                    dynamic_features=dynamic_features, 
                                    static_features=static_features, 
                                    spatial_coords=spatial_coords
                                )  
                                
                                loss = criterion(output[:, warmup_positions:, :], y[:, warmup_positions:, 3:9])

                            total_val_loss += loss.item()
                            val_num_batches += 1
                            pbar_val.update(1)

                        del sub_dataset, X_data, Y_data
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"[Warning] Error validating subcache {os.path.basename(subcache_path)}: {e}")

            pbar_val.close()
            avg_val_loss = total_val_loss / val_num_batches if val_num_batches > 0 else 0.0
            val_losses.append(avg_val_loss)
            if run:
                run.log_metric("val/batch_loss", avg_val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss if avg_val_loss else 'N/A'}")

            save_path = os.path.join(save_dir, f"hybrid_predictor_10sensors_{start_epoch+epoch+1}_epochs.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint: {save_path}")
    
    plot_losses(train_losses, val_losses, output_path=os.path.join(ROOT_DIR, "outputs", "hybrid_train_val_loss.png"))
    return model


def main():
    seq_len = 40
    batch_size = 512
    d_model = 128
    num_heads = 8
    num_layers = 3
    dim_ff = 512
    dropout = 0.1
    num_epochs = 5
    lr = 1e-4
    warmup_positions = 10

    train_path = os.path.join(ROOT_DIR, "data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")
    if not os.path.exists(train_path):
        train_path = os.path.join(ROOT_DIR, "Data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")

    test_path = os.path.join(ROOT_DIR, "data", "test_data_10_sensors_zsectors_ptx_model_preprocessing")
    if not os.path.exists(test_path):
        test_path = os.path.join(ROOT_DIR, "Data", "test_data_10_sensors_zsectors_ptx_model_preprocessing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==================================================")
    print("  Hybrid Spatiotemporal Predictor Training       ")
    print("==================================================")
    print(f"[*] PyTorch Version : {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] Active Device   : CUDA -> {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[!] Active Device   : CPU")
    print("==================================================\n")

    train_cache_dir = os.path.join(ROOT_DIR, "data", "cache", "train")
    test_cache_dir  = os.path.join(ROOT_DIR, "data", "cache", "test")

    train_files = get_subcache_files(train_cache_dir, train_path, seq_len=seq_len)
    test_files  = get_subcache_files(test_cache_dir, test_path, seq_len=seq_len)

    estimated_train_batches = max(1, len(train_files) * 4300)
    estimated_test_batches  = max(1, len(test_files) * 4300)

    model = HybridSpatiotemporalPredictor(
        d_model=d_model, 
        nhead=num_heads, 
        num_layers=num_layers, 
        dim_feedforward=dim_ff, 
        dropout=dropout,
        dynamic_dim=6,
        static_dim=3,
        coords_dim=3
    )
    #Temp: loading latest version
    latest_model_path = os.path.join(ROOT_DIR, "models", "encoder", "hybrid_predictor_10sensors_17_epochs.pth")
    if os.path.exists(latest_model_path):
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
        print(f"[*] Loaded model weights from {latest_model_path}")
    else:
        print(f"[!] No pre-trained model found at {latest_model_path}. Starting training from scratch.")
    print("Starting Hybrid Training...")
    train_model(model, train_files, test_files, device, 
                estimated_train_batches, estimated_test_batches,
                batch_size=batch_size, num_epochs=num_epochs, lr=lr, 
                warmup_positions=warmup_positions, start_epoch=17, 
                run=None, use_amp=True)


if __name__ == "__main__":
    sys.stdout = open(os.path.join(ROOT_DIR,  "models", "encoder", "train_hybrid_log.txt"), "a")
    print(f"[*] Logging training output to {os.path.join(ROOT_DIR,  'models', 'encoder', 'train_hybrid_log.txt')}")
    print(f"[*] Training started at: {pd.Timestamp.now()}")
    main()
