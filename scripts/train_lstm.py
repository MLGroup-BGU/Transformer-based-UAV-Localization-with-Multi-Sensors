"""
train_lstm.py

Script to train the LSTM Decoder using hidden state representations from the trained Transformer Encoder.
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
from tqdm import tqdm

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models.hybrid_encoder import HybridSpatiotemporalPredictor
from src.models.lstm_decoder import LSTMDecoder
from scripts.build_dataset_cache import build_tensor_cache, _resolve_csv_paths


class SubcacheSequenceDataset(torch.utils.data.Dataset):
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


def plot_losses(train_losses, val_losses, output_path="lstm_train_val_loss.png"):
    if plt is None:
        print("[Warning] matplotlib is not installed. Skipping plot generation.")
        return
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Decoder Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_lstm_decoder(encoder, decoder, train_files, test_files, device, 
                       estimated_train_batches, estimated_test_batches, 
                       optimizer, batch_size=512, num_epochs=10, start_epoch=0, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, "models", "decoder")
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    decoder.to(device)
    encoder.to(device)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        encoder.eval()
        decoder.train()
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
                except Exception:
                    X_data = sub_dataset.X

                for i in range(0, N, batch_size):
                    batch_idx = perm[i : i + batch_size]
                    x = X_data[batch_idx].to(device, non_blocking=True)

                    dynamic_features = x[:, :, 3:9]   # sector_1 to sector_6
                    spatial_coords   = x[:, :, 9:12]  # sensor_x, sensor_y, sensor_z
                    static_features  = x[:, :, 12:15] # gain, sinr_threshold, sensitivity

                    with torch.no_grad():
                        encoder_outputs = encoder.get_hidden_state(dynamic_features, static_features, spatial_coords, layer_idx='embedding')

                    pred_xyz_seq = decoder(encoder_outputs)  # [batch, seq_len, 3]
                    
                    target_xyz = x[:, -1, 15:18]             # [batch, 3] target at last timestep
                    pred_xyz = pred_xyz_seq[:, -1, :]        # [batch, 3] prediction at last timestep

                    loss = criterion(pred_xyz, target_xyz)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
                    pbar.update(1)

                del sub_dataset, X_data
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[Warning] Error processing subcache {os.path.basename(subcache_path)}: {e}")

        pbar.close()

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)

        avg_val_loss = None
        if test_files:
            decoder.eval()
            encoder.eval()
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
                        except Exception:
                            X_data = sub_dataset.X

                        for i in range(0, N, batch_size):
                            batch_idx = slice(i, i + batch_size)
                            x = X_data[batch_idx].to(device, non_blocking=True)

                            dynamic_features = x[:, :, 3:9]
                            spatial_coords   = x[:, :, 9:12]
                            static_features  = x[:, :, 12:15]

                            encoder_outputs = encoder.get_hidden_state(dynamic_features, static_features, spatial_coords, layer_idx='embedding')

                            pred_xyz_seq = decoder(encoder_outputs)
                            target_xyz = x[:, -1, 15:18]
                            pred_xyz = pred_xyz_seq[:, -1, :]

                            loss = criterion(pred_xyz, target_xyz)

                            total_val_loss += loss.item()
                            val_num_batches += 1
                            pbar_val.update(1)

                        del sub_dataset, X_data
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"[Warning] Error validating subcache {os.path.basename(subcache_path)}: {e}")

            pbar_val.close()
            avg_val_loss = total_val_loss / val_num_batches if val_num_batches > 0 else 0.0
            val_losses.append(avg_val_loss)

            print(f"Epoch [{start_epoch+epoch+1}/{start_epoch+num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss if avg_val_loss is not None else 'N/A'}")

            save_path = os.path.join(save_dir, f"decoder_10sensors_{start_epoch+epoch+1}_epochs.pth")
            torch.save({
                'epoch': start_epoch + epoch + 1,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, save_path)
            print(f"Saved model checkpoint: {save_path}")

    plot_losses(train_losses, val_losses, output_path=os.path.join(ROOT_DIR, "outputs", "lstm_train_val_loss.png"))
    return decoder


def main():
    seq_len = 40
    batch_size = 512
    d_model = 128
    num_heads = 8
    num_layers = 3
    dim_ff = 512
    dropout = 0.1
    num_epochs = 15
    lr = 1e-4

    train_path = os.path.join(ROOT_DIR, "data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")
    if not os.path.exists(train_path):
        train_path = os.path.join(ROOT_DIR, "Data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")

    test_path = os.path.join(ROOT_DIR, "data", "test_data_10_sensors_zsectors_ptx_model_preprocessing")
    if not os.path.exists(test_path):
        test_path = os.path.join(ROOT_DIR, "Data", "test_data_10_sensors_zsectors_ptx_model_preprocessing")

    encoder_ckpt_path = os.path.join(ROOT_DIR, "models", "encoder", "hybrid_predictor_10sensors_16_epochs.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==================================================")
    print("  LSTM Decoder Training                            ")
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

    # Initialize pre-trained Encoder
    encoder = HybridSpatiotemporalPredictor(
        d_model=d_model, 
        nhead=num_heads, 
        num_layers=num_layers, 
        dim_feedforward=dim_ff, 
        dropout=dropout,
        dynamic_dim=6,
        static_dim=3,
        coords_dim=3
    )

    if os.path.exists(encoder_ckpt_path):
        encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
        print(f"[*] Loaded pre-trained encoder weights from: {encoder_ckpt_path}")
    else:
        print(f"[!] Warning: Pre-trained encoder weights NOT found at: {encoder_ckpt_path}")

    # Initialize LSTM Decoder from scratch
    decoder = LSTMDecoder(
        input_size=d_model,
        hidden_size=d_model * 2,
        num_layers=2,
        dropout=dropout,
        output_size=3
    )
    print("[*] Initialized LSTM Decoder (training from scratch)")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    save_dir = os.path.join(ROOT_DIR, "models", "decoder")
    os.makedirs(save_dir, exist_ok=True)

    print("Starting LSTM Decoder Training...")
    train_lstm_decoder(
        encoder=encoder,
        decoder=decoder,
        train_files=train_files,
        test_files=test_files,
        device=device,
        estimated_train_batches=estimated_train_batches,
        estimated_test_batches=estimated_test_batches,
        optimizer=optimizer,
        batch_size=batch_size,
        num_epochs=num_epochs,
        start_epoch=0,
        save_dir=save_dir
    )


if __name__ == "__main__":
    log_dir = os.path.join(ROOT_DIR, "models", "decoder")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_lstm_log.txt")
    sys.stdout = open(log_file, "a")
    print(f"[*] Logging training output to {log_file}")
    print(f"[*] Training started at: {pd.Timestamp.now()}")
    main()
