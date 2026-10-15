"""
finetune_decoder.py

Fine-tunes the pre-trained LSTM Decoder on the Mountains dataset (Data/MountainsData).

  * Encoder : HybridSpatiotemporalPredictor, best epoch (16) checkpoint - frozen, eval mode.
  * Decoder : LSTMDecoder initialised from the best original decoder checkpoint (epoch 12, lowest val loss)
              and fine-tuned with a lower learning rate.
  * Data    : non-anomaly Mountains folders, train_data_..._mountains_transformer/5K_0..5K_39 for training and
              test_data_..._mountains_transformer/5K_0..5K_4 for validation (5K_anomalies is skipped),
              preprocessed by preprocess_mountains.py into the same feature layout as normalized_data.csv.
              Windows / targets are identical to scripts/train/train_lstm.py: 40-row sliding windows,
              MSE on the (x, y, z) prediction at the last timestep.

All outputs (cache, checkpoints, logs, plots) are written inside the scripts/finetune folder.

Usage:
    python scripts/finetune/finetune_decoder.py
    python scripts/finetune/finetune_decoder.py --epochs 5 --lr 5e-6 --train_stride 4
"""

import sys
import os
import re
import glob
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for p in (ROOT_DIR, SCRIPT_DIR):
    if p not in sys.path:
        sys.path.append(p)

from src.models.hybrid_encoder import HybridSpatiotemporalPredictor
from src.models.lstm_decoder import LSTMDecoder
from preprocess_mountains import preprocess_all, RAW_DIR, CACHE_DIR

DEFAULT_ENCODER_CKPT = os.path.join(ROOT_DIR, "models", "encoder", "hybrid_predictor_10sensors_16_epochs.pth")
DEFAULT_DECODER_CKPT = os.path.join(ROOT_DIR, "models", "decoder", "decoder_10sensors_12_epochs.pth")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "finetune_history.csv")


class Tee:
    """Writes stdout to both the console and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class ExperimentWindows:
    """
    Holds one preprocessed file (all experiment rows) on the target device and cuts sliding windows
    on the fly - equivalent to build_dataset_cache.process_single_csv_to_cache:
        for i in range(0, len(features) - seq_len, step): x = features[i : i + seq_len]
    """
    def __init__(self, data_path, offsets_path, device, seq_len=40, stride=1, max_exps=None):
        offsets = np.load(offsets_path)
        if max_exps is not None:
            offsets = offsets[: max_exps + 1]
        data = np.load(data_path, mmap_mode="r")[: offsets[-1]]

        starts = [np.arange(s, e - seq_len, stride, dtype=np.int64)
                  for s, e in zip(offsets[:-1], offsets[1:]) if e - s > seq_len]
        self.starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self.data = torch.from_numpy(np.array(data)).to(device)
        self.window = torch.arange(seq_len, device=device)

    def __len__(self):
        return len(self.starts)

    def batches(self, batch_size, shuffle):
        order = torch.randperm(len(self.starts), device=self.starts.device) if shuffle else None
        for i in range(0, len(self.starts), batch_size):
            idx = order[i : i + batch_size] if shuffle else slice(i, i + batch_size)
            rows = self.starts[idx].unsqueeze(1) + self.window   # [B, seq_len]
            yield self.data[rows]                                 # [B, seq_len, 18]


def forward_batch(encoder, decoder, x):
    dynamic_features = x[:, :, 3:9]   # sector_1 to sector_6
    spatial_coords   = x[:, :, 9:12]  # sensor_x, sensor_y, sensor_z
    static_features  = x[:, :, 12:15] # gain, sinr_threshold, sensitivity

    with torch.no_grad():
        encoder_outputs = encoder.get_hidden_state(dynamic_features, static_features, spatial_coords, layer_idx='embedding')

    pred_xyz_seq = decoder(encoder_outputs)  # [batch, seq_len, 3]
    target_xyz = x[:, -1, 15:18]             # [batch, 3] target at last timestep
    pred_xyz = pred_xyz_seq[:, -1, :]        # [batch, 3] prediction at last timestep
    return pred_xyz, target_xyz


def count_windows(files, seq_len, stride, max_exps):
    total = 0
    for _, off_path in files:
        offsets = np.load(off_path)
        if max_exps is not None:
            offsets = offsets[: max_exps + 1]
        lengths = np.diff(offsets)
        total += int(np.sum(np.ceil(np.maximum(lengths - seq_len, 0) / stride)))
    return total


def evaluate(encoder, decoder, val_files, device, args, desc):
    """Returns (mean MSE loss, mean 3D error [m]) over all validation windows."""
    decoder.eval()
    criterion = nn.MSELoss()
    total_loss, total_err, n_batches, n_samples = 0.0, 0.0, 0, 0
    total = -(-count_windows(val_files, args.seq_len, args.val_stride, args.max_val_exps) // args.batch_size)

    pbar = tqdm(desc=desc, total=total, unit="batch", dynamic_ncols=True, file=sys.__stderr__)
    with torch.no_grad():
        for data_path, off_path in val_files:
            shard = ExperimentWindows(data_path, off_path, device, args.seq_len, args.val_stride, args.max_val_exps)
            for x in shard.batches(args.batch_size, shuffle=False):
                pred_xyz, target_xyz = forward_batch(encoder, decoder, x)
                total_loss += criterion(pred_xyz, target_xyz).item()
                total_err += torch.linalg.norm(pred_xyz - target_xyz, dim=1).sum().item() * 1000.0  # km -> m
                n_batches += 1
                n_samples += len(x)
                pbar.update(1)
            del shard
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    pbar.close()
    return total_loss / max(n_batches, 1), total_err / max(n_samples, 1)


def plot_losses(history, output_path):
    if plt is None:
        print("[Warning] matplotlib is not installed. Skipping plot generation.")
        return
    hist = pd.DataFrame(history)
    train = hist.dropna(subset=['train_loss'])
    plt.figure(figsize=(8, 5))
    plt.plot(train['epoch'], train['train_loss'], label="Train Loss")
    plt.plot(hist['epoch'], hist['val_loss'], label="Validation Loss")  # epoch 0 = before fine-tuning
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Decoder Fine-Tuning (Mountains) - Train & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def finetune_decoder(encoder, decoder, train_files, val_files, device, optimizer, args, start_epoch=0, history=None):
    """
    Fine-tunes for args.epochs epochs, numbered start_epoch+1 .. start_epoch+args.epochs.
    history: list of {'epoch', 'train_loss', 'val_loss', 'val_mean_3d_error_m'} rows from a previous run
             (when resuming); if None, the pre-trained decoder is evaluated first as epoch 0.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad = False
    decoder.to(device)
    criterion = nn.MSELoss()
    last_epoch = start_epoch + args.epochs

    if history is None:
        # Baseline: original decoder on Mountains validation data before any fine-tuning
        val_loss, val_err = evaluate(encoder, decoder, val_files, device, args, desc="Epoch 0 [Val - baseline]")
        history = [{'epoch': 0, 'train_loss': np.nan, 'val_loss': val_loss, 'val_mean_3d_error_m': val_err}]
        print(f"Epoch [0/{last_epoch}] (pre-trained decoder), Val Loss: {val_loss:.6f}, Val Mean 3D Error: {val_err:.1f} m")
    best_val = min(h['val_loss'] for h in history)

    train_batches = -(-count_windows(train_files, args.seq_len, args.train_stride, args.max_train_exps) // args.batch_size)

    for epoch in range(start_epoch + 1, last_epoch + 1):
        decoder.train()
        total_loss, num_batches = 0.0, 0

        current_train_files = list(train_files)
        random.shuffle(current_train_files)

        pbar = tqdm(desc=f"Epoch {epoch}/{last_epoch} [Train]", total=train_batches, unit="batch",
                    dynamic_ncols=True, file=sys.__stderr__)
        for data_path, off_path in current_train_files:
            shard = ExperimentWindows(data_path, off_path, device, args.seq_len, args.train_stride, args.max_train_exps)
            for x in shard.batches(args.batch_size, shuffle=True):
                pred_xyz, target_xyz = forward_batch(encoder, decoder, x)
                loss = criterion(pred_xyz, target_xyz)

                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.update(1)
            del shard
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        pbar.close()

        avg_train_loss = total_loss / max(num_batches, 1)

        val_loss, val_err = evaluate(encoder, decoder, val_files, device, args, desc=f"Epoch {epoch}/{last_epoch} [Val]")
        history.append({'epoch': epoch, 'train_loss': avg_train_loss, 'val_loss': val_loss, 'val_mean_3d_error_m': val_err})
        print(f"Epoch [{epoch}/{last_epoch}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, "
              f"Val Mean 3D Error: {val_err:.1f} m")

        ckpt = {
            'epoch': epoch,
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'val_loss': val_loss,
            'val_mean_3d_error_m': val_err,
            'encoder_ckpt': args.encoder_ckpt,
            'base_decoder_ckpt': args.decoder_ckpt,
        }
        save_path = os.path.join(CHECKPOINT_DIR, f"decoder_mountains_finetuned_{epoch}_epochs.pth")
        torch.save(ckpt, save_path)
        print(f"Saved model checkpoint: {save_path}")

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "decoder_mountains_finetuned_best.pth")
            torch.save(ckpt, best_path)
            print(f"[*] New best val loss {val_loss:.6f} -> {best_path}")

        # Keep the plot / history up to date after every epoch
        plot_losses(history, os.path.join(OUTPUT_DIR, "finetune_train_val_loss.png"))
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    return decoder


def latest_finetuned_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """Returns (path, epoch) of the highest-epoch decoder_mountains_finetuned_{N}_epochs.pth, or (None, 0)."""
    latest, latest_epoch = None, 0
    for path in glob.glob(os.path.join(checkpoint_dir, "decoder_mountains_finetuned_*_epochs.pth")):
        m = re.search(r"_(\d+)_epochs\.pth$", path)
        if m and int(m.group(1)) > latest_epoch:
            latest, latest_epoch = path, int(m.group(1))
    return latest, latest_epoch


def load_history(upto_epoch, ckpt):
    """Loss history up to (and including) upto_epoch; falls back to the checkpoint's own metrics."""
    if os.path.exists(HISTORY_PATH):
        hist = pd.read_csv(HISTORY_PATH)
        hist = hist[hist['epoch'] <= upto_epoch]
        if len(hist):
            return hist.to_dict('records')
    return [{'epoch': upto_epoch, 'train_loss': ckpt.get('loss', np.nan), 'val_loss': ckpt['val_loss'],
             'val_mean_3d_error_m': ckpt.get('val_mean_3d_error_m', np.nan)}]


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the LSTM decoder on the Mountains dataset.")
    parser.add_argument("--raw_dir", default=RAW_DIR)
    parser.add_argument("--cache_dir", default=CACHE_DIR)
    parser.add_argument("--encoder_ckpt", default=DEFAULT_ENCODER_CKPT)
    parser.add_argument("--decoder_ckpt", default=DEFAULT_DECODER_CKPT,
                        help="Pre-trained decoder to start from (default: epoch 12, best original val loss).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5, help="Fine-tuning LR (original training used 1e-4).")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max grad norm (<= 0 disables).")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=40)
    parser.add_argument("--train_stride", type=int, default=1,
                        help="Step between training windows (1 = same as original cache; larger = faster epochs).")
    parser.add_argument("--val_stride", type=int, default=1)
    parser.add_argument("--max_train_exps", type=int, default=None, help="Limit experiments used per train file.")
    parser.add_argument("--max_val_exps", type=int, default=None, help="Limit experiments used per val file.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(resume=False):
    args = parse_args()

    # Model hyper-parameters - must match the pre-trained encoder / decoder
    d_model = 128
    num_heads = 8
    num_layers = 3
    dim_ff = 512
    dropout = 0.1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n==================================================")
    print("  LSTM Decoder Fine-Tuning - Mountains Data        ")
    print("==================================================")
    print(f"[*] PyTorch Version : {torch.__version__}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[*] Active Device   : CUDA -> {gpu_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("[!] Active Device   : CPU")
    print(f"[*] Arguments       : {vars(args)}")
    print("==================================================\n")

    caches = preprocess_all(args.raw_dir, args.cache_dir)
    train_files, val_files = caches["train"], caches["val"]
    print(f"[*] Train files: {len(train_files)}, Val files: {len(val_files)}")
    if not train_files or not val_files:
        raise RuntimeError("Need at least one train and one validation Mountains file.")

    # Pre-trained Encoder (frozen)
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
    encoder.load_state_dict(torch.load(args.encoder_ckpt, map_location=device))
    print(f"[*] Loaded pre-trained encoder weights from: {args.encoder_ckpt}")

    # Pre-trained Decoder (fine-tuned)
    decoder = LSTMDecoder(
        input_size=d_model,
        hidden_size=d_model * 2,
        num_layers=2,
        dropout=dropout,
        output_size=3
    )
    # Move to device before building / loading the optimizer, so its restored state lands on the same device
    decoder.to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, history = 0, None

    if resume:
        # Continue from the latest fine-tuned checkpoint (decoder + optimizer state + loss history)
        resume_path, start_epoch = latest_finetuned_checkpoint()
        if resume_path is None:
            raise FileNotFoundError(f"No fine-tuned checkpoints found in {CHECKPOINT_DIR} - run finetune_decoder.py first.")
        ckpt = torch.load(resume_path, map_location=device)
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for group in optimizer.param_groups:  # --lr / --weight_decay of this run take effect
            group['lr'] = args.lr
            group['weight_decay'] = args.weight_decay
        args.decoder_ckpt = ckpt.get('base_decoder_ckpt', args.decoder_ckpt)
        history = load_history(start_epoch, ckpt)
        print(f"[*] Resuming fine-tuned decoder (epoch {start_epoch}, val loss {ckpt['val_loss']:.6f}) from: {resume_path}")
    else:
        decoder_ckpt = torch.load(args.decoder_ckpt, map_location=device)
        decoder.load_state_dict(decoder_ckpt['decoder_state_dict'])
        print(f"[*] Loaded pre-trained decoder (epoch {decoder_ckpt.get('epoch')}) from: {args.decoder_ckpt}")

    print("Starting LSTM Decoder Fine-Tuning...")
    finetune_decoder(encoder, decoder, train_files, val_files, device, optimizer, args,
                     start_epoch=start_epoch, history=history)
    print(f"[OK] Fine-tuning finished at: {pd.Timestamp.now()}")


def run(resume=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "finetune_decoder_log.txt")
    sys.stdout = Tee(sys.__stdout__, open(log_file, "a"))
    print(f"[*] Logging fine-tuning output to {log_file}")
    print(f"[*] Fine-tuning {'resumed' if resume else 'started'} at: {pd.Timestamp.now()}")
    main(resume=resume)


if __name__ == "__main__":
    run()
