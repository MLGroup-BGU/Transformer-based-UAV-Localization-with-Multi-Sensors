# Transformer-Based UAV Localization with Multi-Sensors

A deep learning pipeline for 3D UAV localization using a passive ground network of RF sensors. The system fuses multi-sensor RSSI readings through a **Hybrid Spatiotemporal Transformer Encoder** and an **LSTM Decoder** to predict continuous 3D UAV coordinates $(x, y, z)$.

---

## Architecture

The core pipeline is an **Encoder → Decoder** design:

```
Multi-Sensor RSSI Input  (10 sensors × T timestamps)
          │
          ▼
┌──────────────────────────────────────────┐
│   HybridSpatiotemporalPredictor          │
│                                          │
│  · SparseDynamicEncoder  ← sector RSSI  │
│  · StaticContextEncoder  ← gain, SINR   │
│  · SpatialPosEncoder     ← sensor (xyz) │
│  · TemporalPosEncoder    ← time index   │
│          │                               │
│   Fusion → LayerNorm                     │
│          │                               │
│   Autoregressive Transformer Decoder     │
│   (causal mask, 3 layers)                │
└──────────────────────────────────────────┘
          │  encoder hidden states
          ▼
┌──────────────────────────────────────────┐
│   LSTMDecoder                            │
│   InputProj → LSTM → FC                  │
│   → (x, y, z) in km                      │
└──────────────────────────────────────────┘
```

The encoder fuses dynamic RSSI readings per antenna sector (up to 6 sectors per sensor) with static hardware parameters (gain, SINR threshold, sensitivity) and both spatial and temporal positional encodings. The LSTM decoder converts the encoder's hidden states into trajectory waypoints.

---

## Setup

```bash
pip install -r requirements.txt
```

**Core dependencies**: Python 3.8+, PyTorch (CUDA recommended), Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib.

---

## Training

### 1. Download Raw Data
```bash
python data/download_drive_data.py
```
Places raw sensor CSVs in `data/train_data_10_sensors_zsectors_ptx_model_preprocessing/` and `data/test_data_10_sensors_zsectors_ptx_model_preprocessing/`.

### 2. Preprocess & Normalize
```bash
python data/data_preprocessing.py
```
Merges sensor positions, hardware parameters, and sector detections into a dense `10 sensors × T timestamps` grid. Spatial coordinates are scaled to km (÷1000), SINR threshold (÷10), sensitivity (÷90).

### 3. Build Tensor Cache
```bash
python scripts/build_dataset_cache.py
```
Converts preprocessed CSVs into memory-mapped PyTorch tensor blocks at `data/cache/{train,test}/subcache_*.pt`.

### 4. Train the Model
```bash
python scripts/train_hybrid.py
```

| Hyperparameter | Value |
|---|---|
| Sequence Window (`seq_len`) | 40 timestamps |
| Batch Size | 512 |
| Embedding Dim (`d_model`) | 128 |
| Transformer Heads | 8 |
| Encoder Layers | 3 |
| Optimizer | Adam, lr=1e-4, AMP |

Checkpoints are saved per epoch to `models/encoder/`. The predictor automatically loads the latest available `.pth` file.

### 5. Train the LSTM Decoder
```bash
python scripts/train_lstm.py
```

Requires the encoder checkpoint from step 4 at its best val loss (We used the 16 epochs checkpoint). The encoder weights are frozen during this phase — only the LSTM decoder is trained from scratch using the encoder's hidden state representations as input.

| Hyperparameter | Value |
|---|---|
| Sequence Window (`seq_len`) | 40 timestamps |
| Batch Size | 512 |
| LSTM Hidden Size | 256 (2× d_model) |
| LSTM Layers | 2 |
| Epochs | 15 |
| Optimizer | Adam, lr=1e-4 |

Decoder checkpoints are saved per epoch to `models/decoder/`. Training and validation loss curves are written to `outputs/lstm_train_val_loss.png`.

---

## Evaluation

### Run a quick evaluation
```bash
# Evaluate all test experiments
python main.py

# Evaluate a subset
python main.py --max_exps 10 --output_dir outputs/
```

### Noise robustness experiment
Noise is injected in the **linear power domain** ($P_{\text{linear}} = 10^{P_{\text{dBm}}/10}$) to simulate measurement uncertainty:

```bash
# Single noise level (5% relative std)
python main.py --power_noise_std 0.05 --output_dir outputs/results_rssi_05

# Full noise sweep (1%–10%) via batch script
scripts/rssi_exp.bat
```

### Plot results
```bash
python src/utils/plot_results.py --base_dir outputs
```

Generates plots under `outputs/plots/`: error vs. noise curves, CDF plots, trajectory comparisons, latency charts.

---

## Experimental Setting

To benchmark the encoder-decoder pipeline, it is compared side-by-side against two classical RSSI-based localization methods on the same test set (`5K_4`, ~5000 trajectory steps):

| Predictor | Method |
|---|---|
| **Encoder-Decoder** | Hybrid Transformer Encoder + LSTM Decoder (this work) |
| **RSSI WCL** | Weighted Centroid Localization — weights sensor positions by linear received power |
| **RSSI Trilateration** | Path-loss multilateration — estimates range from RSSI via log-distance model, solves 3D position |

Each predictor is evaluated on 3D Mean Error, RMSE, Median Error, P95 Error, 2D/Z errors, and inference latency. The noise sweep (1%–10% Gaussian linear noise on received power) tests robustness under measurement uncertainty — the encoder-decoder's learned feature space degrades far more gracefully than the math-based methods that depend directly on raw RSSI values.

All three predictors share the same `BaseUAVPredictor` interface and receive identical raw CSV inputs per experiment, ensuring a fair comparison.

---

## Dataset Overview

The dataset (`5K_4`) covers realistic UAV trajectories over a 5 km × 5 km area monitored by 10 ground RF sensors.

| Property | Distribution |
|---|---|
| **3D sensor-UAV distance** | Bell-shaped, peak ~4–5 km, range 0–8 km |
| **UAV altitude (Z)** | Roughly uniform, 0–5 km |
| **2D trajectory coverage** | Uniformly dense across the full 5 km × 5 km grid |
| **Active sensors / timestamp** | Strongly skewed toward 9–10 simultaneous active sensors |
| **UAV transmit power (P_tx)** | Near-uniform, 10–30 dBm, mean ≈ 19.9 dBm |
| **RSSI vs. distance** | Follows log-distance path-loss (fitted exponent n ≈ 1.83) with high variance |

![Dataset Analysis Dashboard](outputs/data_analysis/dataset_analysis_dashboard.png)

---

## CLI Reference

### `main.py` / `scripts/main.py`

| Flag | Default | Description |
|---|---|---|
| `--max_exps` | all | Max experiments to evaluate |
| `--power_noise_std` | 0.0 | Gaussian noise std on received power (linear, relative) |
| `--ptx_noise_std` | 0.0 | Gaussian noise std on transmit power (linear, relative) |
| `--noise_mode` | `relative` | `relative` (% multiplier) or `absolute` (mW) |
| `--seed` | 42 | Random seed for noise reproducibility |
| `--output_dir` | `outputs/` | Output folder for result CSVs |

### `src/utils/plot_results.py`

| Flag | Default | Description |
|---|---|---|
| `--base_dir` | `outputs/` | Root folder with `results_rssi_*` subdirectories |
| `--output_dir` | `outputs/plots/` | Destination for generated PNG figures |
| `--show` | False | Display plots interactively |

