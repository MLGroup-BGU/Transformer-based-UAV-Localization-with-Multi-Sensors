# Transformer-Based UAV Localization with Multi-Sensors

A deep learning framework for real-time 3D UAV (drone) localization using a passive network of ground-based RF sensors. The system processes multi-sensor RSSI (Received Signal Strength Indicator) power readings across antenna sectors, fusing spatial, temporal, and hardware characteristics through a Hybrid Spatiotemporal Transformer Encoder and an LSTM Decoder to predict 3D UAV coordinates $(x, y, z)$ in kilometers.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Prerequisites & Setup](#prerequisites--setup)
- [Quick Start](#quick-start)
- [Training the Model from Scratch](#training-the-model-from-scratch)
  - [Step 1: Download / Extract Raw Datasets](#step-1-download--extract-raw-datasets)
  - [Step 2: Run Data Preprocessing & SQL Normalization](#step-2-run-data-preprocessing--sql-normalization)
  - [Step 3: Build Binary Tensor Dataset Caches](#step-3-build-binary-tensor-dataset-caches)
  - [Step 4: Train the Transformer Encoder](#step-4-train-the-transformer-encoder)
  - [Step 5: Model Weights & Checkpoint Management](#step-5-model-weights--checkpoint-management)
- [Testing & Experimenting](#testing--experimenting)
  - [Step 1: Run Predictor Evaluation](#step-1-run-predictor-evaluation)
  - [Step 2: Linear Power Noise Injection Testing](#step-2-linear-power-noise-injection-testing)
  - [Step 3: Execute Batch Noise Matrix](#step-3-execute-batch-noise-matrix)
  - [Step 4: Plotting & Visualizing Results](#step-4-plotting--visualizing-results)
- [CLI Reference](#cli-reference)
- [Evaluation Metrics & Normalization Rules](#evaluation-metrics--normalization-rules)

---

## System Overview

The system addresses the challenge of 3D drone position estimation using ground RF sensor arrays. Each sensor detects power signals across up to 6 directional sectors at each timestamp. The pipeline:

1. **Ingests** raw multi-sensor detection logs from `data/`.
2. **Preprocesses** and densifies data into a Cartesian Product skeleton ($10 \text{ sensors} \times N \text{ timestamps}$).
3. **Encodes** dynamic RSSI sector power, static sensor parameters (gain, SINR threshold, sensitivity), 3D spatial coordinates, and temporal sequence indices using a **Hybrid Spatiotemporal Transformer Encoder**.
4. **Decodes** hidden states into continuous 3D spatial positions $(x, y, z)$ via an **LSTM Decoder**.
5. **Optionally refines** coordinates using an **XGBoost Post-Processor**.
6. **Compares** deep learning predictors against a baseline **RSSI Path-Loss Multilateration** solver across varying noise levels.

---

## Architecture

```
Sensor Data (10 sensors x T timestamps)
       |
       v
+---------------------------------------------+
|         HybridSpatiotemporalPredictor        |  (src/models/hybrid_encoder.py)
|                                             |
|  +------------------+                       |
|  | SparseDynamicEnc |  <- sector_1..6 (RSSI)|
|  +------------------+                       |
|  +------------------+                       |
|  | StaticContextEnc |  <- gain, SINR, sens. |
|  +------------------+                       |
|  +------------------+                       |
|  | SpatialPosEnc    |  <- sensor (x, y, z)  |
|  +------------------+       (Dual Basis)    |
|  +------------------+                       |
|  | TemporalPosEnc   |  <- time step index   |
|  +------------------+       (1D Sinusoidal) |
|          |                                  |
|  Fusion Layer -> LayerNorm                  |
|          |                                  |
|  Autoregressive Transformer Decoder         |
|          |  (causal mask, n=3 layers)       |
+----------+----------------------------------+
           | Encoder hidden states (embedding)
           v
+------------------------------+
|         LSTMDecoder          |  (src/models/lstm_decoder.py)
|  Input Proj -> LSTM -> FC    |
|  -> (x_d, y_d, z_d) in km   |
+------------------------------+
           |
           v (Optional)
+------------------------------+
|    XGBoost Post-Processor    |  (src/postprocessing/online_post_processor.py)
|   Refines 3D coordinates     |
|   using temporal context &   |
|   sensor geometry features   |
+------------------------------+
```

---

## Repository Layout

```
Transformer-based-UAV-Localization-with-Multi-Sensors/
│
├── .gitignore                      # Git ignore rules (checkpoints, cache, outputs)
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python package dependencies
├── main.py                         # Root entrypoint CLI wrapper
│
├── data/                           # Data directory (Raw CSVs & Cache)
│   ├── .gitkeep
│   ├── data_preprocessing.py       # SQL normalization & preprocessing script
│   └── download_drive_data.py      # Google Drive dataset & checkpoint downloader
│
├── models/                         # Model weights directory [Ignored by Git except .gitkeep]
│   ├── .gitkeep
│   ├── encoder/                    # Transformer Encoder checkpoints (.pth)
│   ├── decoder/                    # LSTM Decoder checkpoints (.pth)
│   └── xgboost/                    # XGBoost post-processor model (.joblib)
│
├── src/                            # Central Reusable Python Package
│   ├── __init__.py
│   ├── models/                     # Model architectures
│   │   ├── __init__.py
│   │   ├── positional_encoding.py  # 1D Sinusoidal, 3D RFF, Dual Basis encodings
│   │   ├── hybrid_encoder.py       # Hybrid Spatiotemporal Transformer Encoder
│   │   └── lstm_decoder.py         # LSTM Trajectory Decoder
│   │
│   ├── predictors/                 # High-level UAV location predictor classes
│   │   ├── __init__.py
│   │   ├── base_predictor.py       # Predictor interface & metric evaluation
│   │   ├── rssi_wcl_predictor.py   # RSSI weighted centroid localization predictor
│   │   ├── rssi_trilateration_predictor.py # RSSI path-loss multilateration predictor
│   │   └── encoder_decoder_predictor.py  # Transformer + LSTM predictor
│   │
│   ├── postprocessing/             # Post-processing refiners
│   │   ├── __init__.py
│   │   └── online_post_processor.py# Online XGBoost feature extractor & refiner
│   │
│   └── utils/                      # Plotting and visualization tools
│       ├── __init__.py
│       └── plot_results.py         # Summary aggregator & figure generator
│
├── scripts/                        # Execution Scripts & CLI Entrypoints
│   ├── main.py                     # Main evaluation CLI script
│   ├── train_hybrid.py             # Transformer Encoder training script
│   ├── build_dataset_cache.py      # Binary tensor cache builder
│   └── rssi_exp.bat                # Windows batch noise experiment runner
│
├── docs/                           # Documentation & Specs
│   └── data_preprocessing_pipeline.md # Preprocessing specification & schema
│
└── outputs/                        # Default folder for predictions & plots [Ignored by Git]
    └── .gitkeep
```

---

## Prerequisites & Setup

### Environment Dependencies
Install required Python libraries:

```bash
pip install -r requirements.txt
```

Core packages include:
- **Python**: 3.8+
- **PyTorch**: 1.10+ (CUDA recommended for training and fast inference)
- **Pandas / NumPy / Scikit-Learn / XGBoost**: Data handling & post-processing
- **Matplotlib / Seaborn**: Result plotting and CDF visualizations
- **gdown**: Automated dataset downloading from Google Drive

---

## Quick Start

```bash
# 1. Download datasets and pre-trained weights from Google Drive
python data/download_drive_data.py

# 2. Run predictor evaluation on 10 test experiments
python main.py --max_exps 10

# 3. Aggregate results and plot performance figures
python src/utils/plot_results.py --base_dir outputs
```

---

## Training the Model from Scratch

Follow these step-by-step instructions to train the model pipeline end-to-end starting from raw sensor logs.

### Step 1: Download / Extract Raw Datasets
Fetch the raw multi-sensor detection datasets:

```bash
python data/download_drive_data.py
```
This places the raw sensor detection logs, sensor hardware profiles, distance matrices, and ground truth trajectory CSVs inside `data/train_data_10_sensors_zsectors_ptx_model_preprocessing/` and `data/test_data_10_sensors_zsectors_ptx_model_preprocessing/`.

### Step 2: Run Data Preprocessing & SQL Normalization
Execute SQLite in-memory normalization to merge sensor positions, hardware gains/sensitivities, sector detections, and ground truth coordinates into normalized tables (`normalized_data.csv`):

```bash
python data/data_preprocessing.py
```
What this does:
- Scales spatial coordinates $(x, y, z)$ from meters to kilometers ($\div 1000.0$).
- Normalizes SINR threshold ($\div 10.0$) and sensitivity ($\div 90.0$).
- Densifies asynchronous sensor events into a structured $10 \text{ sensors} \times T \text{ timestamps}$ grid.

### Step 3: Build Binary Tensor Dataset Caches
Convert preprocessed CSV files into contiguous, memory-mapped PyTorch tensor blocks (`subcache_*.pt`) for fast GPU streaming:

```bash
python scripts/build_dataset_cache.py
```
Outputs are written to:
- `data/cache/train/subcache_*.pt`
- `data/cache/test/subcache_*.pt`

### Step 4: Train the Transformer Encoder
Run the training script for the `HybridSpatiotemporalPredictor`:

```bash
python scripts/train_hybrid.py
```

**Training Configuration**:
- **Sequence Window Length (`seq_len`)**: 40 timestamps
- **Batch Size**: 512
- **Embedding Dim (`d_model`)**: 128
- **Transformer Heads (`num_heads`)**: 8
- **Layers (`num_layers`)**: 3
- **Learning Rate**: $1 \times 10^{-4}$ with Adam optimizer and automatic mixed precision (AMP).

### Step 5: Model Weights & Checkpoint Management
Trained checkpoint weights are automatically saved per epoch to `models/encoder/`:
- `models/encoder/hybrid_predictor_10sensors_1_epochs.pth`
- `models/encoder/hybrid_predictor_10sensors_10_epochs.pth`

The predictor automatically detects and uses the latest `.pth` checkpoint file available in `models/encoder/` or `models/`.

---

## Testing & Experimenting

### Step 1: Run Predictor Evaluation
To evaluate predictors on raw test set experiments:

```bash
# Evaluate all experiments in the test set
python main.py

# Evaluate a subset (e.g., first 10 experiments)
python main.py --max_exps 10
```

### Step 2: Linear Power Noise Injection Testing
Test model robustness against measurement noise by injecting Gaussian normal noise in the **linear power domain** ($P_{\text{linear}} = 10^{P_{\text{dBm}}/10}$):

```bash
# Test with 1% linear power noise std dev
python main.py --power_noise_std 0.01 --output_dir outputs/results_rssi_01

# Test with 5% linear power noise std dev
python main.py --power_noise_std 0.05 --output_dir outputs/results_rssi_05

# Test with 10% linear power noise std dev in absolute mode (mW)
python main.py --power_noise_std 0.10 --noise_mode absolute --output_dir outputs/results_rssi_10
```

### Step 3: Execute Batch Noise Matrix
Run evaluation across 10 noise levels ($1\%$ to $10\%$) using the Windows batch script:

```cmd
scripts/rssi_exp.bat
```
This generates evaluation summaries and detailed timestamp predictions inside `outputs/results_rssi_01/` through `outputs/results_rssi_10/`.

### Step 4: Plotting & Visualizing Results
Aggregate evaluation CSVs across noise levels and produce visualization plots:

```bash
python src/utils/plot_results.py --base_dir outputs
```

**Generated Plots** (saved in `outputs/plots/`):
1. `01_noise_vs_3d_error_metrics.png` — 3D Mean, Median, RMSE, and P95 errors vs noise percentage.
2. `02_noise_vs_2d_and_z_error.png` — 2D Horizontal (XY) vs Z-Axis (Altitude) mean errors.
3. `03_noise_vs_mse_loss.png` — PyTorch MSE Loss ($\text{km}^2$) across noise levels (log scale).
4. `04_error_cdfs_across_noise.png` — Cumulative Distribution Function (CDF) curves of 3D spatial error.
5. `05_trajectory_3d_and_2d_comparison.png` — 2D XY plane & 3D flight trajectory plots comparing predictions to Ground Truth.
6. `06_error_over_time.png` — Instantaneous 3D localization error over timestamp steps.
7. `07_latency_and_throughput.png` — Bar chart comparing inference latency (ms/step) and throughput (steps/sec).

---

## CLI Reference

### Main Evaluation Script (`main.py` / `scripts/main.py`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--max_exps` | `int` | `None` (all) | Maximum number of test set experiments to evaluate |
| `--power_noise_std` | `float` | `0.0` | Std dev of Gaussian noise added in linear power domain (e.g. `0.05` = 5%) |
| `--ptx_noise_std` | `float` | `0.0` | Std dev of Gaussian noise added to drone transmit power ($P_{\text{tx}}$) |
| `--noise_mode` | `str` | `relative` | Noise injection mode: `relative` (percentage multiplier) or `absolute` (mW) |
| `--seed` | `int` | `42` | Random seed for reproducible noise generation |
| `--output_dir` | `str` | `outputs/` | Destination folder for summary & detailed prediction CSV files |

### Dataset Cache Builder (`scripts/build_dataset_cache.py`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `seq_len` | `int` | `40` | Sliding window sequence length |
| `overwrite` | `bool` | `False` | Whether to overwrite existing `.pt` cache blocks |

### Result Plotter (`src/utils/plot_results.py`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--base_dir` | `str` | `outputs/` | Base directory containing `results_rssi_*` or `outputs/` folders |
| `--output_dir` | `str` | `outputs/plots/` | Directory to save generated plot PNG figures |
| `--show` | `flag` | `False` | Display plots interactively in a window |

---

## Evaluation Metrics & Normalization Rules

### Evaluated Error Metrics

| Metric | Unit | Definition / Math Formula |
| :--- | :--- | :--- |
| **3D Mean Error** | meters (m) | Average 3D Euclidean distance: $\frac{1}{N} \sum \sqrt{(x - \hat{x})^2 + (y - \hat{y})^2 + (z - \hat{z})^2}$ |
| **3D Median Error** | meters (m) | 50th percentile (median) 3D spatial error |
| **3D RMSE** | meters (m) | Root Mean Squared Error: $\sqrt{\frac{1}{N} \sum E_{3D}^2}$ |
| **3D P95 Error** | meters (m) | 95th percentile 3D spatial error bound |
| **Trajectory Step Loss** | meters (m) | Step displacement continuity error between consecutive steps |
| **2D Mean Error** | meters (m) | Horizontal error in XY plane |
| **Z-Axis Mean Error** | meters (m) | Altitude estimation error |
| **PyTorch MSELoss** | $\text{km}^2$ | Matches PyTorch `nn.MSELoss()` calculation in $\text{km}^2$ |
| **Inference Latency** | ms / step | Execution time per timestamp step |

### Feature Normalization Standards
According to `docs/data_preprocessing_pipeline.md`:

| Feature | Raw Unit | Pipeline Normalization |
| :--- | :--- | :--- |
| `sector_1_power` .. `sector_6_power` | dBm | Raw dBm values ($0.0$ for non-detected inactive sectors) |
| `sensor_x`, `sensor_y`, `sensor_z` | meters | Divided by $1000.0$ (converted to km) |
| `gain` | dB | Raw gain |
| `sinr_threshold` | dB | Divided by $10.0$ |
| `sensitivity` | dBm | Divided by $90.0$ |
| UAV Ground Truth `uav_x`, `uav_y`, `uav_z` | meters | Divided by $1000.0$ (converted to km) |
