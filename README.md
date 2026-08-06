# Transformer-Based UAV Localization with Multi-Sensors

A deep learning pipeline for real-time 3D UAV (drone) localization using a network of ground-based RF sensors. The system fuses multi-sensor RSSI (Received Signal Strength Indicator) measurements through a Hybrid Spatiotemporal Transformer Encoder and an LSTM Decoder to predict the drone's 3D position (x, y, z) in kilometers.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Data Format](#data-format)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Step 0 (Optional): Pre-compute Sensor Distances](#step-0-optional-pre-compute-sensor-distances)
  - [Step 1: Run Inference — Single Experiment](#step-1-run-inference--single-experiment)
  - [Step 2: Run Inference — All Experiments (Batch)](#step-2-run-inference--all-experiments-batch)
  - [Step 3 (Optional): Online XGBoost Post-Processing](#step-3-optional-online-xgboost-post-processing)
  - [Step 4 (Optional): Offline XGBoost Training and Evaluation](#step-4-optional-offline-xgboost-training-and-evaluation)
- [CLI Reference](#cli-reference)
- [Output Files](#output-files)
- [Running on an HPC Cluster (SLURM)](#running-on-an-hpc-cluster-slurm)
- [Normalization Rules](#normalization-rules)
- [Model Checkpoints](#model-checkpoints)

---

## System Overview

The system addresses the problem of **UAV 3D localization** using a passive network of 10 ground-based RF sensors. Each sensor detects RSSI power readings across up to 6 antenna sectors at each timestamp. The pipeline:

1. **Ingests** raw multi-sensor detection CSVs from the `Data/` folder.
2. **Preprocesses** the data into a dense Cartesian Product representation (10 sensors x N timestamps).
3. **Encodes** the fused spatiotemporal sensor signals with a Hybrid Transformer Encoder.
4. **Decodes** the encoder hidden states through an LSTM Decoder to produce 3D (x, y, z) UAV coordinates.
5. **Optionally refines** predictions using an XGBoost/Gradient Boosting post-processor.
6. **Reports** localization error metrics (2D/3D mean, median, P95).

---

## Architecture

`
Sensor Data (10 sensors x T timestamps)
       |
       v
+---------------------------------------------+
|         HybridSpatiotemporalPredictor        |  (HybridEncoder.py)
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
|         LSTMDecoder          |  (Linear_Layer_embDec/LSTM_decoder30seq.py)
|  Input Proj -> LSTM -> FC    |
|  -> (x_d, y_d, z_d) in km   |
+------------------------------+
           |
           v (Optional)
+------------------------------+
|    XGBoost Post-Processor    |  (Post Processing XGBoost/)
|   Refines 3D coordinates     |
|   using temporal context &   |
|   sensor geometry features   |
+------------------------------+
`

**Sliding Window Inference**: Inference runs on overlapping windows of `seq_len=30` consecutive sensor rows. Each window's last-timestep prediction is aggregated per timestamp (mean over overlapping windows) to produce the final trajectory.

---

## Repository Structure

`
Transformer-based-UAV-Localization-with-Multi-Sensors/
|-- Data/                                              # Raw experiment dataset (CSV)
|   |-- *_detections.csv                              # Per-sensor RSSI detections
|   |-- *_drone_trajectories.csv                      # Ground-truth UAV positions
|   |-- *_sensor_distances.csv                        # Sensor pair coordinates
|   +-- *_sensor_information.csv                      # Static sensor hardware properties
|
+-- Encoder-Decoder/                                  # Core inference pipeline
    |-- HybridEncoder.py                              # Transformer Encoder architecture
    |-- positional_encoding.py                        # 1D & 3D positional encoding modules
    |-- predict_uav_location.py                       # Main inference & evaluation script
    |-- calculate_active_sensors_distance.py          # Pre-computes UAV-to-sensor distances
    |-- hybrid_predictor_10sensors_1_epochs.pth       # Trained Encoder checkpoint
    |-- experiment_active_sensor_distances.json       # Precomputed distance filter map
    |-- experiments_summary.csv                       # Per-experiment evaluation summary
    |-- predicted_uav_trajectories_all.csv            # Full batch trajectory predictions
    |
    |-- Linear_Layer_embDec/                          # LSTM Decoder
    |   |-- LSTM_decoder30seq.py                     # LSTM Decoder architecture & training
    |   |-- decoder_10sensors_29_epochs_baseline_30seq.pth  # Trained Decoder checkpoint
    |   +-- run_decoder.sbatch                       # SLURM HPC training script
    |
    +-- Post Processing XGBoost/                      # Post-processing refinement
        |-- online_post_processor.py                  # Online inference-time XGBoost refiner
        +-- uav_location_predictor.py                 # Offline k-fold XGBoost training & eval
`

---

## Data Format

The system expects four raw CSV files in the `Data/` directory. Files are auto-detected by keyword matching (preferring filenames starting with `no_anomalies_`):

| File Suffix | Description |
|---|---|
| `*_detections.csv` | RSSI readings: `experiment_id`, `sensor_id`, `timestamp`, `sector_detected`, `total_power_dBm`, `drone_positions` |
| `*_sensor_information.csv` | Static hardware: `sensor_id`, `gain` (dB), `sinr_threshold` (dB), `sensitivity` (dBm) |
| `*_sensor_distances.csv` | Sensor geometry: `exp_id`, `sensor1_id`, `sensor2_id`, `x1`, `y1`, `z1`, `x2`, `y2`, `z2` (meters) |
| `*_drone_trajectories.csv` | Ground-truth UAV positions (used for evaluation only) |

---

## Requirements

Install the required Python packages:

```bash
pip install torch numpy pandas tqdm scikit-learn xgboost joblib
```

- **Python**: 3.8+
- **PyTorch**: 1.12+ (CUDA optional but recommended for large-batch evaluation)

---

## Quick Start

```bash
cd Encoder-Decoder

# Run inference on experiment ID 1
python predict_uav_location.py --exp_id 1

# Run batch inference on all experiments
python predict_uav_location.py --all

# Batch with distance filter (only experiments where sensors are within 3000 m of UAV)
python predict_uav_location.py --all --max_m 3000
```

---

## Usage

### Step 0 (Optional): Pre-compute Sensor Distances

Generate `experiment_active_sensor_distances.json` for use by the `--max_m` distance filter. Only needs to be run once per dataset.

```bash
cd Encoder-Decoder
python calculate_active_sensors_distance.py

# With custom paths
python calculate_active_sensors_distance.py \
    --data_dir ../Data \
    --max_exps 500 \
    --output_csv experiment_distances.csv
```

**Outputs:**
- `experiment_active_sensor_distances.json` — used by the main inference script
- `experiment_active_sensor_distances.csv` — human-readable per-experiment summary

---

### Step 1: Run Inference — Single Experiment

```bash
cd Encoder-Decoder

# Using raw CSVs (auto-detects Data/ folder relative to the script)
python predict_uav_location.py --exp_id 1

# Explicitly specify data directory
python predict_uav_location.py --exp_id 5 --data_dir /path/to/Data

# Using a pre-normalized CSV (skips raw preprocessing)
python predict_uav_location.py --csv_path /path/to/normalized.csv

# Custom output path
python predict_uav_location.py --exp_id 1 --output_csv results/exp1_predictions.csv

# Run on GPU
python predict_uav_location.py --exp_id 1 --device cuda
```

---

### Step 2: Run Inference — All Experiments (Batch)

```bash
cd Encoder-Decoder

# Evaluate all experiments
python predict_uav_location.py --all

# Limit to first 100 experiments
python predict_uav_location.py --all --max_exps 100

# Filter by distance threshold (requires Step 0)
python predict_uav_location.py --all --max_m 3000

# Custom sensor distance map path
python predict_uav_location.py --all --max_m 3000 \
    --sensor_dist_map /path/to/experiment_active_sensor_distances.json

# Save to a specific output file
python predict_uav_location.py --all --output_csv results/all_predictions.csv
```

**Outputs:**
- `predicted_uav_trajectories_all.csv` — timestep-level predictions for all experiments
- `experiments_summary.csv` — per-experiment 2D/3D error summary

---

### Step 3 (Optional): Online XGBoost Post-Processing

Apply a trained XGBoost/Gradient Boosting model at inference time to refine the Transformer+LSTM predictions:

```bash
cd Encoder-Decoder

python predict_uav_location.py --exp_id 1 \
    --post_process \
    --post_model_path "Post Processing XGBoost/post_processing_models.joblib"
```

The post-processor appends `post_pred_x_d`, `post_pred_y_d`, `post_pred_z_d`, `post_error_2d_m`, and `post_error_3d_m` columns to the output CSV and prints a comparison of refined vs. original error metrics.

---

### Step 4 (Optional): Offline XGBoost Training and Evaluation

To train and evaluate the XGBoost post-processing model using k-fold cross-validation on existing predictions:

```bash
cd "Encoder-Decoder/Post Processing XGBoost"
python uav_location_predictor.py
```

This script:
- Loads Transformer+LSTM predictions from a CSV
- Engineers temporal context, sensor geometry, and cross-axis correlation features
- Trains separate XGBoost models for x, y, z refinement
- Runs 5-fold trajectory-level cross-validation
- Outputs aggregated error metrics and feature importance rankings

---

## CLI Reference

```
predict_uav_location.py

  --data_dir PATH         Path to Data/ directory with raw CSVs (default: ../Data)
  --csv_path PATH         Path to a pre-normalized CSV (bypasses raw preprocessing)
  --raw                   Force raw CSV preprocessing even if csv_path is given
  --exp_id INT            Experiment ID for single-experiment mode (default: 1)
  --all, --all_exps       Run inference over ALL experiments in the Data/ folder
  --max_exps INT          Maximum number of experiments in batch mode
  --max_m FLOAT           Filter: only evaluate experiments where the mean
                          active-sensor-to-UAV distance <= this value (metres).
                          Requires experiment_active_sensor_distances.json.
  --sensor_dist_map PATH  Path to experiment_active_sensor_distances.json
  --window_size INT       Sliding window length in timestamps (default: 30)
  --device STR            Compute device: 'cuda' or 'cpu'
  --output_csv PATH       Path to save the output predictions CSV
  --post_process          Apply online XGBoost post-processing refinement
  --post_model_path PATH  Path to post_processing_models.joblib
```

```
calculate_active_sensors_distance.py

  --data_dir PATH         Path to Data/ directory (default: ../Data)
  --max_exps INT          Maximum number of experiments to process
  --output_csv PATH       Path to the detailed output CSV
```

---

## Output Files

| File | Description |
|---|---|
| `predicted_uav_trajectories.csv` | Single-experiment predictions: `timestamp`, `pred_x_d`, `pred_y_d`, `pred_z_d`, `gt_x_d`, `gt_y_d`, `gt_z_d`, `error_2d_m`, `error_3d_m`, `error_z_m` |
| `predicted_uav_trajectories_all.csv` | Batch predictions: same columns plus `exp_id` |
| `experiments_summary.csv` | Per-experiment: `exp_id`, `2D_Mean_Error (m)`, `3D_Mean_Error (m)`, `2D_Median_Error (m)`, `3D_P95_Error (m)`, average altitude, valid timestep count |

All coordinates are in **kilometers (km)**; all error metrics are in **meters (m)**.

---

## Running on an HPC Cluster (SLURM)

A SLURM batch script is provided for training the LSTM Decoder on an HPC cluster:

```bash
# From the Linear_Layer_embDec/ directory
sbatch run_decoder.sbatch
```

Before submitting, edit `run_decoder.sbatch`:
- `#SBATCH --partition` — set to your cluster's partition name
- `#SBATCH --gpus` — set GPU type/count
- `source activate my_env` — replace `my_env` with your conda environment name

---

## Normalization Rules

All preprocessing is aligned with `data_preprocessing_pipeline (1).md`:

| Feature | Preprocessing | Training Statistics |
|---|---|---|
| `gain` (dB) | Z-score normalization | mean = -90.005570, std = 5.775258 |
| `sinr_threshold` (dB) | Z-score normalization | mean = 9.999188, std = 2.888816 |
| `sensitivity` (dBm) | Z-score normalization | mean = -0.003132, std = 2.887262 |
| Sensor coords x, y, z | Divide by 1000 (m -> km) | — |
| UAV coords x_d, y_d, z_d | Divide by 1000 (m -> km) | — |
| `sector_1`..`sector_6` (RSSI) | Raw dBm, zero-padded for inactive | — |

Sequence structure: 10 sensors x 60 timesteps (Cartesian Product skeleton), sorted by `[exp_id, timestamp_number, sensor_id]`. Missing sensor/sector measurements are zero-padded.

---

## Model Checkpoints

| File | Description |
|---|---|
| `Encoder-Decoder/hybrid_predictor_10sensors_1_epochs.pth` | Trained `HybridSpatiotemporalPredictor`: d_model=128, nhead=8, num_layers=3, dim_feedforward=512 |
| `Encoder-Decoder/Linear_Layer_embDec/decoder_10sensors_29_epochs_baseline_30seq.pth` | Trained `LSTMDecoder`: input_size=128, hidden_size=256, num_layers=2, trained for 29 epochs on 30-step windows |

The inference script automatically resolves these paths relative to the script directory — no manual path configuration is required as long as the directory structure is preserved.
