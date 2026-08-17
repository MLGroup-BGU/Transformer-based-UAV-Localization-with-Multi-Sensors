# System Instruction Manual: UAV Detection Preprocessing & Input Pipeline

This document defines the strict algorithmic rules, directory structures, and data-handling constraints for compiling the data preprocessing pipeline.

---

## 📂 Directory & File Structure Configuration

The pipeline utilizes the following paths and directory structure:

```python
INPUT_DIR = "10_sensors"
OUTPUT_DIR = "data"

# Raw inputs inside INPUT_DIR:
# Detections: *_detections.csv
# Sensor Distances: *_sensor_distances.csv
# Trajectories: *_drone_trajectories.csv
# Sensor Information: *_sensor_information.csv

# Step 1 Outputs (Intermediate Files):
CLEAN_DRONE_POS = "clean_drone_positions.csv"
TRAIN_RAW = "processed_detections_train.csv"
TEST_RAW = "processed_detections_test.csv"
DRONE_RAW = "processed_drone_data_all.csv"

# Step 2 Outputs (Final Files inside OUTPUT_DIR):
TRAIN_PROCESSED = "processed_train_normalized.csv"
TEST_PROCESSED = "processed_test_normalized.csv"
```

---

## 📌 Pipeline Overview

The preprocessing script transforms raw asynchronous sensor logs and drone trajectories into a synchronized, structured sequence of sensor readings.

```mermaid
graph TD
    classDef raw fill:#ffe3e3,stroke:#ff8080,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef clean fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef tensor fill:#fff8e1,stroke:#ffc107,stroke-width:2px;

    subgraph Raw Data ("10_sensors/ folder")
        R_det["Detections<br>(*_detections.csv)"]:::raw
        R_dist["Sensor Distances<br>(*_sensor_distances.csv)"]:::raw
        R_traj["Trajectories<br>(*_drone_trajectories.csv)"]:::raw
        R_info["Sensor Config<br>(*_sensor_information.csv)"]:::raw
    end

    subgraph Step 1: Densification ("data_preprocessing.py")
        P1["Parse Sector IDs & Pivot total_power_dBm<br>(Creates sector_1 to sector_6)"]:::process
        P2["Normalize Timestamps per Experiment<br>(Rank-based: 0 to 59)"]:::process
        P3["Cartesian Product Skeleton<br>(timestamp x 10 sensors)"]:::process
        P4["Left Join & Zero-Padding<br>(NaN power -> 0.0)"]:::process
        P5["Scale Spatial Coordinates<br>(Divided by 1000.0)"]:::process
        P6["Sort Rows by<br>(exp_id, timestamp, sensor_id)"]:::process
    end

    subgraph Intermediate Data
        I_det["processed_detections_train.csv<br>processed_detections_test.csv"]:::clean
        I_xyz["processed_drone_data_all.csv"]:::clean
    end

    subgraph Step 2: Normalization & Merge
        N1["Compute Training Stats (Mean, Std) on:<br>gain, sinr_threshold, sensitivity"]:::process
        N2["Z-Score Normalize Hardware Features"]:::process
        N3["Vectorized Merge with Drone positions<br>(x_d, y_d, z_d)"]:::process
    end

    subgraph Final Files
        F_norm["processed_train_normalized.csv<br>processed_test_normalized.csv"]:::clean
    end

    R_det --> P1
    R_det --> P2
    R_dist --> P5
    R_info --> P4
    
    P1 --> P3
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> P6
    
    P6 --> I_det
    R_det --> I_xyz
    
    I_det --> N1
    N1 --> N2
    I_xyz --> N3
    N2 --> N3
    
    N3 --> F_norm
```

---

## 1. Raw Data Schema & Safe Parsing Requirements

The raw experiment data resides in the `data/` folder, partitioned in chunks.

### A. Stringified Type Handling Constraints
1. **Sector Detection Strings (`sensor_sectors_detected`)**:
   - Stringified list containing float values (e.g. `'[1.0]'` or `'[3]'`).
   - Use `ast.literal_eval(s)[0]` or regex `r"\d+"`.
2. **Drone Position Strings (`drone_positions`)**:
   - Stringified list containing tuple (e.g., `'[(283940.23, 192834.11, 4593.22)]'`).

### B. Raw Schema Mapping

| File Type | Pattern / Example | Key Columns & Core Constraints |
| :--- | :--- | :--- |
| **Detections** | `*_detections.csv` | `exp_id`/`experiment_id`, `timestamp`, `sensor_id`, `sensor_sectors_detected`, `total_power_dBm`, `drone_positions`. |
| **Sensor Distances** | `*_sensor_distances.csv` | `exp_id`, `sensor1_id`, `x1`, `y1`, `z1`, `sensor2_id`, `x2`, `y2`, `z2`. |
| **Trajectories** | `*_drone_trajectories.csv` | `exp_id`, `timestamp`, `start_x`, `start_y`, `start_z`. |
| **Sensor Information**| `*_sensor_information.csv` | Static hardware profiles: `sensor_id`, `gain`, `sinr_threshold`, `sensitivity`. |

---

## 2. Step 1: Densification & Structuring

1. **Timestamp Normalization**: Integer ranks $0 \dots T_{\text{max}} - 1$.
2. **Pivoting Sector Detections**: Split power into `sector_1` through `sector_6`.
3. **Cartesian Skeleton**: $N_{\text{exp}} = 60 \text{ timestamps} \times 10 \text{ sensors} = 600 \text{ rows}$.
4. **Coordinate Scaling**: Divide meter coordinates by $1000.0$ for kilometers.

---

## 3. Final Dataset Schema

| Index | Column Name | Type | Description |
| :--- | :--- | :--- | :--- |
| **0** | `exp_id` | Integer | Experiment ID |
| **1** | `sensor_id` | Integer | Sensor ID (0..9) |
| **2** | `timestamp` | Integer | Timestamp index (0..59) |
| **3..8** | `sector_1`..`sector_6` | Float | Power in dBm (0.0 if not detected) |
| **9..11** | `x`, `y`, `z` | Float | Sensor coordinates in km |
| **12..14** | `gain`, `sinr_threshold`, `sensitivity` | Float | Normalized sensor properties |
| **15..17** | `x_d`, `y_d`, `z_d` | Float | **Target**: Ground truth UAV position in km |
