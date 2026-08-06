# System Instruction Manual: UAV Detection Preprocessing & Input Pipeline

This document defines the strict algorithmic rules, directory structures, and data-handling constraints for compiling the data preprocessing pipeline. The code generator must implement this pipeline to transform raw asynchronous sensor logs into normalized, sequential CSV files for training and inference.

---

## 📂 Directory & File Structure Configuration

The pipeline must utilize the following paths and directory structure. The code generator must implement all file paths relative to these configuration parameters:

```python
INPUT_DIR = "10_sensors"
OUTPUT_DIR = "model training"

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

The preprocessing script must transform raw asynchronous sensor logs and drone trajectories into a synchronized, structured sequence of sensor readings.

```mermaid
graph TD
    %% Define Styles
    classDef raw fill:#ffe3e3,stroke:#ff8080,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef clean fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef tensor fill:#fff8e1,stroke:#ffc107,stroke-width:2px;

    %% Raw Data Nodes
    subgraph Raw Data ("10_sensors/ folder")
        R_det["Detections<br>(*_detections.csv)"]:::raw
        R_dist["Sensor Distances<br>(*_sensor_distances.csv)"]:::raw
        R_traj["Trajectories<br>(*_drone_trajectories.csv)"]:::raw
        R_info["Sensor Config<br>(*_sensor_information.csv)"]:::raw
    end

    %% Processing Nodes
    subgraph Step 1: Densification ("file_processing.py")
        P1["Parse Sector IDs & Pivot total_power_dBm<br>(Creates sector_1 to sector_6)"]:::process
        P2["Normalize Timestamps per Experiment<br>(Rank-based: 0 to 59)"]:::process
        P3["Cartesian Product Skeleton<br>(timestamp x 10 sensors)"]:::process
        P4["Left Join & Zero-Padding<br>(NaN power -> 0.0)"]:::process
        P5["Scale Spatial Coordinates<br>(Divided by 1000.0)"]:::process
        P6["Sort Rows by<br>(exp_id, timestamp, sensor_id)"]:::process
    end

    %% Intermediate Data Nodes
    subgraph Intermediate Data
        I_det["processed_detections_train.csv<br>processed_detections_test.csv"]:::clean
        I_xyz["processed_drone_data_all.csv"]:::clean
    end

    %% Normalization & Merging Nodes
    subgraph Step 2: Normalization & Merge ("files_creation.py")
        N1["Compute Training Stats (Mean, Std) on:<br>gain, sinr_threshold, sensitivity"]:::process
        N2["Z-Score Normalize Hardware Features"]:::process
        N3["Vectorized Merge with Drone positions<br>(x_d, y_d, z_d)"]:::process
    end

    %% Final Outputs
    subgraph Final Files
        F_norm["processed_train_normalized.csv<br>processed_test_normalized.csv"]:::clean
    end

    %% Connections
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

The raw experiment data resides in the `10_sensors/` folder, partitioned in chunks of 25,000 or 50,000 experiments. The generator must process the following raw schemas:

### A. Stringified Type Handling Constraints
Raw CSV files contain string-serialized collections that the parser must convert using safe parsing strategies to prevent execution errors:

1. **Sector Detection Strings (`sensor_sectors_detected`)**:
   * *Format*: Stringified list containing float values (e.g. `'[1.0]'` or `'[3]'`).
   * *Required Parsing Method*: Use `ast.literal_eval(s)[0]` if it is a string. The parser must fall back to regex numeric extraction `r"\d+"` to fetch the first digit if evaluation fails, converting the result to a standard `int`.
2. **Drone Position Strings (`drone_positions`)**:
   * *Format*: Stringified list containing a tuple (e.g., `'[(283940.23, 192834.11, 4593.22)]'`) representing coordinates in meters/millimeters.
   * *Required Parsing Method*: The parser must safely evaluate strings using `ast.literal_eval(s)[0]`. If parsing fails, it must catch the exception and return a tuple of NaNs: `(np.nan, np.nan, np.nan)`.

### B. Raw Schema Mapping

| File Type | Pattern / Example | Key Columns & Core Constraints |
| :--- | :--- | :--- |
| **Detections** | `*_detections.csv` | `exp_id`/`experiment_id`, `timestamp` (string), `sensor_id`, `sensor_sectors_detected` (string list), `total_power_dBm` (power in dBm), `drone_positions` (string tuple `[(x, y, z)]` in meters/millimeters). |
| **Sensor Distances** | `*_sensor_distances.csv` | `exp_id`, `sensor1_id`, `x1`, `y1`, `z1` and `sensor2_id`, `x2`, `y2`, `z2` (coordinates of sensors in meters). |
| **Trajectories** | `*_drone_trajectories.csv` | `exp_id`, `timestamp`, and binary markers indicating anomaly states (e.g. `is_anomaly`). |
| **Sensor Information**| `*_sensor_information.csv` | Static hardware profiles: `sensor_id`, `gain` (dB), `sinr_threshold` (dB), `sensitivity` (dBm). |

---

## 2. Step 1: Densification & Structuring (`file_processing.py`)

A raw detection file only logs active events. Because deep learning models require structured inputs, the generator must implement the **densification** algorithm so that for **every timestamp** of **every experiment**, there is a record for **all 10 sensors**.

### Timestamp Normalization
Timestamps are string-represented or non-contiguous floats. The generator must convert these into continuous ranks starting at `0` for each experiment:
```python
det["timestamp_number"] = det.groupby("exp_id", sort=False)["timestamp"].rank(method="dense").astype("int32") - 1
```
This forces timestamps to be integers from $0$ to $T_{\text{max}} - 1$ (where $T_{\text{max}} = 60$ for a standard 60-timestamp experiment).

### Pivoting Sector Detections
Each sensor contains 6 directional sectors. The generator must parse the sector index from `sensor_sectors_detected` and pivot the table to split the received power (`total_power_dBm`) into 6 dedicated columns:
```python
pivot_det = det.pivot_table(
    index=["exp_id", "sensor_id", "timestamp_number"],
    columns="sector",
    values="total_power_dBm",
    fill_value=0
).reset_index()
```

### Cartesian Product Skeleton & Zero-Padding
To ensure all 10 sensors are represented at every timestamp, the generator must construct a Cartesian product (cross-join) skeleton:
$$\text{Skeleton} = \text{Unique Timestamps} \times \text{All 10 Sensors}$$
For each experiment, the target size of the skeleton sequence must satisfy:
$$N_{\text{exp}} = T_{\text{exp}} \times S_{\text{exp}} = 60 \text{ timestamps} \times 10 \text{ sensors} = 600 \text{ rows}$$

The script must perform a left-join of the skeleton with the pivoted detections. Missing detections (inactive sensors) result in `NaN` values, which the script must pad with `0.0`:
```python
sector_cols = [f"sector_{i}" for i in range(1, 7)]
merged[sector_cols] = merged[sector_cols].fillna(0.0)
```

### Static Parameters & Coordinate Scaling
1. **Sensor Coordinates**: The generator must extract sensor coordinates from the distances file, and scale them to kilometers by dividing by `1000.0`:
   ```python
   sensor_positions[["x", "y", "z"]] /= 1000.0
   ```
2. **Sensor Hardware Properties**: Static variables (`gain`, `sinr_threshold`, `sensitivity`) from the sensor information file must be joined to each sensor row.
3. **Sorting Constraint**: The entire dataset must be sorted by `["exp_id", "timestamp", "sensor_id"]`.
   > [!IMPORTANT]
   > Sorting is a hard constraint. Each experiment must yield exactly 600 consecutive rows ordered by timestamp, and then by sensor ID inside each timestamp block.

### Edge-Case: Missing Sensor Configuration
> [!CAUTION]
> If a `sensor_id` present in the Cartesian product skeleton is missing from the physical hardware config files (`*_sensor_information.csv` or `*_sensor_distances.csv`), the script must write a warning log and fall back to the following default configuration:
> * Coordinates: $x = 0.0$, $y = 0.0$, $z = 0.0$ km
> * Gain: $0.0$ dB
> * SINR Threshold: $0.0$ dB
> * Sensitivity: $0.0$ dBm
>
> The script must **never** drop the missing rows, as this would break the 600-rows-per-experiment schema constraint.

---

## 3. Step 2: Normalization & Target Merging (`files_creation.py`)

This step processes the intermediate files to compute normalization parameters, scale hardware features, and attach the target coordinates.

### Memory Optimization Requirement (Streaming and Chunking)
To avoid RAM/Memory out-of-bounds exceptions when processing multi-gigabyte CSV files, the generator must chunk the datasets:
* Read and compute statistics using a streaming iterator: `pd.read_csv(..., chunksize=100000)`.
* Save output files by appending chunks progressively: `to_csv(..., mode='a', header=first_chunk, index=False)`.

### Z-Score Normalization Math
To prevent data leakage, normalization parameters must be calculated **only** on the training dataset.
The generator must normalize the hardware features (`gain`, `sinr_threshold`, and `sensitivity`) using:
$$x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Where the mean ($\mu_{\text{train}}$) and standard deviation ($\sigma_{\text{train}}$) are defined as:
$$\mu_{\text{train}} = \frac{1}{N} \sum_{i=1}^{N} x_i$$
$$\sigma_{\text{train}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu_{\text{train}})^2}$$
> [!TIP]
> To prevent division-by-zero errors when a feature standard deviation is 0, the script must replace $\sigma_{\text{train}} = 0$ with a small constant value:
> $$\text{if } \sigma_{\text{train}} < 10^{-8}, \quad \text{set } \sigma_{\text{train}} = 10^{-8}$$

### Drone Coordinates Integration
Drone coordinates must be extracted from the raw `drone_positions` string, scaled to kilometers by dividing by `1000.0`, and joined to the dataframe based on `['exp_id', 'timestamp']`. This appends the target columns `x_d`, `y_d`, `z_d` to the dataset.

---

## 4. Final Dataset Schema

The generated files (`processed_train_normalized.csv` and `processed_test_normalized.csv`) must contain exactly 18 columns in this precise order:

| Index | Column Name | Type | Raw Units | Normalized Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `exp_id` | Integer | N/A | $[0, \text{num\_exps}-1]$ | Unique identifier for the experiment |
| **1** | `sensor_id` | Integer | N/A | $[0, 9]$ | Identifier for the sensor (0 to 9) |
| **2** | `timestamp` | Integer | N/A | $[0, 59]$ | Normalized timestamp index (0 to 59) |
| **3** | `sector_1` | Float | dBm | Original raw dBm values | Power received in sector 1 (0.0 if no detection) |
| **4** | `sector_2` | Float | dBm | Original raw dBm values | Power received in sector 2 (0.0 if no detection) |
| **5** | `sector_3` | Float | dBm | Original raw dBm values | Power received in sector 3 (0.0 if no detection) |
| **6** | `sector_4` | Float | dBm | Original raw dBm values | Power received in sector 4 (0.0 if no detection) |
| **7** | `sector_5` | Float | dBm | Original raw dBm values | Power received in sector 5 (0.0 if no detection) |
| **8** | `sector_6` | Float | dBm | Original raw dBm values | Power received in sector 6 (0.0 if no detection) |
| **9** | `x` | Float | meters | Converted to km (div 1000) | Sensor X coordinate |
| **10** | `y` | Float | meters | Converted to km (div 1000) | Sensor Y coordinate |
| **11** | `z` | Float | meters | Converted to km (div 1000) | Sensor Z coordinate |
| **12** | `gain` | Float | dB | Z-score Normalized | Normalized sensor gain |
| **13** | `sinr_threshold`| Float | dB | Z-score Normalized | Normalized sensor SINR threshold |
| **14** | `sensitivity` | Float | dBm | Z-score Normalized | Normalized sensor sensitivity |
| **15** | `x_d` | Float | meters | Converted to km (div 1000) | **Target**: Drone X coordinate |
| **16** | `y_d` | Float | meters | Converted to km (div 1000) | **Target**: Drone Y coordinate |
| **17** | `z_d` | Float | meters | Converted to km (div 1000) | **Target**: Drone Z coordinate |
