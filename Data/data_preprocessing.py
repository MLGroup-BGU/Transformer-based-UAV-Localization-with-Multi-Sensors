"""
data_preprocessing.py

Data Preprocessing Script using SQLite in-memory database queries to normalize and align data.
"""

import os
import glob
import sqlite3
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

BASE_DATA_DIR = os.path.join(ROOT_DIR, "data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")
if not os.path.exists(BASE_DATA_DIR):
    BASE_DATA_DIR = os.path.join(ROOT_DIR, "Data", "train_data_10_sensors_zsectors_ptx_model_preprocessing")

OUTPUT_FILENAME = "normalized_data.csv"

SQL_QUERY = """
WITH SensorLocations AS (
    SELECT exp_id, sensor1_id AS sensor_id, x1 AS x, y1 AS y, z1 AS z 
    FROM sensor_distances
    UNION ALL
    SELECT exp_id, sensor2_id AS sensor_id, x2 AS x, y2 AS y, z2 AS z 
    FROM sensor_distances
)
SELECT distinct
    drone_trajectories.experiment_id, 
    drone_trajectories.timestamp, 
    sensor_gain_sectors.sensor_id,
    sensor_detection_summary.total_power_dBm, 
    sensor_detection_summary.sensor_sectors_detected, 
    sensor_information.gain, 
    sensor_information.sensitivity / 90.0 AS sensitivity, 
    sensor_information.sinr_threshold / 10.0 AS sinr_threshold,
    sl.x AS sensor_x,
    sl.y AS sensor_y,
    sl.z AS sensor_z,
    drone_trajectories.start_x as uav_x, 
    drone_trajectories.start_y as uav_y, 
    drone_trajectories.start_z as uav_z
FROM drone_trajectories 
LEFT OUTER JOIN sensor_gain_sectors 
    ON drone_trajectories.experiment_id = sensor_gain_sectors.exp_id 
    AND sensor_gain_sectors.sector_id = 1 
LEFT OUTER JOIN sensor_detection_summary 
    ON sensor_gain_sectors.sensor_id = sensor_detection_summary.sensor_id 
    AND sensor_detection_summary.timestamp = drone_trajectories.timestamp 
INNER JOIN sensor_information 
    ON sensor_gain_sectors.sensor_id = sensor_information.sensor_id
LEFT OUTER JOIN SensorLocations sl
    ON sl.exp_id = drone_trajectories.experiment_id
    AND sl.sensor_id = sensor_gain_sectors.sensor_id
"""


def process_subfolder(folder_path):
    subfolder_name = os.path.basename(folder_path)
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        return

    conn = sqlite3.connect(":memory:")

    for csv_file in csv_files:
        table_name = os.path.splitext(os.path.basename(csv_file))[0]
        try:
            df = pd.read_csv(csv_file)
            df.to_sql(table_name, conn, index=False, if_exists="replace")
        except Exception as e:
            print(f"    [Warning] Failed to load {table_name} in {subfolder_name}: {e}")

    try:
        result_df = pd.read_sql_query(SQL_QUERY, conn)
        out_path = os.path.join(folder_path, OUTPUT_FILENAME)
        result_df.to_csv(out_path, index=False)
        print(f"  [+] Created {OUTPUT_FILENAME} in subfolder: {subfolder_name} ({len(result_df)} rows)")
    except Exception as e:
        print(f"  [ERROR] Failed to run SQL query on {subfolder_name}: {e}")
    finally:
        conn.close()


def main():
    print(f"Target Directory: {BASE_DATA_DIR}")
    if not os.path.exists(BASE_DATA_DIR):
        print(f"[ERROR] Directory does not exist: {BASE_DATA_DIR}")
        return

    subfolders = [f.path for f in os.scandir(BASE_DATA_DIR) if f.is_dir()]
    print(f"Found {len(subfolders)} subfolders to process...\n")

    for sf in subfolders:
        process_subfolder(sf)

    print("\n[OK] All subfolder preprocessing completed successfully!")


if __name__ == "__main__":
    main()
