"""
rssi_wcl_predictor.py

Implementation of the RSSI Weighted Centroid Localization (WCL) UAV Predictor.
Estimates 3D UAV positions using power-weighted averaging of ground sensor positions.
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .base_predictor import BaseUAVPredictor
from .rssi_utils import estimate_position_wcl


class RSSIWCLPredictor(BaseUAVPredictor):
    """
    RSSI Weighted Centroid Localization (WCL) UAV Predictor.
    Estimates 3D UAV positions directly from raw RSSI power measurements.
    """

    def __init__(
        self,
        name: str = "RSSI WCL",
    ):
        super().__init__(name=name)

    def _extract_sensor_positions(self, dist_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        pos_map = {}
        if not dist_df.empty:
            for _, r in dist_df.iterrows():
                s1, s2 = int(r["sensor1_id"]), int(r["sensor2_id"])
                if s1 not in pos_map and "x1" in r:
                    pos_map[s1] = np.array([r["x1"], r["y1"], r["z1"]], dtype=float)
                if s2 not in pos_map and "x2" in r:
                    pos_map[s2] = np.array([r["x2"], r["y2"], r["z2"]], dtype=float)
        return pos_map

    def predict_experiment(self, raw_exp_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        det_df = raw_exp_data["detections"].copy()
        dist_df = raw_exp_data.get("sensor_dist", pd.DataFrame())

        exp_col = "experiment_id" if "experiment_id" in det_df.columns else "exp_id"
        exp_id = det_df[exp_col].iloc[0] if not det_df.empty else 1
        if not det_df.empty:
            det_df = det_df[det_df[exp_col] == exp_id].copy()

        dist_exp_col = "exp_id" if "exp_id" in dist_df.columns else ("experiment_id" if "experiment_id" in dist_df.columns else None)
        if dist_exp_col and not dist_df.empty and dist_exp_col in dist_df.columns:
            exp_dist_df = dist_df[dist_df[dist_exp_col] == exp_id]
        else:
            exp_dist_df = dist_df

        sensor_pos_map = self._extract_sensor_positions(exp_dist_df)

        pow_col = "total_power_dBm" if "total_power_dBm" in det_df.columns else "received_power"
        
        start_time = time.perf_counter()
        
        unique_ts = np.sort(det_df["timestamp"].unique())
        ts_map = {t: idx for idx, t in enumerate(unique_ts)}
        det_df["ts_idx"] = det_df["timestamp"].map(ts_map)

        pred_list = []
        ts_list = []

        for ts_num, group in det_df.groupby("ts_idx"):
            active_sids = group["sensor_id"].unique()
            s_positions = []
            s_rssis = []

            for sid in active_sids:
                if sid in sensor_pos_map:
                    s_pos = sensor_pos_map[sid]
                    s_rows = group[group["sensor_id"] == sid]
                    rssi_val = float(s_rows[pow_col].max())
                    if rssi_val > -120.0:
                        s_positions.append(s_pos)
                        s_rssis.append(rssi_val)

            if len(s_positions) > 0:
                sp_arr = np.array(s_positions, dtype=float)
                sr_arr = np.array(s_rssis, dtype=float)
                pred_pos = estimate_position_wcl(sp_arr, sr_arr)
            else:
                pred_pos = np.zeros(3, dtype=float)

            pred_list.append(pred_pos)
            ts_list.append(ts_num)

        processing_time_sec = time.perf_counter() - start_time

        pred_xyz_m = np.array(pred_list, dtype=float)
        timestamps = np.array(ts_list, dtype=int)

        return pred_xyz_m, timestamps, processing_time_sec
