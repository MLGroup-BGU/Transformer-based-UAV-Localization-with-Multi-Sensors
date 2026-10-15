"""
rssi_trilateration_predictor.py

Implementation of the RSSI Path-Loss Multilateration / Trilateration UAV Predictor.
Uses multi-sensor RSSI power measurements, sensor coordinates, drone transmit parameters
(P_tx, G_tx, frequency f), path loss model, and non-linear Gauss-Newton optimization.
"""

import time
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .base_predictor import BaseUAVPredictor
from .rssi_utils import estimate_position_wcl


class RSSITrilaterationPredictor(BaseUAVPredictor):
    """
    RSSI Path-Loss Multilateration / Trilateration UAV Predictor.
    Estimates 3D UAV positions from raw RSSI power measurements using log-distance path loss propagation.
    """

    def __init__(
        self,
        path_loss_exponent: float = 2.0,
        name: str = "RSSI Path-Loss Multilateration",
    ):
        super().__init__(name=name)
        self.n = path_loss_exponent

    def _extract_drone_tx_power(self, drone_info_df: pd.DataFrame, exp_id: Any) -> Tuple[float, float, float]:
        default_ptx = 20.0
        default_gtx = 0.0
        default_freq = 433000000.0

        if drone_info_df.empty:
            return default_ptx, default_gtx, default_freq

        exp_col = "experiment_id" if "experiment_id" in drone_info_df.columns else "exp_id"
        matching = drone_info_df[drone_info_df[exp_col] == exp_id]
        if matching.empty:
            matching = drone_info_df

        row = matching.iloc[0]
        p_tx = float(row["drone_transmitter_power"]) if "drone_transmitter_power" in row and not pd.isna(row["drone_transmitter_power"]) else (float(row["p_tx"]) if "p_tx" in row and not pd.isna(row["p_tx"]) else None)
        g_tx = float(row["transmitter_antenna_gain"]) if "transmitter_antenna_gain" in row and not pd.isna(row["transmitter_antenna_gain"]) else (float(row["g_tx"]) if "g_tx" in row and not pd.isna(row["g_tx"]) else None)
        f_hz = float(row["freq_min"]) if "freq_min" in row and not pd.isna(row["freq_min"]) else (float(row["center_frequency"]) if "center_frequency" in row and not pd.isna(row["center_frequency"]) else None)

        if p_tx is None or g_tx is None or f_hz is None:
            logging.warning(f"Missing drone TX parameters for exp {exp_id}. Using defaults: P_tx={default_ptx}, G_tx={default_gtx}, f={default_freq}")
        p_tx = p_tx if p_tx is not None else default_ptx
        g_tx = g_tx if g_tx is not None else default_gtx
        f_hz = f_hz if f_hz is not None else default_freq

        return p_tx, g_tx, f_hz

    def _extract_sensor_positions_and_gains(self, dist_df: pd.DataFrame, info_df: pd.DataFrame, exp_id: Any) -> Dict[int, Tuple[np.ndarray, float]]:
        pos_map = {}

        if not dist_df.empty:
            dist_exp_col = "exp_id" if "exp_id" in dist_df.columns else ("experiment_id" if "experiment_id" in dist_df.columns else None)
            exp_dist = dist_df[dist_df[dist_exp_col] == exp_id] if dist_exp_col and not dist_df.empty and dist_exp_col in dist_df.columns else dist_df
            for _, r in exp_dist.iterrows():
                s1, s2 = int(r["sensor1_id"]), int(r["sensor2_id"])
                if s1 not in pos_map and "x1" in r:
                    pos_map[s1] = np.array([r["x1"], r["y1"], r["z1"]], dtype=float)
                if s2 not in pos_map and "x2" in r:
                    pos_map[s2] = np.array([r["x2"], r["y2"], r["z2"]], dtype=float)

        gain_map = {}
        if not info_df.empty and "gain" in info_df.columns:
            info_exp_col = "experiment_id" if "experiment_id" in info_df.columns else ("exp_id" if "exp_id" in info_df.columns else None)
            exp_info = info_df[info_df[info_exp_col] == exp_id] if info_exp_col and info_exp_col in info_df.columns else info_df
            for _, r in exp_info.iterrows():
                s_id = int(r["sensor_id"])
                gain_map[s_id] = float(r["gain"])
            if not gain_map:
                for _, r in info_df.iterrows():
                    s_id = int(r["sensor_id"])
                    gain_map[s_id] = float(r["gain"])

        res = {}
        for s_id, pos in pos_map.items():
            g_rx = gain_map.get(s_id, 0.0)
            res[s_id] = (pos, g_rx)

        return res

    def _estimate_position_trilateration(
        self,
        sensor_positions: np.ndarray,
        rssi_values_dbm: np.ndarray,
        sensor_gains_dbi: np.ndarray,
        p_tx: float,
        g_tx: float,
        f_hz: float,
    ) -> np.ndarray:
        K = len(sensor_positions)
        if K == 0:
            return np.zeros(3, dtype=float)

        # User spec formula: P_rx = P_tx + G_t + G_r - 20log10(d) - 20log10(f) + 147
        offset = 20.0 * np.log10(f_hz) - 147.0
        dists = 10.0 ** ((p_tx + g_tx + sensor_gains_dbi - rssi_values_dbm - offset) / (10.0 * self.n))

        wcl_guess = estimate_position_wcl(sensor_positions, rssi_values_dbm)

        if K < 3:
            return wcl_guess

        try:
            A = 2.0 * (sensor_positions[1:] - sensor_positions[0])
            b = (
                dists[0] ** 2
                - dists[1:] ** 2
                + np.sum(sensor_positions[1:] ** 2, axis=1)
                - np.sum(sensor_positions[0] ** 2)
            )
            est_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            initial_guess = est_lstsq.copy()
            
            # UAVs fly above the ground sensors. Since sensors are roughly co-planar, 
            # the linear algebraic Z estimate is unstable and can easily point underground.
            # We force it to a high positive value so least_squares descends to the correct airborne root.
            initial_guess[2] = np.abs(initial_guess[2])
            if initial_guess[2] < 200.0:
                initial_guess[2] = 1000.0
                
        except Exception:
            initial_guess = wcl_guess.copy()
            initial_guess[2] += 1000.0

        def residuals_func(x):
            calc_dists = np.linalg.norm(sensor_positions - x, axis=1)
            return calc_dists - dists

        # The 'trf' method requires the initial guess to be strictly within the bounds
        initial_guess = np.clip(initial_guess, 0.0, 5000.0)

        try:
            res = least_squares(
                residuals_func, 
                initial_guess, 
                bounds=(0.0, 5000.0), 
                method='trf'
            )
            x_est = res.x
        except Exception:
            x_est = wcl_guess.copy()

        if np.isnan(x_est).any() or np.isinf(x_est).any():
            return wcl_guess

        return x_est

    def predict_experiment(self, raw_exp_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        total_start = time.perf_counter()
        model_time_sec = 0.0

        det_df = raw_exp_data["detections"].copy()
        info_df = raw_exp_data.get("sensor_info", pd.DataFrame())
        dist_df = raw_exp_data.get("sensor_dist", pd.DataFrame())
        drone_info_df = raw_exp_data.get("drone_info", pd.DataFrame())

        exp_col = "experiment_id" if "experiment_id" in det_df.columns else "exp_id"
        exp_id = det_df[exp_col].iloc[0] if not det_df.empty else 1
        if not det_df.empty:
            det_df = det_df[det_df[exp_col] == exp_id].copy()

        p_tx, g_tx, f_hz = self._extract_drone_tx_power(drone_info_df, exp_id)
        sensor_map = self._extract_sensor_positions_and_gains(dist_df, info_df, exp_id)

        pow_col = "total_power_dBm" if "total_power_dBm" in det_df.columns else "received_power"

        unique_ts = np.sort(det_df["timestamp"].unique())
        ts_map = {t: idx for idx, t in enumerate(unique_ts)}
        det_df["ts_idx"] = det_df["timestamp"].map(ts_map)

        pred_list = []
        ts_list = []

        for ts_num, group in det_df.groupby("ts_idx"):
            active_sids = group["sensor_id"].unique()
            s_positions = []
            s_rssis = []
            s_gains = []

            for sid in active_sids:
                if sid in sensor_map:
                    s_pos, s_gain = sensor_map[sid]
                    s_rows = group[group["sensor_id"] == sid]
                    rssi_val = float(s_rows[pow_col].max())
                    if rssi_val > -120.0:
                        s_positions.append(s_pos)
                        s_rssis.append(rssi_val)
                        s_gains.append(s_gain)

            if len(s_positions) > 0:
                sp_arr = np.array(s_positions, dtype=float)
                sr_arr = np.array(s_rssis, dtype=float)
                sg_arr = np.array(s_gains, dtype=float)

                model_start = time.perf_counter()
                pred_pos = self._estimate_position_trilateration(sp_arr, sr_arr, sg_arr, p_tx, g_tx, f_hz)
                model_time_sec += time.perf_counter() - model_start
            else:
                pred_pos = np.zeros(3, dtype=float)

            pred_list.append(pred_pos)
            ts_list.append(ts_num)

        pred_xyz_m = np.array(pred_list, dtype=float)
        timestamps = np.array(ts_list, dtype=int)

        self.last_preprocessing_time_sec = time.perf_counter() - total_start - model_time_sec
        return pred_xyz_m, timestamps, model_time_sec
