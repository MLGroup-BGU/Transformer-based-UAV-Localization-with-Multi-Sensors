"""
rssi_utils.py

Utility functions for RSSI-based UAV position estimation.
"""
import numpy as np

def estimate_position_wcl(sensor_positions: np.ndarray, rssi_values_dbm: np.ndarray) -> np.ndarray:
    """
    Estimates 3D UAV position using RSSI Weighted Centroid Localization (WCL).
    
    Args:
        sensor_positions: np.ndarray of shape (N, 3) with sensor coordinates.
        rssi_values_dbm: np.ndarray of shape (N,) with RSSI values in dBm.
        
    Returns:
        np.ndarray of shape (3,) with the estimated (x, y, z) position.
    """
    weights = 10.0 ** (rssi_values_dbm / 10.0)
    total_w = np.sum(weights)
    if total_w < 1e-12:
        return np.mean(sensor_positions, axis=0)

    pos_est = np.sum(sensor_positions * weights[:, np.newaxis], axis=0) / total_w
    return pos_est
