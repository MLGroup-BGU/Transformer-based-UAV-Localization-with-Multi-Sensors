"""
base_predictor.py

Abstract Base Class for UAV Localization Predictors.
Defines standard prediction interface and comprehensive evaluation metrics computation.
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


class BaseUAVPredictor(ABC):
    """
    Abstract base class for all UAV location predictor implementations.
    """

    def __init__(self, name: str = "BasePredictor"):
        self.name = name
        # Pre/post-processing time (s) of the last predict_experiment call, excluded from processing_time_sec
        self.last_preprocessing_time_sec = 0.0

    @abstractmethod
    def predict_experiment(self, raw_exp_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predicts 3D UAV locations for a single experiment from raw sensor input features ONLY.

        Args:
            raw_exp_data (dict): Dictionary containing raw input DataFrames

        Returns:
            pred_xyz (np.ndarray): Predicted [N, 3] 3D coordinates in meters (x, y, z)
            timestamps (np.ndarray): Timestamps array [N]
            processing_time_sec (float): Model computation duration in seconds (pre/post-processing
                excluded and stored in self.last_preprocessing_time_sec)
        """
        pass

    def evaluate(self, pred_xyz: np.ndarray, true_xyz: np.ndarray, timestamps: np.ndarray, processing_time_sec: float) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        Calculates standard spatial accuracy, trajectory loss, and processing performance metrics.
        """
        if len(pred_xyz) == 0 or len(true_xyz) == 0:
            raise ValueError("Empty prediction or ground truth array provided for evaluation.")

        n_samples = len(pred_xyz)

        diff_3d_m = pred_xyz - true_xyz
        diff_2d_m = pred_xyz[:, :2] - true_xyz[:, :2]
        diff_z_m = pred_xyz[:, 2] - true_xyz[:, 2]

        errors_3d_m = np.linalg.norm(diff_3d_m, axis=-1)
        errors_2d_m = np.linalg.norm(diff_2d_m, axis=-1)
        errors_z_m = np.abs(diff_z_m)

        diff_3d_km = diff_3d_m / 1000.0
        diff_2d_km = diff_2d_m / 1000.0

        valid_mask = ~((true_xyz[:, 0] == 0.0) & (true_xyz[:, 1] == 0.0) & (true_xyz[:, 2] == 0.0))
        if not valid_mask.any():
            valid_mask = np.ones(n_samples, dtype=bool)

        e3d_val = errors_3d_m[valid_mask]
        e2d_val = errors_2d_m[valid_mask]
        ez_val = errors_z_m[valid_mask]

        diff_3d_km_val = diff_3d_km[valid_mask]
        diff_2d_km_val = diff_2d_km[valid_mask]

        pytorch_mse_km2 = float(np.mean(diff_3d_km_val ** 2))

        x_mse_km2 = float(np.mean(diff_3d_km_val[:, 0] ** 2))
        y_mse_km2 = float(np.mean(diff_3d_km_val[:, 1] ** 2))
        z_mse_km2 = float(np.mean(diff_3d_km_val[:, 2] ** 2))

        mse_3d_km2 = float(np.mean(np.sum(diff_3d_km_val ** 2, axis=-1)))
        mae_3d_km = float(np.mean(np.abs(diff_3d_km_val)))

        mse_2d_km2 = float(np.mean(np.sum(diff_2d_km_val ** 2, axis=-1)))
        mae_2d_km = float(np.mean(np.abs(diff_2d_km_val)))
        rmse_3d_m = float(np.sqrt(np.mean(e3d_val ** 2)))

        if n_samples > 1:
            pred_step_diff = pred_xyz[1:] - pred_xyz[:-1]
            true_step_diff = true_xyz[1:] - true_xyz[:-1]
            traj_step_loss_m = float(np.mean(np.linalg.norm(pred_step_diff - true_step_diff, axis=-1)))
        else:
            traj_step_loss_m = 0.0

        latency_ms_per_step = (processing_time_sec * 1000.0 / n_samples) if n_samples > 0 else 0.0
        throughput_fps = (n_samples / processing_time_sec) if processing_time_sec > 0 else 0.0

        metrics = {
            "Predictor_Name": self.name,
            "Total_Timestamps": n_samples,
            "Total_Processing_Time (s)": processing_time_sec,
            "Latency_Per_Timestamp (ms)": latency_ms_per_step,
            "Throughput (steps/s)": throughput_fps,
            "3D_Mean_Error (m)": float(np.mean(e3d_val)),
            "3D_Median_Error (m)": float(np.median(e3d_val)),
            "3D_Std_Error (m)": float(np.std(e3d_val)),
            "3D_RMSE (m)": float(rmse_3d_m),
            "3D_P90_Error (m)": float(np.percentile(e3d_val, 90)),
            "3D_P95_Error (m)": float(np.percentile(e3d_val, 95)),
            "3D_Min_Error (m)": float(np.min(e3d_val)),
            "3D_Max_Error (m)": float(np.max(e3d_val)),
            "Trajectory_Step_Loss (m)": traj_step_loss_m,
            "2D_Mean_Error (m)": float(np.mean(e2d_val)),
            "2D_Median_Error (m)": float(np.median(e2d_val)),
            "2D_Std_Error (m)": float(np.std(e2d_val)),
            "2D_P90_Error (m)": float(np.percentile(e2d_val, 90)),
            "2D_P95_Error (m)": float(np.percentile(e2d_val, 95)),
            "Z_Mean_Error (m)": float(np.mean(ez_val)),
            "Z_Median_Error (m)": float(np.median(ez_val)),
            "Z_P95_Error (m)": float(np.percentile(ez_val, 95)),
            "Axis_X_MSE (km^2)": x_mse_km2,
            "Axis_Y_MSE (km^2)": y_mse_km2,
            "Axis_Z_MSE (km^2)": z_mse_km2,
            "PyTorch_MSELoss (km^2)": pytorch_mse_km2,
            "3D_MSE (km^2)": mse_3d_km2,
            "3D_MAE (km)": mae_3d_km,
            "2D_MSE (km^2)": mse_2d_km2,
            "2D_MAE (km)": mae_2d_km,
        }

        merged_df = pd.DataFrame({
            "timestamp": timestamps,
            "pred_x": pred_xyz[:, 0],
            "pred_y": pred_xyz[:, 1],
            "pred_z": pred_xyz[:, 2],
            "true_x": true_xyz[:, 0],
            "true_y": true_xyz[:, 1],
            "true_z": true_xyz[:, 2],
            "error_2d_m": errors_2d_m,
            "error_3d_m": errors_3d_m,
            "error_z_m": errors_z_m,
        })

        return metrics, merged_df

    def format_metrics_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Formats evaluation metrics into a human-readable summary block.
        """
        lines = [
            f"=== {metrics.get('Predictor_Name', self.name)} Evaluation Summary ===",
            f"  Timestamps Evaluated : {metrics.get('Total_Timestamps', 0)}",
            f"  Processing Time      : {metrics.get('Total_Processing_Time (s)', 0.0):.4f} s",
            f"  Inference Latency    : {metrics.get('Latency_Per_Timestamp (ms)', 0.0):.3f} ms / step",
            f"  Throughput           : {metrics.get('Throughput (steps/s)', 0.0):.2f} steps / s",
            "  -------------------------------------------------------------",
            f"  3D Mean Error (m)    : {metrics.get('3D_Mean_Error (m)', 0.0):.2f} m",
            f"  3D Median Error (m)  : {metrics.get('3D_Median_Error (m)', 0.0):.2f} m",
            f"  3D RMSE (m)          : {metrics.get('3D_RMSE (m)', 0.0):.2f} m",
            f"  3D P90 Error (m)     : {metrics.get('3D_P90_Error (m)', 0.0):.2f} m",
            f"  3D P95 Error (m)     : {metrics.get('3D_P95_Error (m)', 0.0):.2f} m",
            "  -------------------------------------------------------------",
            f"  Trajectory Step Loss : {metrics.get('Trajectory_Step_Loss (m)', 0.0):.2f} m (step displacement loss)",
            "  -------------------------------------------------------------",
            f"  2D Mean Error (m)    : {metrics.get('2D_Mean_Error (m)', 0.0):.2f} m",
            f"  2D Median Error (m)  : {metrics.get('2D_Median_Error (m)', 0.0):.2f} m",
            f"  2D P95 Error (m)     : {metrics.get('2D_P95_Error (m)', 0.0):.2f} m",
            "  -------------------------------------------------------------",
            f"  Z-Axis Mean Error (m): {metrics.get('Z_Mean_Error (m)', 0.0):.2f} m",
            f"  Z-Axis Median (m)    : {metrics.get('Z_Median_Error (m)', 0.0):.2f} m",
            "  -------------------------------------------------------------",
            f"  PyTorch MSELoss (km2): {metrics.get('PyTorch_MSELoss (km^2)', 0.0):.6f} (matches nn.MSELoss)",
            f"  Axis X/Y/Z MSE (km2) : X={metrics.get('Axis_X_MSE (km^2)', 0.0):.6f}, Y={metrics.get('Axis_Y_MSE (km^2)', 0.0):.6f}, Z={metrics.get('Axis_Z_MSE (km^2)', 0.0):.6f}",
            f"  3D Total MSE (km^2)  : {metrics.get('3D_MSE (km^2)', 0.0):.6f}",
            f"  3D MAE (km)          : {metrics.get('3D_MAE (km)', 0.0):.6f}",
            "=============================================================",
        ]
        return "\n".join(lines)
