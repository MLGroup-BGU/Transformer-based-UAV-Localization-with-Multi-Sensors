"""
online_post_processor.py

Online Post-Processing Component for UAV Location Prediction using Gradient Boosting / XGBoost.
Provides a lightweight interface to load trained gradient boosting regressor models
(for x, y, z target refinements) and apply them to raw spatiotemporal predictions.
"""

import os
import sys
try:
    import joblib
except ImportError:
    joblib = None
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import __main__
    if not hasattr(__main__, 'Config'):
        @dataclass
        class Config:
            predictions_file: str = ""
            sensor_positions_file: str = ""
            decimal_precision: int = 3
            context_window: int = 5
            n_folds: int = 5
            test_size: float = 0.2
            random_seed: int = 42
            compute_feature_importance: bool = False
            worst_trajectory_pct: float = 0.05
            model_params: Optional[Dict] = None
        __main__.Config = Config
except Exception:
    pass


@dataclass
class PostProcessorConfig:
    decimal_precision: int = 3
    context_window: int = 5
    model_params: Optional[Dict] = None


def compute_static_sensor_features(sensors_df: pd.DataFrame, exp_id: int) -> Tuple[Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    features = {}
    exp_sensors = sensors_df[sensors_df['exp_id'] == exp_id] if 'exp_id' in sensors_df.columns else sensors_df
    
    # Fallback for when train_xgboost passes a hardcoded exp_id=1 for the whole sensor layout
    if len(exp_sensors) == 0 and 'exp_id' in sensors_df.columns and len(sensors_df['exp_id'].unique()) == 1:
        exp_sensors = sensors_df

    if len(exp_sensors) == 0:
        for stat in ['mean', 'min', 'max']:
            features[f'sensor_dist_{stat}'] = 0.0
            features[f'sensor_inter_dist_{stat}'] = 0.0
        features['n_sensors'] = 0.0
        return features, None, None

    sensor_positions = exp_sensors[['x', 'y', 'z']].values
    features['n_sensors'] = float(len(sensor_positions))

    if len(sensor_positions) > 1:
        inter_distances = []
        for i in range(len(sensor_positions)):
            for j in range(i + 1, len(sensor_positions)):
                dist = np.sqrt(np.sum((sensor_positions[i] - sensor_positions[j]) ** 2))
                inter_distances.append(dist)
        inter_dist_arr = np.array(inter_distances)
        features['sensor_inter_dist_mean'] = float(np.mean(inter_dist_arr))
        features['sensor_inter_dist_min'] = float(np.min(inter_dist_arr))
        features['sensor_inter_dist_max'] = float(np.max(inter_dist_arr))
    else:
        features['sensor_inter_dist_mean'] = 0.0
        features['sensor_inter_dist_min'] = 0.0
        features['sensor_inter_dist_max'] = 0.0

    features['sensor_spread_x'] = float(np.max(sensor_positions[:, 0]) - np.min(sensor_positions[:, 0]))
    features['sensor_spread_y'] = float(np.max(sensor_positions[:, 1]) - np.min(sensor_positions[:, 1]))
    features['sensor_spread_z'] = float(np.max(sensor_positions[:, 2]) - np.min(sensor_positions[:, 2]))

    centroid = np.mean(sensor_positions, axis=0)
    features['sensor_centroid_x'] = float(centroid[0])
    features['sensor_centroid_y'] = float(centroid[1])
    features['sensor_centroid_z'] = float(centroid[2])

    return features, sensor_positions, centroid


class OnlinePostProcessor:
    """
    Online post-processing engine that takes spatiotemporal model predictions,
    builds the full feature representation, and applies trained Gradient Boosting / XGBoost
    models to predict refined (x, y, z) UAV positions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        context_window: int = 5,
        decimal_precision: int = 3
    ):
        self.context_window = context_window
        self.decimal_precision = decimal_precision
        self.models = {}
        self.feature_cols = []

        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

        if model_path is None:
            candidates = [
                os.path.join(root_dir, "models", "xgboost", "post_processing_models.joblib"),
                os.path.join(root_dir, "models", "post_processing_models.joblib"),
                os.path.join(root_dir, "Encoder-Decoder", "Post Processing XGBoost", "post_processing_models.joblib"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif model_path != "__no_load__":
            print(f"[INFO] Post-processing model file not loaded. Running in pass-through mode.")

    def load(self, model_path: str):
        """Loads trained regressor models and feature column names from a joblib file."""
        if joblib is None:
            print("[WARNING] joblib is not installed. Cannot load post-processing models.")
            return
        checkpoint = joblib.load(model_path)
        if isinstance(checkpoint, dict):
            self.models = {
                'true_x': checkpoint.get('model_x'),
                'true_y': checkpoint.get('model_y'),
                'true_z': checkpoint.get('model_z')
            }
            self.feature_cols = checkpoint.get('feature_cols', [])
            if 'config' in checkpoint and hasattr(checkpoint['config'], 'context_window'):
                self.context_window = checkpoint['config'].context_window
        print(f"[OK] OnlinePostProcessor loaded models from: {model_path} ({len(self.feature_cols)} features)")

    def save(self, save_path: str, model_x, model_y, model_z, feature_cols: List[str]):
        """Saves trained models into joblib artifact."""
        if joblib is None:
            raise ImportError("joblib package is required to save models.")
        checkpoint = {
            'model_x': model_x,
            'model_y': model_y,
            'model_z': model_z,
            'feature_cols': feature_cols,
            'context_window': self.context_window
        }
        joblib.dump(checkpoint, save_path)
        print(f"[OK] OnlinePostProcessor saved models to: {save_path}")

    def extract_features(
        self,
        predictions_df: pd.DataFrame,
        sensors_df: pd.DataFrame
    ) -> pd.DataFrame:
        df = predictions_df.copy()

        if 'exp_id' not in df.columns:
            df['exp_id'] = 1

        if 'time_step' not in df.columns:
            if 'timestamp' in df.columns:
                df['time_step'] = df['timestamp']
            else:
                df['time_step'] = np.arange(len(df))

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = lambda x, **kw: x

        print("\n[*] Starting streamlined vectorized feature extraction...")
        trajectories = df['exp_id'].unique()
        all_feature_dfs = []
        cw = self.context_window

        for exp_id in tqdm(trajectories, desc="Extracting Features", dynamic_ncols=True):
            traj_df = df[df['exp_id'] == exp_id]

            # Pre-compute static sensor geometry features ONCE per experiment
            static_sensor_feats, sensor_positions, centroid = compute_static_sensor_features(sensors_df, exp_id)

            traj_sorted = traj_df.sort_values('time_step').reset_index(drop=True)

            px = traj_sorted['pred_x'].values
            py = traj_sorted['pred_y'].values
            pz = traj_sorted['pred_z'].values
            ts_ids_list = traj_sorted['time_step'].tolist()
            T = len(ts_ids_list)

            if T == 0:
                continue

            if T < 2:
                grad_px = np.zeros(T, dtype=np.float32)
                grad_py = np.zeros(T, dtype=np.float32)
                grad_pz = np.zeros(T, dtype=np.float32)
            else:
                grad_px = np.gradient(px)
                grad_py = np.gradient(py)
                grad_pz = np.gradient(pz)

            # Core non-redundant features
            feat_matrix_dict = {
                'pred_x': px,
                'pred_y': py,
                'pred_z': pz,
                'pred_x_trend_direction': np.sign(grad_px),
                'pred_x_trend_strength': np.abs(grad_px),
                'pred_y_trend_direction': np.sign(grad_py),
                'pred_y_trend_strength': np.abs(grad_py),
                'pred_z_trend_direction': np.sign(grad_pz),
                'pred_z_trend_strength': np.abs(grad_pz),
            }

            # Sensor distance features
            if sensor_positions is not None and len(sensor_positions) > 0:
                preds_xyz = np.stack([px, py, pz], axis=1)
                dists = np.linalg.norm(preds_xyz[:, None, :] - sensor_positions[None, :, :], axis=2)  # [T, S]
                feat_matrix_dict['sensor_dist_mean'] = np.mean(dists, axis=1)
                feat_matrix_dict['sensor_dist_min'] = np.min(dists, axis=1)
                feat_matrix_dict['sensor_dist_max'] = np.max(dists, axis=1)
                feat_matrix_dict['dist_to_sensor_centroid'] = np.linalg.norm(preds_xyz - centroid[None, :], axis=1)
            else:
                for stat in ['mean', 'min', 'max']:
                    feat_matrix_dict[f'sensor_dist_{stat}'] = np.zeros(T)
                feat_matrix_dict['dist_to_sensor_centroid'] = np.zeros(T)

            # Add static sensor features
            for k, v in static_sensor_feats.items():
                if k not in feat_matrix_dict:
                    feat_matrix_dict[k] = np.full(T, v, dtype=np.float32)

            feat_keys = list(feat_matrix_dict.keys())
            feat_key_to_idx = {k: i for i, k in enumerate(feat_keys)}
            feat_matrix = np.column_stack([feat_matrix_dict[k] for k in feat_keys]).astype(np.float32)

            # Meta columns
            meta_dict = {
                'exp_id': np.full(T, exp_id),
                'time_step': traj_sorted['time_step'].values,
                'orig_x': px,
                'orig_y': py,
                'orig_z': pz,
            }
            if 'true_x' in traj_sorted.columns:
                meta_dict['true_x'] = traj_sorted['true_x'].values
                meta_dict['true_y'] = traj_sorted['true_y'].values
                meta_dict['true_z'] = traj_sorted['true_z'].values

            meta_df = pd.DataFrame(meta_dict)
            base_df = pd.DataFrame(feat_matrix, columns=feat_keys)

            # Context window features (fast numpy shift)
            context_dict = {}
            for offset in range(1, cw + 1):
                prev_idx = np.clip(np.arange(T) - offset, 0, T - 1)
                next_idx = np.clip(np.arange(T) + offset, 0, T - 1)
                
                prev_mat = feat_matrix[prev_idx]
                next_mat = feat_matrix[next_idx]

                for i, k in enumerate(feat_keys):
                    context_dict[f'{k}_prev_{offset}'] = prev_mat[:, i]
                    context_dict[f'{k}_next_{offset}'] = next_mat[:, i]
                    
                    # Differences only for coordinates
                    if k in ['pred_x', 'pred_y', 'pred_z']:
                        context_dict[f'diff_{k}_prev_{offset}'] = feat_matrix[:, i] - prev_mat[:, i]
                        context_dict[f'diff_{k}_next_{offset}'] = feat_matrix[:, i] - next_mat[:, i]

            # Velocity and acceleration
            for axis in ['x', 'y', 'z']:
                k = f'pred_{axis}'
                if k in feat_key_to_idx:
                    fi = feat_key_to_idx[k]
                    prev_stack = feat_matrix[np.clip(np.arange(T)[:, None] - np.arange(cw + 1), 0, T - 1), fi]
                    next_stack = feat_matrix[np.clip(np.arange(T)[:, None] + np.arange(cw + 1), 0, T - 1), fi]
                    
                    vel_prev = np.mean(np.diff(prev_stack, axis=1), axis=1) if cw > 0 else np.zeros(T)
                    vel_next = np.mean(np.diff(next_stack, axis=1), axis=1) if cw > 0 else np.zeros(T)
                    
                    context_dict[f'velocity_{axis}_prev'] = vel_prev
                    context_dict[f'velocity_{axis}_next'] = vel_next
                    context_dict[f'acceleration_{axis}'] = vel_next - vel_prev

            ctx_df = pd.DataFrame(context_dict)
            traj_feat_df = pd.concat([meta_df, base_df, ctx_df], axis=1)
            all_feature_dfs.append(traj_feat_df)

        if not all_feature_dfs:
            print("[-] No features extracted.")
            return pd.DataFrame()

        print("[*] Concatenating all feature dataframes...")
        feature_df = pd.concat(all_feature_dfs, ignore_index=True)
        print(f"[+] Final feature dataframe shape: {feature_df.shape}\n")
        return feature_df

    def predict_trajectory(
        self,
        predictions_df: pd.DataFrame,
        sensors_df: pd.DataFrame
    ) -> pd.DataFrame:
        if not self.models or 'true_x' not in self.models:
            res = predictions_df.copy()
            res['post_pred_x'] = res['pred_x']
            res['post_pred_y'] = res['pred_y']
            res['post_pred_z'] = res['pred_z']
            return res

        feature_df = self.extract_features(predictions_df, sensors_df)

        missing_cols = [col for col in self.feature_cols if col not in feature_df.columns]
        for col in missing_cols:
            feature_df[col] = 0.0

        X = feature_df[self.feature_cols].values

        pred_x = self.models['true_x'].predict(X)
        pred_y = self.models['true_y'].predict(X)
        pred_z = self.models['true_z'].predict(X)

        # Auto-detect if trained models predict residual offsets (magnitude < 1.0 km) or absolute coordinates
        if np.mean(np.abs(pred_x)) < 1.0 and 'orig_x' in feature_df.columns:
            feature_df['post_pred_x'] = feature_df['orig_x'] + pred_x
            feature_df['post_pred_y'] = feature_df['orig_y'] + pred_y
            feature_df['post_pred_z'] = feature_df['orig_z'] + pred_z
        else:
            feature_df['post_pred_x'] = pred_x
            feature_df['post_pred_y'] = pred_y
            feature_df['post_pred_z'] = pred_z

        if 'true_x' in feature_df.columns:
            diff_3d = feature_df[['post_pred_x', 'post_pred_y', 'post_pred_z']].values - feature_df[['true_x', 'true_y', 'true_z']].values
            diff_2d = feature_df[['post_pred_x', 'post_pred_y']].values - feature_df[['true_x', 'true_y']].values
            feature_df['post_error_3d_m'] = np.linalg.norm(diff_3d, axis=1) * 1000.0
            feature_df['post_error_2d_m'] = np.linalg.norm(diff_2d, axis=1) * 1000.0

        return feature_df
