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


def compute_trend_features(values: np.ndarray) -> Dict[str, float]:
    features = {}
    if len(values) < 2:
        features['trend_direction'] = 0
        features['trend_strength'] = 0
        features['n_fluctuations'] = 0
        features['longest_up_streak'] = 0
        features['longest_down_streak'] = 0
        features['final_vs_initial'] = 0
        features['monotonic'] = 1
        return features

    diffs = np.diff(values)
    total_change = values[-1] - values[0]
    features['trend_direction'] = float(np.sign(total_change))
    features['trend_strength'] = float(abs(total_change))
    features['final_vs_initial'] = float(total_change)

    if len(diffs) > 1:
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        features['n_fluctuations'] = float(sign_changes)
    else:
        features['n_fluctuations'] = 0.0

    up_streak, down_streak, current_up, current_down = 0, 0, 0, 0
    for d in diffs:
        if d > 0:
            current_up += 1
            current_down = 0
            up_streak = max(up_streak, current_up)
        elif d < 0:
            current_down += 1
            current_up = 0
            down_streak = max(down_streak, current_down)
        else:
            current_up, current_down = 0, 0

    features['longest_up_streak'] = float(up_streak)
    features['longest_down_streak'] = float(down_streak)

    is_monotonic_up = np.all(diffs >= 0)
    is_monotonic_down = np.all(diffs <= 0)
    features['monotonic'] = 1.0 if (is_monotonic_up or is_monotonic_down) else 0.0
    return features


def compute_distribution_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
    features = {}
    if len(values) == 0:
        for stat in ['mean', 'std', 'min', 'max', 'median', 'range', 'iqr', 'skew', 'n_samples']:
            features[f'{prefix}_{stat}'] = 0.0
        return features

    features[f'{prefix}_mean'] = float(np.mean(values))
    features[f'{prefix}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
    features[f'{prefix}_min'] = float(np.min(values))
    features[f'{prefix}_max'] = float(np.max(values))
    features[f'{prefix}_median'] = float(np.median(values))
    features[f'{prefix}_range'] = float(np.max(values) - np.min(values))
    features[f'{prefix}_n_samples'] = float(len(values))

    if len(values) >= 4:
        q75, q25 = np.percentile(values, [75, 25])
        features[f'{prefix}_iqr'] = float(q75 - q25)
    else:
        features[f'{prefix}_iqr'] = 0.0

    if len(values) > 2 and features[f'{prefix}_std'] > 0:
        features[f'{prefix}_skew'] = float((np.mean(values) - np.median(values)) / features[f'{prefix}_std'])
    else:
        features[f'{prefix}_skew'] = 0.0

    return features


def extract_time_step_features(time_step_df: pd.DataFrame) -> Dict[str, float]:
    features = {}
    for axis in ['x', 'y', 'z']:
        pred_col = f'pred_{axis}'
        values = time_step_df[pred_col].values
        features.update(compute_distribution_features(values, f'pred_{axis}'))
        trend_feats = compute_trend_features(values)
        for key, val in trend_feats.items():
            features[f'pred_{axis}_{key}'] = val

    pred_x = time_step_df['pred_x'].values
    pred_y = time_step_df['pred_y'].values
    pred_z = time_step_df['pred_z'].values

    if len(pred_x) > 2:
        features['corr_xy'] = float(np.corrcoef(pred_x, pred_y)[0, 1]) if np.std(pred_x) > 0 and np.std(pred_y) > 0 else 0.0
        features['corr_xz'] = float(np.corrcoef(pred_x, pred_z)[0, 1]) if np.std(pred_x) > 0 and np.std(pred_z) > 0 else 0.0
        features['corr_yz'] = float(np.corrcoef(pred_y, pred_z)[0, 1]) if np.std(pred_y) > 0 and np.std(pred_z) > 0 else 0.0
    else:
        features['corr_xy'], features['corr_xz'], features['corr_yz'] = 0.0, 0.0, 0.0

    for key in ['corr_xy', 'corr_xz', 'corr_yz']:
        if np.isnan(features[key]):
            features[key] = 0.0

    return features


def compute_sensor_features(
    time_step_df: pd.DataFrame,
    sensors_df: pd.DataFrame,
    exp_id: int
) -> Dict[str, float]:
    features = {}
    exp_sensors = sensors_df[sensors_df['exp_id'] == exp_id] if 'exp_id' in sensors_df.columns else sensors_df

    if len(exp_sensors) == 0:
        for stat in ['mean', 'std', 'min', 'max', 'median']:
            features[f'sensor_dist_{stat}'] = 0.0
            features[f'sensor_inter_dist_{stat}'] = 0.0
        features['n_sensors'] = 0.0
        return features

    sensor_positions = exp_sensors[['x', 'y', 'z']].values
    features['n_sensors'] = float(len(sensor_positions))

    all_distances = []
    for _, pred_row in time_step_df.iterrows():
        pred_pos = np.array([pred_row['pred_x'], pred_row['pred_y'], pred_row['pred_z']])
        distances = np.sqrt(np.sum((sensor_positions - pred_pos) ** 2, axis=1))
        all_distances.extend(distances)

    all_distances = np.array(all_distances)
    features.update(compute_distribution_features(all_distances, 'sensor_dist'))

    if len(sensor_positions) > 1:
        inter_distances = []
        for i in range(len(sensor_positions)):
            for j in range(i + 1, len(sensor_positions)):
                dist = np.sqrt(np.sum((sensor_positions[i] - sensor_positions[j]) ** 2))
                inter_distances.append(dist)
        features.update(compute_distribution_features(np.array(inter_distances), 'sensor_inter_dist'))
    else:
        for stat in ['mean', 'std', 'min', 'max', 'median', 'range', 'iqr', 'skew', 'n_samples']:
            features[f'sensor_inter_dist_{stat}'] = 0.0

    features['sensor_spread_x'] = float(np.max(sensor_positions[:, 0]) - np.min(sensor_positions[:, 0]))
    features['sensor_spread_y'] = float(np.max(sensor_positions[:, 1]) - np.min(sensor_positions[:, 1]))
    features['sensor_spread_z'] = float(np.max(sensor_positions[:, 2]) - np.min(sensor_positions[:, 2]))

    centroid = np.mean(sensor_positions, axis=0)
    features['sensor_centroid_x'] = float(centroid[0])
    features['sensor_centroid_y'] = float(centroid[1])
    features['sensor_centroid_z'] = float(centroid[2])

    mean_pred = np.array([
        time_step_df['pred_x'].mean(),
        time_step_df['pred_y'].mean(),
        time_step_df['pred_z'].mean()
    ])
    features['dist_to_sensor_centroid'] = float(np.sqrt(np.sum((mean_pred - centroid) ** 2)))
    return features


def extract_temporal_context_features(
    trajectory_time_steps: Dict[int, Dict[str, float]],
    current_step: int,
    context_window: int = 5
) -> Tuple[Dict[str, float], bool]:
    context_features = {}
    all_steps = sorted(trajectory_time_steps.keys())
    current_idx = all_steps.index(current_step)

    if len(all_steps) == 0:
        return {}, False

    current_features = trajectory_time_steps[current_step]

    for offset in range(1, context_window + 1):
        idx = max(0, current_idx - offset)
        prev_step = all_steps[idx]
        prev_features = trajectory_time_steps[prev_step]

        for key, val in prev_features.items():
            context_features[f'{key}_prev_{offset}'] = val

        for axis in ['x', 'y', 'z']:
            for stat in ['mean', 'median', 'max', 'min', 'std']:
                ckey = f'pred_{axis}_{stat}'
                if ckey in current_features and ckey in prev_features:
                    context_features[f'diff_{axis}_{stat}_prev_{offset}'] = (
                        current_features[ckey] - prev_features[ckey]
                    )

    for offset in range(1, context_window + 1):
        idx = min(len(all_steps) - 1, current_idx + offset)
        next_step = all_steps[idx]
        next_features = trajectory_time_steps[next_step]

        for key, val in next_features.items():
            context_features[f'{key}_next_{offset}'] = val

        for axis in ['x', 'y', 'z']:
            for stat in ['mean', 'median', 'max', 'min', 'std']:
                ckey = f'pred_{axis}_{stat}'
                if ckey in current_features and ckey in next_features:
                    context_features[f'diff_{axis}_{stat}_next_{offset}'] = (
                        current_features[ckey] - next_features[ckey]
                    )

    for axis in ['x', 'y', 'z']:
        prev_means = [
            trajectory_time_steps[all_steps[max(0, current_idx - i)]][f'pred_{axis}_mean']
            for i in range(context_window + 1)
        ]
        context_features[f'velocity_{axis}_prev'] = float(np.mean(np.diff(prev_means))) if len(prev_means) > 1 else 0.0

        next_means = [
            trajectory_time_steps[all_steps[min(len(all_steps) - 1, current_idx + i)]][f'pred_{axis}_mean']
            for i in range(context_window + 1)
        ]
        context_features[f'velocity_{axis}_next'] = float(np.mean(np.diff(next_means))) if len(next_means) > 1 else 0.0

        context_features[f'acceleration_{axis}'] = (
            context_features[f'velocity_{axis}_next'] - context_features[f'velocity_{axis}_prev']
        )

    return context_features, True


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
        else:
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

        trajectories = df['exp_id'].unique()
        all_samples = []

        for exp_id in trajectories:
            traj_df = df[df['exp_id'] == exp_id]
            time_step_ids = sorted(traj_df['time_step'].unique())

            time_step_features = {}
            time_step_orig_preds = {}

            for ts_id in time_step_ids:
                ts_df = traj_df[traj_df['time_step'] == ts_id]
                features = extract_time_step_features(ts_df)
                sensor_features = compute_sensor_features(ts_df, sensors_df, exp_id)
                features.update(sensor_features)

                time_step_features[ts_id] = features
                time_step_orig_preds[ts_id] = {
                    'orig_x': ts_df['pred_x'].iloc[0],
                    'orig_y': ts_df['pred_y'].iloc[0],
                    'orig_z': ts_df['pred_z'].iloc[0]
                }
                if 'true_x' in ts_df.columns:
                    time_step_orig_preds[ts_id]['true_x'] = ts_df['true_x'].iloc[0]
                    time_step_orig_preds[ts_id]['true_y'] = ts_df['true_y'].iloc[0]
                    time_step_orig_preds[ts_id]['true_z'] = ts_df['true_z'].iloc[0]

            for ts_id in time_step_ids:
                context_features, is_valid = extract_temporal_context_features(
                    time_step_features, ts_id, self.context_window
                )

                sample = {
                    'exp_id': exp_id,
                    'time_step': ts_id
                }
                sample.update(time_step_features[ts_id])
                sample.update(context_features)
                sample.update(time_step_orig_preds[ts_id])

                all_samples.append(sample)

        feature_df = pd.DataFrame(all_samples)
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

        feature_df['post_pred_x'] = self.models['true_x'].predict(X)
        feature_df['post_pred_y'] = self.models['true_y'].predict(X)
        feature_df['post_pred_z'] = self.models['true_z'].predict(X)

        if 'true_x' in feature_df.columns:
            diff_3d = feature_df[['post_pred_x', 'post_pred_y', 'post_pred_z']].values - feature_df[['true_x', 'true_y', 'true_z']].values
            diff_2d = feature_df[['post_pred_x', 'post_pred_y']].values - feature_df[['true_x', 'true_y']].values
            feature_df['post_error_3d_m'] = np.linalg.norm(diff_3d, axis=1) * 1000.0
            feature_df['post_error_2d_m'] = np.linalg.norm(diff_2d, axis=1) * 1000.0

        return feature_df
