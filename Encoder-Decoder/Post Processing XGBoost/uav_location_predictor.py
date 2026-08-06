"""
UAV Location Prediction using Gradient Boosting Regression

This script trains gradient boosting regression models to predict UAV locations (x, y, z)
based on model predictions and sensor positions. It uses extensive feature
engineering including temporal context, prediction statistics, and sensor-based features.

Uses sklearn's HistGradientBoostingRegressor (similar to XGBoost/LightGBM).
If XGBoost is available, you can easily swap it in by modifying the model instantiation.

Author: Claude (Anthropic)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    """Configuration parameters for the UAV location prediction pipeline."""

    # ==========================================================================
    # FILE PATHS - MODIFY THESE TO POINT TO YOUR DATA FILES
    # ==========================================================================
    predictions_file: str = os.path.join(script_dir, "predictions_best.csv")
    sensor_positions_file: str = os.path.join(script_dir, "sensors_positions.csv")
    # ==========================================================================

    # Data processing
    decimal_precision: int = 3  # Round to meters (3 decimal places for km)

    # Temporal context
    context_window: int = 5  # Number of time steps before/after to consider

    # Cross-validation
    n_folds: int = 5
    test_size: float = 0.2  # 80-20 split for 5-fold CV
    random_seed: int = 42

    # Feature importance (can be slow for large datasets)
    compute_feature_importance: bool = False  # Set to True to compute feature importance

    # Worst trajectory removal for additional metrics
    worst_trajectory_pct: float = 0.05  # Remove top 5% worst trajectories for additional analysis

    # HistGradientBoostingRegressor parameters (similar to XGBoost)
    model_params: Dict = None

    def __post_init__(self):
        if self.model_params is None:
            self.model_params = {
                'max_iter': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_seed,
                'early_stopping': False
            }


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load predictions and sensor positions data from CSV files.

    Args:
        config: Configuration object with file paths

    Returns:
        Tuple of (predictions_df, sensors_df)
    """
    print("Loading data...")

    predictions_df = pd.read_csv(config.predictions_file)
    sensors_df = pd.read_csv(config.sensor_positions_file)

    print(f"  Predictions: {len(predictions_df)} rows")
    print(f"  Sensor positions: {len(sensors_df)} rows")
    print(f"  Unique trajectories in predictions: {predictions_df['exp_id'].nunique()}")
    print(f"  Unique trajectories in sensors: {sensors_df['exp_id'].nunique()}")

    return predictions_df, sensors_df


def round_coordinates(df: pd.DataFrame, columns: List[str], precision: int) -> pd.DataFrame:
    """
    Round coordinate columns to specified precision.

    Args:
        df: DataFrame containing coordinate columns
        columns: List of column names to round
        precision: Number of decimal places

    Returns:
        DataFrame with rounded coordinates
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(precision)
    return df


def identify_time_steps(trajectory_df: pd.DataFrame, precision: int) -> pd.DataFrame:
    """
    Identify time steps within a trajectory based on true location changes.

    A new time step is identified when any of true_x, true_y, or true_z changes.

    Args:
        trajectory_df: DataFrame for a single trajectory
        precision: Decimal precision for rounding

    Returns:
        DataFrame with 'time_step' column added
    """
    df = trajectory_df.copy().reset_index(drop=True)

    # Round true coordinates
    df['true_x_rounded'] = df['true_x'].round(precision)
    df['true_y_rounded'] = df['true_y'].round(precision)
    df['true_z_rounded'] = df['true_z'].round(precision)

    # Create a combined key for identifying time steps
    df['location_key'] = (
        df['true_x_rounded'].astype(str) + '_' +
        df['true_y_rounded'].astype(str) + '_' +
        df['true_z_rounded'].astype(str)
    )

    # Assign time step IDs (increments when location changes)
    df['time_step'] = (df['location_key'] != df['location_key'].shift()).cumsum()

    # Clean up temporary columns
    df = df.drop(columns=['true_x_rounded', 'true_y_rounded', 'true_z_rounded', 'location_key'])

    return df


def preprocess_data(predictions_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Preprocess predictions data: round coordinates and identify time steps.

    Args:
        predictions_df: Raw predictions DataFrame
        config: Configuration object

    Returns:
        Preprocessed DataFrame with time steps identified
    """
    print("Preprocessing data...")

    # Round prediction coordinates
    pred_cols = ['pred_x', 'pred_y', 'pred_z']
    true_cols = ['true_x', 'true_y', 'true_z']
    predictions_df = round_coordinates(predictions_df, pred_cols + true_cols, config.decimal_precision)

    # Process each trajectory to identify time steps
    processed_dfs = []
    for exp_id in tqdm(predictions_df['exp_id'].unique()):
        traj_df = predictions_df[predictions_df['exp_id'] == exp_id].copy()
        traj_df = identify_time_steps(traj_df, config.decimal_precision)
        processed_dfs.append(traj_df)

    result_df = pd.concat(processed_dfs, ignore_index=True)

    n_time_steps = result_df.groupby('exp_id')['time_step'].max().sum()
    print(f"  Total time steps identified: {n_time_steps}")

    return result_df


# =============================================================================
# FEATURE ENGINEERING - PREDICTION STATISTICS
# =============================================================================

def compute_trend_features(values: np.ndarray) -> Dict[str, float]:
    """
    Compute trend features for a sequence of values.

    Args:
        values: Array of prediction values within a time step

    Returns:
        Dictionary of trend features
    """
    features = {}

    if len(values) < 2:
        # Not enough data for trend analysis
        features['trend_direction'] = 0  # 0 = stable
        features['trend_strength'] = 0
        features['n_fluctuations'] = 0
        features['longest_up_streak'] = 0
        features['longest_down_streak'] = 0
        features['final_vs_initial'] = 0
        features['monotonic'] = 1
        return features

    # Calculate differences between consecutive values
    diffs = np.diff(values)

    # Overall trend direction: 1 = up, -1 = down, 0 = stable
    total_change = values[-1] - values[0]
    features['trend_direction'] = np.sign(total_change)
    features['trend_strength'] = abs(total_change)
    features['final_vs_initial'] = total_change

    # Count fluctuations (sign changes in differences)
    if len(diffs) > 1:
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        features['n_fluctuations'] = sign_changes
    else:
        features['n_fluctuations'] = 0

    # Longest up/down streaks
    up_streak = 0
    down_streak = 0
    current_up = 0
    current_down = 0

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
            current_up = 0
            current_down = 0

    features['longest_up_streak'] = up_streak
    features['longest_down_streak'] = down_streak

    # Check if monotonic
    is_monotonic_up = np.all(diffs >= 0)
    is_monotonic_down = np.all(diffs <= 0)
    features['monotonic'] = 1 if (is_monotonic_up or is_monotonic_down) else 0

    return features


def compute_distribution_features(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Compute distribution statistics for a set of values.

    Args:
        values: Array of values
        prefix: Prefix for feature names

    Returns:
        Dictionary of distribution features
    """
    features = {}

    if len(values) == 0:
        # Return zeros for empty arrays
        for stat in ['mean', 'std', 'min', 'max', 'median', 'range', 'iqr', 'skew', 'n_samples']:
            features[f'{prefix}_{stat}'] = 0
        return features

    features[f'{prefix}_mean'] = np.mean(values)
    features[f'{prefix}_std'] = np.std(values) if len(values) > 1 else 0
    features[f'{prefix}_min'] = np.min(values)
    features[f'{prefix}_max'] = np.max(values)
    features[f'{prefix}_median'] = np.median(values)
    features[f'{prefix}_range'] = np.max(values) - np.min(values)
    features[f'{prefix}_n_samples'] = len(values)

    # Interquartile range
    if len(values) >= 4:
        q75, q25 = np.percentile(values, [75, 25])
        features[f'{prefix}_iqr'] = q75 - q25
    else:
        features[f'{prefix}_iqr'] = 0

    # Skewness (simplified)
    if len(values) > 2 and features[f'{prefix}_std'] > 0:
        features[f'{prefix}_skew'] = (
            (np.mean(values) - np.median(values)) / features[f'{prefix}_std']
        )
    else:
        features[f'{prefix}_skew'] = 0

    return features


def extract_time_step_features(time_step_df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract features from predictions within a single time step.

    Args:
        time_step_df: DataFrame containing all predictions for one time step

    Returns:
        Dictionary of features for this time step
    """
    features = {}

    for axis in ['x', 'y', 'z']:
        pred_col = f'pred_{axis}'
        values = time_step_df[pred_col].values

        # Distribution features
        dist_features = compute_distribution_features(values, f'pred_{axis}')
        features.update(dist_features)

        # Trend features
        trend_features = compute_trend_features(values)
        for key, value in trend_features.items():
            features[f'pred_{axis}_{key}'] = value

    # Cross-axis features
    pred_x = time_step_df['pred_x'].values
    pred_y = time_step_df['pred_y'].values
    pred_z = time_step_df['pred_z'].values

    # Correlation between axes (if enough samples)
    if len(pred_x) > 2:
        features['corr_xy'] = np.corrcoef(pred_x, pred_y)[0, 1] if np.std(pred_x) > 0 and np.std(pred_y) > 0 else 0
        features['corr_xz'] = np.corrcoef(pred_x, pred_z)[0, 1] if np.std(pred_x) > 0 and np.std(pred_z) > 0 else 0
        features['corr_yz'] = np.corrcoef(pred_y, pred_z)[0, 1] if np.std(pred_y) > 0 and np.std(pred_z) > 0 else 0
    else:
        features['corr_xy'] = 0
        features['corr_xz'] = 0
        features['corr_yz'] = 0

    # Handle NaN correlations
    for key in ['corr_xy', 'corr_xz', 'corr_yz']:
        if np.isnan(features[key]):
            features[key] = 0

    return features


# =============================================================================
# FEATURE ENGINEERING - TEMPORAL CONTEXT
# =============================================================================

def compute_temporal_difference_features(
    current_features: Dict[str, float],
    context_features: Dict[str, float],
    suffix: str
) -> Dict[str, float]:
    """
    Compute difference features between current and context time steps.

    Args:
        current_features: Features from current time step
        context_features: Features from context time step
        suffix: Suffix for feature names (e.g., 'prev_1', 'next_2')

    Returns:
        Dictionary of difference features
    """
    diff_features = {}

    # Key statistics to compare
    stats_to_compare = ['mean', 'median', 'max', 'min', 'std']

    for axis in ['x', 'y', 'z']:
        for stat in stats_to_compare:
            current_key = f'pred_{axis}_{stat}'
            if current_key in current_features and current_key in context_features:
                diff_features[f'diff_{axis}_{stat}_{suffix}'] = (
                    current_features[current_key] - context_features[current_key]
                )

    return diff_features


def extract_temporal_context_features(
    trajectory_time_steps: Dict[int, Dict[str, float]],
    current_step: int,
    context_window: int
) -> Tuple[Dict[str, float], bool]:
    """
    Extract features from surrounding time steps.

    Args:
        trajectory_time_steps: Dictionary mapping time step ID to features
        current_step: Current time step ID
        context_window: Number of steps to look before/after

    Returns:
        Tuple of (context features dict, is_valid flag)
    """
    context_features = {}
    all_steps = sorted(trajectory_time_steps.keys())
    current_idx = all_steps.index(current_step)

    # Check if we have enough context
    if current_idx < context_window or current_idx >= len(all_steps) - context_window:
        return {}, False

    current_features = trajectory_time_steps[current_step]

    # Features from previous time steps
    for offset in range(1, context_window + 1):
        prev_step = all_steps[current_idx - offset]
        prev_features = trajectory_time_steps[prev_step]

        # Add raw features from previous step
        for key, value in prev_features.items():
            context_features[f'{key}_prev_{offset}'] = value

        # Add difference features
        diff_features = compute_temporal_difference_features(
            current_features, prev_features, f'prev_{offset}'
        )
        context_features.update(diff_features)

    # Features from next time steps
    for offset in range(1, context_window + 1):
        next_step = all_steps[current_idx + offset]
        next_features = trajectory_time_steps[next_step]

        # Add raw features from next step
        for key, value in next_features.items():
            context_features[f'{key}_next_{offset}'] = value

        # Add difference features
        diff_features = compute_temporal_difference_features(
            current_features, next_features, f'next_{offset}'
        )
        context_features.update(diff_features)

    # Aggregate context features
    # Velocity/acceleration-like features (change between consecutive context steps)
    for axis in ['x', 'y', 'z']:
        # Average change rate in previous steps
        prev_means = [
            trajectory_time_steps[all_steps[current_idx - i]][f'pred_{axis}_mean']
            for i in range(context_window + 1)
        ]
        context_features[f'velocity_{axis}_prev'] = np.mean(np.diff(prev_means))

        # Average change rate in next steps
        next_means = [
            trajectory_time_steps[all_steps[current_idx + i]][f'pred_{axis}_mean']
            for i in range(context_window + 1)
        ]
        context_features[f'velocity_{axis}_next'] = np.mean(np.diff(next_means))

        # Acceleration (change in velocity)
        context_features[f'acceleration_{axis}'] = (
            context_features[f'velocity_{axis}_next'] - context_features[f'velocity_{axis}_prev']
        )

    return context_features, True


# =============================================================================
# FEATURE ENGINEERING - SENSOR FEATURES
# =============================================================================

def compute_sensor_features(
    time_step_df: pd.DataFrame,
    sensors_df: pd.DataFrame,
    exp_id: int
) -> Dict[str, float]:
    """
    Compute features based on sensor positions and predictions.

    Args:
        time_step_df: Predictions for current time step
        sensors_df: Sensor positions DataFrame
        exp_id: Experiment/trajectory ID

    Returns:
        Dictionary of sensor-based features
    """
    features = {}

    # Get sensors for this experiment
    exp_sensors = sensors_df[sensors_df['exp_id'] == exp_id]

    if len(exp_sensors) == 0:
        # No sensor data available
        for stat in ['mean', 'std', 'min', 'max', 'median']:
            features[f'sensor_dist_{stat}'] = 0
            features[f'sensor_inter_dist_{stat}'] = 0
        features['n_sensors'] = 0
        return features

    sensor_positions = exp_sensors[['x', 'y', 'z']].values
    features['n_sensors'] = len(sensor_positions)

    # For each prediction, compute distance to each sensor
    all_distances = []

    for _, pred_row in time_step_df.iterrows():
        pred_pos = np.array([pred_row['pred_x'], pred_row['pred_y'], pred_row['pred_z']])

        # Distance from this prediction to each sensor
        distances = np.sqrt(np.sum((sensor_positions - pred_pos) ** 2, axis=1))
        all_distances.extend(distances)

    all_distances = np.array(all_distances)

    # Aggregate distance statistics
    dist_features = compute_distribution_features(all_distances, 'sensor_dist')
    features.update(dist_features)

    # Inter-sensor distances (distances between sensors)
    if len(sensor_positions) > 1:
        inter_distances = []
        for i in range(len(sensor_positions)):
            for j in range(i + 1, len(sensor_positions)):
                dist = np.sqrt(np.sum((sensor_positions[i] - sensor_positions[j]) ** 2))
                inter_distances.append(dist)

        inter_dist_features = compute_distribution_features(
            np.array(inter_distances), 'sensor_inter_dist'
        )
        features.update(inter_dist_features)
    else:
        for stat in ['mean', 'std', 'min', 'max', 'median', 'range', 'iqr', 'skew', 'n_samples']:
            features[f'sensor_inter_dist_{stat}'] = 0

    # Sensor spread (convex hull-like features)
    features['sensor_spread_x'] = np.max(sensor_positions[:, 0]) - np.min(sensor_positions[:, 0])
    features['sensor_spread_y'] = np.max(sensor_positions[:, 1]) - np.min(sensor_positions[:, 1])
    features['sensor_spread_z'] = np.max(sensor_positions[:, 2]) - np.min(sensor_positions[:, 2])

    # Sensor centroid
    centroid = np.mean(sensor_positions, axis=0)
    features['sensor_centroid_x'] = centroid[0]
    features['sensor_centroid_y'] = centroid[1]
    features['sensor_centroid_z'] = centroid[2]

    # Distance from mean prediction to sensor centroid
    mean_pred = np.array([
        time_step_df['pred_x'].mean(),
        time_step_df['pred_y'].mean(),
        time_step_df['pred_z'].mean()
    ])
    features['dist_to_sensor_centroid'] = np.sqrt(np.sum((mean_pred - centroid) ** 2))

    return features


# =============================================================================
# MAIN FEATURE EXTRACTION PIPELINE
# =============================================================================

def build_feature_matrix(
    predictions_df: pd.DataFrame,
    sensors_df: pd.DataFrame,
    config: Config
) -> pd.DataFrame:
    """
    Build the complete feature matrix for all valid time steps.
    """
    print("Building feature matrix...")

    dfs = []          # Store dataframe chunks
    all_samples = []  # Batch accumulator
    BATCH_SIZE = 50000  # Number of rows before converting to DataFrame

    trajectories = predictions_df['exp_id'].unique()
    total_time_steps = 0
    valid_time_steps = 0

    for exp_id in tqdm(trajectories, desc="Trajectories"):
        traj_df = predictions_df[predictions_df['exp_id'] == exp_id]

        # First pass: extract features for each time step
        time_step_features = {}
        time_step_targets = {}
        time_step_ids = traj_df['time_step'].unique()
        total_time_steps += len(time_step_ids)

        for ts_id in time_step_ids:
            ts_df = traj_df[traj_df['time_step'] == ts_id]

            # Extract basic features for this time step
            features = extract_time_step_features(ts_df)

            # Extract sensor features
            sensor_features = compute_sensor_features(ts_df, sensors_df, exp_id)
            features.update(sensor_features)

            time_step_features[ts_id] = features

            # Store targets (true locations - same for all predictions in time step)
            time_step_targets[ts_id] = {
                'true_x': ts_df['true_x'].iloc[0],
                'true_y': ts_df['true_y'].iloc[0],
                'true_z': ts_df['true_z'].iloc[0]
            }

        # Second pass: add temporal context features
        for ts_id in time_step_ids:
            context_features, is_valid = extract_temporal_context_features(
                time_step_features, ts_id, config.context_window
            )

            if not is_valid:
                continue  # Skip edge time steps without enough context

            valid_time_steps += 1

            # Combine all features
            sample = {
                'exp_id': exp_id,
                'time_step': ts_id
            }
            sample.update(time_step_features[ts_id])
            sample.update(context_features)
            sample.update(time_step_targets[ts_id])

            all_samples.append(sample)

            # --- MEMORY OPTIMIZATION: Convert in chunks ---
            if len(all_samples) >= BATCH_SIZE:
                chunk_df = pd.DataFrame(all_samples)
                # Downcast 64-bit floats to 32-bit floats to halve memory usage
                float_cols = chunk_df.select_dtypes(include=['float64']).columns
                chunk_df[float_cols] = chunk_df[float_cols].astype('float32')
                dfs.append(chunk_df)
                all_samples = []

    # Process any remaining samples after the loop finishes
    if all_samples:
        chunk_df = pd.DataFrame(all_samples)
        float_cols = chunk_df.select_dtypes(include=['float64']).columns
        chunk_df[float_cols] = chunk_df[float_cols].astype('float32')
        dfs.append(chunk_df)
        all_samples = []

    print(f"  Total time steps: {total_time_steps}")
    print(f"  Valid time steps (with context): {valid_time_steps}")
    print(f"  Excluded edge time steps: {total_time_steps - valid_time_steps}")

    print("  Concatenating chunks into final feature matrix...")
    feature_df = pd.concat(dfs, ignore_index=True)
    
    print(f"  Feature matrix shape: {feature_df.shape}")
    print(f"  Number of features: {feature_df.shape[1] - 5}")  # Subtract id columns and targets

    return feature_df


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding identifiers and targets).

    Args:
        df: Feature DataFrame

    Returns:
        List of feature column names
    """
    exclude_cols = ['exp_id', 'time_step', 'true_x', 'true_y', 'true_z']
    return [col for col in df.columns if col not in exclude_cols]


def train_and_evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config
) -> Dict[str, any]:
    """
    Train models and evaluate on a single fold.

    Args:
        train_df: Training data
        test_df: Test data
        config: Configuration object

    Returns:
        Dictionary containing predictions and metrics
    """
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    results = {
        'test_df': test_df.copy(),
        'predictions': {},
        'models': {}
    }

    # Train a separate model for each target
    for target in tqdm(['true_x', 'true_y', 'true_z'], desc="Training models", leave=False):
        y_train = train_df[target].values

        model = HistGradientBoostingRegressor(**config.model_params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        results['predictions'][target] = predictions
        results['models'][target] = model

    return results


def compute_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.

    Args:
        true_values: Ground truth values
        pred_values: Predicted values

    Returns:
        Dictionary of metrics
    """
    errors = true_values - pred_values

    return {
        'mse': np.mean(errors ** 2),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mae': np.mean(np.abs(errors)),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors)
    }


def compute_3d_distance(
    true_x: np.ndarray, true_y: np.ndarray, true_z: np.ndarray,
    pred_x: np.ndarray, pred_y: np.ndarray, pred_z: np.ndarray
) -> np.ndarray:
    """
    Compute 3D Euclidean distance between true and predicted locations.

    Args:
        true_x, true_y, true_z: True coordinates
        pred_x, pred_y, pred_z: Predicted coordinates

    Returns:
        Array of distances
    """
    return np.sqrt(
        (true_x - pred_x) ** 2 +
        (true_y - pred_y) ** 2 +
        (true_z - pred_z) ** 2
    )


def compute_2d_distance(
    true_x: np.ndarray, true_y: np.ndarray,
    pred_x: np.ndarray, pred_y: np.ndarray
) -> np.ndarray:
    """
    Compute 2D Euclidean distance (xy) between true and predicted locations.

    Args:
        true_x, true_y: True coordinates
        pred_x, pred_y: Predicted coordinates

    Returns:
        Array of distances
    """
    return np.sqrt(
        (true_x - pred_x) ** 2 +
        (true_y - pred_y) ** 2
    )


def run_cross_validation(
    feature_df: pd.DataFrame,
    config: Config
) -> Dict[str, any]:
    """
    Run k-fold cross-validation at the trajectory level.

    Args:
        feature_df: Complete feature matrix
        config: Configuration object

    Returns:
        Dictionary containing all fold results
    """
    print(f"\nRunning {config.n_folds}-fold cross-validation...")

    # Get unique trajectories
    trajectories = feature_df['exp_id'].unique()
    np.random.seed(config.random_seed)
    np.random.shuffle(trajectories)

    # Create folds at trajectory level
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)

    all_fold_results = []

    for fold_idx, (train_traj_idx, test_traj_idx) in enumerate(tqdm(kf.split(trajectories), total=config.n_folds, desc="Folds")):
        print(f"\n  Fold {fold_idx + 1}/{config.n_folds}")

        train_trajectories = trajectories[train_traj_idx]
        test_trajectories = trajectories[test_traj_idx]

        train_df = feature_df[feature_df['exp_id'].isin(train_trajectories)]
        test_df = feature_df[feature_df['exp_id'].isin(test_trajectories)]

        print(f"    Train: {len(train_trajectories)} trajectories, {len(train_df)} samples")
        print(f"    Test: {len(test_trajectories)} trajectories, {len(test_df)} samples")

        fold_results = train_and_evaluate_fold(train_df, test_df, config)
        fold_results['fold_idx'] = fold_idx
        fold_results['train_trajectories'] = train_trajectories
        fold_results['test_trajectories'] = test_trajectories

        all_fold_results.append(fold_results)

    return all_fold_results


# =============================================================================
# RESULTS REPORTING
# =============================================================================

def print_sample_level_metrics(
    true_x: np.ndarray, true_y: np.ndarray, true_z: np.ndarray,
    pred_x: np.ndarray, pred_y: np.ndarray, pred_z: np.ndarray,
    orig_pred_x: np.ndarray, orig_pred_y: np.ndarray, orig_pred_z: np.ndarray,
    label: str = ""
):
    """Print sample-level metrics for both models."""

    print(f"\n*** Gradient Boosting Model Results{label} ***")

    # Per-axis metrics
    for axis, true_vals, pred_vals in [
        ('X', true_x, pred_x),
        ('Y', true_y, pred_y),
        ('Z', true_z, pred_z)
    ]:
        metrics = compute_metrics(true_vals, pred_vals)
        print(f"\n  {axis}-axis:")
        print(f"    MSE:  {metrics['mse']:.6f}")
        print(f"    RMSE: {metrics['rmse']:.6f}")
        print(f"    MAE:  {metrics['mae']:.6f}")

    # 3D distance
    distances_gb = compute_3d_distance(true_x, true_y, true_z, pred_x, pred_y, pred_z)
    print(f"\n  3D Euclidean Distance:")
    print(f"    Mean:   {np.mean(distances_gb):.6f}")
    print(f"    Std:    {np.std(distances_gb):.6f}")
    print(f"    Median: {np.median(distances_gb):.6f}")
    print(f"    Min:    {np.min(distances_gb):.6f}")
    print(f"    Max:    {np.max(distances_gb):.6f}")

    # 2D distance
    distances_gb_2d = compute_2d_distance(true_x, true_y, pred_x, pred_y)
    print(f"\n  2D Euclidean Distance (xy):")
    print(f"    Mean:   {np.mean(distances_gb_2d):.6f}")
    print(f"    Std:    {np.std(distances_gb_2d):.6f}")
    print(f"    Median: {np.median(distances_gb_2d):.6f}")
    print(f"    Min:    {np.min(distances_gb_2d):.6f}")
    print(f"    Max:    {np.max(distances_gb_2d):.6f}")

    # Comparison with original model
    print(f"\n*** Original Model Results (mean of predictions per time step){label} ***")

    for axis, true_vals, orig_vals in [
        ('X', true_x, orig_pred_x),
        ('Y', true_y, orig_pred_y),
        ('Z', true_z, orig_pred_z)
    ]:
        metrics = compute_metrics(true_vals, orig_vals)
        print(f"\n  {axis}-axis:")
        print(f"    MSE:  {metrics['mse']:.6f}")
        print(f"    RMSE: {metrics['rmse']:.6f}")
        print(f"    MAE:  {metrics['mae']:.6f}")

    distances_orig = compute_3d_distance(true_x, true_y, true_z, orig_pred_x, orig_pred_y, orig_pred_z)
    print(f"\n  3D Euclidean Distance:")
    print(f"    Mean:   {np.mean(distances_orig):.6f}")
    print(f"    Std:    {np.std(distances_orig):.6f}")
    print(f"    Median: {np.median(distances_orig):.6f}")

    distances_orig_2d = compute_2d_distance(true_x, true_y, orig_pred_x, orig_pred_y)
    print(f"\n  2D Euclidean Distance (xy):")
    print(f"    Mean:   {np.mean(distances_orig_2d):.6f}")
    print(f"    Std:    {np.std(distances_orig_2d):.6f}")
    print(f"    Median: {np.median(distances_orig_2d):.6f}")

    # Improvement summary
    print(f"\n*** Improvement (Gradient Boosting vs Original Baseline){label} ***")
    improvement_distance = (np.mean(distances_orig) - np.mean(distances_gb)) / np.mean(distances_orig) * 100
    print(f"  3D Distance Improvement: {improvement_distance:.2f}%")
    improvement_distance_2d = (np.mean(distances_orig_2d) - np.mean(distances_gb_2d)) / np.mean(distances_orig_2d) * 100
    print(f"  2D Distance (xy) Improvement: {improvement_distance_2d:.2f}%")
    print(f"    (Positive = Gradient Boosting is better, Negative = Original is better)")

    for axis, true_vals, pred_vals, orig_vals in [
        ('X', true_x, pred_x, orig_pred_x),
        ('Y', true_y, pred_y, orig_pred_y),
        ('Z', true_z, pred_z, orig_pred_z)
    ]:
        mse_gb = np.mean((true_vals - pred_vals) ** 2)
        mse_orig = np.mean((true_vals - orig_vals) ** 2)
        improvement = (mse_orig - mse_gb) / mse_orig * 100
        print(f"  {axis}-axis MSE Improvement: {improvement:.2f}%")


def print_trajectory_level_metrics(
    true_x: np.ndarray, true_y: np.ndarray, true_z: np.ndarray,
    pred_x: np.ndarray, pred_y: np.ndarray, pred_z: np.ndarray,
    orig_pred_x: np.ndarray, orig_pred_y: np.ndarray, orig_pred_z: np.ndarray,
    exp_ids: np.ndarray,
    label: str = ""
) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Print trajectory-level metrics for both models.

    Returns:
        Tuple of (traj_metrics_gb, traj_metrics_orig, unique_trajectories)
    """
    true_x = np.asarray(true_x).ravel()
    true_y = np.asarray(true_y).ravel()
    true_z = np.asarray(true_z).ravel()
    pred_x = np.asarray(pred_x).ravel()
    pred_y = np.asarray(pred_y).ravel()
    pred_z = np.asarray(pred_z).ravel()
    orig_pred_x = np.asarray(orig_pred_x).ravel()
    orig_pred_y = np.asarray(orig_pred_y).ravel()
    orig_pred_z = np.asarray(orig_pred_z).ravel()
    exp_ids = np.asarray(exp_ids).ravel()

    unique_trajectories = np.unique(exp_ids)

    traj_metrics_gb = {axis: [] for axis in ['X', 'Y', 'Z', 'distance', 'distance_2d']}
    traj_metrics_orig = {axis: [] for axis in ['X', 'Y', 'Z', 'distance', 'distance_2d']}
    traj_ids_list = []

    for exp_id in tqdm(unique_trajectories, desc="Calculating detailed metrics", leave=False):
        mask = exp_ids == exp_id
        traj_ids_list.append(exp_id)

        # Gradient Boosting metrics for this trajectory
        for axis, true_vals, pred_vals in [
            ('X', true_x[mask], pred_x[mask]),
            ('Y', true_y[mask], pred_y[mask]),
            ('Z', true_z[mask], pred_z[mask])
        ]:
            mse = np.mean((true_vals - pred_vals) ** 2)
            traj_metrics_gb[axis].append(mse)

        dist_gb = compute_3d_distance(
            true_x[mask], true_y[mask], true_z[mask],
            pred_x[mask], pred_y[mask], pred_z[mask]
        )
        traj_metrics_gb['distance'].append(np.mean(dist_gb))

        dist_gb_2d = compute_2d_distance(
            true_x[mask], true_y[mask],
            pred_x[mask], pred_y[mask]
        )
        traj_metrics_gb['distance_2d'].append(np.mean(dist_gb_2d))

        # Original model metrics for this trajectory
        for axis, true_vals, orig_vals in [
            ('X', true_x[mask], orig_pred_x[mask]),
            ('Y', true_y[mask], orig_pred_y[mask]),
            ('Z', true_z[mask], orig_pred_z[mask])
        ]:
            mse = np.mean((true_vals - orig_vals) ** 2)
            traj_metrics_orig[axis].append(mse)

        dist_orig = compute_3d_distance(
            true_x[mask], true_y[mask], true_z[mask],
            orig_pred_x[mask], orig_pred_y[mask], orig_pred_z[mask]
        )
        traj_metrics_orig['distance'].append(np.mean(dist_orig))

        dist_orig_2d = compute_2d_distance(
            true_x[mask], true_y[mask],
            orig_pred_x[mask], orig_pred_y[mask]
        )
        traj_metrics_orig['distance_2d'].append(np.mean(dist_orig_2d))

    # Convert to numpy arrays
    for key in traj_metrics_gb:
        traj_metrics_gb[key] = np.array(traj_metrics_gb[key])
        traj_metrics_orig[key] = np.array(traj_metrics_orig[key])

    print(f"\n*** Gradient Boosting Model Results{label} ***")
    for metric_name in ['X', 'Y', 'Z']:
        values = traj_metrics_gb[metric_name]
        print(f"\n  {metric_name}-axis MSE:")
        print(f"    Mean across trajectories: {np.mean(values):.6f}")
        print(f"    Std across trajectories:  {np.std(values):.6f}")

    print(f"\n  3D Euclidean Distance:")
    print(f"    Mean across trajectories: {np.mean(traj_metrics_gb['distance']):.6f}")
    print(f"    Std across trajectories:  {np.std(traj_metrics_gb['distance']):.6f}")

    print(f"\n  2D Euclidean Distance (xy):")
    print(f"    Mean across trajectories: {np.mean(traj_metrics_gb['distance_2d']):.6f}")
    print(f"    Std across trajectories:  {np.std(traj_metrics_gb['distance_2d']):.6f}")

    print(f"\n*** Original Model Results{label} ***")
    for metric_name in ['X', 'Y', 'Z']:
        values = traj_metrics_orig[metric_name]
        print(f"\n  {metric_name}-axis MSE:")
        print(f"    Mean across trajectories: {np.mean(values):.6f}")
        print(f"    Std across trajectories:  {np.std(values):.6f}")

    print(f"\n  3D Euclidean Distance:")
    print(f"    Mean across trajectories: {np.mean(traj_metrics_orig['distance']):.6f}")
    print(f"    Std across trajectories:  {np.std(traj_metrics_orig['distance']):.6f}")

    print(f"\n  2D Euclidean Distance (xy):")
    print(f"    Mean across trajectories: {np.mean(traj_metrics_orig['distance_2d']):.6f}")
    print(f"    Std across trajectories:  {np.std(traj_metrics_orig['distance_2d']):.6f}")

    return traj_metrics_gb, traj_metrics_orig, np.array(traj_ids_list)


def aggregate_and_report_results(fold_results: List[Dict], config: Config):
    """
    Aggregate results across folds and print comprehensive report.

    Args:
        fold_results: List of results from each fold
        config: Configuration object
    """
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Collect all predictions across folds
    all_true_x, all_true_y, all_true_z = [], [], []
    all_pred_x, all_pred_y, all_pred_z = [], [], []
    all_exp_ids = []

    # Also collect original model predictions for comparison
    all_orig_pred_x, all_orig_pred_y, all_orig_pred_z = [], [], []

    for fold_result in fold_results:
        test_df = fold_result['test_df']

        all_true_x.extend(test_df['true_x'].values)
        all_true_y.extend(test_df['true_y'].values)
        all_true_z.extend(test_df['true_z'].values)

        all_pred_x.extend(fold_result['predictions']['true_x'])
        all_pred_y.extend(fold_result['predictions']['true_y'])
        all_pred_z.extend(fold_result['predictions']['true_z'])

        all_exp_ids.extend(test_df['exp_id'].values)

        # Original predictions (mean of predictions in each time step)
        all_orig_pred_x.extend(test_df['pred_x_mean'].values)
        all_orig_pred_y.extend(test_df['pred_y_mean'].values)
        all_orig_pred_z.extend(test_df['pred_z_mean'].values)

    # Convert to numpy arrays
    all_true_x = np.array(all_true_x)
    all_true_y = np.array(all_true_y)
    all_true_z = np.array(all_true_z)
    all_pred_x = np.array(all_pred_x)
    all_pred_y = np.array(all_pred_y)
    all_pred_z = np.array(all_pred_z)
    all_orig_pred_x = np.array(all_orig_pred_x)
    all_orig_pred_y = np.array(all_orig_pred_y)
    all_orig_pred_z = np.array(all_orig_pred_z)
    all_exp_ids = np.array(all_exp_ids)

    # Save postprocessed predictions for external analysis
    results_df = pd.DataFrame({
        'exp_id': all_exp_ids,
        'true_x': all_true_x,
        'true_y': all_true_y,
        'true_z': all_true_z,
        'pred_x': all_pred_x,  # Postprocessed by Gradient Boosting
        'pred_y': all_pred_y,
        'pred_z': all_pred_z,
        'orig_pred_x': all_orig_pred_x,  # Original Transformer predictions
        'orig_pred_y': all_orig_pred_y,
        'orig_pred_z': all_orig_pred_z
    })
    output_predictions_file = os.path.join(script_dir, "postprocessed_predictions.csv")
    results_df.to_csv(output_predictions_file, index=False)
    print(f"\nSaved postprocessed predictions to {output_predictions_file}")

    # ==========================================================================
    # FULL RESULTS (ALL TRAJECTORIES)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FULL RESULTS (ALL TRAJECTORIES)")
    print("=" * 80)

    # Sample-level metrics
    print("\n" + "-" * 80)
    print("SAMPLE-LEVEL METRICS (averaged across all samples)")
    print("-" * 80)

    print_sample_level_metrics(
        all_true_x, all_true_y, all_true_z,
        all_pred_x, all_pred_y, all_pred_z,
        all_orig_pred_x, all_orig_pred_y, all_orig_pred_z
    )

    # Trajectory-level metrics
    print("\n" + "-" * 80)
    print("TRAJECTORY-LEVEL METRICS (averaged per trajectory, then across trajectories)")
    print("-" * 80)

    traj_metrics_gb, traj_metrics_orig, traj_ids = print_trajectory_level_metrics(
        all_true_x, all_true_y, all_true_z,
        all_pred_x, all_pred_y, all_pred_z,
        all_orig_pred_x, all_orig_pred_y, all_orig_pred_z,
        all_exp_ids
    )

    # ==========================================================================
    # RESULTS AFTER REMOVING WORST TRAJECTORIES
    # ==========================================================================
    n_trajectories = len(traj_ids)
    n_to_remove = max(1, int(np.ceil(n_trajectories * config.worst_trajectory_pct)))

    print("\n" + "=" * 80)
    print(f"RESULTS AFTER REMOVING {config.worst_trajectory_pct*100:.1f}% WORST TRAJECTORIES")
    print(f"(Removing {n_to_remove} of {n_trajectories} trajectories based on Gradient Boosting 3D distance)")
    print("=" * 80)

    # Identify worst trajectories based on Gradient Boosting model's average 3D distance
    worst_traj_indices = np.argsort(traj_metrics_gb['distance'])[-n_to_remove:]
    worst_traj_ids = traj_ids[worst_traj_indices]

    print(f"\nWorst trajectory IDs removed: {worst_traj_ids.tolist()}")
    print(f"Their average 3D distances: {[f'{d:.4f}' for d in traj_metrics_gb['distance'][worst_traj_indices]]}")
    print(f"Their average 2D distances (xy): {[f'{d:.4f}' for d in traj_metrics_gb['distance_2d'][worst_traj_indices]]}")

    # Create mask to exclude worst trajectories
    keep_mask = ~np.isin(all_exp_ids, worst_traj_ids)

    # Filter data
    filtered_true_x = all_true_x[keep_mask]
    filtered_true_y = all_true_y[keep_mask]
    filtered_true_z = all_true_z[keep_mask]
    filtered_pred_x = all_pred_x[keep_mask]
    filtered_pred_y = all_pred_y[keep_mask]
    filtered_pred_z = all_pred_z[keep_mask]
    filtered_orig_pred_x = all_orig_pred_x[keep_mask]
    filtered_orig_pred_y = all_orig_pred_y[keep_mask]
    filtered_orig_pred_z = all_orig_pred_z[keep_mask]
    filtered_exp_ids = all_exp_ids[keep_mask]

    print(f"\nSamples remaining: {len(filtered_true_x)} (removed {len(all_true_x) - len(filtered_true_x)})")
    print(f"Trajectories remaining: {n_trajectories - n_to_remove}")

    # Sample-level metrics (filtered)
    print("\n" + "-" * 80)
    print("SAMPLE-LEVEL METRICS (after removing worst trajectories)")
    print("-" * 80)

    print_sample_level_metrics(
        filtered_true_x, filtered_true_y, filtered_true_z,
        filtered_pred_x, filtered_pred_y, filtered_pred_z,
        filtered_orig_pred_x, filtered_orig_pred_y, filtered_orig_pred_z,
        label=" (filtered)"
    )

    # Trajectory-level metrics (filtered)
    print("\n" + "-" * 80)
    print("TRAJECTORY-LEVEL METRICS (after removing worst trajectories)")
    print("-" * 80)

    print_trajectory_level_metrics(
        filtered_true_x, filtered_true_y, filtered_true_z,
        filtered_pred_x, filtered_pred_y, filtered_pred_z,
        filtered_orig_pred_x, filtered_orig_pred_y, filtered_orig_pred_z,
        filtered_exp_ids,
        label=" (filtered)"
    )

    # ==========================================================================
    # PER-FOLD BREAKDOWN
    # ==========================================================================
    print("\n" + "-" * 80)
    print("PER-FOLD BREAKDOWN")
    print("-" * 80)

    for fold_idx, fold_result in enumerate(fold_results):
        test_df = fold_result['test_df']

        true_x = test_df['true_x'].values
        true_y = test_df['true_y'].values
        true_z = test_df['true_z'].values

        pred_x = fold_result['predictions']['true_x']
        pred_y = fold_result['predictions']['true_y']
        pred_z = fold_result['predictions']['true_z']

        distances = compute_3d_distance(true_x, true_y, true_z, pred_x, pred_y, pred_z)
        distances_2d = compute_2d_distance(true_x, true_y, pred_x, pred_y)

        print(f"\n  Fold {fold_idx + 1}:")
        print(f"    Test trajectories: {len(fold_result['test_trajectories'])}")
        print(f"    Test samples: {len(test_df)}")
        print(f"    MSE X: {np.mean((true_x - pred_x)**2):.6f}")
        print(f"    MSE Y: {np.mean((true_y - pred_y)**2):.6f}")
        print(f"    MSE Z: {np.mean((true_z - pred_z)**2):.6f}")
        print(f"    Mean 3D Distance: {np.mean(distances):.6f}")
        print(f"    Mean 2D Distance (xy): {np.mean(distances_2d):.6f}")

    # Feature importance (from last fold's models)
    if config.compute_feature_importance:
        print("\n" + "-" * 80)
        print("TOP 15 MOST IMPORTANT FEATURES (from last fold, using permutation importance)")
        print("-" * 80)

        feature_cols = get_feature_columns(fold_results[-1]['test_df'])
        last_fold_test_df = fold_results[-1]['test_df']
        X_test_importance = last_fold_test_df[feature_cols].values

        for target in tqdm(['true_x', 'true_y', 'true_z'], desc="Feature importance"):
            model = fold_results[-1]['models'][target]
            y_test_importance = last_fold_test_df[target].values

            print(f"\n  {target.replace('true_', '').upper()}-axis model:")

            try:
                # Compute permutation importance
                perm_importance = permutation_importance(
                    model, X_test_importance, y_test_importance,
                    n_repeats=10, random_state=config.random_seed, n_jobs=-1
                )

                # Get top features
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': perm_importance.importances_mean,
                    'std': perm_importance.importances_std
                }).sort_values('importance', ascending=False)

                print(f"    {'Feature':<50} {'Importance':>12} {'Std':>10}")
                print(f"    {'-'*50} {'-'*12} {'-'*10}")

                for _, row in importance_df.head(15).iterrows():
                    feat_name = row['feature'][:48] + '..' if len(row['feature']) > 50 else row['feature']
                    print(f"    {feat_name:<50} {row['importance']:>12.6f} {row['std']:>10.6f}")

            except Exception as e:
                print(f"    Could not compute feature importance: {e}")
    else:
        print("\n(Feature importance computation skipped - set compute_feature_importance=True to enable)")

    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run the UAV location prediction pipeline."""

    # Initialize configuration
    config = Config()

    print("=" * 80)
    print("UAV LOCATION PREDICTION USING GRADIENT BOOSTING")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Predictions file: {config.predictions_file}")
    print(f"  Sensor positions file: {config.sensor_positions_file}")
    print(f"  Decimal precision: {config.decimal_precision} (meters)")
    print(f"  Context window: {config.context_window} time steps")
    print(f"  Cross-validation folds: {config.n_folds}")
    print(f"  Random seed: {config.random_seed}")

    # Load data
    predictions_df, sensors_df = load_data(config)

    print("\nLimiting data to 3000 trajectories...")
    traj_subset = predictions_df['exp_id'].unique()[:3000]
    predictions_df = predictions_df[predictions_df['exp_id'].isin(traj_subset)]

    # Preprocess data
    predictions_df = preprocess_data(predictions_df, config)

    # Build feature matrix
    feature_df = build_feature_matrix(predictions_df, sensors_df, config)

    # Run cross-validation
    fold_results = run_cross_validation(feature_df, config)

    # Report results
    aggregate_and_report_results(fold_results, config)

    # Train and save full dataset model for online post-processing
    print("\n" + "=" * 80)
    print("TRAINING & SAVING FINAL POST-PROCESSING MODEL ARTIFACT")
    print("=" * 80)
    import joblib
    feature_cols = get_feature_columns(feature_df)
    X_full = feature_df[feature_cols].values
    
    final_models = {}
    for target in ['true_x', 'true_y', 'true_z']:
        print(f"Training final model for {target}...")
        mdl = HistGradientBoostingRegressor(**config.model_params)
        mdl.fit(X_full, feature_df[target].values)
        final_models[target] = mdl

    model_path = os.path.join(script_dir, "post_processing_models.joblib")
    joblib.dump({
        'model_x': final_models['true_x'],
        'model_y': final_models['true_y'],
        'model_z': final_models['true_z'],
        'feature_cols': feature_cols,
        'config': config
    }, model_path)
    print(f"[OK] Saved trained post-processor model artifact to:\n  {model_path}")


if __name__ == "__main__":
    main()