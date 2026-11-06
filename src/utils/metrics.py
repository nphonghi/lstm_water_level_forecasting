from typing import Dict
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate common regression metrics: MSE, RMSE, MAE, R2, MAPE.

    Args:
        y_true: Ground truth target values (1D or flattened array).
        y_pred: Predicted target values (1D or flattened array).

    Returns:
        Dictionary containing metric names and their calculated values.
    """
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)

    try:
        mape = mean_absolute_percentage_error(y_true_flat, y_pred_flat)
    except Exception:
        mape = float("nan")

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "MAPE": float(mape)
    }


def calculate_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate regression metrics for each step (horizon) in multi-step predictions.

    Args:
        y_true: Shape (n_samples, horizon, n_features) or (n_samples, horizon).
        y_pred: Shape (n_samples, horizon, n_features) or (n_samples, horizon).

    Returns:
        Dictionary mapping each step (e.g. 't+1') to its metrics dict.
    """
    if y_true.ndim == 3:
        y_true = y_true[:, :, 0]
    if y_pred.ndim == 3:
        y_pred = y_pred[:, :, 0]

    horizon = y_true.shape[1]
    horizon_metrics = {}

    for step in range(horizon):
        step_metrics = calculate_metrics(y_true[:, step], y_pred[:, step])
        horizon_metrics[f"t+{step + 1}"] = step_metrics

    return horizon_metrics

