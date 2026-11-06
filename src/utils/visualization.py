from typing import Optional
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def ensure_parent_dir(file_path: str) -> None:
    """Ensure the parent directory of a file path exists."""
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    station_name: str = "Le Thuy",
    prediction_time: int = 12,
    save_path: Optional[str] = None
) -> None:
    """
    Plot observed vs predicted series for single-step / flat sequence.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label="Observed (y_true)", color="black", linewidth=2)
    plt.plot(y_pred, label="Predicted (y_pred)", color="red", linestyle="--", linewidth=2)

    plt.title(
        f"{station_name} Station - Prediction Horizon = {prediction_time} hours",
        fontsize=15,
        fontweight="bold"
    )
    plt.xlabel("Timestamp" if timestamps is not None else "Time Step", fontsize=12)
    plt.ylabel("Water Level (m)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=11)

    if timestamps is not None:
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()

    plt.tight_layout()
    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.close()


def plot_horizon_sample(
    y_true_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    sample_index: int = 0,
    horizon: int = 12,
    save_path: Optional[str] = None
) -> None:
    """
    Plot single sample comparison over multi-step prediction horizon.
    """
    plt.figure(figsize=(10, 5))
    steps = range(1, horizon + 1)
    plt.plot(steps, y_true_seq, label="Observed (y_true)", marker="o", color="blue")
    plt.plot(steps, y_pred_seq, label="Predicted (y_pred)", marker="x", linestyle="--", color="red")

    plt.title(f"Water Level Forecast - Sample {sample_index} ({horizon} steps ahead)", fontsize=13)
    plt.xlabel("Horizon (t+k)")
    plt.ylabel("Water Level (m)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.close()


def plot_horizon_average(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 12,
    save_path: Optional[str] = None
) -> None:
    """
    Plot mean across all samples for each step in horizon.
    """
    mean_true = y_true.mean(axis=0)
    mean_pred = y_pred.mean(axis=0)

    plt.figure(figsize=(10, 5))
    steps = range(1, horizon + 1)
    plt.plot(steps, mean_true, label="Average Observed", marker="o", color="blue")
    plt.plot(steps, mean_pred, label="Average Predicted", marker="x", linestyle="--", color="red")

    plt.title(f"Average Water Level Across All Samples ({y_true.shape[0]} samples)", fontsize=13)
    plt.xlabel("Horizon (t+k)")
    plt.ylabel("Water Level (m)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.close()


def plot_batch_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batch_size: int = 32,
    cols: int = 8,
    save_path: Optional[str] = None
) -> None:
    """
    Plot subplots for a batch of sample predictions.
    """
    batch_size = min(batch_size, y_true.shape[0])
    rows = (batch_size + cols - 1) // cols
    horizon = y_true.shape[1]
    steps = range(1, horizon + 1)

    plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(batch_size):
        plt.subplot(rows, cols, i + 1)
        plt.plot(steps, y_true[i], label="True", marker="o", markersize=3, linewidth=1)
        plt.plot(steps, y_pred[i], label="Pred", linestyle="--", marker="x", markersize=3, linewidth=1)
        plt.title(f"#{i}", fontsize=10)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.suptitle(f"Predictions on first {batch_size} samples", fontsize=14, y=1.02)
    if save_path:
        ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.close()

