import os
import pandas as pd
from tensorflow.keras.models import load_model
from src.traditional.data_loader import create_sequences
from src.utils.metrics import calculate_metrics
from src.utils.visualization import plot_prediction_comparison

# Configuration
TEST_FILE = 'data/DataPre/Test_WL_2024.csv'
MODEL_PATH = 'models/lstm_model.keras'
LOG_FILE = 'logs/traditional_test.log'
PLOT_SAVE_PATH = 'reports/figures/traditional_prediction_2024.png'

ATTRIBUTES = [
    'WL_KienGiang', 'RF_KienGiang', 'RF_LeThuy',
    'WL_DongHoi', 'RF_DongHoi', 'Tide_DongHoi', 'WL_LeThuy'
]
LEAD_TIME = 24
PREDICT_TIME = 12


def main():
    print(f"Loading test data from: {TEST_FILE}")
    x_raw, y_real = create_sequences(TEST_FILE, ATTRIBUTES, LEAD_TIME, PREDICT_TIME)

    n_features = len(ATTRIBUTES)
    x = x_raw.reshape((x_raw.shape[0], LEAD_TIME, n_features))

    print(f"Loading trained model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print("Running predictions...")
    y_pred = model.predict(x).reshape(-1)

    # Evaluate
    metrics = calculate_metrics(y_real, y_pred)
    print("\nEvaluation Metrics (2024 Test Set):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Log results
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n===== Test Evaluation - {pd.Timestamp.now()} =====\n")
        f.write(f"Lead time: {LEAD_TIME}, Predict time: {PREDICT_TIME}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("=" * 80 + "\n")
    print(f"Metrics saved to: {LOG_FILE}")

    # Plot
    plot_prediction_comparison(
        y_true=y_real,
        y_pred=y_pred,
        station_name="Le Thuy",
        prediction_time=PREDICT_TIME,
        save_path=PLOT_SAVE_PATH
    )


if __name__ == '__main__':
    main()

