import io
import sys
import os
import pandas as pd
from src.traditional.data_loader import create_sequences
from src.traditional.model import BaselineLSTM
from src.utils.metrics import calculate_metrics

# Configuration
TRAIN_FILE = 'data/DataPre/Train_WL_2020_2022.csv'
TEST_FILE = 'data/DataPre/Test_WL_2023.csv'
ATTRIBUTES = [
    'WL_KienGiang', 'RF_KienGiang', 'RF_LeThuy',
    'WL_DongHoi', 'RF_DongHoi', 'Tide_DongHoi', 'WL_LeThuy'
]
LEAD_TIME = 24
PREDICT_TIME = 12
MODEL_SAVE_PATH = 'models/lstm_model.keras'
LOG_FILE = 'logs/traditional_eval.log'


def main():
    print(f"Loading training data from: {TRAIN_FILE}")
    x_train_raw, y_train = create_sequences(TRAIN_FILE, ATTRIBUTES, LEAD_TIME, PREDICT_TIME)
    x_test_raw, y_test = create_sequences(TEST_FILE, ATTRIBUTES, LEAD_TIME, PREDICT_TIME)

    n_features = len(ATTRIBUTES)
    x_train = x_train_raw.reshape((x_train_raw.shape[0], LEAD_TIME, n_features))
    x_test = x_test_raw.reshape((x_test_raw.shape[0], LEAD_TIME, n_features))

    print(f"Train samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

    model = BaselineLSTM(
        input_shape=(LEAD_TIME, n_features),
        lstm_layers=[10],
        dense_layers=[50],
        learning_rate=0.0001
    )
    model.summary()

    print("Training Baseline LSTM model...")
    model.train(x_train, y_train, x_val=x_test, y_val=y_test, epochs=20, batch_size=32, verbose=1)

    # Evaluate
    y_pred = model.predict(x_test)
    metrics = calculate_metrics(y_test, y_pred)
    print("\nTest Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # Capture summary string
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    summary_str = stream.getvalue()

    # Append to log
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n===== Model Evaluation Log - {pd.Timestamp.now()} =====\n")
        f.write("===== Model Architecture =====\n")
        f.write(summary_str + '\n')
        f.write(f"\nLead time: {LEAD_TIME}, Predict time: {PREDICT_TIME}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"Metrics logged to: {LOG_FILE}")


if __name__ == '__main__':
    main()

