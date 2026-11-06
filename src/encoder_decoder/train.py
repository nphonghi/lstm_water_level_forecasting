import io
import os
import sys
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

from src.encoder_decoder.data_loader import create_seq2seq_sequences
from src.encoder_decoder.model import Seq2SeqLSTM
from src.utils.metrics import calculate_metrics, calculate_horizon_metrics
from src.utils.visualization import (
    plot_horizon_sample,
    plot_horizon_average,
    plot_batch_predictions
)

# Configuration
TRAIN_FILE = 'data/DataScaled/Train_WL_2020_2022.csv'
TEST_FILE = 'data/DataScaled/Test_WL_2023.csv'
TARGET_SCALER_PATH = 'data/DataScaled/minmax_scaler2.pkl'
MODEL_SAVE_PATH = 'models/encoder_decoder_lstm_model_v2.keras'
LOG_FILE = 'logs/encoder_decoder_eval.log'

ATTRIBUTES = [
    'WL_KienGiang', 'WL_DongHoi', 'WL_LeThuy',
    'RF_DongHoi', 'Tide_DongHoi', 'RF_KienGiang', 'RF_LeThuy'
]
TARGET_ATTRIBUTE = 'WL_LeThuy'
LOOKBACK_WINDOW = 24  # T
HORIZON = 12          # P
LSTM_UNITS = 64
NUM_LSTM_LAYERS = 3
LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 32
USE_TEACHER_FORCING = False


def main():
    print(f"Loading data: Train={TRAIN_FILE}, Test={TEST_FILE}")
    x_train, y_train = create_seq2seq_sequences(
        TRAIN_FILE, ATTRIBUTES, LOOKBACK_WINDOW, HORIZON, TARGET_ATTRIBUTE
    )
    x_test, y_test = create_seq2seq_sequences(
        TEST_FILE, ATTRIBUTES, LOOKBACK_WINDOW, HORIZON, TARGET_ATTRIBUTE
    )

    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

    train_inputs = {'encoder_inputs': tf.convert_to_tensor(x_train, dtype=tf.float32)}
    if USE_TEACHER_FORCING:
        train_inputs['teacher_forcing_targets'] = y_train_tf
        print("Teacher Forcing is ENABLED.")
    else:
        print("Teacher Forcing is DISABLED.")

    test_inputs = {'encoder_inputs': tf.convert_to_tensor(x_test, dtype=tf.float32)}

    # Build model
    model = Seq2SeqLSTM(
        input_dim=len(ATTRIBUTES),
        output_dim=1,
        lstm_units=LSTM_UNITS,
        num_layers=NUM_LSTM_LAYERS,
        prediction_length=HORIZON
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )

    _ = model(train_inputs)
    model.summary()

    print("\nTraining Seq2Seq LSTM model...")
    history = model.fit(
        train_inputs,
        y_train_tf,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(test_inputs, y_test_tf),
        verbose=1
    )

    print("\nEvaluating on test set...")
    y_pred_scaled = model.predict(test_inputs)

    # Inverse transform
    scaler = joblib.load(TARGET_SCALER_PATH)
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_true_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # Calculate metrics
    overall_metrics = calculate_metrics(y_true_orig, y_pred_orig)
    horizon_metrics = calculate_horizon_metrics(y_true_orig, y_pred_orig)

    print("\nOverall Performance Metrics:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nMetrics by Horizon:")
    for step, metrics_dict in horizon_metrics.items():
        print(f"  Horizon {step}: RMSE={metrics_dict['RMSE']:.4f}, MAE={metrics_dict['MAE']:.4f}, R2={metrics_dict['R2']:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    # Visualizations
    plot_horizon_sample(
        y_true_seq=y_true_orig[0, :, 0],
        y_pred_seq=y_pred_orig[0, :, 0],
        sample_index=0,
        horizon=HORIZON,
        save_path='reports/figures/seq2seq_prediction_sample_0.png'
    )
    plot_horizon_average(
        y_true=y_true_orig[:, :, 0],
        y_pred=y_pred_orig[:, :, 0],
        horizon=HORIZON,
        save_path='reports/figures/seq2seq_average_all_samples.png'
    )
    plot_batch_predictions(
        y_true=y_true_orig[:, :, 0],
        y_pred=y_pred_orig[:, :, 0],
        batch_size=BATCH_SIZE,
        save_path='reports/figures/seq2seq_batch_predictions.png'
    )

    # Log to file
    stream = io.StringIO()
    sys.stdout = stream
    model.summary(expand_nested=True, show_trainable=True)
    sys.stdout = sys.__stdout__
    summary_str = stream.getvalue()

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n===== Model Evaluation Log - {pd.Timestamp.now()} =====\n")
        f.write("----- Model Architecture -----\n")
        f.write(summary_str + '\n')
        f.write(f"----- Configuration -----\n")
        f.write(f"Lead time (T): {LOOKBACK_WINDOW}\n")
        f.write(f"Predict time (P): {HORIZON}\n")
        f.write(f"LSTM Units: {LSTM_UNITS}, Layers: {NUM_LSTM_LAYERS}\n")
        f.write(f"Teacher Forcing: {USE_TEACHER_FORCING}\n")
        f.write(f"Epochs trained: {len(history.history['loss'])}\n\n")

        f.write("----- Overall Performance Metrics -----\n")
        for k, v in overall_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

        f.write("\n----- Performance Metrics by Horizon -----\n")
        for step, metrics_dict in horizon_metrics.items():
            f.write(f"{step} -> RMSE: {metrics_dict['RMSE']:.4f}, MAE: {metrics_dict['MAE']:.4f}, R2: {metrics_dict['R2']:.4f}\n")
        f.write("=" * 80 + "\n")

    print(f"Evaluation log updated: {LOG_FILE}")


if __name__ == '__main__':
    main()

