import os
import joblib
import numpy as np  
import tensorflow as tf

from src.encoder_decoder.data_loader import create_seq2seq_sequences
from src.encoder_decoder.model import Seq2SeqLSTM, EncoderDecoderModel, Encoder, Decoder
from src.utils.metrics import calculate_metrics, calculate_horizon_metrics
from src.utils.visualization import plot_horizon_sample

# Configuration
DATA_FILE = 'data/DataScaled/Test_WL_2024.csv'
MODEL_PATH = 'models/encoder_decoder_lstm_model_v2.keras'
TARGET_SCALER_PATH = 'data/DataScaled/minmax_scaler2.pkl'
FIGURES_DIR = 'reports/figures'

ATTRIBUTES = [
    'WL_KienGiang', 'WL_DongHoi', 'WL_LeThuy',
    'RF_DongHoi', 'Tide_DongHoi', 'RF_KienGiang', 'RF_LeThuy'
]
TARGET_ATTRIBUTE = 'WL_LeThuy'
LOOKBACK_WINDOW = 24  # T
HORIZON = 12          # P


def main():
    print(f"Loading Seq2Seq model from: {MODEL_PATH}")
    custom_objects = {
        'Seq2SeqLSTM': Seq2SeqLSTM,
        'EncoderDecoderModel': EncoderDecoderModel,
        'Encoder': Encoder,
        'Decoder': Decoder
    }

    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading data from: {DATA_FILE}")
    x_test, y_test = create_seq2seq_sequences(
        DATA_FILE, ATTRIBUTES, LOOKBACK_WINDOW, HORIZON, TARGET_ATTRIBUTE
    )
    test_inputs = {'encoder_inputs': tf.convert_to_tensor(x_test, dtype=tf.float32)}

    print(f"Running inference on {x_test.shape[0]} samples...")
    y_pred_scaled = model.predict(test_inputs)

    # Inverse transform
    scaler = joblib.load(TARGET_SCALER_PATH)
    y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_true_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

    # Calculate metrics
    overall_metrics = calculate_metrics(y_true_orig, y_pred_orig)
    horizon_metrics = calculate_horizon_metrics(y_true_orig, y_pred_orig)

    print("\nEvaluation Metrics on 2024 Test Data:")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nMetrics by Horizon:")
    for step, metrics_dict in horizon_metrics.items():
        print(f"  Horizon {step}: RMSE={metrics_dict['RMSE']:.4f}, MAE={metrics_dict['MAE']:.4f}, R2={metrics_dict['R2']:.4f}")

    # Plot first 3 samples
    num_samples_to_plot = min(3, x_test.shape[0])
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for i in range(num_samples_to_plot):
        save_path = os.path.join(FIGURES_DIR, f"seq2seq_prediction_2024_sample_{i}.png")
        plot_horizon_sample(
            y_true_seq=y_true_orig[i, :, 0],
            y_pred_seq=y_pred_orig[i, :, 0],
            sample_index=i,
            horizon=HORIZON,
            save_path=save_path
        )


if __name__ == '__main__':
    main()

