from typing import List, Tuple, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMNS_TO_KEEP = [
    'Time', 'WL_KienGiang', 'WL_LeThuy', 'WL_DongHoi',
    'RF_DongHoi', 'Tide_DongHoi', 'RF_KienGiang', 'RF_LeThuy'
]


def clean_raw_dataframe(input_path: str) -> pd.DataFrame:
    """Read CSV, format datetime and round numeric values."""
    df = pd.read_csv(input_path)
    datetime_col = df.columns[0]

    def _parse_and_format(val):
        try:
            dt = pd.to_datetime(val, errors='coerce', dayfirst=True)
            if pd.isnull(dt):
                return None
            return dt.strftime('%d/%m/%Y %H:%M')
        except Exception:
            return None

    df[datetime_col] = df[datetime_col].astype(str).apply(_parse_and_format)

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

    df = df[[col for col in COLUMNS_TO_KEEP if col in df.columns]]
    return df


def clean_and_scale_dataset(
    train_path: str = 'data/DataRaw/Train_WL_2020_2022.csv',
    val_path: str = 'data/DataRaw/Test_WL_2023.csv',
    test_path: str = 'data/DataRaw/Test_WL_2024.csv',
    output_dir: str = 'data/DataScaled',
    target_column: str = 'WL_LeThuy'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit scalers on train set and transform val/test sets, saving scalers and CSV files.
    """
    train_df = clean_raw_dataframe(train_path)
    val_df = clean_raw_dataframe(val_path)
    test_df = clean_raw_dataframe(test_path)

    value_columns = [col for col in COLUMNS_TO_KEEP if col != 'Time']

    # Scaler for all features
    feature_scaler = MinMaxScaler()
    train_scaled = feature_scaler.fit_transform(train_df[value_columns])
    val_scaled = feature_scaler.transform(val_df[value_columns])
    test_scaled = feature_scaler.transform(test_df[value_columns])

    # Separate scaler for target variable for inverse transform
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[target_column]])

    os.makedirs(output_dir, exist_ok=True)
    feature_scaler_path = os.path.join(output_dir, 'minmax_scaler.pkl')
    target_scaler_path = os.path.join(output_dir, 'minmax_scaler2.pkl')

    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    # Reconstruct dataframes with Time column
    train_scaled_df = pd.concat([train_df[['Time']], pd.DataFrame(train_scaled, columns=value_columns)], axis=1)
    val_scaled_df = pd.concat([val_df[['Time']], pd.DataFrame(val_scaled, columns=value_columns)], axis=1)
    test_scaled_df = pd.concat([test_df[['Time']], pd.DataFrame(test_scaled, columns=value_columns)], axis=1)

    train_scaled_df.to_csv(os.path.join(output_dir, 'Train_WL_2020_2022.csv'), index=False)
    val_scaled_df.to_csv(os.path.join(output_dir, 'Test_WL_2023.csv'), index=False)
    test_scaled_df.to_csv(os.path.join(output_dir, 'Test_WL_2024.csv'), index=False)

    return train_scaled_df, val_scaled_df, test_scaled_df


def create_seq2seq_sequences(
    filename: str,
    attributes: List[str],
    lookback_window: int,
    horizon: int,
    target_attribute: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for Seq2Seq multi-step forecasting.

    Args:
        filename: Path to CSV data file.
        attributes: List of all input features to use.
        lookback_window: Number of historical steps for encoder (T).
        horizon: Number of future steps for decoder (P).
        target_attribute: Name of the target column to forecast.

    Returns:
        encoder_inputs: Shape (num_samples, lookback_window, num_input_features)
        decoder_targets: Shape (num_samples, horizon, 1)
    """
    df = pd.read_csv(filename, index_col=False, usecols=attributes, encoding='utf-8')[attributes]
    data = np.asarray(df.values, dtype=np.float32)

    if target_attribute not in attributes:
        raise ValueError(f"Target attribute '{target_attribute}' not found in attributes list.")

    target_idx = attributes.index(target_attribute)
    num_timesteps = data.shape[0]

    encoder_inputs_list = []
    decoder_targets_list = []
    total_window = lookback_window + horizon

    for i in range(num_timesteps - total_window + 1):
        # Encoder Inputs: T historical steps with all features
        enc_slice = data[i : i + lookback_window, :]
        encoder_inputs_list.append(enc_slice)

        # Decoder Targets: P future steps with target column only
        target_slice = data[i + lookback_window : i + total_window, target_idx]
        decoder_targets_list.append(target_slice.reshape(horizon, 1))

    return np.array(encoder_inputs_list, dtype=np.float32), np.array(decoder_targets_list, dtype=np.float32)


# Backwards compatibility alias
prepare_data = create_seq2seq_sequences


