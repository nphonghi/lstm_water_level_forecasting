from typing import List, Tuple, Optional
import os
import numpy as np
import pandas as pd

DEFAULT_COLUMNS_TO_KEEP = [
    'Time', 'WL_LongDai', 'WL_HamNinh', 'WL_AnLac',
    'WL_KienGiang', 'RF_KienGiang', 'WL_LeThuy',
    'RF_LeThuy', 'WL_DongHoi', 'RF_DongHoi', 'Tide_DongHoi'
]


def clean_raw_csv(
    input_path: str,
    output_path: str,
    columns_to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Parse datetime column and round numeric features, saving cleaned CSV.

    Args:
        input_path: Path to raw CSV file.
        output_path: Path to destination cleaned CSV file.
        columns_to_keep: List of columns to keep.

    Returns:
        Cleaned pandas DataFrame.
    """
    if columns_to_keep is None:
        columns_to_keep = DEFAULT_COLUMNS_TO_KEEP

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

    # Parse datetime
    df[datetime_col] = df[datetime_col].astype(str).apply(_parse_and_format)

    # Convert numeric columns
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

    # Filter columns
    df = df[[col for col in columns_to_keep if col in df.columns]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def create_sequences(
    filename: str,
    attributes: List[str],
    lead_time: int,
    predict_time: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for traditional single-target forecasting.

    Args:
        filename: Path to CSV data file.
        attributes: List of feature names. The last attribute is assumed to be target.
        lead_time: Number of historical time steps (T).
        predict_time: Number of steps ahead to forecast (P).

    Returns:
        X: Historical feature array of shape (num_samples, lead_time * num_features).
        y: Target array of shape (num_samples,).
    """
    df = pd.read_csv(filename, index_col=False, usecols=attributes, encoding='utf-8')[attributes]
    data = np.asarray(df.values, dtype=np.float32)

    x_list, y_list = [], []
    num_timesteps = len(data)
    total_window = lead_time + predict_time

    for i in range(num_timesteps - total_window + 1):
        x_window = data[i : i + lead_time, :]
        x_list.append(x_window.reshape(-1))

        # Target value at (lead_time + predict_time) from the last attribute column
        y_label = data[i + total_window - 1, -1]
        y_list.append(y_label)

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# Backwards compatibility alias
format_data = create_sequences


