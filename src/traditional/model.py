from typing import List, Tuple, Dict, Optional
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from src.utils.metrics import calculate_metrics


class BaselineLSTM: 
    """
    Standard sequential LSTM model for time series forecasting.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        lstm_layers: Optional[List[int]] = None,
        dense_layers: Optional[List[int]] = None,
        learning_rate: float = 0.001
    ):
        """
        Args:
            input_shape: Tuple (timesteps, features).
            lstm_layers: List of unit counts for each LSTM layer.
            dense_layers: List of unit counts for Dense layers.
            learning_rate: Learning rate for Adam optimizer.
        """
        self.input_shape = input_shape
        self.lstm_layers = lstm_layers if lstm_layers is not None else [100]
        self.dense_layers = dense_layers if dense_layers is not None else [50, 30, 20]
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        model = Sequential()
        for i, units in enumerate(self.lstm_layers):
            return_seq = i < len(self.lstm_layers) - 1
            model.add(LSTM(
                units,
                activation='tanh',
                return_sequences=return_seq,
                kernel_initializer='he_normal',
                input_shape=self.input_shape if i == 0 else None
            ))

        for units in self.dense_layers:
            model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))

        model.add(Dense(1))
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """Train the model."""
        val_data = (x_val, y_val) if (x_val is not None and y_val is not None) else None
        return self.model.fit(
            x_train,
            y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions and reshape to 1D."""
        return self.model.predict(x).reshape(-1)

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate predictions using common regression metrics."""
        return calculate_metrics(y_true, y_pred)

    def save(self, filepath: str) -> None:
        """Save model weights to filepath."""
        self.model.save(filepath)

    def summary(self) -> None:
        """Print model architecture summary."""
        self.model.summary()


# Backwards compatibility alias
Model = BaselineLSTM


