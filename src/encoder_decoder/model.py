from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.saving.register_keras_serializable(package="Custom", name="Encoder")
class Encoder(layers.Layer):
    """
    LSTM Encoder layer.
    Processes historical time series sequence and returns final hidden & cell states.
    """

    def __init__(self, features: int, lstm_units: int, num_layers: int, name: str = 'encoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.features = features
        self.lstm_units = lstm_units
        self.num_layers = num_layers

        self.lstm_layers = []
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            self.lstm_layers.append(layers.LSTM(
                lstm_units,
                return_sequences=not is_last_layer,
                return_state=True,
                name=f'encoder_lstm_{i}'
            ))

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, list]:
        """
        Args:
            inputs: Tensor of shape (batch_size, timesteps, features).

        Returns:
            encoder_output: Output of last LSTM layer (batch_size, lstm_units).
            encoder_states: List of [h, c] pairs from all LSTM layers.
        """
        encoder_states = []
        x = inputs
        for layer in self.lstm_layers:
            x, state_h, state_c = layer(x)
            encoder_states.append([state_h, state_c])

        return x, encoder_states

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'features': self.features,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
        })
        return config


@keras.saving.register_keras_serializable(package="Custom", name="Decoder")
class Decoder(layers.Layer):
    """
    LSTM Decoder layer.
    Receives hidden/cell states and generates step-by-step predictions autoregressively.
    """

    def __init__(self, output_dim: int, lstm_units: int, num_layers: int, name: str = 'decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers

        self.lstm_layers = []
        for i in range(num_layers):
            self.lstm_layers.append(layers.LSTM(
                lstm_units,
                return_sequences=True,
                return_state=True,
                name=f'decoder_lstm_{i}'
            ))

        self.dense_output = layers.Dense(output_dim, name='decoder_output')

    def call(self, inputs: tf.Tensor, initial_states: list) -> Tuple[tf.Tensor, list]:
        """
        Execute one decoding step.

        Args:
            inputs: Shape (batch_size, 1, output_dim).
            initial_states: List of [h, c] pairs for each layer.

        Returns:
            prediction: Shape (batch_size, 1, output_dim).
            decoder_states_new: List of updated [h, c] states.
        """
        x = inputs
        decoder_states_new = []

        for i, layer in enumerate(self.lstm_layers):
            x, state_h, state_c = layer(x, initial_state=initial_states[i])
            decoder_states_new.append([state_h, state_c])

        prediction = self.dense_output(x)
        return prediction, decoder_states_new

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'output_dim': self.output_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
        })
        return config


@keras.saving.register_keras_serializable(package="Custom", name="EncoderDecoderModel")
class EncoderDecoderModel(keras.Model):
    """
    LSTM Encoder-Decoder architecture for multi-step time series forecasting.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lstm_units: int,
        num_layers: int,
        prediction_length: int,
        name: str = 'encoder_decoder_model',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.prediction_length = prediction_length

        self.encoder = Encoder(input_dim, lstm_units, num_layers)
        self.decoder = Decoder(output_dim, lstm_units, num_layers)
        self.encoder_transformer = layers.Dense(output_dim, name='encoder_transformer')

    def call(self, inputs: dict, training: bool = None) -> tf.Tensor:
        encoder_inputs = inputs['encoder_inputs']
        teacher_forcing_targets = inputs.get('teacher_forcing_targets', None)

        encoder_output, encoder_states = self.encoder(encoder_inputs)
        encoder_output = layers.Reshape((1, self.lstm_units))(encoder_output)
        initial_decoder_input = self.encoder_transformer(encoder_output)

        all_predictions = []
        decoder_input_current = initial_decoder_input
        decoder_states_current = encoder_states

        for i in range(self.prediction_length):
            prediction, decoder_states_new = self.decoder(
                decoder_input_current, initial_states=decoder_states_current
            )
            all_predictions.append(prediction)
            decoder_states_current = decoder_states_new

            if training and teacher_forcing_targets is not None:
                decoder_input_current = teacher_forcing_targets[:, i:i+1, :]
            else:
                decoder_input_current = prediction

        return tf.concat(all_predictions, axis=1)

    def get_config(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'lstm_units': self.lstm_units,
            'num_layers': self.num_layers,
            'prediction_length': self.prediction_length,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Modern alias for clean naming
Seq2SeqLSTM = EncoderDecoderModel

