from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.config import TARGET_FRAMES, NUM_FEATURES, NUM_CLASSES, LEARNING_RATE, DROPOUT_RATE


def build_lstm_model(
    input_shape: tuple = (TARGET_FRAMES, NUM_FEATURES),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """Build and compile the 3-layer LSTM sequential model for BdSL gesture classification."""
    layers = [Input(shape=input_shape)]

    if dropout_rate > 0.0:
        layers.extend([
            LSTM(128, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(64, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(32),
            Dropout(dropout_rate),
            Dense(num_classes, activation="softmax"),
        ])
    else:
        # Baseline notebook architecture without dropout
        layers.extend([
            LSTM(128, return_sequences=True),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(num_classes, activation="softmax"),
        ])

    model = Sequential(layers)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
