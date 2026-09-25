import argparse
from pathlib import Path
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.config import (
    BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASSES_PATH,
    DROPOUT_RATE,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)
from src.model import build_lstm_model


def train_model(
    data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    dropout_rate: float = DROPOUT_RATE,
):
    """Load preprocessed splits, train LSTM with EarlyStopping & Checkpointing, and save weights."""
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading preprocessed arrays from {data_dir}...")
    X_train = np.load(str(data_dir / "X_train.npy"))
    y_train = np.load(str(data_dir / "y_train.npy"))
    X_val = np.load(str(data_dir / "X_val.npy"))
    y_val = np.load(str(data_dir / "y_val.npy"))
    X_test = np.load(str(data_dir / "X_test.npy"))
    y_test = np.load(str(data_dir / "y_test.npy"))

    num_classes = len(np.unique(y_train))
    print(f"Train samples: {X_train.shape[0]} | Val samples: {X_val.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"Sequence length: {X_train.shape[1]} | Features: {X_train.shape[2]} | Classes: {num_classes}")

    # Build model
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    model.summary()

    # Callbacks
    best_model_path = models_dir / "ishara_lstm_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=str(best_model_path), monitor="val_accuracy", save_best_only=True),
    ]

    print("\nStarting training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # Save final model
    final_model_path = models_dir / "ishara_lstm_final.keras"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ingeet LSTM Sign Language Classifier")
    parser.add_argument("--data", type=str, default=str(PROCESSED_DATA_DIR), help="Path to processed data dir")
    parser.add_argument("--models", type=str, default=str(MODELS_DIR), help="Output directory for saved models")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE, help="Dropout regularization rate")
    args = parser.parse_args()

    train_model(
        data_dir=Path(args.data),
        models_dir=Path(args.models),
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout,
    )
