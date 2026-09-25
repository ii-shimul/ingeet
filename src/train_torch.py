from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import io
import zipfile
import numpy as np
import h5py
from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    DEFAULT_MODEL_PATH,
    TARGET_FRAMES,
    NUM_FEATURES,
    NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    DROPOUT_RATE,
)

def train_and_export(
    data_dir: Path = PROCESSED_DATA_DIR,
    output_model_path: Path = DEFAULT_MODEL_PATH,
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = LEARNING_RATE,
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("[Error] PyTorch is not installed in the active environment.")
        print("Please run: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

    print("=" * 60)
    print("Ingeet: Training BdSL LSTM Classifier via PyTorch Engine")
    print("=" * 60)
    print(f"Data directory:   {data_dir}")
    print(f"Output model:     {output_model_path}")
    print(f"Target sequence:  {TARGET_FRAMES} frames x {NUM_FEATURES} features -> {NUM_CLASSES} classes")

    # 1. Load data
    print("\nLoading preprocessed numpy arrays...")
    X_train = np.load(str(data_dir / "X_train.npy")).astype(np.float32)
    y_train = np.load(str(data_dir / "y_train.npy")).astype(np.int64)
    X_val = np.load(str(data_dir / "X_val.npy")).astype(np.float32)
    y_val = np.load(str(data_dir / "y_val.npy")).astype(np.int64)
    X_test = np.load(str(data_dir / "X_test.npy")).astype(np.float32)
    y_test = np.load(str(data_dir / "y_test.npy")).astype(np.int64)

    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=batch_size,
        shuffle=False,
    )

    # 2. Build 3-layer LSTM Architecture matching notebook specs
    class BdSLLSTM(nn.Module):
        def __init__(self, in_features=258, num_classes=60, dropout=0.2):
            super().__init__()
            self.lstm1 = nn.LSTM(in_features, 128, batch_first=True)
            self.drop1 = nn.Dropout(dropout)
            self.lstm2 = nn.LSTM(128, 64, batch_first=True)
            self.drop2 = nn.Dropout(dropout)
            self.lstm3 = nn.LSTM(64, 32, batch_first=True)
            self.drop3 = nn.Dropout(dropout)
            self.fc = nn.Linear(32, num_classes)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.drop1(out)
            out, _ = self.lstm2(out)
            out = self.drop2(out)
            out, _ = self.lstm3(out)
            out = self.drop3(out[:, -1, :])  # Take last time step
            logits = self.fc(out)
            return logits

    device = torch.device("cpu")
    model = BdSLLSTM(NUM_FEATURES, NUM_CLASSES, DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    print("\nStarting model training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            train_correct += (out.argmax(dim=1) == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x)
                loss = criterion(out, batch_y)
                val_loss += loss.item() * len(batch_y)
                val_correct += (out.argmax(dim=1) == batch_y).sum().item()
                val_total += len(batch_y)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EarlyStopping] Validation loss did not improve for {patience} epochs. Stopping early.")
                break

    # Load best weights
    model.load_state_dict(best_state)

    # Evaluate on test set
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            out = model(batch_x)
            loss = criterion(out, batch_y)
            test_loss += loss.item() * len(batch_y)
            test_correct += (out.argmax(dim=1) == batch_y).sum().item()
            test_total += len(batch_y)

    test_loss /= test_total
    test_acc = test_correct / test_total
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")
    print("=" * 60)

    # 3. Export trained weights into Keras 3 format for zero-dependency NumpyLSTMModel
    print(f"\nExporting trained weights to {output_model_path}...")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert PyTorch LSTM gates [i, f, g, o] to Keras format:
    # PyTorch: W_ii, W_if, W_ig, W_io (order: i, f, g/c, o)
    # Keras:   W_i, W_f, W_c, W_o     (order: i, f, c, o) -> EXACT MATCH!
    def get_keras_lstm_weights(lstm_layer):
        w_ih = lstm_layer.weight_ih_l0.detach().numpy().T  # (in_features, 4*units)
        w_hh = lstm_layer.weight_hh_l0.detach().numpy().T  # (units, 4*units)
        b = (lstm_layer.bias_ih_l0 + lstm_layer.bias_hh_l0).detach().numpy()  # (4*units,)
        return w_ih, w_hh, b

    l1_k, l1_rk, l1_b = get_keras_lstm_weights(model.lstm1)
    l2_k, l2_rk, l2_b = get_keras_lstm_weights(model.lstm2)
    l3_k, l3_rk, l3_b = get_keras_lstm_weights(model.lstm3)
    dense_w = model.fc.weight.detach().numpy().T  # (32, 60)
    dense_b = model.fc.bias.detach().numpy()       # (60,)

    # Write model.weights.h5 in memory
    h5_bytes = io.BytesIO()
    with h5py.File(h5_bytes, "w") as h:
        h.create_dataset("layers/lstm/cell/vars/0", data=l1_k)
        h.create_dataset("layers/lstm/cell/vars/1", data=l1_rk)
        h.create_dataset("layers/lstm/cell/vars/2", data=l1_b)

        h.create_dataset("layers/lstm_1/cell/vars/0", data=l2_k)
        h.create_dataset("layers/lstm_1/cell/vars/1", data=l2_rk)
        h.create_dataset("layers/lstm_1/cell/vars/2", data=l2_b)

        h.create_dataset("layers/lstm_2/cell/vars/0", data=l3_k)
        h.create_dataset("layers/lstm_2/cell/vars/1", data=l3_rk)
        h.create_dataset("layers/lstm_2/cell/vars/2", data=l3_b)

        h.create_dataset("layers/dense/vars/0", data=dense_w)
        h.create_dataset("layers/dense/vars/1", data=dense_b)

    # Save into .keras zip archive
    with zipfile.ZipFile(str(output_model_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.weights.h5", h5_bytes.getvalue())
        z.writestr("config.json", '{"name": "ishara_lstm"}')
        z.writestr("metadata.json", '{"keras_version": "3.x"}')

    print(f"Successfully saved trained model to {output_model_path}!")
    print("Zero-dependency real-time inference is now ready!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Ingeet LSTM Sign Model using PyTorch")
    parser.add_argument("--epochs", type=int, default=35, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train_and_export(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
