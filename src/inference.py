from collections import deque
from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np
from src.config import (
    TARGET_FRAMES,
    NUM_FEATURES,
    CONFIDENCE_THRESHOLD,
    DEBOUNCE_FRAMES,
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASSES_PATH,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class NumpyLSTMModel:
    """Zero-dependency pure NumPy LSTM inference engine for Keras 3 models.

    Avoids TensorFlow/MediaPipe C++ Abseil SetTimeZone collisions and NumPy C-API conflicts.
    """

    def __init__(self, keras_path: str):
        import io
        import zipfile
        import h5py

        with zipfile.ZipFile(keras_path, "r") as z:
            f = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")
            self.l1_k = f["layers/lstm/cell/vars/0"][:]
            self.l1_rk = f["layers/lstm/cell/vars/1"][:]
            self.l1_b = f["layers/lstm/cell/vars/2"][:]

            self.l2_k = f["layers/lstm_1/cell/vars/0"][:]
            self.l2_rk = f["layers/lstm_1/cell/vars/1"][:]
            self.l2_b = f["layers/lstm_1/cell/vars/2"][:]

            self.l3_k = f["layers/lstm_2/cell/vars/0"][:]
            self.l3_rk = f["layers/lstm_2/cell/vars/1"][:]
            self.l3_b = f["layers/lstm_2/cell/vars/2"][:]

            self.dense_w = f["layers/dense/vars/0"][:]
            self.dense_b = f["layers/dense/vars/1"][:]

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        seq = x_seq[0] if x_seq.ndim == 3 else x_seq
        # Layer 1 (128 units, return_sequences=True)
        x1 = np.dot(seq, self.l1_k) + self.l1_b
        h1 = np.zeros(128, dtype=np.float32)
        c1 = np.zeros(128, dtype=np.float32)
        out1 = np.empty((44, 128), dtype=np.float32)
        for t in range(44):
            g = x1[t] + np.dot(h1, self.l1_rk)
            c1 = _sigmoid(g[128:256]) * c1 + _sigmoid(g[:128]) * np.tanh(g[256:384])
            h1 = _sigmoid(g[384:]) * np.tanh(c1)
            out1[t] = h1

        # Layer 2 (64 units, return_sequences=True)
        x2 = np.dot(out1, self.l2_k) + self.l2_b
        h2 = np.zeros(64, dtype=np.float32)
        c2 = np.zeros(64, dtype=np.float32)
        out2 = np.empty((44, 64), dtype=np.float32)
        for t in range(44):
            g = x2[t] + np.dot(h2, self.l2_rk)
            c2 = _sigmoid(g[64:128]) * c2 + _sigmoid(g[:64]) * np.tanh(g[128:192])
            h2 = _sigmoid(g[192:]) * np.tanh(c2)
            out2[t] = h2

        # Layer 3 (32 units, return_sequences=False)
        x3 = np.dot(out2, self.l3_k) + self.l3_b
        h3 = np.zeros(32, dtype=np.float32)
        c3 = np.zeros(32, dtype=np.float32)
        for t in range(44):
            g = x3[t] + np.dot(h3, self.l3_rk)
            c3 = _sigmoid(g[32:64]) * c3 + _sigmoid(g[:32]) * np.tanh(g[64:96])
            h3 = _sigmoid(g[96:]) * np.tanh(c3)

        # Dense + Softmax
        logits = np.dot(h3, self.dense_w) + self.dense_b
        exp_l = np.exp(logits - np.max(logits))
        return exp_l / np.sum(exp_l)


class RealtimeRecognizer:
    """Sliding-window buffer and debounce smoother for real-time sign recognition."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        classes_path: Path = DEFAULT_CLASSES_PATH,
        sequence_length: int = TARGET_FRAMES,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        debounce_frames: int = DEBOUNCE_FRAMES,
    ):
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.debounce_frames = debounce_frames

        # Buffer holding the last N frame feature vectors
        self.buffer = deque(maxlen=sequence_length)

        # Model and class labels
        self.model: Optional[Any] = None
        self.class_names: list = []

        if Path(model_path).exists():
            try:
                self.model = NumpyLSTMModel(str(model_path))
            except Exception:
                from tensorflow.keras.models import load_model
                self.model = load_model(str(model_path))
        if Path(classes_path).exists():
            if str(classes_path).endswith(".json"):
                import json
                with open(classes_path, "r", encoding="utf-8") as f:
                    self.class_names = json.load(f)
            else:
                self.class_names = list(np.load(str(classes_path), allow_pickle=True))
        else:
            from src.config import CLASS_NAMES_LIST
            self.class_names = list(CLASS_NAMES_LIST)

        # Debounce state tracker
        self.candidate_word: Optional[str] = None
        self.candidate_count: int = 0
        self.current_confirmed_word: str = "..."
        self.current_confidence: float = 0.0
        self.latest_top_word: str = "..."
        self.latest_top_prob: float = 0.0

    def add_frame(self, feature_vector: np.ndarray) -> Tuple[str, float, bool]:
        """Add one frame feature vector (258,) and compute inference if buffer is full.

        Returns:
            (confirmed_word, confidence, is_new_word_confirmed)
        """
        assert feature_vector.shape == (NUM_FEATURES,), f"Expected shape ({NUM_FEATURES},), got {feature_vector.shape}"
        self.buffer.append(feature_vector)

        if len(self.buffer) < self.sequence_length or self.model is None or not self.class_names:
            return self.current_confirmed_word, 0.0, False

        # Input tensor shape: (1, 44, 258)
        buffer_arr = np.array(self.buffer, dtype=np.float32)

        # Hand slices: Left Hand [132:195], Right Hand [195:258]
        lh_slice = buffer_arr[:, 132:195]
        rh_slice = buffer_arr[:, 195:258]

        # 1. Check if hands are visible in recent frames
        lh_active = bool(np.any(lh_slice[-15:] != 0))
        rh_active = bool(np.any(rh_slice[-15:] != 0))

        if not lh_active and not rh_active:
            self.candidate_word = None
            self.candidate_count = 0
            self.latest_top_word = "NO_HANDS"
            self.latest_top_prob = 0.0
            return self.current_confirmed_word, 0.0, False

        # 2. Check if hands are moving (differentiate signing from idle/resting pose)
        lh_motion = float(np.std(lh_slice[lh_slice != 0])) if lh_active and np.any(lh_slice != 0) else 0.0
        rh_motion = float(np.std(rh_slice[rh_slice != 0])) if rh_active and np.any(rh_slice != 0) else 0.0
        max_hand_motion = max(lh_motion, rh_motion)

        # Still hand has std ~0.002; active sign movement has std >= 0.007
        if max_hand_motion < 0.006:
            self.candidate_word = None
            self.candidate_count = 0
            self.latest_top_word = "STATIONARY"
            self.latest_top_prob = 0.0
            return self.current_confirmed_word, 0.0, False

        input_tensor = np.expand_dims(buffer_arr, axis=0)

        # Run pure NumPy inference (~14ms on CPU)
        if hasattr(self.model, "predict"):
            predictions = self.model.predict(input_tensor)
        else:
            predictions = self.model(input_tensor, training=False).numpy()[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        self.current_confidence = confidence
        self.latest_top_word = str(self.class_names[predicted_idx]) if self.class_names else "..."
        self.latest_top_prob = confidence

        is_new_word = False
        if confidence >= self.confidence_threshold:
            word = str(self.class_names[predicted_idx])

            if word == self.candidate_word:
                self.candidate_count += 1
            else:
                self.candidate_word = word
                self.candidate_count = 1

            # Debounce confirmed
            if self.candidate_count >= self.debounce_frames and self.current_confirmed_word != word:
                self.current_confirmed_word = word
                is_new_word = True
        else:
            self.candidate_word = None
            self.candidate_count = 0

        return self.current_confirmed_word, confidence, is_new_word

    def reset_buffer(self):
        """Clear temporal buffer and tracking state."""
        self.buffer.clear()
        self.candidate_word = None
        self.candidate_count = 0
        self.current_confirmed_word = "..."
        self.current_confidence = 0.0
        self.latest_top_word = "..."
        self.latest_top_prob = 0.0
