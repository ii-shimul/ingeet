from collections import deque
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from tensorflow.keras.models import load_model, Model
from src.config import (
    TARGET_FRAMES,
    NUM_FEATURES,
    CONFIDENCE_THRESHOLD,
    DEBOUNCE_FRAMES,
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASSES_PATH,
)


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
        self.model: Optional[Model] = None
        self.class_names: list = []

        if Path(model_path).exists():
            self.model = load_model(str(model_path))
        if Path(classes_path).exists():
            self.class_names = list(np.load(str(classes_path), allow_pickle=True))

        # Debounce state tracker
        self.candidate_word: Optional[str] = None
        self.candidate_count: int = 0
        self.current_confirmed_word: str = "..."
        self.current_confidence: float = 0.0

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
        input_tensor = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)

        # Direct call is faster than model.predict for single-item streaming inference
        predictions = self.model(input_tensor, training=False).numpy()[0]
        predicted_idx = int(np.argmax(predictions))
        confidence = float(predictions[predicted_idx])
        self.current_confidence = confidence

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
