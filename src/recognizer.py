"""Ingeet Real-Time Gesture Recognition and Tracking Engine.

Combines:
1. Geometric invariant detection for universal conversational hand signs
2. High-accuracy neural classifier for alphabet fingerspelling
3. Multi-frame smoothing buffer to eliminate video flicker
"""

import sys
import types
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if "tensorflow" not in sys.modules:
    _tf = types.ModuleType("tensorflow")
    _tf_tools = types.ModuleType("tensorflow.tools")
    _tf_docs = types.ModuleType("tensorflow.tools.docs")
    _doc_ctrl = types.ModuleType("tensorflow.tools.docs.doc_controls")
    _doc_ctrl.do_not_generate_docs = lambda x: x
    _tf_docs.doc_controls = _doc_ctrl
    _tf_tools.docs = _tf_docs
    _tf.tools = _tf_tools
    sys.modules["tensorflow"] = _tf
    sys.modules["tensorflow.tools"] = _tf_tools
    sys.modules["tensorflow.tools.docs"] = _tf_docs
    sys.modules["tensorflow.tools.docs.doc_controls"] = _doc_ctrl

import pickle
import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FONT_PATH,
    DEFAULT_MEAN_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_STD_PATH,
    HOLD_FRAMES_TO_COMMIT,
    SMOOTHING_HISTORY_SIZE,
)
from src.dictionary import SIGN_DICTIONARY, get_translation
from src.features import FeatureExtractor

ENGLISH_CLASSES_26 = [chr(ord("A") + i) for i in range(26)]

# Common English to Bengali dictionary for word-builder translation
WORD_TRANSLATIONS = {
    "HELLO": "হ্যালো",
    "HI": "হ্যালো",
    "BYE": "বিদায়",
    "YES": "হ্যাঁ",
    "NO": "না",
    "OK": "ঠিক আছে",
    "LOVE": "ভালোবাসা",
    "PEACE": "শান্তি",
    "GOOD": "ভালো",
    "HELP": "সাহায্য",
    "WATER": "পানি",
    "FOOD": "খাবার",
    "BOOK": "বই",
    "CAT": "বিড়াল",
    "DOG": "কুকুর",
    "BANGLA": "বাংলা",
    "BANGLADESH": "বাংলাদেশ",
    "CLASS": "ক্লাস",
    "TEST": "পরীক্ষা",
    "FRIEND": "বন্ধু",
}


class SignRecognizer:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        mean_path: Path = DEFAULT_MEAN_PATH,
        std_path: Path = DEFAULT_STD_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        history_size: int = SMOOTHING_HISTORY_SIZE,
        hold_frames: int = HOLD_FRAMES_TO_COMMIT,
    ):
        self.conf_threshold = confidence_threshold
        self.history_size = history_size
        self.hold_frames = hold_frames

        self.feature_extractor = FeatureExtractor(mean_path=str(mean_path), std_path=str(std_path))
        self.classes = ENGLISH_CLASSES_26

        # Load trained classifier model
        model_file = Path(model_path)
        if model_file.exists():
            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            self.model_loaded = True
        else:
            self.model = None
            self.model_loaded = False

        # MediaPipe Hands with sleek minimal drawing specs
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=1, circle_radius=2
        )
        self.connection_spec = self.mp_drawing.DrawingSpec(
            color=(0, 245, 160), thickness=2
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # State tracking
        self.prediction_history: deque = deque(maxlen=history_size)
        self.current_sign: str = "..."
        self.current_conf: float = 0.0
        self.current_bn: str = ""
        self.current_desc: str = ""

        # Letter accumulation buffer for words
        self.accumulated_text: str = ""
        self.letter_counter: int = 0
        self.last_committed_letter: Optional[str] = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        show_skeleton: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """Process a webcam BGR frame, draw hand landmarks, and return recognition result."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        hand_detected = False
        raw_pred = "..."
        conf = 0.0
        is_conversational = False

        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw sleek delicate skeleton
            if show_skeleton:
                self.mp_drawing.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_spec,
                    self.connection_spec,
                )

            # Flatten 63 landmarks
            raw_63 = []
            for lm in hand_landmarks.landmark:
                raw_63.extend([lm.x, lm.y, lm.z])
            raw_63 = np.array(raw_63, dtype=np.float32)

            # 1. Check geometric conversational signs first (I Love You, Peace, Good, Okay, Yes, Hello)
            conv_gesture, conv_conf = self.feature_extractor.detect_conversational_gesture(raw_63)
            if conv_gesture is not None:
                raw_pred = conv_gesture
                conf = conv_conf
                is_conversational = True
            elif self.model_loaded:
                # 2. Extract 86 features and run classifier on alphabet
                feats_86 = self.feature_extractor.extract_86_features(raw_63)
                feats_norm = self.feature_extractor.normalize(feats_86).reshape(1, -1)
                probs = self.model.predict_proba(feats_norm)[0]
                best_idx = int(np.argmax(probs))
                conf = float(probs[best_idx])
                raw_pred = self.classes[best_idx]

        # Multi-frame smoothing
        if hand_detected and conf >= self.conf_threshold:
            self.prediction_history.append(raw_pred)
            # Majority vote
            most_common, count = Counter(self.prediction_history).most_common(1)[0]
            if count >= max(2, len(self.prediction_history) // 2):
                self.current_sign = most_common
                self.current_conf = conf
            else:
                self.current_sign = raw_pred
                self.current_conf = conf

            # Check for word-builder letter hold
            if len(self.current_sign) == 1 and self.current_sign.isalpha():
                if self.current_sign == self.last_committed_letter:
                    self.letter_counter += 1
                else:
                    self.last_committed_letter = self.current_sign
                    self.letter_counter = 1

                # If held for hold_frames, append to word
                if self.letter_counter == self.hold_frames:
                    self.accumulated_text += self.current_sign
        else:
            self.prediction_history.clear()
            self.letter_counter = 0
            if not hand_detected:
                self.current_sign = "NO_HAND"
                self.current_conf = 0.0
            else:
                self.current_sign = "UNCERTAIN"
                self.current_conf = conf

        # Lookup Bengali translation
        trans_info = get_translation(self.current_sign)
        self.current_bn = trans_info["bn"]
        self.current_desc = trans_info["desc"]

        # Word translation for accumulated text
        word_bn = WORD_TRANSLATIONS.get(self.accumulated_text, "")

        result = {
            "hand_detected": hand_detected,
            "sign_en": self.current_sign,
            "sign_bn": self.current_bn,
            "desc": self.current_desc,
            "confidence": self.current_conf,
            "is_conversational": is_conversational,
            "accumulated_text": self.accumulated_text,
            "accumulated_bn": word_bn,
            "color": trans_info.get("color", (0, 255, 128)),
        }
        return frame_bgr, result

    def clear_text(self):
        self.accumulated_text = ""
        self.letter_counter = 0
        self.last_committed_letter = None

    def backspace_text(self):
        if len(self.accumulated_text) > 0:
            self.accumulated_text = self.accumulated_text[:-1]

    def add_space(self):
        self.accumulated_text += " "


# Backward-compatible alias
ASLRecognizer = SignRecognizer
