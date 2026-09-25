"""Ingeet Global Configuration and Paths."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

DEFAULT_MODEL_PATH = MODELS_DIR / "sign_model.pkl"
DEFAULT_MEAN_PATH = MODELS_DIR / "mean.npy"
DEFAULT_STD_PATH = MODELS_DIR / "std.npy"
DEFAULT_FONT_PATH = FONTS_DIR / "Kalpurush.ttf"

# Real-Time Inference Defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_HISTORY_SIZE = 5
HOLD_FRAMES_TO_COMMIT = 15
