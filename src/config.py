from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

DEFAULT_MODEL_PATH = MODELS_DIR / "ishara_lstm_baseline.keras"
DEFAULT_CLASSES_PATH = MODELS_DIR / "class_names.npy"
DEFAULT_FONT_PATH = FONTS_DIR / "Kalpurush.ttf"

# Dataset & Feature Constants
TARGET_FRAMES = 44  # Normalized temporal sequence length
NUM_CLASSES = 60    # Number of sign vocabulary words

# Landmark Feature Dimensions
# MediaPipe Holistic breakdown:
# - Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
# - Left Hand: 21 landmarks * 3 (x, y, z) = 63
# - Right Hand: 21 landmarks * 3 (x, y, z) = 63
# Total features per frame: 132 + 63 + 63 = 258
POSE_LANDMARKS_COUNT = 33
POSE_COORDS_COUNT = 4
HAND_LANDMARKS_COUNT = 21
HAND_COORDS_COUNT = 3
NUM_FEATURES = (POSE_LANDMARKS_COUNT * POSE_COORDS_COUNT) + (2 * HAND_LANDMARKS_COUNT * HAND_COORDS_COUNT)

# Model Training Hyperparameters
BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2

# Real-Time Inference
CONFIDENCE_THRESHOLD = 0.80  # Minimum softmax probability to accept prediction
DEBOUNCE_FRAMES = 10         # Consecutive stable frames required before switching word
