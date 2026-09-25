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
CONFIDENCE_THRESHOLD = 0.35  # Realistic threshold for 60-class Softmax distribution
DEBOUNCE_FRAMES = 5          # 5 consecutive stable frames (~160ms) before confirming word

# 60 Word Vocabulary from BdSLW60
CLASS_NAMES_LIST = [
    "aam", "aaple", "ac", "aids", "alu", "anaros", "angur", "apartment", "attio", "audio cassette",
    "ayna", "baandej", "baat", "baba", "balti", "balu", "bhai", "biscuts", "bon", "boroi",
    "bottam", "bou", "cake", "capsule", "cha", "chacha", "chachi", "chadar", "chal", "chikissha",
    "chini", "chips", "chiruni", "chocolate", "chokh utha", "chosma", "churi", "clip", "cream", "dada",
    "dadi", "daeitto", "dal", "debor", "denadar", "dengue", "doctor", "dongson", "dulavai", "durbol",
    "jomoj", "juta", "konna", "maa", "tattha", "toothpaste", "tshirt", "tubelight", "tupi", "tv"
]

# Bengali Script Translation Map for Unicode Rendering
BENGALI_LABEL_MAP = {
    "aam": "আম",
    "aaple": "আপেল",
    "ac": "এসি",
    "aids": "এইডস",
    "alu": "আলু",
    "anaros": "আনারস",
    "angur": "আঙুর",
    "apartment": "অ্যাপার্টমেন্ট",
    "attio": "আত্মীয়",
    "audio cassette": "অডিও ক্যাসেট",
    "ayna": "আয়না",
    "baandej": "ব্যান্ডেজ",
    "baat": "ভাত",
    "baba": "বাবা",
    "balti": "বালতি",
    "balu": "বালু",
    "bhai": "ভাই",
    "biscuts": "বিস্কুট",
    "bon": "বোন",
    "boroi": "বরই",
    "bottam": "বোতাম",
    "bou": "বউ",
    "cake": "কেক",
    "capsule": "ক্যাপসুল",
    "cha": "চা",
    "chacha": "চাচা",
    "chachi": "চাচী",
    "chadar": "চাদর",
    "chal": "চাল",
    "chikissha": "চিকিৎসা",
    "chini": "চিনি",
    "chips": "চিপস",
    "chiruni": "চিরুনি",
    "chocolate": "চকলেট",
    "chokh utha": "চোখ ওঠা",
    "chosma": "চশমা",
    "churi": "চুরি",
    "clip": "ক্লিপ",
    "cream": "ক্রিম",
    "dada": "দাদা",
    "dadi": "দাদী",
    "daeitto": "দায়িত্ব",
    "dal": "ডাল",
    "debor": "দেবর",
    "denadar": "দেনাদার",
    "dengue": "ডেঙ্গু",
    "doctor": "ডাক্তার",
    "dongson": "দংশন",
    "dulavai": "দুলাভাই",
    "durbol": "দুর্বল",
    "jomoj": "জমজ",
    "juta": "জুতা",
    "konna": "কন্যা",
    "maa": "মা",
    "tattha": "তথ্য",
    "toothpaste": "টুথপেস্ট",
    "tshirt": "টি-শার্ট",
    "tubelight": "টিউবলাইট",
    "tupi": "টুপি",
    "tv": "টিভি",
}
