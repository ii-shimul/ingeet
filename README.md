# Ingeet (ইঙ্গিত)

Real-time Bengali Sign Language (BdSL) recognition system using MediaPipe Holistic landmarks and temporal Long Short-Term Memory (LSTM) neural networks.

---

## Project Structure

```
ingeet/
├── data/
│   ├── raw/                 # Unprocessed landmark archives (.npz)
│   └── processed/           # Normalized train/val/test splits (.npy)
├── models/                  # Trained model checkpoints & label maps
│   ├── ishara_lstm_baseline.keras
│   └── class_names.npy
├── assets/
│   └── fonts/               # Bengali Unicode fonts (e.g. Kalpurush.ttf)
├── notebooks/               # Research & experimental analysis
│   ├── 01_data_acquisition_and_preprocessing.ipynb
│   └── 02_lstm_model_architecture.ipynb
├── src/                     # Core production pipeline
│   ├── __init__.py
│   ├── config.py            # Global constants, paths, thresholds
│   ├── preprocessing.py     # Frame normalization & landmark extraction (258 features)
│   ├── model.py             # LSTM network architectures
│   ├── inference.py         # Real-time sliding window & debounce engine
│   └── visualizer.py        # OpenCV HUD & Bengali font rendering via PIL
├── tests/                   # Pipeline validation tests
│   └── test_pipeline.py
├── app.py                   # Live webcam desktop inference application
├── requirements.txt         # Python dependencies
├── plan.md                  # Development roadmap and specification
└── README.md                # Documentation
```

---

## Technical Specifications

- **Dataset:** BdSLW60 (9,307 samples, 60 Bengali sign words).
- **Features per frame (258 total):**
  - Pose: 33 landmarks × 4 coords ($x, y, z, \text{visibility}$) = 132
  - Left Hand: 21 landmarks × 3 coords ($x, y, z$) = 63
  - Right Hand: 21 landmarks × 3 coords ($x, y, z$) = 63
- **Sequence Length:** 44 frames (zero-padded or linearly subsampled).
- **Classifier:** 3-layer LSTM stack (128 → 64 → 32) + Dense Softmax (60 classes).

---

## Installation & Setup

```bash
# Clone repository
git clone https://github.com/ii-shimul/ingeet.git
cd ingeet

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Research & Training
Run notebooks inside `notebooks/` locally or on Google Colab:
- `notebooks/01_data_acquisition_and_preprocessing.ipynb`: Dataset acquisition, EDA, cleaning, and normalization.
- `notebooks/02_lstm_model_architecture.ipynb`: Model training, hyperparameter experiments, and evaluation.

### 2. Live Webcam Recognition
Ensure trained model weights and class labels exist in `models/`:
- `models/ishara_lstm_baseline.keras`
- `models/class_names.npy`
- Optional: Add Bengali TrueType font to `assets/fonts/Kalpurush.ttf` for proper complex Unicode glyph rendering.

Launch application:
```bash
python app.py --camera 0 --conf 0.80 --debounce 10
```

Controls:
- `q`: Quit application.
- `c`: Clear temporal rolling buffer.

### 3. Run Pipeline Tests
```bash
python -m tests.test_pipeline
```
