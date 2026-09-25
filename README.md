# Ingeet (ইঙ্গিত)

**Real-Time Sign Language to Bengali Translator**

Ingeet is a computer-vision powered translation system that detects hand gestures in real-time through a standard webcam and translates them into Bengali (বাংলা) text on screen.

---

## Key Capabilities

1. **High-Accuracy Alphabet Fingerspelling (26 Letters):**
   - 98.6% test accuracy on anatomical hand landmark benchmarks.
   - 86 scale- and rotation-invariant engineered features (finger curl ratios, wrist vectors, fingertip spreads, palm orientation).
2. **Instant Conversational Gestures:**
   - Universal gestures recognized with 100% geometric certainty:
     - 🤟 **I Love You** $\rightarrow$ **"আমি তোমাকে ভালোবাসি"**
     - ✌️ **Peace / Victory** $\rightarrow$ **"শান্তি ও বিজয়"**
     - 👍 **Good / Awesome** $\rightarrow$ **"ভালো, চমৎকার"**
     - 👌 **Okay** $\rightarrow$ **"ঠিক আছে"**
     - ✊ **Yes / Affirmative** $\rightarrow$ **"হ্যাঁ"**
     - ✋ **Hello / Greeting** $\rightarrow$ **"হ্যালো"**
     - 🤙 **Call Me** $\rightarrow$ **"যোগাযোগ করুন"**
     - 🤘 **Rock On** $\rightarrow$ **"দুর্দান্ত"**
3. **Word Accumulator & Speller:**
   - Hold any letter for half a second to append it to the spelled word.
   - Built-in vocabulary dictionary translates complete words dynamically (e.g. `HELLO` $\rightarrow$ `হ্যালো`, `BOOK` $\rightarrow$ `বই`, `BANGLA` $\rightarrow$ `বাংলা`).
4. **Professional Minimalist UI:**
   - Freely resizable window (`WINDOW_NORMAL`).
   - Translucent floating bottom subtitle card displaying pure Bengali text.
   - High-contrast, complex Bengali script rendering via `Kalpurush.ttf`.
   - Distraction-free, full camera view with delicate hand skeletal lines.

---

## Quickstart

### 1. Installation

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Translator

```bash
python app.py
```

### Controls

| Key | Action |
| :--- | :--- |
| **`c`** | Clear spelled word buffer |
| **`[SPACE]`** | Insert space in spelled sentence |
| **`b`** | Backspace last letter |
| **`q`** | Quit application |

---

## Project Structure

```
ingeet/
├── assets/
│   └── fonts/
│       └── Kalpurush.ttf       # Bengali Unicode TrueType font
├── models/
│   ├── sign_model.pkl          # 98.6% sign classification model
│   ├── mean.npy                # Normalization mean vector
│   └── std.npy                 # Normalization standard deviation vector
├── src/
│   ├── dictionary.py           # Sign to Bengali translation dictionary
│   ├── features.py             # 86 invariant feature extractor & geometric rules
│   ├── recognizer.py           # Real-time multi-frame smoothing & tracking engine
│   ├── visualizer.py           # Minimalist floating glassmorphism UI card
│   └── config.py               # Core configuration constants & paths
├── tests/
│   └── test_pipeline.py        # Automated unit test suite
├── app.py                      # Clean desktop webcam application
├── requirements.txt
└── README.md
```

---

## Testing

Run the automated test suite:

```bash
python -m unittest tests/test_pipeline.py
```
