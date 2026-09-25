import numpy as np
from src.config import TARGET_FRAMES, NUM_FEATURES
from src.inference import RealtimeRecognizer
from src.visualizer import draw_hud


def test_recognizer_buffer():
    recognizer = RealtimeRecognizer(
        model_path="nonexistent.keras",
        classes_path="models/class_names.npy",
        sequence_length=TARGET_FRAMES,
    )
    assert len(recognizer.buffer) == 0

    # Feed 20 frames (less than sequence_length)
    for _ in range(20):
        word, conf, is_new = recognizer.add_frame(np.zeros(NUM_FEATURES, dtype=np.float32))
        assert word == "..."
        assert conf == 0.0
        assert not is_new

    assert len(recognizer.buffer) == 20

    # Feed 30 more frames (exceeds sequence_length 44)
    for _ in range(30):
        word, conf, is_new = recognizer.add_frame(np.zeros(NUM_FEATURES, dtype=np.float32))

    assert len(recognizer.buffer) == TARGET_FRAMES

    recognizer.reset_buffer()
    assert len(recognizer.buffer) == 0


def test_hud_rendering():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_hud(
        frame,
        confirmed_word="baat",
        confidence=0.92,
        buffer_len=44,
        target_frames=TARGET_FRAMES,
    )
    assert rendered.shape == (480, 640, 3)
    # Ensure overlay is drawn (not completely black)
    assert np.any(rendered != 0)


if __name__ == "__main__":
    test_recognizer_buffer()
    test_hud_rendering()
    print("Inference and HUD unit tests passed successfully!")
