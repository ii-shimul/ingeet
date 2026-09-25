import numpy as np
from src.config import TARGET_FRAMES, NUM_FEATURES
from src.preprocessing import normalize_frames


def test_normalize_frames_shorter():
    # 20 frames -> pad to 44
    sample = np.ones((20, NUM_FEATURES))
    res = normalize_frames(sample, target_frames=TARGET_FRAMES)
    assert res.shape == (TARGET_FRAMES, NUM_FEATURES)
    assert np.all(res[20:] == 0)


def test_normalize_frames_longer():
    # 60 frames -> subsample to 44
    sample = np.random.rand(60, NUM_FEATURES)
    res = normalize_frames(sample, target_frames=TARGET_FRAMES)
    assert res.shape == (TARGET_FRAMES, NUM_FEATURES)


def test_normalize_frames_exact():
    # 44 frames -> unchanged
    sample = np.random.rand(TARGET_FRAMES, NUM_FEATURES)
    res = normalize_frames(sample, target_frames=TARGET_FRAMES)
    assert res.shape == (TARGET_FRAMES, NUM_FEATURES)
    assert np.array_equal(res, sample)


if __name__ == "__main__":
    test_normalize_frames_shorter()
    test_normalize_frames_longer()
    test_normalize_frames_exact()
    print("All pipeline unit tests passed successfully!")
