import numpy as np
from src.config import (
    TARGET_FRAMES,
    NUM_FEATURES,
    POSE_LANDMARKS_COUNT,
    POSE_COORDS_COUNT,
    HAND_LANDMARKS_COUNT,
    HAND_COORDS_COUNT,
)


def normalize_frames(landmarks: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Standardize temporal frame count to fixed sequence length.

    - Fewer than target_frames: Zero-padding appended at the end.
    - Greater than target_frames: Uniform temporal subsampling via linspace.
    - Exactly target_frames: Identity.
    """
    num_frames = landmarks.shape[0]

    if num_frames < target_frames:
        padding = np.zeros((target_frames - num_frames, landmarks.shape[1]), dtype=landmarks.dtype)
        return np.vstack((landmarks, padding))
    elif num_frames > target_frames:
        indices = np.linspace(0, num_frames - 1, target_frames).astype(int)
        return landmarks[indices]
    return landmarks


def extract_landmarks(results) -> np.ndarray:
    """Extract and flatten MediaPipe Holistic landmarks into a 258-dim feature vector.

    Matches BdSLW60 landmark schema:
    - Pose: 33 landmarks * 4 (x, y, z, visibility) = 132
    - Left Hand: 21 landmarks * 3 (x, y, z) = 63
    - Right Hand: 21 landmarks * 3 (x, y, z) = 63
    Total: 258 features
    """
    # Pose landmarks (33 * 4 = 132)
    if results.pose_landmarks:
        pose = np.array(
            [[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        pose = np.zeros(POSE_LANDMARKS_COUNT * POSE_COORDS_COUNT, dtype=np.float32)

    # Left Hand landmarks (21 * 3 = 63)
    if results.left_hand_landmarks:
        lh = np.array(
            [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        lh = np.zeros(HAND_LANDMARKS_COUNT * HAND_COORDS_COUNT, dtype=np.float32)

    # Right Hand landmarks (21 * 3 = 63)
    if results.right_hand_landmarks:
        rh = np.array(
            [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark],
            dtype=np.float32,
        ).flatten()
    else:
        rh = np.zeros(HAND_LANDMARKS_COUNT * HAND_COORDS_COUNT, dtype=np.float32)

    features = np.concatenate([pose, lh, rh])
    assert features.shape[0] == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {features.shape[0]}"
    return features
