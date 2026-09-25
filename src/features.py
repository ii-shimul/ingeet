"""Hand Landmark Feature Engineering and Geometric Gesture Detection.

Extracts 86 invariant features from MediaPipe 21 hand landmarks, and provides
geometric detection for universal conversational hand gestures (Hello, I Love You,
Peace, Good, Okay, Yes, Call Me, etc.).
"""

import numpy as np


class FeatureExtractor:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self, mean_path="models/mean.npy", std_path="models/std.npy"):
        try:
            self.mean = np.load(mean_path)
            self.std = np.load(std_path)
            self.std[self.std == 0] = 1.0
            self.has_norm = True
        except Exception:
            self.has_norm = False

    @staticmethod
    def extract_point(landmarks: np.ndarray, index: int) -> np.ndarray:
        return landmarks[index * 3 : (index + 1) * 3]

    @staticmethod
    def euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        cos_val = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        return float(np.arccos(cos_val) * 180.0 / np.pi)

    def extract_86_features(self, landmarks_63: np.ndarray) -> np.ndarray:
        """Engineer 86 features from raw 63 landmarks (21 * 3)."""
        feats = []
        feats.extend(landmarks_63)

        wrist = self.extract_point(landmarks_63, self.WRIST)
        t_tip = self.extract_point(landmarks_63, self.THUMB_TIP)
        i_tip = self.extract_point(landmarks_63, self.INDEX_TIP)
        m_tip = self.extract_point(landmarks_63, self.MIDDLE_TIP)
        r_tip = self.extract_point(landmarks_63, self.RING_TIP)
        p_tip = self.extract_point(landmarks_63, self.PINKY_TIP)

        t_mcp = self.extract_point(landmarks_63, self.THUMB_MCP)
        i_mcp = self.extract_point(landmarks_63, self.INDEX_MCP)
        m_mcp = self.extract_point(landmarks_63, self.MIDDLE_MCP)
        r_mcp = self.extract_point(landmarks_63, self.RING_MCP)
        p_mcp = self.extract_point(landmarks_63, self.PINKY_MCP)

        # 1. Distances from wrist (5 features)
        feats.append(self.euclidean_dist(t_tip, wrist))
        feats.append(self.euclidean_dist(i_tip, wrist))
        feats.append(self.euclidean_dist(m_tip, wrist))
        feats.append(self.euclidean_dist(r_tip, wrist))
        feats.append(self.euclidean_dist(p_tip, wrist))

        # 2. Adjacent fingertip spreads (4 features)
        feats.append(self.euclidean_dist(t_tip, i_tip))
        feats.append(self.euclidean_dist(i_tip, m_tip))
        feats.append(self.euclidean_dist(m_tip, r_tip))
        feats.append(self.euclidean_dist(r_tip, p_tip))

        # 3. Finger curl - tip to MCP (5 features)
        feats.append(self.euclidean_dist(t_tip, t_mcp))
        feats.append(self.euclidean_dist(i_tip, i_mcp))
        feats.append(self.euclidean_dist(m_tip, m_mcp))
        feats.append(self.euclidean_dist(r_tip, r_mcp))
        feats.append(self.euclidean_dist(p_tip, p_mcp))

        # 4. Palm orientation (3 features)
        palm_vec = m_mcp - wrist
        feats.extend([palm_vec[0], palm_vec[1], palm_vec[2]])

        # 5. Finger angles relative to palm (5 features)
        for tip, mcp in [
            (t_tip, t_mcp),
            (i_tip, i_mcp),
            (m_tip, m_mcp),
            (r_tip, r_mcp),
            (p_tip, p_mcp),
        ]:
            f_vec = tip - mcp
            feats.append(self.angle_between(f_vec, palm_vec))

        # 6. Hand span (1 feature)
        tips = [t_tip, i_tip, m_tip, r_tip, p_tip]
        max_span = max(
            self.euclidean_dist(tips[i], tips[j])
            for i in range(len(tips))
            for j in range(i + 1, len(tips))
        )
        feats.append(max_span)

        return np.array(feats, dtype=np.float32)

    def normalize(self, feats_86: np.ndarray) -> np.ndarray:
        if self.has_norm:
            return (feats_86 - self.mean) / self.std
        return feats_86

    def detect_conversational_gesture(self, landmarks_63: np.ndarray):
        """Geometric detection of universal conversational gestures.

        Returns (gesture_name, confidence) or (None, 0.0)
        """
        wrist = self.extract_point(landmarks_63, self.WRIST)
        t_tip = self.extract_point(landmarks_63, self.THUMB_TIP)
        i_tip = self.extract_point(landmarks_63, self.INDEX_TIP)
        m_tip = self.extract_point(landmarks_63, self.MIDDLE_TIP)
        r_tip = self.extract_point(landmarks_63, self.RING_TIP)
        p_tip = self.extract_point(landmarks_63, self.PINKY_TIP)

        t_mcp = self.extract_point(landmarks_63, self.THUMB_MCP)
        i_mcp = self.extract_point(landmarks_63, self.INDEX_MCP)
        m_mcp = self.extract_point(landmarks_63, self.MIDDLE_MCP)
        r_mcp = self.extract_point(landmarks_63, self.RING_MCP)
        p_mcp = self.extract_point(landmarks_63, self.PINKY_MCP)

        i_pip = self.extract_point(landmarks_63, self.INDEX_PIP)
        m_pip = self.extract_point(landmarks_63, self.MIDDLE_PIP)
        r_pip = self.extract_point(landmarks_63, self.RING_PIP)
        p_pip = self.extract_point(landmarks_63, self.PINKY_PIP)

        palm_scale = max(self.euclidean_dist(wrist, m_mcp), 0.05)

        # Extended tests
        i_ext = (self.euclidean_dist(i_tip, wrist) > self.euclidean_dist(i_pip, wrist) * 1.15)
        m_ext = (self.euclidean_dist(m_tip, wrist) > self.euclidean_dist(m_pip, wrist) * 1.15)
        r_ext = (self.euclidean_dist(r_tip, wrist) > self.euclidean_dist(r_pip, wrist) * 1.15)
        p_ext = (self.euclidean_dist(p_tip, wrist) > self.euclidean_dist(p_pip, wrist) * 1.15)
        t_ext = (self.euclidean_dist(t_tip, wrist) > palm_scale * 1.1)

        # 1. "I LOVE YOU" (🤟)
        if t_ext and i_ext and p_ext and not m_ext and not r_ext:
            return "I_LOVE_YOU", 0.98

        # 2. "GOOD / THUMBS UP" (👍)
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if t_ext or (wrist[1] - t_tip[1] > palm_scale * 0.5):
                return "GOOD", 0.97
            return "YES", 0.95

        # 3. "PEACE / VICTORY" (✌️)
        if i_ext and m_ext and not r_ext and not p_ext:
            spread = self.euclidean_dist(i_tip, m_tip)
            if spread > palm_scale * 0.35:
                return "PEACE", 0.98

        # 4. "OKAY" (👌)
        ti_dist = self.euclidean_dist(t_tip, i_tip)
        if ti_dist < palm_scale * 0.45 and m_ext and r_ext and p_ext:
            return "OKAY", 0.97

        # 5. "CALL ME" (🤙)
        if t_ext and p_ext and not i_ext and not m_ext and not r_ext:
            return "CALL_ME", 0.98

        # 6. "ROCK ON" (🤘)
        if i_ext and p_ext and not m_ext and not r_ext and not t_ext:
            return "ROCK", 0.97

        # 7. "HELLO / OPEN PALM" (✋)
        if t_ext and i_ext and m_ext and r_ext and p_ext:
            return "HELLO", 0.96

        return None, 0.0


# Backward-compatible alias
ASLFeatureExtractor = FeatureExtractor
