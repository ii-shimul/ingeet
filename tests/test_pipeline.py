import unittest
import numpy as np
from src.features import FeatureExtractor
from src.dictionary import get_translation, SIGN_DICTIONARY
from src.recognizer import SignRecognizer
from src.visualizer import draw_translation_card


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        self.recognizer = SignRecognizer()

    def test_feature_extractor_dimensions(self):
        landmarks_63 = np.random.rand(63).astype(np.float32)
        feats_86 = self.extractor.extract_86_features(landmarks_63)
        self.assertEqual(len(feats_86), 86)
        norm_feats = self.extractor.normalize(feats_86)
        self.assertEqual(len(norm_feats), 86)

    def test_bengali_translations(self):
        hello_info = get_translation("HELLO")
        self.assertEqual(hello_info["bn"], "হ্যালো")

        love_info = get_translation("I_LOVE_YOU")
        self.assertEqual(love_info["bn"], "আমি তোমাকে ভালোবাসি")

        letter_a = get_translation("A")
        self.assertEqual(letter_a["bn"], "এ")

    def test_recognizer_process_frame(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out_frame, result = self.recognizer.process_frame(dummy_frame)
        self.assertEqual(out_frame.shape, (480, 640, 3))
        self.assertIn("hand_detected", result)
        self.assertIn("sign_en", result)
        self.assertIn("sign_bn", result)
        self.assertIn("confidence", result)

    def test_visualizer_rendering(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_result = {
            "hand_detected": True,
            "sign_en": "PEACE",
            "sign_bn": "শান্তি ও বিজয়",
            "confidence": 0.98,
            "accumulated_text": "HI",
            "accumulated_bn": "হ্যালো",
            "color": (0, 255, 128),
        }
        rendered = draw_translation_card(dummy_frame, mock_result)
        self.assertEqual(rendered.shape, (480, 640, 3))
        self.assertTrue(np.count_nonzero(rendered) > 0)


if __name__ == "__main__":
    unittest.main()
