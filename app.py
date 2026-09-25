"""Ingeet: Real-Time Sign Language to Bengali Translator.

Clean, professional, distraction-free cross-lingual sign language translator.
Detects hand gestures and translates them directly into Bengali Unicode on screen.
"""

import argparse
import sys
import types
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prevent NumPy C-API conflicts
if "tensorflow" not in sys.modules:
    _tf = types.ModuleType("tensorflow")
    _tf_tools = types.ModuleType("tensorflow.tools")
    _tf_docs = types.ModuleType("tensorflow.tools.docs")
    _doc_ctrl = types.ModuleType("tensorflow.tools.docs.doc_controls")
    _doc_ctrl.do_not_generate_docs = lambda x: x
    _tf_docs.doc_controls = _doc_ctrl
    _tf_tools.docs = _tf_docs
    _tf.tools = _tf_tools
    sys.modules["tensorflow"] = _tf
    sys.modules["tensorflow.tools"] = _tf_tools
    sys.modules["tensorflow.tools.docs"] = _tf_docs
    sys.modules["tensorflow.tools.docs.doc_controls"] = _doc_ctrl

from src.config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_FONT_PATH
from src.recognizer import SignRecognizer
from src.visualizer import draw_translation_card


def main():
    parser = argparse.ArgumentParser(description="Ingeet: Real-Time Sign Language Translator")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional video file input")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--no-skeleton", action="store_true", help="Hide hand tracking skeletal lines")
    parser.add_argument("--no-gui", action="store_true", help="Headless mode (no graphical window)")
    parser.add_argument("--max-frames", type=int, default=None, help="Exit after N frames")
    args = parser.parse_args()

    print("=" * 60)
    print("Ingeet: Real-Time Sign Language Translator")
    print("=" * 60)
    source_desc = f"Video: {args.video}" if args.video else f"Webcam (Device {args.camera})"
    print(f"Source:   {source_desc}")
    print(f"Controls: [C] Clear word  |  [SPACE] Space  |  [Q] Quit")
    print("=" * 60)

    recognizer = SignRecognizer(confidence_threshold=args.conf)
    cap = cv2.VideoCapture(args.video if args.video else args.camera)

    if not cap.isOpened():
        print(f"[Error] Could not open camera/video source {args.video or args.camera}.")
        return

    frame_counter = 0
    last_printed_sign = None

    window_name = "Ingeet - Sign Language Translator"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if args.max_frames and frame_counter > args.max_frames:
                break

            if not args.video:
                frame = cv2.flip(frame, 1)

            # Process hand tracking & translation
            frame, result = recognizer.process_frame(frame, show_skeleton=not args.no_skeleton)

            # Terminal log
            sign_en = result["sign_en"]
            sign_bn = result["sign_bn"]
            if sign_en not in ("...", "NO_HAND", "UNCERTAIN") and sign_en != last_printed_sign:
                last_printed_sign = sign_en
                print(f"[অনুবাদ] {sign_bn} ({sign_en})")

            if not args.no_gui:
                display_frame = draw_translation_card(frame, result, font_path=DEFAULT_FONT_PATH)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    recognizer.clear_text()
                elif key == ord(" ") or key == 32:
                    recognizer.add_space()
                elif key == ord("b") or key == 8:
                    recognizer.backspace_text()

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
