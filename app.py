import sys
import types
# Isolate mediapipe from pulling broken tensorflow C++ binaries at runtime
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

import os
import warnings
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
if not hasattr(np, "long"):
    np.long = int
if not hasattr(np, "ulong"):
    np.ulong = int

import argparse
from pathlib import Path
from typing import Optional
import cv2
import mediapipe as mp
from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASSES_PATH,
    DEFAULT_FONT_PATH,
    TARGET_FRAMES,
    CONFIDENCE_THRESHOLD,
    DEBOUNCE_FRAMES,
    BENGALI_LABEL_MAP,
)
from src.inference import RealtimeRecognizer
from src.preprocessing import extract_landmarks
from src.visualizer import draw_hud


def run_app(
    camera_id: int = 0,
    video_path: Optional[str] = None,
    model_path: Path = DEFAULT_MODEL_PATH,
    classes_path: Path = DEFAULT_CLASSES_PATH,
    font_path: Path = DEFAULT_FONT_PATH,
    conf_thresh: float = CONFIDENCE_THRESHOLD,
    debounce_frames: int = DEBOUNCE_FRAMES,
    no_gui: bool = False,
    max_frames: Optional[int] = None,
):
    print("=" * 60)
    print("Ingeet: Real-Time Bengali Sign Language Recognition")
    print("=" * 60)
    source_name = f"Video ({video_path})" if video_path else f"Webcam (Device {camera_id})"
    print(f"Input Source:      {source_name}")
    print(f"Model path:        {model_path} (Exists: {Path(model_path).exists()})")
    print(f"Classes path:      {classes_path} (Exists: {Path(classes_path).exists()})")
    print(f"Font path:         {font_path} (Exists: {Path(font_path).exists()})")
    print("Controls: Press 'q' to quit, 'c' to clear buffer.")
    print("=" * 60)

    recognizer = RealtimeRecognizer(
        model_path=model_path,
        classes_path=classes_path,
        sequence_length=TARGET_FRAMES,
        confidence_threshold=conf_thresh,
        debounce_frames=debounce_frames,
    )

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path if video_path else camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path or camera_id}.")
        return

    frame_counter = 0

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of stream or failed to grab frame.")
                    break

                frame_counter += 1
                if max_frames and frame_counter > max_frames:
                    print(f"Reached max frames limit ({max_frames}). Exiting.")
                    break

                # Convert BGR to RGB for MediaPipe processing on unmirrored coordinates
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)
                image_rgb.flags.writeable = True

                # Extract 258 features matching BdSLW60 orientation
                features = extract_landmarks(results)

                # Stream into inference engine
                confirmed_word, confidence, is_new = recognizer.add_frame(features)

                # Live hand tracking flags
                lh_ok = results.left_hand_landmarks is not None
                rh_ok = results.right_hand_landmarks is not None

                # Console streaming feedback
                if is_new:
                    bengali_label = BENGALI_LABEL_MAP.get(confirmed_word, confirmed_word)
                    print(f"\n[RECOGNIZED] Sign: {bengali_label} ({confirmed_word}) | Confidence: {confidence * 100:.1f}%\n")
                elif frame_counter % 15 == 0:
                    if recognizer.latest_top_word in ("STATIONARY", "NO_HANDS", "..."):
                        live_str = f"State: {recognizer.latest_top_word}"
                    else:
                        live_label = BENGALI_LABEL_MAP.get(recognizer.latest_top_word, recognizer.latest_top_word)
                        live_str = f"Live: {live_label} ({recognizer.latest_top_word}) {recognizer.latest_top_prob * 100:.0f}%"

                    status_str = (
                        f"[FRAME {frame_counter:04d}] "
                        f"Buf: {len(recognizer.buffer)}/{TARGET_FRAMES} | "
                        f"{live_str} | "
                        f"Hands: LH={'[OK]' if lh_ok else '[--]'} RH={'[OK]' if rh_ok else '[--]'}"
                    )
                    print(f"\r{status_str}", end="", flush=True)

                if not no_gui:
                    # Draw landmarks on true coordinates
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    # Mirror video feed for natural user display
                    if not video_path:
                        frame = cv2.flip(frame, 1)

                    # Draw HUD with Bengali text AFTER flip
                    frame = draw_hud(
                        frame,
                        confirmed_word=confirmed_word,
                        confidence=confidence,
                        buffer_len=len(recognizer.buffer),
                        target_frames=TARGET_FRAMES,
                        live_word=recognizer.latest_top_word,
                        live_conf=recognizer.latest_top_prob,
                        debounce_count=recognizer.candidate_count,
                        debounce_max=debounce_frames,
                        hands_detected=(lh_ok, rh_ok),
                    )

                    cv2.imshow("Ingeet - Bengali Sign Language Recognition", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("c"):
                        recognizer.reset_buffer()
                        print("Buffer cleared.")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C).")
    finally:
        cap.release()
        if not no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingeet Real-Time BSL Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (optional)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to .keras model")
    parser.add_argument("--classes", type=str, default=str(DEFAULT_CLASSES_PATH), help="Path to class_names.npy or .json")
    parser.add_argument("--font", type=str, default=str(DEFAULT_FONT_PATH), help="Path to Bengali TTF font")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--debounce", type=int, default=DEBOUNCE_FRAMES, help="Debounce frame count")
    parser.add_argument("--no-gui", action="store_true", help="Run without graphical display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process before exiting")
    args = parser.parse_args()

    run_app(
        camera_id=args.camera,
        video_path=args.video,
        model_path=Path(args.model),
        classes_path=Path(args.classes),
        font_path=Path(args.font),
        conf_thresh=args.conf,
        debounce_frames=args.debounce,
        no_gui=args.no_gui,
        max_frames=args.max_frames,
    )

