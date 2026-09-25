import argparse
from pathlib import Path
import cv2
import mediapipe as mp
from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_CLASSES_PATH,
    DEFAULT_FONT_PATH,
    TARGET_FRAMES,
    CONFIDENCE_THRESHOLD,
    DEBOUNCE_FRAMES,
)
from src.inference import RealtimeRecognizer
from src.preprocessing import extract_landmarks
from src.visualizer import draw_hud


def run_app(
    camera_id: int = 0,
    model_path: Path = DEFAULT_MODEL_PATH,
    classes_path: Path = DEFAULT_CLASSES_PATH,
    font_path: Path = DEFAULT_FONT_PATH,
    conf_thresh: float = CONFIDENCE_THRESHOLD,
    debounce_frames: int = DEBOUNCE_FRAMES,
):
    print("=" * 50)
    print("Ingeet: Real-Time Bengali Sign Language Recognition")
    print("=" * 50)
    print(f"Model path:        {model_path} (Exists: {Path(model_path).exists()})")
    print(f"Classes path:      {classes_path} (Exists: {Path(classes_path).exists()})")
    print(f"Font path:         {font_path} (Exists: {Path(font_path).exists()})")
    print("Controls: Press 'q' to quit, 'c' to clear buffer.")
    print("=" * 50)

    recognizer = RealtimeRecognizer(
        model_path=model_path,
        classes_path=classes_path,
        sequence_length=TARGET_FRAMES,
        confidence_threshold=conf_thresh,
        debounce_frames=debounce_frames,
    )

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}.")
        return

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab camera frame.")
                break

            # Mirror frame for natural user interaction
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True

            # Extract 258 features
            features = extract_landmarks(results)

            # Stream into inference engine
            confirmed_word, confidence, is_new = recognizer.add_frame(features)
            if is_new:
                print(f"[RECOGNIZED] Word: {confirmed_word} (Confidence: {confidence * 100:.1f}%)")

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Draw HUD with Bengali text
            frame = draw_hud(
                frame,
                confirmed_word=confirmed_word,
                confidence=confidence,
                buffer_len=len(recognizer.buffer),
                target_frames=TARGET_FRAMES,
            )

            cv2.imshow("Ingeet - Bengali Sign Language Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                recognizer.reset_buffer()
                print("Buffer cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingeet Real-Time BSL Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to .keras model")
    parser.add_argument("--classes", type=str, default=str(DEFAULT_CLASSES_PATH), help="Path to class_names.npy")
    parser.add_argument("--font", type=str, default=str(DEFAULT_FONT_PATH), help="Path to Bengali TTF font")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--debounce", type=int, default=DEBOUNCE_FRAMES, help="Debounce frame count")
    args = parser.parse_args()

    run_app(
        camera_id=args.camera,
        model_path=Path(args.model),
        classes_path=Path(args.classes),
        font_path=Path(args.font),
        conf_thresh=args.conf,
        debounce_frames=args.debounce,
    )
