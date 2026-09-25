"""Generate a dummy test video for automated headless inference testing."""
from pathlib import Path
import cv2
import numpy as np


def create_test_video(output_path: Path = Path("tests/sample_gesture.mp4"), num_frames: int = 60):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, 30.0, (640, 480))

    for i in range(num_frames):
        # Create blank image with an animated circle moving across to simulate motion
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        center_x = int(100 + (440 * (i / num_frames)))
        center_y = int(240 + 50 * np.sin(i / 5.0))
        cv2.circle(frame, (center_x, center_y), 40, (255, 200, 150), -1)
        out.write(frame)

    out.release()
    print(f"Created synthetic test video: {output_path} ({num_frames} frames)")


if __name__ == "__main__":
    create_test_video()
