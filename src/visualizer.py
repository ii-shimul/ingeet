from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.config import DEFAULT_FONT_PATH


def draw_bengali_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (20, 40),
    font_size: int = 36,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_path: Path = DEFAULT_FONT_PATH,
) -> np.ndarray:
    """Render Bengali Unicode text onto OpenCV BGR image using PIL TrueType fonts.

    Falls back to Hershey ASCII rendering if font file is missing.
    """
    if Path(font_path).exists():
        # OpenCV BGR -> PIL RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        try:
            font = ImageFont.truetype(str(font_path), font_size)
            draw.text(position, text, font=font, fill=text_color)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    # Fallback to standard OpenCV text
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (text_color[2], text_color[1], text_color[0]),
        2,
        cv2.LINE_AA,
    )
    return image


def draw_hud(
    image: np.ndarray,
    confirmed_word: str,
    confidence: float,
    buffer_len: int,
    target_frames: int,
) -> np.ndarray:
    """Draw a clean top HUD overlay displaying current recognition status."""
    h, w = image.shape[:2]

    # Semi-transparent dark banner at top
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # Progress bar for buffer fill
    fill_ratio = min(1.0, buffer_len / target_frames)
    bar_width = int((w - 40) * fill_ratio)
    cv2.rectangle(image, (20, 80), (20 + bar_width, 85), (0, 255, 128), -1)

    # Status info
    status_text = f"Buffer: {buffer_len}/{target_frames} | Conf: {confidence * 100:.1f}%"
    cv2.putText(image, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    # Predicted word
    display_text = f"Word: {confirmed_word}"
    image = draw_bengali_text(image, display_text, position=(20, 42), font_size=28, text_color=(0, 255, 200))

    return image
