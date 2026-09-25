from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.config import DEFAULT_FONT_PATH, BENGALI_LABEL_MAP


from functools import lru_cache

@lru_cache(maxsize=16)
def _get_font(font_path_str: str, font_size: int):
    try:
        return ImageFont.truetype(font_path_str, font_size)
    except Exception:
        return None


def draw_bengali_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (20, 40),
    font_size: int = 36,
    bgr_color: Tuple[int, int, int] = (0, 255, 128),
    font_path: Path = DEFAULT_FONT_PATH,
) -> Tuple[np.ndarray, int]:
    """Render Bengali Unicode text onto OpenCV BGR image using PIL TrueType fonts.

    Returns:
        (image, rendered_width_px)
    """
    font = _get_font(str(font_path), font_size)
    if font is not None:
        try:
            # OpenCV BGR -> PIL RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            # Convert BGR to RGB for PIL fill
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            draw.text(position, text, font=font, fill=rgb_color)
            out_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            text_width = int(font.getlength(text))
            return out_img, text_width
        except Exception:
            pass

    # Fallback to standard OpenCV text
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        bgr_color,
        2,
        cv2.LINE_AA,
    )
    return image, len(text) * 18


def draw_hud(
    image: np.ndarray,
    confirmed_word: str,
    confidence: float,
    buffer_len: int,
    target_frames: int,
    live_word: str = "...",
    live_conf: float = 0.0,
    debounce_count: int = 0,
    debounce_max: int = 10,
    hands_detected: Tuple[bool, bool] = (False, False),
) -> np.ndarray:
    """Draw clean top HUD with English UI labels and pure Bengali word rendering."""
    h, w = image.shape[:2]

    # Semi-transparent dark banner at top
    banner_height = 95
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.80, image, 0.20, 0, image)

    # Top Row: Hands and Buffer status (pure English)
    lh_detected, rh_detected = hands_detected
    hands_text = f"Hands: LH[{'OK' if lh_detected else '--'}] RH[{'OK' if rh_detected else '--'}]"
    cv2.putText(image, hands_text, (w - 240, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    status_text = f"Buffer: {buffer_len}/{target_frames}"
    cv2.putText(image, status_text, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # Progress bar for buffer
    fill_ratio = min(1.0, buffer_len / target_frames)
    bar_width = int((w - 40) * fill_ratio)
    bar_color = (0, 255, 128) if fill_ratio >= 1.0 else (100, 100, 100)
    cv2.rectangle(image, (20, 85), (20 + bar_width, 89), bar_color, -1)

    # Main Row: Clean rendering
    y_text = 62
    if confirmed_word and confirmed_word not in ("...", "STATIONARY", "NO_HANDS"):
        # Confirmed sign
        bengali_word = BENGALI_LABEL_MAP.get(confirmed_word, confirmed_word)
        prefix = "CONFIRMED: "
        cv2.putText(image, prefix, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 128), 2, cv2.LINE_AA)
        prefix_len = 155

        # Render only pure Bengali word through PIL
        image, bengali_w = draw_bengali_text(
            image,
            bengali_word,
            position=(20 + prefix_len, y_text - 20),
            font_size=28,
            bgr_color=(0, 255, 128),
        )

        suffix = f"({confirmed_word})  {confidence * 100:.0f}%"
        cv2.putText(
            image,
            suffix,
            (20 + prefix_len + bengali_w + 12, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 128),
            1,
            cv2.LINE_AA,
        )

    elif buffer_len < target_frames:
        msg = f"STATUS: Initializing buffer... ({buffer_len}/{target_frames})"
        cv2.putText(image, msg, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

    elif live_word in ("NO_HANDS", "..."):
        msg = "STATUS: Waiting for hands in camera view..."
        cv2.putText(image, msg, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 160, 255), 1, cv2.LINE_AA)

    elif live_word == "STATIONARY":
        msg = "STATUS: Ready (make gesture to sign)"
        cv2.putText(image, msg, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 200, 180), 1, cv2.LINE_AA)

    else:
        # Active live candidate prediction
        bengali_live = BENGALI_LABEL_MAP.get(live_word, live_word)
        prefix = "LIVE: "
        color = (0, 215, 255) if live_conf >= 0.5 else (180, 180, 220)
        cv2.putText(image, prefix, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)
        prefix_len = 70

        image, bengali_w = draw_bengali_text(
            image,
            bengali_live,
            position=(20 + prefix_len, y_text - 20),
            font_size=28,
            bgr_color=color,
        )

        suffix = f"({live_word})  {live_conf * 100:.0f}%  [{debounce_count}/{debounce_max}]"
        cv2.putText(
            image,
            suffix,
            (20 + prefix_len + bengali_w + 12, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color,
            1,
            cv2.LINE_AA,
        )

    return image


