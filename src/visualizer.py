"""Ingeet Minimalist Bengali Translation Visualizer.

Renders crisp Bengali Unicode typography (via Kalpurush.ttf) in a floating
glassmorphism subtitle card at bottom-center of the camera feed.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import DEFAULT_FONT_PATH


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
    bgr_color: Tuple[int, int, int] = (0, 245, 160),
    font_path: Path = DEFAULT_FONT_PATH,
) -> Tuple[np.ndarray, int]:
    """Render crisp Bengali Unicode text onto OpenCV image using PIL."""
    font = _get_font(str(font_path), font_size)
    if font is not None:
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
            draw.text(position, text, font=font, fill=rgb_color)
            out_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            text_width = int(font.getlength(text))
            return out_img, text_width
        except Exception:
            pass

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


def draw_translation_card(
    image: np.ndarray,
    result: Dict,
    font_path: Path = DEFAULT_FONT_PATH,
    **kwargs,
) -> np.ndarray:
    """Draw a sleek, distraction-free floating translation card at bottom-center."""
    h, w = image.shape[:2]
    hand_detected = result.get("hand_detected", False)
    sign_en = result.get("sign_en", "...")
    sign_bn = result.get("sign_bn", "")
    accumulated = result.get("accumulated_text", "")
    accumulated_bn = result.get("accumulated_bn", "")

    # Dimensions for floating bottom card
    card_w = min(int(w * 0.75), 520)
    card_h = 68
    x1 = (w - card_w) // 2
    y1 = h - card_h - 24
    x2 = x1 + card_w
    y2 = y1 + card_h

    # Semi-transparent dark glass card
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (16, 18, 22), -1)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)

    # Subtle sleek rounded outline
    cv2.rectangle(image, (x1, y1), (x2, y2), (65, 70, 82), 1, cv2.LINE_AA)

    font_bengali = _get_font(str(font_path), 40)

    if hand_detected and sign_en not in ("...", "NO_HAND", "UNCERTAIN"):
        # Calculate text width for true centering
        bn_len = int(font_bengali.getlength(sign_bn)) if font_bengali else len(sign_bn) * 22
        bn_x = max(x1 + 16, x1 + (card_w - bn_len) // 2)

        # Pure prominent Bengali translation
        bn_color = (0, 245, 160)
        image, _ = draw_bengali_text(
            image,
            sign_bn,
            position=(bn_x, y1 + 12),
            font_size=40,
            bgr_color=bn_color,
            font_path=font_path,
        )

    elif accumulated:
        # Display spelled translation or letters
        display_str = accumulated_bn if accumulated_bn else accumulated
        bn_len = int(font_bengali.getlength(display_str)) if font_bengali else len(display_str) * 22
        bn_x = max(x1 + 16, x1 + (card_w - bn_len) // 2)
        image, _ = draw_bengali_text(
            image,
            display_str,
            position=(bn_x, y1 + 12),
            font_size=40,
            bgr_color=(0, 245, 160) if accumulated_bn else (255, 255, 255),
            font_path=font_path,
        )

    else:
        # Subtle idle prompt in Bengali
        idle_msg = "হাত তুলে ইশারা করুন"
        font_idle = _get_font(str(font_path), 24)
        msg_len = int(font_idle.getlength(idle_msg)) if font_idle else len(idle_msg) * 14
        msg_x = max(x1 + 16, x1 + (card_w - msg_len) // 2)
        image, _ = draw_bengali_text(
            image,
            idle_msg,
            position=(msg_x, y1 + 18),
            font_size=24,
            bgr_color=(150, 155, 165),
            font_path=font_path,
        )

    return image


# Aliases
draw_minimal_pill = draw_translation_card
draw_hud = draw_translation_card
draw_asl_hud = draw_translation_card
