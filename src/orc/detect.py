import cv2
import numpy as np
from typing import List, Tuple

def detect_blocks(image_path: str, min_area: int = 300) -> List[Tuple[int, int, int, int]]:
    """
    Находит текстовые блоки (прямоугольники) на изображении.

    Args:
        image_path (str): путь до изображения (PNG/JPG).
        min_area (int): минимальная площадь блока для фильтрации мусора.

    Returns:
        List[Tuple[int, int, int, int]]: список блоков в формате (x, y, w, h).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morphed = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:
            blocks.append((x, y, w, h))

    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    return blocks
