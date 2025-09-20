import cv2
import pytesseract
from pytesseract import Output
from typing import List, Dict


def page_paragraphs_from_image(
    img_path: str,
    mode: str = "paragraph",
    lang: str = "rus+eng",
    pad: int = 4,
    min_conf: int = 40
) -> List[Dict]:
    """
    Разбивает страницу на абзацы или строки и возвращает список блоков.

    Args:
        img_path (str): путь до изображения
        mode (str): "paragraph" или "line" (как группировать текст)
        lang (str): языки для Tesseract (пример: "rus+eng")
        pad (int): отступ вокруг текста при обрезке
        min_conf (int): минимальная уверенность OCR (0–100)

    Returns:
        List[Dict]: список блоков:
            {
                "bbox": (x, y, w, h),
                "ocr_text": str,
                "crop": numpy.ndarray,
                "conf": float,
                "top": int
            }
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Не удалось открыть изображение: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OCR с координатами
    data = pytesseract.image_to_data(gray, output_type=Output.DICT, lang=lang)

    N = len(data['text'])
    entries = []

    for i in range(N):
        txt = data['text'][i].strip()
        conf = int(data['conf'][i]) if data['conf'][i].isdigit() else -1
        if txt == "" or conf < min_conf:
            continue
        entries.append({
            "block_num": data['block_num'][i],
            "par_num": data['par_num'][i],
            "line_num": data['line_num'][i],
            "word_num": data['word_num'][i],
            "left": int(data['left'][i]),
            "top": int(data['top'][i]),
            "width": int(data['width'][i]),
            "height": int(data['height'][i]),
            "text": txt,
            "conf": conf
        })

    if not entries:
        return []

    groups = {}
    if mode == "paragraph":
        for e in entries:
            key = (e['block_num'], e['par_num'])
            groups.setdefault(key, []).append(e)
    else:  # line mode
        for e in entries:
            key = (e['block_num'], e['par_num'], e['line_num'])
            groups.setdefault(key, []).append(e)

    results = []
    for key, items in groups.items():
        left = min(it['left'] for it in items)
        top = min(it['top'] for it in items)
        right = max(it['left'] + it['width'] for it in items)
        bottom = max(it['top'] + it['height'] for it in items)

        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(img.shape[1], right + pad)
        bottom = min(img.shape[0], bottom + pad)

        w = right - left
        h = bottom - top

        items_sorted = sorted(
            items,
            key=lambda x: (x['line_num'], x['word_num'])
        )
        ocr_text = " ".join(it['text'] for it in items_sorted)

        avg_conf = sum(it['conf'] for it in items_sorted) / len(items_sorted)

        crop = img[top:bottom, left:right].copy()

        results.append({
            "bbox": (left, top, w, h),
            "ocr_text": ocr_text,
            "crop": crop,
            "conf": avg_conf,
            "top": top
        })

    results = sorted(results, key=lambda x: x['top'])
    return results
