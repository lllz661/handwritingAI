import cv2
import numpy as np
from pytesseract import Output
import pytesseract
from typing import List, Dict
from pathlib import Path
from src.utils.io import ensure_dir

def page_paragraphs_from_image(
    img_path: str,
    mode: str = "paragraph",
    lang: str = "rus+eng",
    pad: int = 4,
    min_conf: int = 30,
    save_crops_dir: str = None
) -> List[Dict]:
    """
    Разбивает страницу на абзацы/строки с помощью pytesseract.image_to_data.
    Возвращает список словарей: {"bbox":(x,y,w,h), "ocr_text":..., "conf":avg_conf, "crop_path": optional}
    Если save_crops_dir указан — сохраняет вырезанные кропы туда и возвращает путь в crop_path.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(gray, output_type=Output.DICT, lang=lang)
    N = len(data["text"])

    entries = []
    for i in range(N):
        txt = (data["text"][i] or "").strip()
        conf_raw = data["conf"][i]
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1.0
        if txt == "" or conf < min_conf:
            continue
        entries.append({
            "block_num": data.get("block_num", [None]*N)[i],
            "par_num": data.get("par_num", [None]*N)[i],
            "line_num": data.get("line_num", [None]*N)[i],
            "word_num": data.get("word_num", [None]*N)[i],
            "left": int(data.get("left", [0]*N)[i]),
            "top": int(data.get("top", [0]*N)[i]),
            "width": int(data.get("width", [0]*N)[i]),
            "height": int(data.get("height", [0]*N)[i]),
            "text": txt,
            "conf": conf
        })

    if not entries:
        return []

    # grouping
    groups = {}
    if mode == "paragraph":
        for e in entries:
            key = (e["block_num"], e["par_num"])
            groups.setdefault(key, []).append(e)
    else:
        for e in entries:
            key = (e["block_num"], e["par_num"], e["line_num"])
            groups.setdefault(key, []).append(e)

    results = []
    for key, items in groups.items():
        left = min(it["left"] for it in items)
        top = min(it["top"] for it in items)
        right = max(it["left"] + it["width"] for it in items)
        bottom = max(it["top"] + it["height"] for it in items)

        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(img.shape[1], right + pad)
        bottom = min(img.shape[0], bottom + pad)

        items_sorted = sorted(items, key=lambda x: (x.get("line_num") or 0, x.get("word_num") or 0))
        ocr_text = " ".join(it["text"] for it in items_sorted)
        avg_conf = sum(it["conf"] for it in items_sorted) / len(items_sorted)

        crop = img[top:bottom, left:right].copy()
        crop_path = None
        if save_crops_dir:
            ensure_dir(save_crops_dir)
            stem = Path(img_path).stem
            crop_fname = f"{stem}_p{top}_{left}.png"
            crop_path = str(Path(save_crops_dir) / crop_fname)
            cv2.imwrite(crop_path, crop)

        results.append({
            "bbox": (left, top, right-left, bottom-top),
            "ocr_text": ocr_text,
            "conf": avg_conf,
            "crop_path": crop_path,
            "top": top
        })

    results = sorted(results, key=lambda x: x["top"])
    return results
