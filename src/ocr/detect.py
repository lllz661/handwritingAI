from pathlib import Path
import cv2
import numpy as np
from src.utils.io import ensure_dir
import pytesseract

def _morph_prepare(gray, ksize=(15,7)):
    # adaptive threshold + dilate to merge words into paragraph blocks
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV,11,2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    morphed = cv2.dilate(th, kernel, iterations=1)
    return morphed

def detect_blocks(image_path, out_dir=None, min_area=2000, ksize=(15,7)):
    """
    Возвращает список блоков: [(x,y,w,h), ...] и сохраняет debug-картинку.
    """
    p = Path(image_path)
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    morphed = _morph_prepare(gray, ksize=ksize)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < min_area:
            continue
        rects.append((x,y,w,h))

    # sort top->bottom left->right
    rects = sorted(rects, key=lambda r: (r[1], r[0]))

    if out_dir:
        ensure_dir(out_dir)
        debug = img.copy()
        for i,(x,y,w,h) in enumerate(rects,1):
            cv2.rectangle(debug,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(debug, str(i),(x,y-6), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
        cv2.imwrite(str(Path(out_dir)/ (p.stem + "_blocks.jpg")), debug)
    return rects

def crop_and_save(img_path, rect, save_path):
    img = cv2.imread(str(img_path))
    x,y,w,h = rect
    crop = img[y:y+h, x:x+w].copy()
    cv2.imwrite(str(save_path), crop)
    return str(save_path)

def is_handwritten_by_heuristic(crop_img_path, ocr_conf_threshold=40):
    """
    Эвристика: если pytesseract дает низкую уверенность для печатного распознавания,
    велика вероятность, что это рукопись. Возвращает True/False и mean_conf.
    """
    try:
        data = pytesseract.image_to_data(crop_img_path, output_type=pytesseract.Output.DICT, lang='rus+eng')
        confs = []
        for c in data.get('conf', []):
            try:
                conf = float(c)
            except:
                conf = -1.0
            if conf >= 0:
                confs.append(conf)
        mean_conf = (sum(confs)/len(confs)) if confs else -1.0
    except Exception:
        mean_conf = -1.0

    # Эвристически: если средняя уверенность низкая — вероятно рукопись
    return (mean_conf < ocr_conf_threshold), mean_conf
