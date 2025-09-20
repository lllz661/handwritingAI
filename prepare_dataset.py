import os
import re
import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher

import cv2
import pytesseract
from pytesseract import Output
from PIL import Image

def normalize_text(s: str):
    s = s.replace("\r", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def similarity(a: str, b: str):
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()

def page_paragraphs_from_image(img_path: str, mode="paragraph", lang="rus+eng", pad=4):
    """
    Возвращает список объектов: {"bbox":(x,y,w,h), "ocr_text": "...", "crop": numpy_image}
    mode = "paragraph" или "line"
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(gray, output_type=Output.DICT, lang=lang)

    # data contains lists: 'level','page_num','block_num','par_num','line_num','word_num','left','top','width','height','conf','text'
    entries = []
    N = len(data['text'])
    for i in range(N):
        txt = data['text'][i].strip()
        if txt == "":
            continue
        entries.append({
            "block_num": data.get('block_num', [None]*N)[i],
            "par_num": data.get('par_num', [None]*N)[i],
            "line_num": data.get('line_num', [None]*N)[i],
            "word_num": data.get('word_num', [None]*N)[i],
            "left": int(data.get('left', [0]*N)[i]),
            "top": int(data.get('top', [0]*N)[i]),
            "width": int(data.get('width', [0]*N)[i]),
            "height": int(data.get('height', [0]*N)[i]),
            "text": txt
        })

    groups = {}
    if mode == "paragraph":
        # group by (block_num, par_num)
        for e in entries:
            key = (e['block_num'], e['par_num'])
            groups.setdefault(key, []).append(e)
    else:
        # group by (block_num, par_num, line_num)
        for e in entries:
            key = (e['block_num'], e['par_num'], e['line_num'])
            groups.setdefault(key, []).append(e)

    results = []
    for key, items in groups.items():
        # compute bbox union
        left = min(it['left'] for it in items)
        top = min(it['top'] for it in items)
        right = max(it['left'] + it['width'] for it in items)
        bottom = max(it['top'] + it['height'] for it in items)
        # padding
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(img.shape[1], right + pad)
        bottom = min(img.shape[0], bottom + pad)
        w = right - left
        h = bottom - top
        # assemble text in order of word_num
        items_sorted = sorted(items, key=lambda x: (x.get('line_num') or 0, x.get('word_num') or 0))
        ocr_text = " ".join(it['text'] for it in items_sorted)
        crop = img[top:bottom, left:right].copy()
        results.append({
            "bbox": (left, top, w, h),
            "ocr_text": ocr_text,
            "crop": crop,
            "top": top
        })

    # sort top->bottom
    results = sorted(results, key=lambda x: x['top'])
    return results

def split_ground_truth(gt_text: str):
    parts = [p.strip() for p in re.split(r'\n\s*\n', gt_text.strip()) if p.strip()]
    if len(parts) <= 1:
        lines = [l.strip() for l in gt_text.splitlines() if l.strip()]
        if len(lines) > 1:
            parts = lines
        else:
            parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', gt_text.strip()) if s.strip()]
    return parts

def prepare_dataset(pages_dir, gt_dir, out_images="dataset/images", out_labels="dataset/labels",
                    mode="paragraph", lang="rus+eng", sim_threshold=0.25, pad=4, debug=False):
    pages_dir = Path(pages_dir)
    gt_dir = Path(gt_dir)
    out_images = Path(out_images); out_labels = Path(out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    report = {"files": []}

    for page_path in sorted(pages_dir.iterdir()):
        if page_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tiff"]:
            continue
        stem = page_path.stem
        gt_file = gt_dir / f"{stem}.txt"
        if not gt_file.exists():
            print(f"[WARN] Отсутствует эталонный текст для {page_path.name}, пропускаем")
            continue

        print(f"[INFO] Обрабатываем {page_path.name}")
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()
        gt_parts = split_ground_truth(gt_text)
        gt_parts_norm = [normalize_text(p) for p in gt_parts]

        detected = page_paragraphs_from_image(str(page_path), mode=mode, lang=lang, pad=pad)
        det_texts = [normalize_text(d['ocr_text']) for d in detected]

        mapping = []  # список кортежей (crop_filename, matched_gt_index, gt_text, sim)
        used_gt = [False]*len(gt_parts)

        gi = 0  # pointer to current ground-truth index (preserve order)
        for di, det in enumerate(detected):
            best_j = None
            best_sim = -1.0
            K = 4
            for j in range(gi, min(len(gt_parts), gi + K)):
                sim = similarity(det_texts[di], gt_parts_norm[j])
                if sim > best_sim:
                    best_sim = sim; best_j = j
            if best_sim < sim_threshold:
                combo_text = det_texts[di]
                combo_img = det['crop']
                merged = False
                for di2 in range(di+1, min(len(detected), di+3)):
                    combo_text = normalize_text(combo_text + " " + detected[di2]['ocr_text'])
                    sim2 = similarity(combo_text, gt_parts_norm[gi])
                    if sim2 > best_sim:
                        best_sim = sim2
                        best_j = gi
                        merged = True
                        # note: we do NOT actually merge images; we will map current crop to gt and leave others unmapped
                        break
            if best_sim < sim_threshold:
                for j in range(gi+K, len(gt_parts)):
                    sim = similarity(det_texts[di], gt_parts_norm[j])
                    if sim > best_sim:
                        best_sim = sim; best_j = j
            if best_j is None:
                best_j = min(gi, len(gt_parts)-1)
            mapping.append((di, best_j, gt_parts[best_j], best_sim))
            used_gt[best_j] = True
            gi = best_j
        file_report = {"page": str(page_path), "mapping": []}
        for di, det in enumerate(detected):
            crop = det['crop']
            out_name = f"{stem}_p{di+1:03d}.png"
            out_img_path = out_images / out_name
            cv2.imwrite(str(out_img_path), crop)
            mapped = next((m for m in mapping if m[0] == di), None)
            if mapped:
                _, gt_idx, gt_text, sim = mapped
                label_text = gt_text
                out_label_path = out_labels / f"{out_name.rsplit('.',1)[0]}.txt"
                with open(out_label_path, "w", encoding="utf-8") as lf:
                    lf.write(label_text)
                file_report["mapping"].append({
                    "crop": str(out_img_path),
                    "gt_index": int(gt_idx),
                    "gt_text_preview": label_text[:200],
                    "similarity": float(sim)
                })
            else:
                out_label_path = out_labels / f"{out_name.rsplit('.',1)[0]}.txt"
                with open(out_label_path, "w", encoding="utf-8") as lf:
                    lf.write("")
                file_report["mapping"].append({
                    "crop": str(out_img_path),
                    "gt_index": None,
                    "gt_text_preview": "",
                    "similarity": 0.0
                })

        report['files'].append(file_report)

    report_path = out_images.parent / "mapping_report.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)

    print(f"[DONE] Сохранено изображений в {out_images} и меток в {out_labels}")
    print(f"[REPORT] {report_path}")
    return report_path

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by cropping paragraphs/lines and matching with ground-truth text")
    parser.add_argument("--pages_dir", type=str, required=True, help="folder with page images (png/jpg)")
    parser.add_argument("--gt_dir", type=str, required=True, help="folder with ground-truth .txt (same stem)")
    parser.add_argument("--out_images", type=str, default="dataset/images", help="where to save cropped images")
    parser.add_argument("--out_labels", type=str, default="dataset/labels", help="where to save labels (.txt)")
    parser.add_argument("--mode", type=str, default="paragraph", choices=["paragraph", "line"])
    parser.add_argument("--lang", type=str, default="rus+eng")
    parser.add_argument("--threshold", type=float, default=0.25, help="similarity threshold for matching")
    parser.add_argument("--pad", type=int, default=4, help="crop padding in pixels")
    args = parser.parse_args()

    prepare_dataset(
        pages_dir=args.pages_dir,
        gt_dir=args.gt_dir,
        out_images=args.out_images,
        out_labels=args.out_labels,
        mode=args.mode,
        lang=args.lang,
        sim_threshold=args.threshold,
        pad=args.pad
    )

if __name__ == "__main__":
    main()
