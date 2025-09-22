from pathlib import Path
from pdf2image import convert_from_path
from src.ocr.detect import detect_blocks, crop_and_save, is_handwritten_by_heuristic
from src.utils.io import ensure_dir, save_json
import cv2

OUT_PAGES = Path("outputs/pages")

def pdf_to_page_images(pdf_path, dpi=300):
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    base = Path(pdf_path).stem
    out_dir = OUT_PAGES / base
    ensure_dir(out_dir)
    image_paths = []
    for i, page in enumerate(pages, start=1):
        p = out_dir / f"{base}_page{i:03d}.png"
        page.save(p, "PNG")
        image_paths.append(str(p))
    return image_paths

def extract_handwritten_paragraphs_from_page(page_image_path, save_root=None, min_area=2000):
    """
    Детектим блоки, фильтруем рукопись (эвристически) и сохраняем кропы.
    Возвращаем список dicts: [{"crop_path":..., "bbox":..., "mean_conf":...}, ...]
    """
    rects = detect_blocks(page_image_path, out_dir=Path(page_image_path).parent, min_area=min_area)
    saved = []
    page_parent = Path(page_image_path).parent
    base = Path(page_image_path).stem
    # create folder for crops
    crops_dir = Path(save_root) if save_root else (page_parent / "crops")
    ensure_dir(crops_dir)
    for idx, rect in enumerate(rects, start=1):
        crop_name = f"{base}_para{idx:03d}.png"
        crop_path = crops_dir / crop_name
        crop_and_save(page_image_path, rect, crop_path)
        is_hand, mean_conf = is_handwritten_by_heuristic(str(crop_path))
        if is_hand:
            saved.append({"crop_path": str(crop_path), "bbox": rect, "mean_conf": mean_conf})
        else:
            # удаляем не-ручные кропы, чтобы не засорять датасет
            try:
                Path(crop_path).unlink()
            except:
                pass
    return saved

def process_pdf_keep_handwriting(pdf_path, save_root=None, dpi=300):
    pages = pdf_to_page_images(pdf_path, dpi=dpi)
    mapping = []
    for p in pages:
        saved = extract_handwritten_paragraphs_from_page(p, save_root=save_root)
        for s in saved:
            mapping.append(s)
    return mapping
