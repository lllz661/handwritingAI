import sys
import logging
from pathlib import Path
from src.dataset.prepare import build_mapping_from_outputs_pages
from src.dataset.split import split_mapping
from src.ocr.paragraphs import process_pdf_keep_handwriting, pdf_to_page_images
from src.ocr.recognize import recognize_folder
from src.utils.io import ensure_dir, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def cmd_prepare(src_pdf_or_folder):
    # Convert PDFs to pages (if PDF), else copy images into outputs/pages manually via earlier scripts
    p = Path(src_pdf_or_folder)
    if p.is_file() and p.suffix.lower() in [".pdf"]:
        logging.info("Конвертация PDF → outputs/pages ...")
        pdf_to_page_images(str(p))
    else:
        logging.info("Если вход — папка, убедись, что pages уже в outputs/pages/...")

def cmd_build_mapping():
    logging.info("Сбор mapping.json ...")
    m = build_mapping_from_outputs_pages()
    logging.info(f"mapping создан — {len(m)} пар")

def cmd_split():
    logging.info("Split mapping → train/val")
    train_n, val_n = split_mapping()
    logging.info(f"Train: {train_n}, Val: {val_n}")

def cmd_preview(n=10, backend="auto"):
    # возьмём небольшой сэмпл из dataset/prepared/train/images и распознаём
    from random import sample
    import glob
    imgs = list(Path("dataset/prepared/train/images").glob("*.png"))
    if not imgs:
        logging.error("Нет подготовленных изображений в dataset/prepared/train/images")
        return
    sel = sample(imgs, min(n, len(imgs)))
    for p in sel:
        text = recognize_folder(str(p.parent), backend=backend)  # recognize_folder expects folder
        logging.info(f"Preview {p}: {text}")

def cmd_infer(pdf_path):
    logging.info("Инференс: PDF -> pages -> extract -> recognize")
    # 1) convert pages
    pages = pdf_to_page_images(pdf_path)
    # 2) for each page, extract handwritten paragraphs and recognize
    results = []
    for page in pages:
        saved = process_pdf_keep_handwriting(pdf_path)
        # saved is list of {'crop_path':..., ...}
        for s in saved:
            res = recognize_folder(Path(s['crop_path']).parent, backend="auto")  # better to call per single crop if needed
            results.extend(res)
    out = Path("outputs/final_results.json")
    save_json(results, out)
    logging.info(f"Saved inference results -> {out}")

def usage():
    print("Usage: python runner.py <command> [args]")
    print("commands: prepare <pdf|folder>, build_mapping, split, preview, infer <pdf>")

if __name__=="__main__":
    if len(sys.argv) < 2:
        usage(); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "prepare":
        cmd_prepare(sys.argv[2] if len(sys.argv)>2 else ".")
    elif cmd == "build_mapping":
        cmd_build_mapping()
    elif cmd == "split":
        cmd_split()
    elif cmd == "preview":
        cmd_preview()
    elif cmd == "infer":
        if len(sys.argv)<3:
            print("infer requires pdf path"); sys.exit(1)
        cmd_infer(sys.argv[2])
    else:
        usage()
