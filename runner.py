import sys
import logging
from pathlib import Path
from src.dataset.prepare import prepare_dataset
from src.ocr.detect import process_folder, process_file, load_model
from src.dataset.split import split_ground_truth
from src.ocr.recognize import recognize_folder
from src.utils.io import ensure_dir

LOG_FILE = Path("outputs/logs/runner.log")
FINAL_OUTPUT = Path("outputs/final_results.txt")

def setup_logging():
    ensure_dir(str(LOG_FILE.parent))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    if len(sys.argv) < 2:
        print("Использование: python runner.py <путь_к_папке_с_файлами_или_pdf>")
        sys.exit(1)

    setup_logging()
    src_path = Path(sys.argv[1])
    logging = logging.getLogger()

    logging.info(f"Старт пайплайна. Вход: {src_path}")

    # 0. Подготовка: скопировать файлы в dataset/
    ds = prepare_dataset(str(src_path), target_root="dataset")
    images_dir = Path(ds["images"])
    gt_dir = Path(ds["ground_truth"])

    # 1. Конвертация всех PDF/изображений -> outputs/pages/<stem>/
    ensure_dir("outputs/pages")
    model = load_model()
    crop_folders = []
    if images_dir.exists():
        crop_folders = process_folder(str(images_dir), model)
    else:
        logging.error("dataset/images не найден")
        return

    # 2. Разрезаем ground_truth на абзацы рядом с картинками (outputs/pages)
    if gt_dir.exists():
        report = split_ground_truth(str(gt_dir), out_pages_root="outputs/pages")
        logging.info("Разрезаны ground_truth -> outputs/pages (см. report)")
    else:
        logging.warning("dataset/ground_truth не найден — пропускаем шаг split")

    # 3. OCR: для каждой папки с картинками распознаём по-кропам (если они есть)
    results = []
    for folder in crop_folders:
        logging.info(f"OCR папки {folder}")
        res = recognize_folder(folder, out_txt=None)
        results.extend(res)

    # 4. Сохраняем финальные результаты
    ensure_dir(str(FINAL_OUTPUT.parent))
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        for img, txt in results:
            f.write(f"[{img}] {txt}\n")
    logging.info(f"Готово — финальный результат: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()
