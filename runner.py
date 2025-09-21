import os
import sys
import logging

sys.path.append(os.path.abspath("src"))

from src.ocr.detect import process_file, process_folder, load_model
from src.ocr.paragraphs import page_paragraphs_from_image
from src.utils.io import save_txt, ensure_dir

FINAL_OUTPUT = "outputs/final_results.txt"
LOG_FILE = "outputs/logs/runner.log"


def setup_logging():
    ensure_dir("outputs/logs")
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main():
    if len(sys.argv) < 2:
        print("Использование: python runner.py <pdf_файл | папка_с_изображениями>")
        sys.exit(1)

    input_path = sys.argv[1]
    ensure_dir("outputs/pages")

    setup_logging()
    logging.info(f"▶ Старт обработки: {input_path}")

    try:
        # Загружаем модель
        model = load_model()

        # 1. Детекция блоков
        if os.path.isdir(input_path):
            crop_folders = process_folder(input_path, model)
        elif os.path.isfile(input_path):
            crop_folders = process_file(input_path, model)
        else:
            logging.error(f"Неверный путь: {input_path}")
            sys.exit(1)

        # 2. OCR по кропам
        results = []
        for crop_folder in crop_folders:
            if os.path.exists(crop_folder):
                for img in os.listdir(crop_folder):
                    img_path = os.path.join(crop_folder, img)
                    try:
                        text_blocks = page_paragraphs_from_image(img_path)
                        for i, txt in enumerate(text_blocks, 1):
                            results.append((f"{img}_p{i}", txt))
                    except Exception as e:
                        logging.error(f"Ошибка OCR для {img_path}: {e}")

        # 3. Сохраняем финальный результат
        with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
            for img, txt in results:
                f.write(f"[{img}] {txt}\n")

        logging.info(f"✅ Финальный результат сохранён: {FINAL_OUTPUT}")

    except Exception as e:
        logging.exception(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
