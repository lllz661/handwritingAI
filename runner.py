import os
import sys
from detect_blocks import process_file, process_folder, load_model
from ocr_handwriting import recognize_folder

FINAL_OUTPUT = "outputs/final_results.txt"

def main():
    if len(sys.argv) < 2:
        print("Использование: python runner.py <pdf_файл | папка_с_изображениями>")
        sys.exit(1)

    input_path = sys.argv[1]
    os.makedirs("outputs", exist_ok=True)

    model = load_model()

    # 1. Блоки
    if os.path.isdir(input_path):
        crop_folders = process_folder(input_path, model)
    elif os.path.isfile(input_path):
        crop_folders = process_file(input_path, model)
    else:
        print("Неверный путь:", input_path)
        sys.exit(1)

    # 2. OCR по кропам
    results = []
    for crop_folder in crop_folders:
        if os.path.exists(crop_folder):
            folder_results = recognize_folder(crop_folder)
            results.extend(folder_results)

    # 3. Сохраняем финальный результат
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        for img, txt in results:
            f.write(f"[{img}] {txt}\n")

    print(f"\n Финальный результат сохранён: {FINAL_OUTPUT}")


if __name__ == "__main__":
    main()
