import os
import argparse
from pathlib import Path
from pdf2image import convert_from_path

DATASET_DIR = Path("dataset")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"


def pdf_to_images(pdf_path: str, dpi: int = 300, fmt: str = "jpg"):
    """
    Конвертирует PDF в изображения и создаёт соответствующие txt-файлы для меток.

    Args:
        pdf_path (str): путь к PDF файлу
        dpi (int): разрешение для конвертации
        fmt (str): формат изображений ("jpg" или "png")
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Файл {pdf_path} не найден")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Конвертация {pdf_path} в изображения...")
    pages = convert_from_path(pdf_path, dpi=dpi)

    for idx, page in enumerate(pages, start=1):
        image_filename = IMAGES_DIR / f"{pdf_path.stem}_page_{idx:03d}.{fmt}"
        label_filename = LABELS_DIR / f"{pdf_path.stem}_page_{idx:03d}.txt"

        page.save(image_filename, fmt.upper())
        print(f"[INFO] Сохранено изображение: {image_filename}")

        if not label_filename.exists():
            with open(label_filename, "w", encoding="utf-8") as f:
                f.write("")  # сюда вручную вписываете текст
            print(f"[INFO] Создан файл меток: {label_filename}")

    print("[INFO] Конвертация завершена.")


def main():
    parser = argparse.ArgumentParser(description="Конвертация PDF в изображения для датасета")
    parser.add_argument("pdf_path", type=str, help="Путь к PDF файлу")
    parser.add_argument("--dpi", type=int, default=300, help="Разрешение при конвертации (по умолчанию 300)")
    parser.add_argument("--fmt", type=str, default="jpg", choices=["jpg", "png"], help="Формат сохранения")
    args = parser.parse_args()

    pdf_to_images(args.pdf_path, dpi=args.dpi, fmt=args.fmt)


if __name__ == "__main__":
    main()
