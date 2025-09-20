from pathlib import Path
from pdf2image import convert_from_path
import os

def load_model():
    """
    Заглушка для модели.
    Если понадобится — сюда подключим Detectron2 или другую модель.
    """
    print("⚠️  Внимание: модель пока не используется (load_model заглушка).")
    return None

def process_file(file_path, output_dir=None):
    """
    Конвертирует PDF в PNG страницы.
    Возвращает путь к папке с картинками.
    """
    file_path = Path(file_path)

    if output_dir is None:
        out_dir = Path("outputs") / "pages" / file_path.stem
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Конвертируем {file_path.name} в PNG...")

    pages = convert_from_path(str(file_path), dpi=300)

    for i, page in enumerate(pages, start=1):
        page_path = out_dir / f"{file_path.stem}_page{i}.png"
        page.save(page_path, "PNG")

    return str(out_dir)


def process_folder(folder_path, model, output_dir="outputs"):
    """
    Обрабатывает все PDF из папки.
    Возвращает список папок с PNG.
    """
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_folders = []

    for file_path in folder_path.glob("*.pdf"):
        print(f"Обработка {file_path} ...")
        folder = process_file(file_path, output_dir / file_path.stem)
        crop_folders.append(folder)

    return crop_folders