from pathlib import Path
from pdf2image import convert_from_path
import os
from src.utils.io import ensure_dir

def load_model():
    """Заглушка. Позже можно подключить layoutparser/detectron."""
    return None

def process_file(file_path: str, output_dir: str = None, dpi: int = 300) -> str:
    """
    Если file_path — PDF, конвертируем в PNG страницу(ы) в папку outputs/pages/<stem>/
    Если это изображение — копируем в ту же папку.
    Возвращает путь к папке с изображениями (строкой).
    """
    file_path = Path(file_path)
    if output_dir is None:
        out_dir = Path("outputs") / "pages" / file_path.stem
    else:
        out_dir = Path(output_dir) / file_path.stem
    ensure_dir(str(out_dir))

    if file_path.suffix.lower() == ".pdf":
        pages = convert_from_path(str(file_path), dpi=dpi)
        for i, page in enumerate(pages, start=1):
            fn = out_dir / f"{file_path.stem}_page{i:03d}.png"
            page.save(fn, "PNG")
    else:
        # single image
        dest = out_dir / file_path.name
        if not dest.exists():
            from shutil import copy2
            copy2(file_path, dest)

    return str(out_dir)

def process_folder(folder_path: str, model=None, exts=(".pdf", ".png", ".jpg", ".jpeg")):
    folder = Path(folder_path)
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            out_dir = process_file(str(p))
            out.append(out_dir)
    return out
