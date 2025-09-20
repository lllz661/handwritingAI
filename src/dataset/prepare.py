
import os
import shutil
from pathlib import Path

def prepare_dataset(raw_dir: str, target_dir: str) -> None:
    """
    Готовит датасет: копирует изображения и тексты в структуру проекта.

    Args:
        raw_dir (str): путь к сырым данным (pdf/jpg/png/txt).
        target_dir (str): путь к целевой папке датасета.
    """
    raw_dir = Path(raw_dir)
    target_dir = Path(target_dir)

    images_dir = target_dir / "images"
    gt_dir = target_dir / "ground_truth"

    images_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for file in raw_dir.iterdir():
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            shutil.copy(file, images_dir / file.name)
        elif file.suffix.lower() == ".txt":
            shutil.copy(file, gt_dir / file.name)

    print(f"[OK] Датасет подготовлен: {target_dir}")
