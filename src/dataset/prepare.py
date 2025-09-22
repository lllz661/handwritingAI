from pathlib import Path
from shutil import copy2
from src.utils.io import ensure_dir

def prepare_dataset(src_folder: str, target_root: str = "dataset"):
    """
    Копирует в target_root/images и target_root/ground_truth:
    - все PNG/JPG/PDF в images (PDF оставляем, конвертация — позже)
    - все TXT в ground_truth
    """
    src = Path(src_folder)
    tgt = Path(target_root)
    img_tgt = tgt / "images"
    gt_tgt = tgt / "ground_truth"
    ensure_dir(str(img_tgt))
    ensure_dir(str(gt_tgt))

    for p in sorted(src.iterdir()):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf"):
            copy2(str(p), str(img_tgt / p.name))
        elif p.suffix.lower() == ".txt":
            copy2(str(p), str(gt_tgt / p.name))

    return {"images": str(img_tgt), "ground_truth": str(gt_tgt)}
