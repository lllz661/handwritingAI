
import os
from pathlib import Path

def split_paragraphs(gt_dir: str, out_dir: str) -> None:
    """
    Делит текстовые файлы разметки на абзацы и сохраняет их в отдельные файлы.

    Args:
        gt_dir (str): директория с ground truth (.txt).
        out_dir (str): директория для сохранённых абзацев.
    """
    gt_dir = Path(gt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in gt_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        save_dir = out_dir / txt_file.stem
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, para in enumerate(paragraphs, start=1):
            out_file = save_dir / f"para_{i:03d}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(para)

        print(f"[OK] {txt_file} → {len(paragraphs)} абзацев сохранено в {save_dir}")
