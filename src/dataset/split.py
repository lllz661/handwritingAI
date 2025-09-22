import re
from pathlib import Path
from src.utils.io import ensure_dir

def split_into_paragraphs(text: str):
    parts = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(parts) <= 1:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) > 1:
            parts = lines
        else:
            parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    return parts

def split_ground_truth(gt_dir: str, out_pages_root: str = "outputs/pages"):
    gt_dir = Path(gt_dir)
    out_root = Path(out_pages_root)
    ensure_dir(str(out_root))

    report = {"files": []}
    for txt in sorted(gt_dir.glob("*.txt")):
        text = txt.read_text(encoding="utf-8")
        parts = split_into_paragraphs(text)
        doc_folder = out_root / txt.stem
        ensure_dir(str(doc_folder))
        mapping = []
        for i, p in enumerate(parts, start=1):
            out_file = doc_folder / f"{txt.stem}_page1_paragraph{i}.txt"
            out_file.write_text(p, encoding="utf-8")
            mapping.append(str(out_file))
        report["files"].append({"source": str(txt), "paragraphs": mapping})
    return report
