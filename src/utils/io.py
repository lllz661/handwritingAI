from pathlib import Path
import json

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_txt(path, text, encoding="utf-8"):
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding=encoding)

def load_txt(path, encoding="utf-8"):
    return Path(path).read_text(encoding=encoding)

def save_json(obj, path):
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
