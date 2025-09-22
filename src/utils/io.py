from pathlib import Path
import json
import os


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_txt(text: str, path: str, encoding: str = "utf-8"):
    ensure_dir(Path(path).parent.as_posix())
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


def load_txt(path: str, encoding: str = "utf-8") -> str:
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def save_json(obj, path: str):
    ensure_dir(Path(path).parent.as_posix())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
