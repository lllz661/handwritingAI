from PIL import Image
import pytesseract
from pathlib import Path
from src.utils.io import ensure_dir

def recognize_image_pytesseract(image_path: str, lang: str = "rus+eng") -> str:
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img, lang=lang)
    return text.strip()

def recognize_folder(folder_path: str, out_txt: str = None, lang: str = "rus+eng"):
    """
    Проходит по всем изображениям в папке и распознаёт их.
    Возвращает список (img_path, text).
    Если out_txt указан — также сохранит в файл.
    """
    folder = Path(folder_path)
    exts = (".png", ".jpg", ".jpeg")
    results = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts:
            try:
                txt = recognize_image_pytesseract(str(p), lang=lang)
            except Exception as e:
                txt = ""
            results.append((str(p), txt))

    if out_txt:
        ensure_dir(str(Path(out_txt).parent))
        with open(out_txt, "w", encoding="utf-8") as f:
            for img, txt in results:
                f.write(f"[{img}] {txt}\n")
    return results
