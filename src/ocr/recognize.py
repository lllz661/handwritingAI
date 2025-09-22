from pathlib import Path
from PIL import Image
import pytesseract
from typing import List, Tuple

# Optional: HuggingFace TrOCR backend (if available)
HAS_TROCR = False
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    HAS_TROCR = True
except Exception:
    HAS_TROCR = False

_trOCR_cache = {}

def load_trocr_model(model_dir_or_name="microsoft/trocr-base-handwritten", device=None):
    if not HAS_TROCR:
        raise RuntimeError("Transformers/torch not available for TrOCR")
    if model_dir_or_name in _trOCR_cache:
        return _trOCR_cache[model_dir_or_name]
    processor = TrOCRProcessor.from_pretrained(model_dir_or_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir_or_name)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    _trOCR_cache[model_dir_or_name] = (processor, model, dev)
    return processor, model, dev

def recognize_with_tesseract(image_path, lang="rus+eng"):
    txt = pytesseract.image_to_string(Image.open(image_path), lang=lang)
    return txt.strip()

def recognize_with_trocr(image_path, model_name="microsoft/trocr-base-handwritten"):
    processor, model, device = load_trocr_model(model_name)
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def recognize_folder(folder_path, backend="auto", out_txt=None):
    """
    Проходит по всем PNG в folder_path и распознаёт.
    backend: "auto" (trocr if available else tesseract), "tesseract", "trocr"
    Возвращает list of (img_path, text)
    """
    folder = Path(folder_path)
    imgs = sorted(folder.glob("*.png"))
    results = []
    if backend=="auto":
        backend = "trocr" if HAS_TROCR else "tesseract"

    for img in imgs:
        try:
            if backend=="trocr":
                text = recognize_with_trocr(str(img))
            else:
                text = recognize_with_tesseract(str(img))
        except Exception as e:
            text = ""
        results.append((str(img), text))

    if out_txt:
        with open(out_txt, "w", encoding="utf-8") as f:
            for img,t in results:
                f.write(f"[{img}] {t}\n")
    return results
