import os
import sys
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from pdf_to_images import pdf_to_images
from pathlib import Path

MODEL_DIR = "outputs/model"
OUTPUT_FILE = "outputs/recognized.txt"

def load_model():
    """
    Загружаем обученную модель из outputs/model,
    если её нет — скачиваем с HuggingFace
    """
    if os.path.exists(MODEL_DIR):
        print(f"Загружаем обученную модель из {MODEL_DIR}")
        processor = TrOCRProcessor.from_pretrained(MODEL_DIR)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)
    else:
        print("Обученная модель не найдена, скачиваем базовую microsoft/trocr-base-handwritten")
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def recognize_image(image_path, processor, model, device):
    """
    Распознавание текста с изображения
    """
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=512)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def recognize_file(input_path):
    """
    Распознавание одного файла (pdf или изображения)
    """
    processor, model, device = load_model()
    results = []

    if input_path.lower().endswith(".pdf"):
        print(f"Конвертация PDF {input_path} в изображения...")
        images = pdf_to_images(input_path, output_folder="outputs/temp_images")
        for img_path in images:
            text = recognize_image(img_path, processor, model, device)
            results.append((img_path, text))
    else:
        text = recognize_image(input_path, processor, model, device)
        results.append((input_path, text))

    # Сохраняем результат
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for img, txt in results:
            f.write(f"[{img}] {txt}\n")

    print(f"Результаты сохранены в {OUTPUT_FILE}")
    return results

def recognize_folder(folder_path):
    """
    Распознавание всех картинок в папке
    """
    processor, model, device = load_model()
    results = []
    exts = (".jpg", ".jpeg", ".png")

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(exts):
            img_path = os.path.join(folder_path, fname)
            text = recognize_image(img_path, processor, model, device)
            results.append((img_path, text))

    # Сохраняем результат
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for img, txt in results:
            f.write(f"[{img}] {txt}\n")

    print(f"Результаты сохранены в {OUTPUT_FILE}")
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python ocr_handwriting.py <путь_к_файлу_или_папке>")
        sys.exit(1)

    input_path = sys.argv[1]
    if os.path.isdir(input_path):
        recognize_folder(input_path)
    elif os.path.isfile(input_path):
        recognize_file(input_path)
    else:
        print("Указан неверный путь:", input_path)
