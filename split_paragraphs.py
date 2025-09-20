import os
import re

GROUND_TRUTH_DIR = "dataset/ground_truth"
OUTPUT_DIR = "outputs/pages"

def split_paragraphs(text):
    """Разделение текста на абзацы (по пустым строкам или \n\n)."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs

def process_file(file_path, output_folder, base_name):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = split_paragraphs(text)

    os.makedirs(output_folder, exist_ok=True)

    for idx, para in enumerate(paragraphs, start=1):
        out_file = os.path.join(output_folder, f"{base_name}_page1_paragraph{idx}.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(para)

    print(f"[OK] {file_path} → {len(paragraphs)} абзацев сохранено в {output_folder}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file_name in os.listdir(GROUND_TRUTH_DIR):
        if file_name.endswith(".txt"):
            input_path = os.path.join(GROUND_TRUTH_DIR, file_name)
            doc_name = os.path.splitext(file_name)[0]

            # Папка для картинок и текста
            output_folder = os.path.join(OUTPUT_DIR, doc_name)

            process_file(input_path, output_folder, doc_name)

if __name__ == "__main__":
    main()
