import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm
from utils import augment_image

def prepare_dataset(img_dir, labels_dir, output_dir, img_size=640):
    os.makedirs(output_dir, exist_ok=True)

    images_out = Path(output_dir) / "images"
    labels_out = Path(output_dir) / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    mapping = {}
    class_id = 0

    for img_name in tqdm(os.listdir(img_dir)):
        img_path = Path(img_dir) / img_name
        label_path = Path(labels_dir) / (Path(img_name).stem + ".json")

        if not img_path.exists() or not label_path.exists():
            continue

        # Загружаем изображение
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (img_size, img_size))

        # Аугментация (добавляем вариации)
        augmented = augment_image(img)

        # Сохраняем
        out_name = Path(img_name).stem + ".jpg"
        cv2.imwrite(str(images_out / out_name), augmented)

        # Загружаем и обновляем аннотации
        with open(label_path, "r") as f:
            ann = json.load(f)

        for obj in ann["objects"]:
            if obj["class"] not in mapping:
                mapping[obj["class"]] = class_id
                class_id += 1
            obj["class_id"] = mapping[obj["class"]]

        with open(labels_out / (Path(img_name).stem + ".json"), "w") as f:
            json.dump(ann, f, indent=4, ensure_ascii=False)

    # сохраняем mapping.json
    with open(Path(output_dir) / "mapping.json", "w") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
