import os
import shutil
import random
from pathlib import Path

def split_dataset(dataset_dir, val_size=0.15, test_size=0.1):
    images = list((Path(dataset_dir) / "images").glob("*.jpg"))
    random.shuffle(images)

    n_total = len(images)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)

    splits = {
        "train": images[n_test + n_val:],
        "val": images[n_test:n_test + n_val],
        "test": images[:n_test],
    }

    for split, files in splits.items():
        img_out = Path(dataset_dir) / split / "images"
        lbl_out = Path(dataset_dir) / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            lbl_path = Path(dataset_dir) / "labels" / (img_path.stem + ".json")
            shutil.copy(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out / lbl_path.name)

    print(f"✅ Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
