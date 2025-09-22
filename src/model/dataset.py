import torch
from torch.utils.data import Dataset
import cv2
import json
from pathlib import Path

class DetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images = list(Path(images_dir).glob("*.jpg"))
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels_dir / (img_path.stem + ".json")

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(lbl_path, "r") as f:
            ann = json.load(f)

        boxes, labels = [], []
        for obj in ann["objects"]:
            x, y, w, h = obj["bbox"]  # формат [x, y, w, h]
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["class_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes, labels=labels)
            img = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)

        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        return img, target
