import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from typing import Tuple

class OCRDataset(Dataset):
    """
    Simple dataset that returns (image_tensor, text_str).
    Images are transformed to fixed height; width is resized preserving aspect ratio and then padded.
    """
    def __init__(self, labels_json: str, img_h: int = 32, max_width: int = 512):
        with open(labels_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = list(data.items())  # [(img_path, text), ...]
        self.img_h = img_h
        self.max_width = max_width
        self.transform = T.Compose([
            T.ToTensor(),  # will produce [C,H,W] in 0..1
        ])

    def __len__(self):
        return len(self.items)

    def _load_image(self, path):
        img = Image.open(path).convert("L")  # grayscale
        w, h = img.size
        new_h = self.img_h
        new_w = max(10, int(w * (new_h / float(h))))
        # cap width at max_width
        if new_w > self.max_width:
            new_w = self.max_width
        img = img.resize((new_w, new_h), Image.BILINEAR)
        img = self.transform(img)  # 1 x H x W
        # pad width to max_width
        if img.shape[2] < self.max_width:
            pad = self.max_width - img.shape[2]
            import torch
            img = torch.nn.functional.pad(img, (0, pad, 0, 0), value=1.0)  # pad with white
        return img

    def __getitem__(self, idx):
        img_path, text = self.items[idx]
        img = self._load_image(img_path)
        return img, text
