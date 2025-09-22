import torch
from dataset import DetectionDataset
from transforms import get_val_transforms
from train import get_model, collate_fn
from torch.utils.data import DataLoader
import json

def evaluate(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("dataset/prepared/mapping.json", "r") as f:
        mapping = json.load(f)
    num_classes = len(mapping) + 1

    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    dataset_val = DetectionDataset("dataset/prepared/val/images",
                                   "dataset/prepared/val/labels",
                                   transforms=get_val_transforms())
    loader_val = DataLoader(dataset_val, batch_size=2, collate_fn=collate_fn)

    # здесь можно подключить pycocotools для расчета mAP
    print("🔹 Модель загружена и готова к оценке (метрики можно дописать)")
