import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import DetectionDataset
from transforms import get_train_transforms, get_val_transforms
from pathlib import Path

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DetectionDataset("dataset/prepared/train/images",
                               "dataset/prepared/train/labels",
                               transforms=get_train_transforms())
    dataset_val = DetectionDataset("dataset/prepared/val/images",
                                   "dataset/prepared/val/labels",
                                   transforms=get_val_transforms())

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn)

    with open("dataset/prepared/mapping.json", "r") as f:
        import json
        mapping = json.load(f)
    num_classes = len(mapping) + 1  # + background

    model = get_model(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch}: loss={losses.item()}")

        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()
