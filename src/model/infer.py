import torch
import cv2
from train import get_model
import json

def infer(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("dataset/prepared/mapping.json", "r") as f:
        mapping = json.load(f)
    id2class = {v: k for k, v in mapping.items()}
    num_classes = len(mapping) + 1

    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        preds = model(tensor)[0]

    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        if score > 0.5:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{id2class[label.item()]} {score:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite("result.jpg", img)
    print("✅ Результат сохранен в result.jpg")

if __name__ == "__main__":
    infer("test.jpg", "model_epoch_9.pth")
