import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from src.models.crnn import CRNN
from src.dataset.dataset import OCRDataset
import json
import numpy as np


def load_model(checkpoint_path: str, device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[int, str]]:
    """Load model and vocabulary from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on. If None, will use GPU if available.
        
    Returns:
        A tuple containing:
            - The loaded model
            - String to index mapping (stoi)
            - Index to string mapping (itos)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get vocabulary mappings
    stoi = checkpoint.get('stoi', {})
    itos = checkpoint.get('itos', {})
    
    # Initialize model
    nclass = len(stoi) + 1  # +1 for blank token
    model = CRNN(imgH=32, nc=1, nclass=nclass, nh=256)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    return model, stoi, itos

def load_checkpoint(path):
    chk = torch.load(path, map_location="cpu")
    stoi = chk.get("stoi")
    itos = chk.get("itos")
    nclass = len(stoi) + 1
    model = CRNN(imgH=32, nc=1, nclass=nclass, nh=256)
    model.load_state_dict(chk["model_state"])
    model.eval()
    return model, stoi, itos

def greedy_decode(probs, itos):
    """
    probs: numpy array [T, nclass] (log-probs or probs)
    itos: dict idx->char (1..)
    returns string
    """
    idxs = np.argmax(probs, axis=1)
    prev = -1
    out = []
    for i in idxs:
        if i != prev and i != 0:
            ch = itos.get(int(i), "")
            out.append(ch)
        prev = i
    return "".join(out)

def predict_image(img_path, model_checkpoint_path="outputs/models/crnn.pth", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, stoi, itos = load_checkpoint(model_checkpoint_path)
    model = model.to(device)
    # load and preprocess image using same pipeline as dataset
    from PIL import Image
    img = Image.open(img_path).convert("L")
    w, h = img.size
    new_h = 32
    new_w = max(10, int(w * (new_h / float(h))))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    import torchvision.transforms as T
    t = T.ToTensor()
    tensor = t(img).unsqueeze(0)  # [1,1,H,W]
    # pad to width used during training (dataset defaults to max_width=512)
    maxw = 512
    if tensor.shape[3] < maxw:
        import torch.nn.functional as F
        tensor = F.pad(tensor, (0, maxw - tensor.shape[3], 0, 0), value=1.0)
    tensor = tensor.to(device)
    with torch.no_grad():
        preds = model(tensor)  # [W, B, nclass]
        probs = preds.squeeze(1).cpu().numpy()  # [W, nclass]
        # preds already log_softmax; convert to exp
        probs = np.exp(probs)
    text = greedy_decode(probs, itos)
    return text
