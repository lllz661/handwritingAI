from typing import List, Tuple
import torch

def collate_fn(batch: List[Tuple]):
    """
    batch: list of (image_tensor [1,H,W], text_str)
    Return: images tensor [B,1,H,W], targets_concat, target_lengths, list_texts
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    texts = [b[1] for b in batch]
    return images, texts
