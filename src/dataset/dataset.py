import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
from typing import Dict, List, Tuple, Optional, Union, Any
import cv2
import imgaug.augmenters as iaa

class RandomAugment:
    """Random data augmentation for OCR."""
    def __init__(self, p: float = 0.5):
        self.p = p
        self.augmenter = iaa.Sequential([
            # Geometric transformations
            iaa.Sometimes(0.3, iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-5, 5),
                shear=(-5, 5),
                order=[0, 1],
                cval=255,
                mode='edge'
            )),
            # Noise and blur
            iaa.Sometimes(0.2, iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 0.5)),
                iaa.MotionBlur(k=3, angle=[-45, 45])
            ])),
            # Contrast and brightness
            iaa.Sometimes(0.3, iaa.OneOf([
                iaa.LinearContrast((0.8, 1.2)),
                iaa.MultiplyBrightness((0.8, 1.2)),
            ])),
            # Erosion and dilation
            iaa.Sometimes(0.1, iaa.OneOf([
                iaa.arithmetic.Erode(1, (1, 2)),
                iaa.arithmetic.Dilate(1, (1, 2))
            ]))
        ])

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            return self.augmenter(image=image)
        return image

class ResizeNormalize:
    """Resize and normalize image for CRNN input."""
    def __init__(self, img_height: int = 32, keep_ratio: bool = True):
        self.img_height = img_height
        self.keep_ratio = keep_ratio
        
    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
            
        # Resize
        if self.keep_ratio:
            w, h = img.size
            ratio = h / self.img_height
            new_w = int(w / ratio)
            img = img.resize((new_w, self.img_height), Image.BICUBIC)
        else:
            img = img.resize((self.img_height * 4, self.img_height), Image.BICUBIC)
            
        # Convert to numpy array and normalize
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        
        # Add channel dimension
        img = img[np.newaxis, :, :]
        return torch.FloatTensor(img)

class OCRDataset(Dataset):
    """Dataset for OCR training and evaluation."""
    
    def __init__(
        self,
        root_dir: str,
        labels_file: str = None,
        img_height: int = 32,
        is_training: bool = True,
        augment: bool = True,
        charset: str = ""
    ):
        """
        Args:
            root_dir: Directory containing images
            labels_file: JSON file with image-text mappings
            img_height: Height to resize images to
            is_training: Whether dataset is for training
            augment: Whether to apply data augmentation
            charset: String of all possible characters (if None, will be built from data)
        """
        self.root_dir = root_dir
        self.img_height = img_height
        self.is_training = is_training
        self.augment = augment and is_training
        
        # Load labels
        self.labels = self._load_labels(labels_file) if labels_file else {}
        
        # Get image paths
        if not self.labels:
            # If no labels file, use all images in directory
            self.image_paths = [
                os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
        else:
            self.image_paths = [
                os.path.join(root_dir, img_name) 
                for img_name in self.labels.keys()
                if os.path.exists(os.path.join(root_dir, img_name))
            ]
        
        # Build character set if not provided
        if not charset and self.labels:
            self.charset = self._build_charset()
        else:
            self.charset = charset
            
        # Create character to index mapping
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.charset)}
        self.char2idx[''] = 0  # Blank for CTC
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) if self.augment else transforms.Lambda(lambda x: x),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Setup augmentation
        self.augmenter = RandomAugment(p=0.7) if self.augment else None
        self.resize = ResizeNormalize(img_height=img_height)
        
    def _load_labels(self, labels_file: str) -> Dict[str, str]:
        """Load labels from JSON file."""
        with open(labels_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _build_charset(self) -> str:
        """Build character set from labels."""
        charset = set()
        for text in self.labels.values():
            charset.update(text)
        return ''.join(sorted(charset))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get item by index."""
        img_path = self.image_paths[idx]
        img_name = os.path.basename(img_path)
        
        # Load image
        try:
            img = Image.open(img_path).convert('L')  # Convert to grayscale
        except IOError:
            # Return a random image if loading fails
            return self[random.randint(0, len(self) - 1)]
        
        # Get label
        if self.labels:
            text = self.labels.get(img_name, '')
        else:
            # For inference when no labels are provided
            text = ''
            
        # Apply transforms
        if self.augmenter:
            img_np = np.array(img)
            img_np = self.augmenter(img_np)
            img = Image.fromarray(img_np)
            
        # Resize and normalize
        img = self.resize(img)
        
        return img, text
    
    def get_char2idx(self) -> Dict[str, int]:
        """Get character to index mapping."""
        return self.char2idx
    
    def get_idx2char(self) -> Dict[int, str]:
        """Get index to character mapping."""
        return self.idx2char
    
    def get_charset(self) -> str:
        """Get character set."""
        return self.charset
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights based on text length for balanced sampling."""
        if not self.labels:
            return torch.ones(len(self))
            
        text_lengths = [len(text) for text in self.labels.values()]
        max_len = max(text_lengths) if text_lengths else 1
        weights = [1.0 / (length / max_len + 1e-5) for length in text_lengths]
        return torch.tensor(weights, dtype=torch.float32)


def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (image, text) tuples
        
    Returns:
        Tuple of (images, targets, target_lengths)
            images: Padded image tensor (B, C, H, W)
            targets: Flattened target indices (sum(target_lengths),)
            target_lengths: Length of each target sequence (B,)
    """
    images, texts = zip(*batch)
    
    # Stack images
    max_width = max(img.size(2) for img in images)
    batch_size = len(images)
    
    # Create padded tensor
    padded_images = torch.zeros(batch_size, 1, images[0].size(1), max_width)
    for i, img in enumerate(images):
        padded_images[i, :, :, :img.size(2)] = img
    
    # Convert texts to indices
    targets = []
    target_lengths = []
    for text in texts:
        # Convert characters to indices (assuming char2idx is available)
        # In practice, you'll need to pass char2idx to the collate function
        # or make it available globally
        indices = [ord(c) for c in text]  # Simple ASCII encoding as fallback
        targets.extend(indices)
        target_lengths.append(len(indices))
    
    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return padded_images, targets, target_lengths
