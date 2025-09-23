import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import shutil

from src.dataset.dataset import OCRDataset, collate_fn
from src.models.crnn import CRNN
from src.utils.text_utils import TextEncoder, calculate_cer, calculate_wer, ctc_decode

class ModelCheckpoint:
    """Save model checkpoints and track the best model."""
    
    def __init__(
        self, 
        save_dir: str, 
        monitor: str = 'val_loss', 
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor ('val_loss', 'val_cer', etc.)
            mode: 'min' or 'max' for the monitored metric
            save_best_only: If True, only save when the monitored metric improves
            save_weights_only: If True, only save model weights, not the full model
            verbose: Whether to print messages when saving models
        """
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.best_score = float('inf') if mode == 'min' else -float('inf')
        self.best_epoch = -1
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, epoch: int, model: nn.Module, metrics: Dict[str, float]) -> None:
        """Save model checkpoint if the monitored metric has improved."""
        if self.monitor not in metrics:
            if self.verbose:
                print(f"Warning: Metric '{self.monitor}' not found in metrics. Available metrics: {list(metrics.keys())}")
            return
        
        current_score = metrics[self.monitor]
        is_best = False
        
        # Check if current score is better than best score
        if (self.mode == 'min' and current_score < self.best_score) or \
           (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            is_best = True
        
        # Save checkpoint
        if not self.save_best_only or is_best:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': model.optimizer.state_dict() if hasattr(model, 'optimizer') else None,
                'scheduler': model.scheduler.state_dict() if hasattr(model, 'scheduler') else None,
                'metrics': metrics,
                'best_score': self.best_score,
                'best_epoch': self.best_epoch,
            }
            
            # Add text encoder if available
            if hasattr(model, 'text_encoder'):
                checkpoint['text_encoder'] = model.text_encoder
            
            # Save checkpoint
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch:03d}.pth'
            best_path = self.save_dir / 'model_best.pth'
            
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                shutil.copyfile(checkpoint_path, best_path)
                if self.verbose:
                    print(f"\nSaved best model to {best_path} with {self.monitor} = {current_score:.4f}")
            
            if self.verbose and not is_best:
                print(f"\nSaved checkpoint to {checkpoint_path}")


def train_model(
    data_dir: str,
    labels_file: str,
    output_dir: str = 'outputs',
    img_height: int = 32,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 10,
    checkpoint_monitor: str = 'val_loss',
    checkpoint_mode: str = 'min',
    log_dir: str = 'logs',
    num_workers: int = 4,
    seed: int = 42,
    gpu_id: int = 0,
    resume: Optional[str] = None
) -> None:
    """
    Train a CRNN model for text recognition.
    
    Args:
        data_dir: Directory containing training images
        labels_file: Path to JSON file with image-text mappings
        output_dir: Directory to save model checkpoints and logs
        img_height: Height to resize images to
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        patience: Number of epochs to wait before early stopping
        checkpoint_monitor: Metric to monitor for checkpointing
        checkpoint_mode: 'min' or 'max' for the monitored metric
        log_dir: Directory to save TensorBoard logs
        num_workers: Number of workers for data loading
        seed: Random seed
        gpu_id: GPU ID to use (-1 for CPU)
        resume: Path to checkpoint to resume training from
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id >= 0 else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(output_dir)
    model_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load dataset
    train_dataset = OCRDataset(
        root_dir=data_dir,
        labels_file=labels_file,
        img_height=img_height,
        is_training=True,
        augment=True
    )
    
    # Split into train and validation sets
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    charset = train_dataset.get_charset()
    num_classes = len(charset) + 1  # +1 for CTC blank
    
    model = CRNN(
        imgH=img_height,
        nc=1,  # grayscale
        nclass=num_classes,
        nh=256,
        leaky_relu=True
    ).to(device)
    
    # Initialize text encoder
    text_encoder = TextEncoder(charset)
    model.text_encoder = text_encoder  # Attach to model for checkpointing
    
    # Loss function
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Optimizer and learning rate scheduler
    optimizer = model.get_optimizer(lr=lr, weight_decay=weight_decay)
    model.optimizer = optimizer  # Attach to model for checkpointing
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    model.scheduler = scheduler  # Attach to model for checkpointing
    
    # Initialize model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        save_dir=model_dir,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_best_only=True,
        verbose=True
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume and os.path.isfile(resume):
        print(f"Loading checkpoint from {resume}")
        checkpoint = torch.load(resume, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load optimizer state
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load scheduler state
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Update start epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # Update best score
        if 'best_score' in checkpoint:
            checkpoint_callback.best_score = checkpoint['best_score']
        
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        train_loss, train_metrics = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs
        )
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'lr': optimizer.param_groups[0]['lr']
        }
        
        # Log to TensorBoard
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | " + 
              f"Val Loss: {val_loss:.4f} | " +
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save checkpoint
        checkpoint_callback(epoch, model, metrics)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs without improvement")
                break
    
    # Close TensorBoard writer
    writer.close()
    print("Training complete!")


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> Tuple[float, Dict[str, float]]:
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0
    
    # Initialize progress bar
    pbar = tqdm(
        data_loader,
        desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
        dynamic_ncols=True
    )
    
    for batch_idx, (images, texts) in enumerate(pbar):
        # Move data to device
        images = images.to(device, non_blocking=True)
        
        # Encode targets
        targets, target_lengths = model.text_encoder.encode_batch(texts)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        # Forward pass
        logits = model(images)
        
        # Calculate loss
        input_lengths = torch.full(
            size=(logits.size(1),),
            fill_value=logits.size(0),
            dtype=torch.long,
            device=device
        )
        
        loss = criterion(
            logits.log_softmax(2).permute(1, 0, 2),  # (T, B, C) -> (B, T, C)
            targets,
            input_lengths,
            target_lengths
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # Gradient clipping
        optimizer.step()
        
        # Calculate metrics
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Calculate CER and WER
        pred_texts = model.text_encoder.decode_batch(logits.argmax(dim=2).permute(1, 0))  # (T, B) -> (B, T)
        
        for pred, target in zip(pred_texts, texts):
            total_cer += calculate_cer(pred, target)
            total_wer += calculate_wer(pred, target)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / total_samples,
            'cer': total_cer / (batch_idx + 1),
            'wer': total_wer / (batch_idx + 1)
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / total_samples
    avg_cer = total_cer / len(data_loader)
    avg_wer = total_wer / len(data_loader)
    
    metrics = {
        'cer': avg_cer,
        'wer': avg_wer,
        'accuracy': 1.0 - avg_cer  # Approximate accuracy
    }
    
    return avg_loss, metrics


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_samples = 0
    
    with torch.no_grad():
        # Initialize progress bar
        pbar = tqdm(
            data_loader,
            desc=f"Epoch {epoch+1} [Val]",
            dynamic_ncols=True,
            leave=False
        )
        
        for batch_idx, (images, texts) in enumerate(pbar):
            # Move data to device
            images = images.to(device, non_blocking=True)
            
            # Encode targets
            targets, target_lengths = model.text_encoder.encode_batch(texts)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            logits = model(images)
            
            # Calculate loss
            input_lengths = torch.full(
                size=(logits.size(1),),
                fill_value=logits.size(0),
                dtype=torch.long,
                device=device
            )
            
            loss = criterion(
                logits.log_softmax(2).permute(1, 0, 2),  # (T, B, C) -> (B, T, C)
                targets,
                input_lengths,
                target_lengths
            )
            
            # Calculate metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate CER and WER
            pred_texts = model.text_encoder.decode_batch(logits.argmax(dim=2).permute(1, 0))  # (T, B) -> (B, T)
            
            for pred, target in zip(pred_texts, texts):
                total_cer += calculate_cer(pred, target)
                total_wer += calculate_wer(pred, target)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total_samples,
                'cer': total_cer / (batch_idx + 1),
                'wer': total_wer / (batch_idx + 1)
            })
    
    # Calculate epoch metrics
    avg_loss = total_loss / total_samples
    avg_cer = total_cer / len(data_loader)
    avg_wer = total_wer / len(data_loader)
    
    metrics = {
        'cer': avg_cer,
        'wer': avg_wer,
        'accuracy': 1.0 - avg_cer  # Approximate accuracy
    }
    
    return avg_loss, metrics

    return model_save
