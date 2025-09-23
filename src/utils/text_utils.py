"""Utility functions for text processing in OCR."""
from typing import List, Dict, Tuple, Optional, Union
import re
import unicodedata
import torch

class TextEncoder:
    """Handles text encoding and decoding for OCR tasks."""
    
    def __init__(self, charset: str = ""):
        """
        Initialize text encoder with character set.
        
        Args:
            charset: String containing all possible characters in the dataset.
                    If empty, will be built from data.
        """
        self.charset = charset
        self.char2idx = {}
        self.idx2char = {}
        self._build_mappings()
        
    def _build_mappings(self) -> None:
        """Build character to index and index to character mappings."""
        # Add special tokens
        self.char2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        # Add characters from charset
        for i, char in enumerate(self.charset, start=len(self.char2idx)):
            self.char2idx[char] = i
            
        # Build reverse mapping
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to sequence of indices.
        
        Args:
            text: Input text string
            max_length: Optional maximum length (pads or truncates)
            
        Returns:
            List of indices
        """
        # Convert to unicode and normalize
        text = self.normalize_text(text)
        
        # Convert characters to indices
        indices = [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]
        
        # Add start/end tokens and handle length
        indices = [self.char2idx['<SOS>']] + indices + [self.char2idx['<EOS>']]
        
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length-1] + [self.char2idx['<EOS>']]
            else:
                indices += [self.char2idx['<PAD>']] * (max_length - len(indices))
                
        return indices
    
    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Decode sequence of indices to text.
        
        Args:
            indices: List of indices
            remove_special: Whether to remove special tokens
            
        Returns:
            Decoded text string
        """
        chars = []
        for idx in indices:
            if idx in self.idx2char:
                char = self.idx2char[idx]
                if remove_special and char in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                    continue
                chars.append(char)
        return ''.join(chars)
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for consistent encoding.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to unicode
        text = str(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters
        text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size including special tokens."""
        return len(self.char2idx)
    
    def save(self, path: str) -> None:
        """Save encoder to file."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'charset': self.charset,
                'char2idx': self.char2idx,
                'idx2char': self.idx2char
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TextEncoder':
        """Load encoder from file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        encoder = cls(data['charset'])
        encoder.char2idx = {k: int(v) for k, v in data['char2idx'].items()}
        encoder.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        return encoder


def calculate_cer(pred: str, target: str) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        pred: Predicted text
        target: Target text
        
    Returns:
        CER as a float (lower is better)
    """
    import numpy as np
    
    # Handle empty strings
    if not target:
        return 1.0 if pred else 0.0
    
    # Initialize matrix for dynamic programming
    dp = np.zeros((len(pred) + 1, len(target) + 1))
    dp[0] = np.arange(len(target) + 1)
    dp[:, 0] = np.arange(len(pred) + 1)
    
    # Fill the matrix
    for i in range(1, len(pred) + 1):
        for j in range(1, len(target) + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[-1][-1] / len(target)


def calculate_wer(pred: str, target: str) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        pred: Predicted text
        target: Target text
        
    Returns:
        WER as a float (lower is better)
    """
    # Split into words
    pred_words = pred.split()
    target_words = target.split()
    
    # Handle empty strings
    if not target_words:
        return 1.0 if pred_words else 0.0
    
    # Initialize matrix for dynamic programming
    dp = [[0] * (len(target_words) + 1) for _ in range(len(pred_words) + 1)]
    
    # Fill first row and column
    for i in range(len(pred_words) + 1):
        dp[i][0] = i
    for j in range(len(target_words) + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(pred_words) + 1):
        for j in range(1, len(target_words) + 1):
            if pred_words[i-1] == target_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[-1][-1] / len(target_words)


def ctc_decode(
    logits: torch.Tensor, 
    method: str = 'greedy', 
    beam_width: int = 10,
    blank: int = 0,
    **kwargs
) -> List[str]:
    """
    Decode CTC output to text.
    
    Args:
        logits: Model output tensor of shape (T, B, C)
        method: Decoding method ('greedy' or 'beam_search')
        beam_width: Beam width for beam search
        blank: Index of blank token
        
    Returns:
        List of decoded strings (one per batch item)
    """
    if method == 'greedy':
        return _ctc_greedy_decode(logits, blank=blank)
    elif method == 'beam_search':
        return _ctc_beam_search(logits, beam_width=beam_width, blank=blank, **kwargs)
    else:
        raise ValueError(f"Unknown decoding method: {method}")


def _ctc_greedy_decode(logits: torch.Tensor, blank: int = 0) -> List[str]:
    """Greedy decoding for CTC."""
    _, max_indices = torch.max(logits, dim=2)  # (T, B)
    max_indices = max_indices.transpose(0, 1)  # (B, T)
    
    decoded = []
    for indices in max_indices:
        # Remove repeated characters and blanks
        prev = -1
        text = []
        for idx in indices:
            idx = idx.item()
            if idx != prev and idx != blank:
                text.append(idx)
            prev = idx if idx != blank else -1
        decoded.append(text)
    
    return decoded


def _ctc_beam_search(logits: torch.Tensor, beam_width: int, blank: int = 0) -> List[str]:
    """Beam search decoding for CTC."""
    # This is a simplified version - in practice, you'd want to use a more efficient implementation
    # like the one in torchaudio or other libraries
    
    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=2)  # (T, B, C)
    probs = probs.transpose(0, 1)  # (B, T, C)
    
    batch_size = probs.size(0)
    decoded = []
    
    for b in range(batch_size):
        # Initialize beams (probability, sequence)
        beams = [([], 1.0)]
        
        # Process each time step
        for t in range(probs.size(1)):
            new_beams = []
            
            # Expand each beam
            for seq, score in beams:
                # Option 1: Take the most probable character
                topk_probs, topk_indices = torch.topk(probs[b, t], k=beam_width)
                
                for i in range(beam_width):
                    char = topk_indices[i].item()
                    prob = topk_probs[i].item()
                    
                    new_seq = seq.copy()
                    # If same as previous, merge
                    if char != blank and (not new_seq or char != new_seq[-1]):
                        new_seq.append(char)
                    
                    new_beams.append((new_seq, score * prob))
            
            # Keep only top-k beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Get the best sequence
        if beams:
            best_seq = max(beams, key=lambda x: x[1])[0]
            decoded.append(best_seq)
        else:
            decoded.append([])
    
    return decoded
