import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

class SEModule(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM with layer normalization and dropout."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (T, B, C)
        Returns:
            Output tensor of shape (T, B, hidden_size * 2)
        """
        self.rnn.flatten_parameters()
        out, _ = self.rnn(x)  # (T, B, hidden_size * 2)
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for text recognition.
    
    Args:
        imgH: Height of input images (must be multiple of 16)
        nc: Number of input channels (1 for grayscale, 3 for RGB)
        nclass: Number of output classes (including blank for CTC)
        nh: Hidden size of LSTM layers
        leaky_relu: Whether to use LeakyReLU instead of ReLU
        lstm_dropout: Dropout probability for LSTM layers
    """
    def __init__(
        self, 
        imgH: int = 32, 
        nc: int = 1, 
        nclass: int = 80, 
        nh: int = 256,
        leaky_relu: bool = False,
        lstm_dropout: float = 0.3
    ):
        super().__init__()
        assert imgH % 16 == 0, "imgH must be multiple of 16"
        
        # CNN parameters
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]  # number of channels
        
        # Activation function
        self.activation = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
        
        # Build CNN
        cnn = nn.Sequential()
        
        def conv_block(i, batch_norm=False):
            nIn = nc if i == 0 else nm[i-1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            cnn.add_module(f'activation{i}', self.activation)
            if i in [1, 3, 5]:  # Add SE block after some layers
                cnn.add_module(f'se{i}', SEModule(nOut, reduction=16))
        
        # Block 1
        conv_block(0)
        cnn.add_module('pool0', nn.MaxPool2d(2, 2))  # 1/2
        
        # Block 2
        conv_block(1)
        cnn.add_module('pool1', nn.MaxPool2d(2, 2))  # 1/4
        
        # Block 3-4
        conv_block(2, batch_norm=True)
        conv_block(3)
        cnn.add_module('pool2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 1/8
        
        # Block 5-6
        conv_block(4, batch_norm=True)
        conv_block(5)
        cnn.add_module('pool3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 1/16
        
        # Final conv
        conv_block(6, batch_norm=True)
        
        self.cnn = cnn
        
        # RNN
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, num_layers=2, dropout=lstm_dropout),
            nn.Dropout(0.3),
            BidirectionalLSTM(nh * 2, nh, num_layers=1, dropout=lstm_dropout)
        )
        
        # Output layer
        self.embedding = nn.Linear(nh * 2, nclass)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CRNN model.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (T, B, nclass) with log-softmax applied
        """
        # CNN
        conv = self.cnn(x)
        
        # Prepare for RNN
        b, c, h, w = conv.size()
        assert h == 1, f"Expected height=1, got {h}"
        
        # Collapse height dimension and permute for RNN (T, B, C)
        conv = conv.squeeze(2)  # (B, C, W)
        conv = conv.permute(2, 0, 1)  # (W, B, C)
        
        # RNN
        rnn_out = self.rnn(conv)  # (T, B, nh*2)
        
        # Output layer
        output = self.embedding(rnn_out)  # (T, B, nclass)
        
        # Apply log-softmax for CTC loss
        return F.log_softmax(output, dim=2)
    
    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-5):
        """Create an Adam optimizer with weight decay for this model."""
        return torch.optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            amsgrad=True
        )


class SEModule(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
