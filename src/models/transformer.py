"""
Transformer model for sequence classification (built from scratch)
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer

    Adds positional information to input sequences since
    Transformers don't have inherent notion of sequence order

    Args:
        d_model (int): Dimension of model
        max_len (int): Maximum sequence length
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer for Sequence Classification

    Uses multi-head self-attention to capture temporal dependencies

    Args:
        input_dim (int): Input feature dimension
        d_model (int): Dimension of transformer model
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer encoder layers
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
    """

    def __init__(self, input_dim=2048, d_model=512, nhead=8, num_layers=4,
                 num_classes=50, dropout=0.3):
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # Project input to d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Project input to d_model
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling across sequence
        x = torch.mean(x, dim=1)

        # Apply dropout and classification
        x = self.dropout(x)
        out = self.fc(x)

        return out