"""
Swin Transformer-inspired model for sequence classification
"""

import torch
import torch.nn as nn


class SwinTransformerModel(nn.Module):
    """
    Swin Transformer-inspired architecture for sequence classification

    Simplified version using multi-head attention and feed-forward networks

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
    """

    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=50, dropout=0.3):
        super(SwinTransformerModel, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, num_classes)
        """
        # Input projection
        x = self.input_proj(x)

        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Classification
        x = self.dropout(x)
        out = self.fc(x)

        return out