"""
Stacked LSTM model for sequence classification
"""

import torch
import torch.nn as nn


class StackedLSTM(nn.Module):
    """
    Stacked (Multi-layer) LSTM Network

    Multiple LSTM layers stacked on top of each other
    for hierarchical temporal feature learning

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden state dimension
        num_layers (int): Number of stacked LSTM layers
        num_classes (int): Number of output classes
        dropout (float): Dropout rate (applied between layers)
    """

    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2,
                 num_classes=50, dropout=0.3):
        super(StackedLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,  # Multiple layers
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout between layers
        )

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
        # Stacked LSTM forward pass
        out, (hidden, cell) = self.lstm(x)

        # Take last time step output
        out = out[:, -1, :]

        # Apply dropout and classification
        out = self.dropout(out)
        out = self.fc(out)

        return out