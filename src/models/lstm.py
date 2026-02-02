"""
LSTM model for sequence classification
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory Network

    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden state dimension
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
    """

    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=50, dropout=0.3):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0 if dropout == 0 else dropout
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
        # LSTM forward pass
        out, (hidden, cell) = self.lstm(x)

        # Take last time step output
        out = out[:, -1, :]

        # Apply dropout and classification
        out = self.dropout(out)
        out = self.fc(out)

        return out