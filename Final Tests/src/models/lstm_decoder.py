import torch
import torch.nn as nn
import numpy as np
from .positional_encoding import PositionalEncoding1D


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder that takes encoder hidden states and predicts xyz coordinates.
    """
    def __init__(self, input_size, d_model=128, num_heads=8, num_layers=2, dim_feedforward=512, dropout=0.1, output_size=3):
        super().__init__()
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_size, d_model) if input_size != d_model else nn.Identity()
        self.pos_encoder = PositionalEncoding1D(d_model, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: [batch, seq_len, input_size]
        Returns:
            out: [batch, seq_len, 3] - predicted xyz coordinates
        """
        x = self.input_proj(encoder_outputs)
        x = self.pos_encoder(x)
        out = self.transformer_decoder(tgt=x, memory=x)
        out = self.fc_out(out)
        return out


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder that takes encoder hidden states and predicts xyz coordinates.
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1, output_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs):
        """
        Args:
            encoder_outputs: [batch, seq_len, input_size]
        Returns:
            out: [batch, seq_len, output_size]
        """
        x = self.input_proj(encoder_outputs)
        out, _ = self.lstm(x)
        out = self.fc_out(out)
        return out
