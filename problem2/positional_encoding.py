"""
Positional encoding implementations for length extrapolation analysis.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention is All You Need'.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len <= self.max_len:
            positional_emb = self.pe[:seq_len, :].unsqueeze(0)
        else:
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            positional_emb = pe.unsqueeze(0)
        return x + positional_emb


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional embeddings.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embeddings = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(seq_len, device=x.device)
        positions = torch.clamp(positions, max=self.max_len - 1)
        positional_emb = self.position_embeddings(positions).unsqueeze(0).expand(batch_size, -1, -1)
        return x + positional_emb


class NoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        return x


def get_positional_encoding(encoding_type, d_model, max_len=5000):
    encodings = {
        'sinusoidal': SinusoidalPositionalEncoding,
        'learned': LearnedPositionalEncoding,
        'none': NoPositionalEncoding
    }
    if encoding_type not in encodings:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    return encodings[encoding_type](d_model, max_len)


def visualize_positional_encoding(encoding_module, max_len=128, d_model=128):
    input = torch.zeros(1, max_len, d_model)
    with torch.no_grad():
        encoded = encoding_module(input)
        encoding = encoded - input
    return encoding.squeeze(0).cpu().numpy()
