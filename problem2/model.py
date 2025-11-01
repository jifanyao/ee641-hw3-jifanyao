"""
Transformer encoder model for sequence classification (Problem 2: Positional Encoding and Length Extrapolation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from positional_encoding import get_positional_encoding


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention implementation.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len]  1=valid, 0=pad

        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        B, T, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        # Linear projections
        Q = self.W_q(x).view(B, T, H, Hd).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, Hd).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, Hd).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Hd)

        if mask is not None:
            key_mask = (mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        output = self.W_o(context)
        return output, attn


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder block.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        # ---- Self-Attention ----
        attn_out, _ = self.self_attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # ---- Feed-Forward ----
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SortingClassifier(nn.Module):
    """
    Transformer encoder for sorting detection task.
    Trains with sinusoidal / learned / no positional encoding to study extrapolation.
    """

    def __init__(
        self,
        vocab_size=101,
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=256,
        dropout=0.1,
        max_len=5000,
        encoding_type='sinusoidal'
    ):
        super().__init__()

        self.pad_id = 100
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=self.pad_id)

        self.pos_encoding = get_positional_encoding(encoding_type, d_model, max_len)
        self.encoding_type = encoding_type

        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for linear layers."""
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids, lengths):
        """
        Forward pass.

        Args:
            input_ids: [B, T]  integer sequence
            lengths: [B]  sequence lengths

        Returns:
            logits: [B, 2]
        """
        mask = (input_ids != self.pad_id).long()
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        x = self.encoder(x, mask)

        # mean pooling
        mask_float = mask.unsqueeze(-1).float()
        x_sum = (x * mask_float).sum(dim=1)
        lens = mask_float.sum(dim=1).clamp(min=1.0)
        pooled = x_sum / lens

        logits = self.classifier(pooled)
        return logits

    def predict(self, input_ids, lengths):
        logits = self.forward(input_ids, lengths)
        return logits.argmax(dim=-1)


def create_model(encoding_type='sinusoidal', **kwargs):
    """
    Create SortingClassifier with chosen positional encoding.
    """
    return SortingClassifier(encoding_type=encoding_type, **kwargs)
