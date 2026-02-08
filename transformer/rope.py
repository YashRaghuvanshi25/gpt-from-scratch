"""
Rotary Positional Embedding (RoPE) implementation used to inject relative
positional information into attention queries and keys by rotating pairs of
feature dimensions. RoPE enables the model to encode token positions without
adding explicit positional vectors.
"""

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.d_k = d_k

        # Compute inverse frequency terms for each pair of feature dimensions
        k = torch.arange(0, d_k, 2, device=device)
        inv_freq = 1.0 / (theta ** (k / d_k))

        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions, inv_freq)

        # Precompute cosine and sine values for all positions up to max_seq_len
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        # x has shape (B, T, H, D) where D is the per-head feature dimension
        B, T, H, D = x.shape

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Select precomputed positional rotations corresponding to current sequence length
        cos = self.cos[:T].to(x.device)  # (T, D/2)
        sin = self.sin[:T].to(x.device)

        # Reshape for broadcasting across batch and heads
        cos = cos.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D/2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = rotated_even
        x_rotated[..., 1::2] = rotated_odd

        return x_rotated