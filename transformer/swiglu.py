"""
SwiGLU feed-forward network used inside Transformer blocks. SwiGLU replaces
standard MLP layers with a gated SiLU activation, improving expressiveness
and training stability in modern language models.
"""
import torch
from torch import nn
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        # Hidden dimension follows SwiGLU convention: ~8/3 × d_model, rounded for hardware efficiency
        d_ff = int((8 / 3) * d_model)
        d_ff = (d_ff + 63) // 64 * 64  # round up to multiple of 64

        # Learnable projection matrices for the gated feed-forward transformation
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model) token representations from attention block

        # Project inputs into two parallel feature spaces for gating
        x1 = einsum(x, self.W1, "... d, f d -> ... f")
        x2 = einsum(x, self.W3, "... d, f d -> ... f")

        # Apply SiLU (Swish) non-linearity to the first projection
        silu = x1 * torch.sigmoid(x1)

        # Gate the activated features with the second projection
        gated = silu * x2

        # Project gated features back to model dimension
        out = einsum(gated, self.W2, "... f, d f -> ... d")

        return out