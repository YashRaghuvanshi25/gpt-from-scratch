import torch
from torch import nn
from einops import einsum


"""
Fully connected linear layer used throughout the Transformer to project
representations into new feature spaces (e.g., Q, K, V projections and
feed-forward layers).
"""
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable weight matrix of shape (out_features, in_features)
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Truncated normal initialization helps stabilize training for deep networks
        sigma = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=sigma,
            a=-3 * sigma,
            b=3 * sigma,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a linear projection y = W x.

        Supports arbitrary leading batch dimensions.
        Input shape:  (..., in_features)
        Output shape: (..., out_features)
        """
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")