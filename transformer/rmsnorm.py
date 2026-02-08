"""
Root Mean Square Layer Normalization (RMSNorm) used in modern Transformer
architectures. RMSNorm normalizes activations by their RMS value without
subtracting the mean, which improves training stability and efficiency in
large language models.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # Learnable gain parameter applied after normalization
        self.g = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to the last dimension.

        Input shape:  (..., d_model)
        Output shape: (..., d_model)
        """
        in_dtype = x.dtype

        # Upcast to float32 for numerical stability during RMS computation
        x = x.to(torch.float32)

        # Compute root-mean-square across the feature dimension
        rms = torch.sqrt(
            x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        # Normalize activations and apply the learnable gain
        out = (x / rms) * self.g

        # Cast back to original dtype to preserve memory efficiency
        return out.to(in_dtype)