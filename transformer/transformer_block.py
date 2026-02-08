"""
Pre‑norm Transformer block combining multi‑head self‑attention with a SwiGLU
feed‑forward network. Residual connections are applied around both sublayers
for stable deep Transformer training.
"""

import torch
from torch import nn

from transformer.rmsnorm import RMSNorm
from transformer.multihead_self_attention import MultiHeadSelfAttentionWithRoPE
from transformer.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        self.ln1 = RMSNorm(d_model, eps=1e-5)
        self.ln2 = RMSNorm(d_model, eps=1e-5)

        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Apply RMSNorm before attention (pre‑norm) and add residual connection
        x = x + self.attn(self.ln1(x), token_positions)

        # Apply RMSNorm before feed‑forward network and add residual connection
        x = x + self.ffn(self.ln2(x))

        return x