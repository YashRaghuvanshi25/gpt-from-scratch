"""
Multi-head self-attention implementations used in the Transformer.
This module contains both standard causal self-attention and a RoPE-enhanced
version for positional encoding. Attention is computed per head and then
merged back into the model dimension.
"""
import torch
from torch import nn
from einops import rearrange

from transformer.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Learnable projection matrices for Q, K, V, and output projection
        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., T, d_model) where T is sequence length
        T = x.shape[-2]

        # Project token representations into query, key, and value spaces
        Q = x @ self.Wq.T
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        # Reshape into multiple heads so attention can be computed independently per head
        Q = rearrange(Q, "... T (h d) -> ... h T d", h=self.num_heads)
        K = rearrange(K, "... T (h d) -> ... h T d", h=self.num_heads)
        V = rearrange(V, "... T (h d) -> ... h T d", h=self.num_heads)

        # Causal mask prevents tokens from attending to future positions
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        # Compute scaled dot-product attention across all heads
        out = scaled_dot_product_attention(Q, K, V, mask)

        # Merge per-head outputs back into the model dimension
        out = rearrange(out, "... h T d -> ... T (h d)")

        # Final linear projection after attention aggregation
        out = out @ self.Wo.T
        return out

from transformer.rope import RotaryPositionalEmbedding


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.Wq = nn.Parameter(torch.empty(d_model, d_model))
        self.Wk = nn.Parameter(torch.empty(d_model, d_model))
        self.Wv = nn.Parameter(torch.empty(d_model, d_model))
        self.Wo = nn.Parameter(torch.empty(d_model, d_model))

        self.rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=self.head_dim,
            max_seq_len=max_seq_len,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        T = x.shape[-2]

        # Project token representations into query, key, and value spaces
        Q = x @ self.Wq.T
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        # Reshape into multiple heads for parallel attention computation
        Q = rearrange(Q, "... T (h d) -> ... h T d", h=self.num_heads)
        K = rearrange(K, "... T (h d) -> ... h T d", h=self.num_heads)
        V = rearrange(V, "... T (h d) -> ... h T d", h=self.num_heads)

        # Apply Rotary Positional Embedding to queries and keys only
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # Causal mask prevents tokens from attending to future positions
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        # Compute scaled dot-product attention across all heads
        out = scaled_dot_product_attention(Q, K, V, mask)

        # Merge per-head outputs back into the model dimension
        out = rearrange(out, "... h T d -> ... T (h d)")

        # Final linear projection after attention aggregation
        out = out @ self.Wo.T
        return out