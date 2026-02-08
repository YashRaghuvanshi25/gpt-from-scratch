"""
Implementation of scaled dot-product attention, the core operation inside
multi-head self-attention. Computes attention weights between queries and keys,
then applies them to values to produce context-aware token representations.
"""
import torch
from einops import einsum

from transformer.softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention.

    Q: (..., queries, d_k)
    K: (..., keys, d_k)
    V: (..., keys, d_v)
    mask: (..., queries, keys) boolean mask preventing attention to certain positions
    """

    d_k = Q.shape[-1]

    # Compute attention scores by comparing queries with all keys
    scores = einsum(Q, K, "... q d, ... k d -> ... q k")

    # Scale by sqrt(d_k) to prevent large dot-product magnitudes
    scores = scores / (d_k ** 0.5)

    # Apply causal or padding mask to block invalid attention positions
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Convert scores into attention probabilities over keys
    attn = softmax(scores, dim=-1)

    # Weight values by attention probabilities to produce output representations
    out = einsum(attn, V, "... q k, ... k d -> ... q d")

    return out