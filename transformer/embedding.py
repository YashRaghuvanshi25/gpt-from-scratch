import torch
from torch import nn


"""
Token embedding layer mapping token ids to dense vectors. This is the first
stage of the Transformer where discrete tokens become continuous representations
that the model can process.
"""
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Learnable embedding matrix of shape (vocab_size, d_model)
        self.W = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )

        # Truncated normal initialization improves stability for large embedding tables
        torch.nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for token ids.

        token_ids: (...,) long tensor of token indices
        returns:   (..., d_model) dense token representations
        """
        return self.W[token_ids]