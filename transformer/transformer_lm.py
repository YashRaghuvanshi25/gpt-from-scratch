"""
Top-level Transformer Language Model composed of token embeddings, multiple
Transformer blocks, final normalization, and a linear language modeling head
that projects hidden states back to vocabulary logits.
"""

import torch
from torch import nn

from transformer.embedding import Embedding
from transformer.rmsnorm import RMSNorm
from transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, eps=1e-5)

        # Learnable language modeling head mapping hidden states back to vocabulary space
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer language model.

        token_ids: (B, T) integer token indices
        returns:   (B, T, vocab_size) logits for next-token prediction
        """
        B, T = token_ids.shape

        # Generate token position indices used by RoPE inside attention layers
        token_positions = (
            torch.arange(T, device=token_ids.device)
            .unsqueeze(0)
            .expand(B, T)
        )

        # Convert token ids into dense vector representations
        x = self.token_embeddings(token_ids)

        # Pass representations through stacked Transformer blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # Apply final RMS normalization before output projection
        x = self.ln_final(x)

        # Compute vocabulary logits using the language modeling head
        logits = torch.einsum("b t d, v d -> b t v", x, self.lm_head)

        return logits