"""
Text generation script for the trained Transformer language model.
Loads tokenizer and model checkpoint, then performs autoregressive sampling
from a user-provided prompt.
"""

import torch
from transformer.transformer_lm import TransformerLM
from tokenizer.tokenizer import Tokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Model configuration must match the one used during training
vocab_size = 10_000
context_length = 128
d_model = 512
num_layers = 8
num_heads = 8
d_ff = 2048
rope_theta = 10_000.0

checkpoint_path = "tiny_40M_ckpt.pt"

# Load trained tokenizer vocabulary and merge rules
tokenizer = Tokenizer.load_tokenizer(
    "artifacts/tiny_vocab.json",
    "artifacts/tiny_merges.json",
)

# Build model and load trained weights
model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
).to(DEVICE)

#
# Checkpoint was saved as a dict with model/optimizer/iteration
state = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(state["model"])
model.eval()


# Autoregressive token generation using temperature sampling
@torch.no_grad()
def generate(prompt, max_new_tokens=200, temperature=0.7):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE)[None, :]

    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -context_length:]
        logits = model(tokens_cond)

        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


# Interactive loop for manual prompting
while True:
    prompt = input("\nPrompt: ")
    print("\n" + generate(prompt) + "\n")