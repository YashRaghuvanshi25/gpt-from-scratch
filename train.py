"""
Training script for the GPT-style Transformer language model.
Loads pre-tokenized TinyStories data and trains the model using
AdamW, gradient clipping, and a cosine learning rate schedule.
"""

import os
import numpy as np
import torch
from tqdm import trange

from transformer.transformer_lm import TransformerLM

# --------------------- Utility Functions ---------------------

def get_batch(tokens, batch_size, context_length, device):
    max_start = len(tokens) - context_length - 1
    starts = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([
        torch.tensor(tokens[s : s + context_length], dtype=torch.long)
        for s in starts
    ])

    y = torch.stack([
        torch.tensor(tokens[s + 1 : s + context_length + 1], dtype=torch.long)
        for s in starts
    ])

    return x.to(device), y.to(device)


def cross_entropy(logits, targets):
    return torch.nn.functional.cross_entropy(logits, targets)


def clip_grad(parameters, max_norm):
    torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def lr_schedule(it, max_lr, min_lr, warmup_iters, cosine_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters

    if it > cosine_iters:
        return min_lr

    progress = (it - warmup_iters) / (cosine_iters - warmup_iters)
    cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.1415926535)))
    return min_lr + (max_lr - min_lr) * cosine.item()


def save_ckpt(model, optimizer, it, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": it,
    }, path)


def load_ckpt(path, model, optimizer):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state["iter"]


# --------------------- Model Configuration ---------------------

MODEL_NAME = "tiny_40M"

vocab_size = 10_000
context_length = 128
d_model = 512
num_layers = 8
num_heads = 8
d_ff = 2048
rope_theta = 10_000.0

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

batch_size = 16
max_iters = 4000
eval_interval = 200

max_lr = 3e-4
min_lr = 3e-5
warmup_iters = 100
cosine_iters = 2000

checkpoint_path = f"{MODEL_NAME}_ckpt.pt"
report_path = f"{MODEL_NAME}_report.txt"

# --------------------- Load Data ---------------------

print("Loading TinyStories tokens...")
train_tokens = np.load("artifacts/tiny_train_tokens.npy")
dev_tokens = np.load("artifacts/tiny_valid_tokens.npy")

# --------------------- Initialize Model ---------------------

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

# --------------------- Resume From Checkpoint ---------------------

start_iter = 0
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    start_iter = load_ckpt(checkpoint_path, model, optimizer)
    print("Resuming from iter", start_iter)

# --------------------- Evaluation ---------------------

@torch.no_grad()
def estimate_loss(split_tokens):
    model.eval()
    losses = []

    for _ in range(20):
        x, y = get_batch(split_tokens, batch_size, context_length, DEVICE)
        logits = model(x)
        B, T, V = logits.shape

        loss = cross_entropy(
            logits.view(B * T, V),
            y.view(B * T),
        )
        losses.append(loss.item())

    model.train()
    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity


# --------------------- Training Loop ---------------------

print("Training on", DEVICE)

for it in trange(start_iter, max_iters):

    lr = lr_schedule(it, max_lr, min_lr, warmup_iters, cosine_iters)
    for g in optimizer.param_groups:
        g["lr"] = lr

    x, y = get_batch(train_tokens, batch_size, context_length, DEVICE)

    logits = model(x)
    B, T, V = logits.shape

    loss = cross_entropy(
        logits.view(B * T, V),
        y.view(B * T),
    )

    optimizer.zero_grad()
    loss.backward()
    clip_grad(model.parameters(), 1.0)
    optimizer.step()

    if it % eval_interval == 0:
        train_loss, train_ppl = estimate_loss(train_tokens)
        val_loss, val_ppl = estimate_loss(dev_tokens)

        print(
            f"\niter {it}\n"
            f"train loss {train_loss:.4f} | ppl {train_ppl:.2f}\n"
            f"val   loss {val_loss:.4f} | ppl {val_ppl:.2f}\n"
        )

    if it % 500 == 0 and it > 0:
        save_ckpt(model, optimizer, it, checkpoint_path)