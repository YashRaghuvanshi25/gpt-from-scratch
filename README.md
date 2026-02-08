# GPT from Scratch

A byte‑level BPE tokenizer and GPT‑style decoder‑only Transformer implemented from first principles (RoPE, SwiGLU, RMSNorm, causal attention) and trained on TinyStories.

This repository demonstrates the complete pipeline:

**Text → Tokens → Tensors → Transformer → Training → Generation**

---

## Architecture Overview

```
Tokenizer
  ↓
Token IDs (.npy)
  ↓
Embedding (B, T, d_model)
  ↓
[ Transformer Block × N ]
      ├─ RMSNorm
      ├─ Multi‑Head Self Attention (RoPE + causal mask)
      ├─ RMSNorm
      └─ SwiGLU FFN
  ↓
Linear Head → Softmax → Next Token
```

---

## What this repository contains

### Tokenizer
- Byte‑level BPE
- GPT‑style regex pre‑tokenization
- Encode / Decode
- Vocabulary and merges training

### Transformer Language Model
- Embedding, Linear layers, RMSNorm (no high‑level libraries)
- Rotary Positional Embeddings (RoPE)
- Numerically stable Softmax
- Scaled dot‑product attention with causal masking
- Multi‑head self‑attention
- SwiGLU feed‑forward network
- Pre‑norm Transformer blocks
- AdamW optimizer
- Gradient clipping
- Cosine learning rate schedule

---

## Project Structure

```
tokenizer/      → Byte‑level BPE tokenizer
transformer/    → Transformer model components
train.py        → Training script
generate.py     → Text generation script
```

---

## Training Details

- ~40M parameters
- Context length: 128
- 8 layers, 8 heads
- d_model: 512, d_ff: 2048
- RoPE positional encoding
- RMSNorm + PreNorm architecture
- SwiGLU feed‑forward
- AdamW with cosine LR schedule
- Trained on TinyStories dataset
- Checkpoint resume support
- Trained for demonstration on TinyStories with limited compute; larger data and longer training improve coherence.
- Designed to be easily scaled to larger datasets and models

---

## Sample Generation Output

**Prompt**

> Once upon a time

**Model output after training on TinyStories**

Output varies based on training duration. With short TinyStories training, the model produces grammatically structured but partially incoherent text, which improves with more data and training steps.

---

## How to Run

### Setup environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy tqdm regex
```

Alternatively, install dependencies using:

```bash
pip install -r requirements.txt
```

### Train the model

```bash
python train.py
```

### Generate text

```bash
python generate.py
```

---

## Pretrained Artifacts

To run generation without retraining, place the following inside an `artifacts/` folder:

```
artifacts/
  ├─ tiny_vocab.json
  ├─ tiny_merges.json
  └─ tiny_40M_ckpt.pt
```

Google Drive link: https://drive.google.com/drive/folders/15c9qwzh65EaeQA5YPWN1NSbEIV7iNGTG?usp=drive_link

---

## Dataset Used

- TinyStories dataset for language modeling
```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

---

## Goal of this Project

To understand, implement, train, and debug a GPT‑style language model at the tensor level without relying on high‑level deep learning abstractions.