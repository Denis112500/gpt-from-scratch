# gpt-from-scratch

Building a GPT-style transformer from scratch in PyTorch.

The goal isn't to produce a production model it's to understand every piece of the architecture before using it for serious work.

---

## What's being built

| Step | Topic | Status |
|------|-------|--------|
| 1 | Micrograd — backprop from scratch | ✅ |
| 2 | Bigram language model | ✅ |
| 3 | Tokenization & embeddings | ✅ |
| 4 | Self-attention mechanism | ✅ |
| 5 | Multi-head attention + Feed-forward blocks | ✅ |
| 6 | Full GPT-2 architecture (nanoGPT) | ✅ |
| 7 | Training on custom dataset | 🔄 |

---

## Stack

- Python 3.11+
- PyTorch
- NumPy

No external ML frameworks. No Hugging Face. Everything from scratch.

---
## Hardware

Trained locally on an RTX 3090 SUPRIM X (borrowed).

---

## Why

Foundation work for my bachelor's thesis on predictive maintenance and my Year 3 AI course (MLP + backpropagation from scratch). Building it before using it.
