import torch
from model import GPT

# config — small but real proportions
vocab_size  = 50257   # GPT-2 vocabulary size (tiktoken)
d_model     = 768     # vector width
num_heads   = 12      # attention heads
num_layers  = 12      # decoder blocks
max_seq_len = 1024    # max sequence length

# instantiate and move to GPU
model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len).to("cuda")

# count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params / 1e6:.1f}M")

# dummy input — batch of 4 sequences, each 128 tokens long
x = torch.randint(0, vocab_size, (4, 128)).to("cuda")

# forward pass
with torch.no_grad():
    out = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {out.shape}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")