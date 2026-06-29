import torch
import numpy as np
from model import GPT
import wandb

# config — small but real proportions
vocab_size  = 50257   # GPT-2 vocabulary size (tiktoken)
block_size  = 128     # context length
batch_size  = 32       # mini-batch size
d_model     = 768     # vector width
num_heads   = 12      # attention heads
num_layers  = 12      # decoder blocks
max_seq_len = 1024    # max sequence length
max_steps   = 5000    # number of training steps

#load dataset
data = np.memmap('data_train.bin', dtype=np.uint16, mode='r')

def get_batch(split):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = [data[i : i + block_size] for i in ix]
    y = [data[i + 1 : i + block_size + 1] for i in ix]
    x = torch.tensor(np.array(x), dtype=torch.long).to("cuda")
    y = torch.tensor(np.array(y), dtype=torch.long).to("cuda")
    return x, y

model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

wandb.init(project="gpt-from-scratch", config={
    "vocab_size": vocab_size,
    "block_size": block_size,
    "batch_size": batch_size,
    "d_model": d_model,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "max_steps": max_steps,
    "lr": 3e-4,
})

for step in range(max_steps):
    x, y = get_batch('train')
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out.view(-1, vocab_size), y.view(-1))
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        wandb.log({"loss": loss.item(), "step": step})
        print(f"Step {step}: loss = {loss.item():.4f}")
    if step % 500 == 0:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"checkpoint_step_{step}.pt")