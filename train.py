import torch
import numpy as np
from model import GPT
from torch.utils.tensorboard import SummaryWriter
import math
import os

# config — small but real proportions
vocab_size  = 50257
block_size  = 128
batch_size  = 32
d_model     = 768
num_heads   = 12
num_layers  = 12
max_seq_len = 1024
max_steps   = 5000

eval_interval = 500
eval_iters    = 50
warmup_steps  = 100
min_lr        = 3e-5
max_lr        = 3e-4
grad_clip     = 1.0

# load dataset and split into train/val
full_data = np.memmap('data_train.bin', dtype=np.uint16, mode='r')
n = len(full_data)
train_data = full_data[: int(n * 0.9)]
val_data   = full_data[int(n * 0.9) :]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = [data[i : i + block_size] for i in ix]
    y = [data[i + 1 : i + block_size + 1] for i in ix]
    x = torch.tensor(np.array(x), dtype=torch.long).to("cuda")
    y = torch.tensor(np.array(y), dtype=torch.long).to("cuda")
    return x, y

@torch.no_grad()
def estimate_val_loss():
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch('val')
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)

model = GPT(vocab_size, d_model, num_heads, num_layers, max_seq_len).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

# --- resume from checkpoint if one exists ---
start_step = 0
checkpoint_path = "checkpoint_step_500.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_step = checkpoint['step'] + 1
    print(f"Resumed from step {start_step}")

writer = SummaryWriter(log_dir="runs/gpt_run1")

for step in range(start_step, max_steps):
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x, y = get_batch('train')
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out.view(-1, vocab_size), y.view(-1))
    loss.backward()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    if step % 100 == 0:
        writer.add_scalar("train_loss", loss.item(), step)
        writer.add_scalar("lr", lr, step)
        writer.add_scalar("grad_norm", total_norm.item(), step)
        print(f"Step {step}: loss = {loss.item():.4f}, lr = {lr:.6f}, grad_norm = {total_norm.item():.4f}")

    if step % eval_interval == 0:
        val_loss = estimate_val_loss()
        writer.add_scalar("val_loss", val_loss, step)
        print(f"Step {step}: val_loss = {val_loss:.4f}")

    if step % 500 == 0:
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"checkpoint_step_{step}.pt")

writer.close()