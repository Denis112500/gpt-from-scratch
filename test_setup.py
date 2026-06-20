import torch
import wandb

# Test GPU
device = torch.device("cuda")
x = torch.randn(1000, 1000).to(device)
y = x @ x
print(f"GPU tensor works: {y.shape}")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Test W&B
wandb.init(project="gpt-from-scratch", name="setup-test")
wandb.log({"test": 1.0})
wandb.finish()

print("All good.")