import numpy as np
import torch
from torch.utils.data import Dataset

class CodeDataset(Dataset):

    def __init__(self, filepath, block_size):
        self.data = np.memmap(filepath, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CodeDataset("data_train.bin", block_size=1024)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    x, y = next(iter(loader))
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"Dataset size: {len(dataset):,} samples")