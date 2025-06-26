import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ModularOpDataset(Dataset):
    def __init__(self, n_samples, seq_length, modulo=2):
        self.X = np.random.randint(0, modulo, size=(n_samples, seq_length))
        self.Y = np.cumsum(self.X, axis=1) % modulo
        self.n_samples = n_samples
        self.X, self.Y = self.X.astype(int), self.Y.astype(int)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def collate_fn(batch):
    X, y = zip(*batch)
    lens = [len(x) for x in X]
    max_len = max(lens)
    X = [np.pad(x, (0, max_len - len(x))) for x in X]
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    return X, y


def get_loaders(data_config):
    train_set = ModularOpDataset(
        n_samples=data_config.n_samples,
        seq_length=data_config.seq_length,
        modulo=data_config.modulo
    )
    val_set = ModularOpDataset(
        n_samples=data_config.n_samples,
        seq_length=data_config.seq_length,
        modulo=data_config.modulo
    )
    
    loader_train = DataLoader(
        train_set, 
        batch_size=data_config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, num_workers=data_config.num_workers
    )
    loader_val = DataLoader(
        val_set, 
        batch_size=data_config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, num_workers=data_config.num_workers
    )
    
    return loader_train, loader_val
