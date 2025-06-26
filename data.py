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
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


def collate_fn(batch):
    X, Y = zip(*batch)
    lens = [x.size(0) for x in X]
    max_len = max(lens)
    X = [torch.nn.functional.pad(x, (0, max_len - x.size(0))) for x in X]
    Y = [torch.nn.functional.pad(y, (0, max_len - y.size(0))) for y in Y]
    X = torch.stack(X)
    Y = torch.stack(Y)
    return X, Y


def get_loaders(data_config):
    train_set = ModularOpDataset(
        n_samples=data_config.n_train_samples,
        seq_length=data_config.seq_length,
        modulo=data_config.modulo
    )
    val_set = ModularOpDataset(
        n_samples=data_config.n_val_samples,
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
