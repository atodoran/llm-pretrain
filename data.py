import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sympy.combinatorics.permutations import Permutation
import math

class ModularAddition(Dataset):
    def __init__(self, n_samples, seq_length, modulo=2):
        self.X = np.random.randint(0, modulo, size=(n_samples, seq_length))
        self.Y = np.cumsum(self.X, axis=1) % modulo
        self.X, self.Y = self.X.astype(int), self.Y.astype(int)
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


class InContextRecall(Dataset):
    def __init__(self, n_samples, seq_length, num_keys=16, num_values=16):
        assert seq_length % 2 == 0, "Sequence length must be even for in-context recall."
        value_of_key = np.random.randint(0, num_values, size=num_keys)

        keys = np.random.randint(0, num_keys, size=(n_samples, seq_length // 2))
        values = np.array([value_of_key[key] for key in keys.flatten()]).reshape(n_samples, seq_length // 2)

        kv_array = np.empty((n_samples, seq_length), dtype=keys.dtype)
        kv_array[:, 0::2] = keys
        kv_array[:, 1::2] = values
        self.X = kv_array[:, :-1].copy()

        kv_array_masked = kv_array.copy()
        kv_array_masked[:, 1::2] = -100
        self.Y = kv_array_masked[:, 1:]

        self.X, self.Y = self.X.astype(int), self.Y.astype(int)
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


class BitwiseXOR(Dataset):
    def __init__(self, n_samples, seq_length, max_num=16):
        self.X = np.random.randint(0, max_num, size=(n_samples, seq_length))
        self.Y = np.bitwise_xor.accumulate(self.X, axis=1)
        self.X, self.Y = self.X.astype(int), self.Y.astype(int)
        self.n_samples = n_samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


def compose(i: int, j: int, n: int = 5) -> int:
    p = Permutation.unrank_lex(n, i) 
    q = Permutation.unrank_lex(n, j)
    r = p * q
    return r.rank()

class PermutationComposition(Dataset):
    def __init__(self, n_samples, seq_length, n=3):
        self.X = np.random.randint(0, math.factorial(n), size=(n_samples, seq_length))
        self.Y = np.empty_like(self.X)
        for i in range(n_samples):
            curr = 0
            for j in range(seq_length):
                curr = compose(curr, self.X[i,j], n)
                self.Y[i,j] = curr
        
        self.X, self.Y = self.X.astype(int), self.Y.astype(int)
        self.n_samples = n_samples
    
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
    task = data_config.task

    Dataset = {
        "modular_op": ModularAddition,
        "in_context_recall": InContextRecall,
        "bitwise_xor": BitwiseXOR,
        "permutation_composition": PermutationComposition
    }.get(task)
    if Dataset is None:
        raise ValueError(f"Unknown task: {task}")

    task_args = data_config.task_kwargs(task)
    
    train_set = Dataset(
        n_samples=data_config.n_train_samples, 
        seq_length=data_config.seq_length, 
        **task_args
    )
    val_set = Dataset(
        n_samples=data_config.n_val_samples,
        seq_length=data_config.seq_length,
        **task_args
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
