import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sympy.combinatorics.permutations import Permutation
import math
import tqdm
from omegaconf import DictConfig
import inspect

class ModularAddition(Dataset):
    def __init__(self, n_samples, seq_length, modulo=2):
        self.X = np.random.randint(0, modulo, size=(n_samples, seq_length))
        self.Y = np.cumsum(self.X, axis=1) % modulo
        self.n_samples = n_samples
        self.X = torch.from_numpy(self.X).long()
        self.Y = torch.from_numpy(self.Y).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class InContextRecall(Dataset):
    def __init__(self, n_samples, seq_length, num_keys=8, num_values=8):
        assert seq_length % 2 == 0, "Sequence length must be even for in-context recall."
        value_of_key = np.random.randint(0, num_values, size=(n_samples, num_keys), dtype=np.int32)

        keys = np.random.randint(0, num_keys, size=(n_samples, seq_length // 2), dtype=np.int32)
        values = np.empty((n_samples, seq_length // 2), dtype=np.int32)
        for i in range(n_samples):
            for j in range(seq_length // 2):
                values[i, j] = value_of_key[i, keys[i, j]]
        
        values_masked = values.copy()
        for i in range(n_samples):
            seen_keys = set()
            for j in range(seq_length // 2):
                if keys[i, j] not in seen_keys:
                    values_masked[i, j] = -100
                    seen_keys.add(keys[i, j])

        kv_array = np.empty((n_samples, seq_length), dtype=np.int32)
        kv_array[:, 0::2] = keys
        kv_array[:, 1::2] = values
        self.X = kv_array[:, :-1].copy()

        kv_array_masked = np.empty((n_samples, seq_length), dtype=np.int32)
        kv_array_masked[:, 0::2] = -100
        kv_array_masked[:, 1::2] = values_masked
        self.Y = kv_array_masked[:, 1:]

        self.n_samples = n_samples
        self.X = torch.from_numpy(self.X).long()
        self.Y = torch.from_numpy(self.Y).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class BitwiseXOR(Dataset):
    def __init__(self, n_samples, seq_length, max_num=16):
        self.X = np.random.randint(0, max_num, size=(n_samples, seq_length))
        self.Y = np.bitwise_xor.accumulate(self.X, axis=1)
        self.n_samples = n_samples
        self.X = torch.from_numpy(self.X).long()
        self.Y = torch.from_numpy(self.Y).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class AdditionWithCarry(Dataset):
    def __init__(self, n_samples, seq_length):
        assert (seq_length - 2) % 3 == 0, "Sequence length must be 3k+2 for addition with carry."
        num_digits = (seq_length - 2) // 3
        self.X = np.random.randint(0, 10, size=(n_samples, seq_length), dtype=np.int32)
        self.X[:, num_digits] = 10 # Addition sign
        self.X[:, num_digits * 2 + 1] = 11 # Equals sign
        for i in range(n_samples):
            num1 = self.X[i, :num_digits]
            num2 = self.X[i, num_digits + 1 : num_digits * 2 + 1]
            sum12 = np.zeros(num_digits, dtype=np.int32)
            carry = 0
            for j in range(num_digits):
                digit_sum = num1[j] + num2[j] + carry
                sum12[j] = digit_sum % 10
                carry = digit_sum // 10
            self.X[i, num_digits * 2 + 2:] = sum12
        self.Y = self.X.copy()[:, 1:]
        self.X = self.X.copy()[:, :-1]
        self.Y[:, :num_digits * 2 + 1] = -100
        self.n_samples = n_samples
        self.X = torch.from_numpy(self.X).long()
        self.Y = torch.from_numpy(self.Y).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def compose(i: int, j: int, n: int) -> int:
    p = Permutation.unrank_lex(n, i) 
    q = Permutation.unrank_lex(n, j)
    r = p * q
    return r.rank()

class PermutationComposition(Dataset):
    def __init__(self, n_samples, seq_length, n=3):
        table = np.array([[compose(i, j, n) for j in range(math.factorial(n))] for i in range(math.factorial(n))])
        self.X = np.random.randint(0, math.factorial(n), size=(n_samples, seq_length))
        self.Y = np.empty_like(self.X)
        for i in tqdm.tqdm(range(n_samples), desc=f"Dataset"):
            curr = 0
            for j in range(seq_length):
                curr = table[curr, self.X[i,j]]
                self.Y[i,j] = curr
        
        self.n_samples = n_samples
        self.X = torch.from_numpy(self.X).long()
        self.Y = torch.from_numpy(self.Y).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def collate_fn(batch):
    X, Y = zip(*batch)
    lens = [x.size(0) for x in X]
    max_len = max(lens)
    X = [torch.nn.functional.pad(x, (0, max_len - x.size(0))) for x in X]
    Y = [torch.nn.functional.pad(y, (0, max_len - y.size(0))) for y in Y]
    X = torch.stack(X)
    Y = torch.stack(Y)
    return X, Y


def get_loaders(config: DictConfig):
    task_type = config.task.type

    Dataset = {
        "modular_addition": ModularAddition,
        "in_context_recall": InContextRecall,
        "bitwise_xor": BitwiseXOR,
        "permutation_composition": PermutationComposition,
        "addition_with_carry": AdditionWithCarry
    }.get(task_type)
    if Dataset is None:
        raise ValueError(f"Unknown task type: {task_type}")

    # Filter out dataset parameters that are not in the Dataset class
    dataset_params = inspect.signature(Dataset.__init__).parameters
    valid_keys = [k for k in dataset_params if k != 'self']
    task_args = {k: v for k, v in config.task.items() if k in valid_keys}
    
    train_set = Dataset(
        n_samples=config.data.n_train_samples, 
        seq_length=config.data.seq_length, 
        **task_args
    )
    val_set = Dataset(
        n_samples=config.data.n_val_samples,
        seq_length=config.data.seq_length,
        **task_args
    )
    
    loader_train = DataLoader(
        train_set, 
        batch_size=config.train.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, num_workers=config.data.num_workers
    )
    loader_val = DataLoader(
        val_set, 
        batch_size=config.train.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, num_workers=config.data.num_workers
    )
    
    return loader_train, loader_val
