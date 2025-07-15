import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sympy.combinatorics.permutations import Permutation
import math
import tqdm
from omegaconf import DictConfig
import inspect

def zipf_probs(vocab, skewness=0):
    ranks = np.arange(1, vocab + 1)
    p = ranks ** -skewness
    np.random.shuffle(p)
    return p / p.sum()


class ModularAddition(Dataset):
    def __init__(self, n_samples, seq_length, use_tqdm=True, modulo=2, skewness=0):
        self.X = np.random.choice(np.arange(modulo), size=(n_samples, seq_length), p=zipf_probs(modulo, skewness))
        self.Y = np.cumsum(self.X, axis=1) % modulo
        self.n_samples = n_samples
        self.X = torch.from_numpy(self.X).long()
        self.Y = torch.from_numpy(self.Y).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


import torch
import numpy as np
from torch.utils.data import Dataset

class InContextRecall(Dataset):
    def __init__(self, n_samples, seq_length, use_tqdm=True, key_vocab_size=120, value_vocab_size=120, num_keys=12):
        assert seq_length % 2 == 0, "Sequence length must be even for in-context recall."
        assert num_keys <= key_vocab_size, "num_keys must be less than or equal to key_vocab_size"

        seq_pairs = seq_length // 2

        sampled_keys = np.stack([
            np.random.choice(key_vocab_size, size=num_keys, replace=False)
            for i in range(n_samples)
        ])

        key_to_value = np.random.randint(0, value_vocab_size, size=(n_samples, num_keys), dtype=np.int32)
        key_indices = np.random.randint(0, num_keys, size=(n_samples, seq_pairs), dtype=np.int32)
        keys = np.take_along_axis(sampled_keys, key_indices, axis=1)
        values = np.take_along_axis(key_to_value, key_indices, axis=1)

        values_masked = values.copy()
        for i in range(n_samples):
            seen = set()
            for j in range(seq_pairs):
                k = keys[i, j]
                if k not in seen:
                    values_masked[i, j] = -100
                    seen.add(k)

        kv_array = np.empty((n_samples, seq_length), dtype=np.int32)
        kv_array[:, 0::2] = keys
        kv_array[:, 1::2] = values

        kv_array_masked = np.full((n_samples, seq_length), fill_value=-100, dtype=np.int32)
        kv_array_masked[:, 1::2] = values_masked

        self.X = torch.from_numpy(kv_array[:, :-1].copy()).long()
        self.Y = torch.from_numpy(kv_array_masked[:, 1:].copy()).long()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class BitwiseXOR(Dataset):
    def __init__(self, n_samples, seq_length, use_tqdm=True, max_num=16, skewness=0):
        self.X = np.random.choice(np.arange(max_num), size=(n_samples, seq_length), p=zipf_probs(max_num, skewness))
        self.Y = np.bitwise_xor.accumulate(self.X, axis=1)
        self.n_samples = n_samples
        self.X = torch.from_numpy(self.X).long()
        self.Y = torch.from_numpy(self.Y).long()
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class AdditionWithCarry(Dataset):
    def __init__(self, n_samples, seq_length, use_tqdm=True):
        assert (seq_length - 2) % 3 == 0, "Sequence length must be 3k+2 for addition with carry."
        num_digits = (seq_length - 2) // 3
        self.X = np.random.randint(0, 10, size=(n_samples, seq_length), dtype=np.int32)
        self.X[:, num_digits] = 10 # Addition sign
        self.X[:, num_digits * 2 + 1] = 11 # Equals sign
        iterator = tqdm.tqdm(range(n_samples), desc="Dataset") if use_tqdm else range(n_samples)
        for i in iterator:
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
    def __init__(self, n_samples, seq_length, use_tqdm=True, n=3, skewness=0):
        vocab = math.factorial(n)
        table = np.array([[compose(i, j, n) for j in range(vocab)] for i in range(vocab)])
        self.X = np.random.choice(np.arange(vocab), size=(n_samples, seq_length), p=zipf_probs(vocab, skewness))
        self.Y = np.empty_like(self.X)
        iterator = tqdm.tqdm(range(n_samples), desc="Dataset") if use_tqdm else range(n_samples)
        for i in iterator:
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


def get_loaders(config: DictConfig, which=("train", "val")):
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

    loaders = {}

    if "train" in which:
        train_set = Dataset(
            n_samples=config.data.n_train_samples, 
            seq_length=config.data.seq_length, 
            use_tqdm=config.data.use_tqdm,
            **task_args
        )
        loaders["train"] = DataLoader(
            train_set, 
            batch_size=config.train.batch_size, 
            shuffle=True, 
            collate_fn=collate_fn, num_workers=config.data.num_workers
        )

    if "val" in which:
        val_set = Dataset(
            n_samples=config.data.n_val_samples,
            seq_length=config.data.seq_length,
            use_tqdm=config.data.use_tqdm,
            **task_args
        )
        loaders["val"] = DataLoader(
            val_set, 
            batch_size=config.train.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, num_workers=config.data.num_workers
        )

    # Return single loader if only one requested, else tuple in ("train", "val") order
    if len(which) == 1:
        return loaders[which[0]]
    return tuple(loaders[w] for w in ("train", "val") if w in loaders)
