import yaml
from dataclasses import dataclass, fields

def load_yml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@dataclass
class Config:
    def __post_init__(self):
        pass

    @classmethod
    def from_dict(cls, kwargs: dict):
        obj = cls()
        valid_keys = {field.name for field in fields(obj)}
        for key, value in kwargs.items():
            if key in valid_keys:
                setattr(obj, key, value)
        return obj

@dataclass
class TrainConfig(Config):
    learning_rate: int = 0.0001
    epochs: int = 10

@dataclass
class ModelConfig(Config):
    depth: int = 22
    dim: int = 512
    attn_heads: int = 8
    max_seq_len: int = 128
    vocab_size: int = 16

@dataclass
class DataConfig(Config):
    seed: int = 0
    n_train_samples: int = 1000000
    n_val_samples: int = 10000
    seq_length: int = 32
    modulo: int = 2
    batch_size: int = 128
    num_workers: int = 4