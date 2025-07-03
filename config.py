import yaml
from dataclasses import dataclass, fields, field

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
    task: str = 'modular_addition'
    n_train_samples: int = 1000000
    n_val_samples: int = 10000
    seq_length: int = 32
    batch_size: int = 128
    num_workers: int = 4
    regenerate: bool = False

    # Task-specific parameters
    modulo:     int = field(default=2,  metadata={"task": ["modular_addition"]})
    num_keys:   int = field(default=16, metadata={"task": ["in_context_recall"]})
    num_values: int = field(default=16, metadata={"task": ["in_context_recall"]})
    max_num:    int = field(default=32, metadata={"task": ["bitwise_xor"]})

    def task_kwargs(self, task: str = None) -> dict:
        task = task or self.task
        return {f.name: getattr(self, f.name)
                for f in fields(self)
                if task in f.metadata.get("task", [])}