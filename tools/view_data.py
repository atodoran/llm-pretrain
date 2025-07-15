import sys
sys.path.append('./')

import torch
import numpy as np
import os

from data import get_loaders
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    config.data.n_val_samples = 5
    config.train.batch_size = 5
    loader_val = get_loaders(config, which=("val",))

    batch = next(iter(loader_val))
    X, Y = batch
    
    X = X.numpy().astype(int)
    Y = Y.numpy().astype(int)
    
    for i in range(len(X)):
        print(f"Sample {i+1}:")
        print("X:", ' '.join(str(x) if x != -100 else '-' for x in X[i]))
        print("Y:", ' '.join(str(y) if y != -100 else '-' for y in Y[i]))
        print()

if __name__ == "__main__":
    main()