import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data import get_loaders
from config import DataConfig, load_yml
import os

def main():
    data_config_path = os.path.join('configs', 'data.yaml')
    data_config = DataConfig.from_dict(kwargs=load_yml(data_config_path))
    
    np.random.seed(data_config.seed)
    torch.manual_seed(data_config.seed)
    torch.cuda.manual_seed(data_config.seed)

    data_config.batch_size = 5
    _, loader_val = get_loaders(data_config)

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