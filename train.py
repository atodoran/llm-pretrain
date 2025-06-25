import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from model import get_model
from data import get_loaders
from config import TrainConfig, ModelConfig, DataConfig, load_yml

def train(model, train_loader, val_loader, train_config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    epochs = train_config.epochs

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_acc = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            epoch_acc += (predicted == targets).float().mean().item()

        train_losses.append(epoch_loss / len(train_loader))
        train_accs.append(epoch_acc / len(train_loader))

        val_loss, val_acc = val_loader(model, val_loader, loss_fn)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, "
              f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

def main():
    print("Loading configs...")
    train_config_path = os.path.join('configs', 'train.yaml')
    data_config_path = os.path.join('configs', 'data.yaml')
    model_config_path = os.path.join('configs', 'model.yaml')

    train_config = TrainConfig.from_dict(kwargs=load_yml(train_config_path))
    data_config = DataConfig.from_dict(kwargs=load_yml(data_config_path))
    model_config = ModelConfig.from_dict(kwargs=load_yml(model_config_path))
    
    # Data
    print("Building the dataset...")
    np.random.seed(data_config.seed)
    torch.manual_seed(data_config.seed)
    torch.cuda.manual_seed(data_config.seed)
    loader_train, loader_val = get_loaders(data_config)

    # Model
    print("Building the model...")
    model = get_model(model_config)
    model = model.cuda()

    train(model=model, train_loader=loader_train, val_loader=loader_val, train_config=train_config)

if __name__ == "__main__":
    main()