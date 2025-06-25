import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from model import get_model
from data import get_loaders
from config import TrainConfig, ModelConfig, DataConfig, load_yml

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def train(model, train_loader, val_loader, train_config: TrainConfig, save_path=None, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    epochs = train_config.epochs

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    start_epoch = 0
    # Resume from checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Resumed from epoch {start_epoch}")

    print("Starting training...")
    model.train()
    for epoch in range(start_epoch, epochs):
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

        # Save checkpoint after each epoch
        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch+1, checkpoint_path)

    # Save final model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

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

    # Checkpoint and model save paths
    checkpoint_path = os.path.join("models", "checkpoint.pth")
    save_path = os.path.join("models", getattr(train_config, "save_name", "final_model.pth"))

    train(
        model=model,
        train_loader=loader_train,
        val_loader=loader_val,
        train_config=train_config,
        save_path=save_path,
        checkpoint_path=checkpoint_path
    )

if __name__ == "__main__":
    main()