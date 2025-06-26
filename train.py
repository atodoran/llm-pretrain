import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
import wandb

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
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    epochs = train_config.epochs

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Resumed from epoch {start_epoch}")

    wandb.init(
        project="llm-pretraining",
        config={
            "learning_rate": train_config.learning_rate,
            "epochs": epochs,
            "batch_size": train_loader.batch_size
        }
    )
    wandb.watch(model, log="all")

    print("Starting training...")
    for epoch in range(start_epoch, epochs):

        # Training loop
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs).permute(0, 2, 1)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == targets).float().mean().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation loop
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for val_inputs, val_targets in tqdm(val_loader, desc=f"[EVAL] Epoch {epoch+1}/{epochs}"):
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device).long()
                val_outputs = model(val_inputs).permute(0, 2, 1)
                loss = loss_fn(val_outputs, val_targets)
                val_loss += loss.item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc += (val_predicted == val_targets).float().mean().item()
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

        # 3. Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Save checkpoint after each epoch
        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch+1, checkpoint_path)

    # Save final model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the final model.')
    args = parser.parse_args()

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
    checkpoint_path = os.path.join("logs", "checkpoint.pth")
    save_path = os.path.join("models", args.save_path) if args.save_path else None

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