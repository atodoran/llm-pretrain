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
import random
import string

def save_checkpoint(model, optimizer, epoch, path, wandb_id):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "wandb_id": wandb_id,
        },
        path,
    )

def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    wandb_id = ckpt.get("wandb_id")
    return epoch, wandb_id

def train(
    model,
    data_config: DataConfig,
    train_config: TrainConfig,
    model_config: ModelConfig,
    save_path=None,
    checkpoint_path=None,
    wandb_enabled=True,
    regenerate_data=False,
):
    print("Building the dataset...")
    train_loader, val_loader = get_loaders(data_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    epochs = train_config.epochs

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    start_epoch, run_id = 0, None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, run_id = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Resumed from epoch {start_epoch}")
    
    if wandb_enabled:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        def format_task_name(task):
            return task.replace("_", "-")

        def human_readable_params(num):
            if num >= 1e9:
                return f"{num/1e9:.0f}B"
            elif num >= 1e6:
                return f"{num/1e6:.0f}M"
            elif num >= 1e3:
                return f"{num/1e3:.0f}K"
            else:
                return str(num)

        def random_string(length=6):
            return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

        lr_str = f"{train_config.learning_rate:.0e}" if train_config.learning_rate < 1e-3 else f"{train_config.learning_rate:.3f}"
        run_name = (
            f"{format_task_name(data_config.task)}-"
            f"{human_readable_params(num_params)}-"
            f"lr={lr_str}-"
            f"{random_string()}"
        )

        run = wandb.init(
            project="llm-pretraining",
            id=run_id,
            resume="allow",
            name=run_name,
            config={
                "task": data_config.task,
                "num_params": num_params,
                "n_train_samples": data_config.n_train_samples,
                "n_val_samples": data_config.n_val_samples,
                "seq_length": data_config.seq_length,
                "regenerate": regenerate_data,
                "depth": model_config.depth,
                "dim": model_config.dim,
                "attn_heads": model_config.attn_heads,
                "learning_rate": train_config.learning_rate,
                "batch_size": train_loader.batch_size,
            },
        )
        run_id = run.id
        wandb.watch(model, log="all")

    print("Starting training...")
    for epoch in range(start_epoch, epochs):
        if regenerate_data and epoch > start_epoch:
            print("Regenerating data...")
            train_loader, val_loader = get_loaders(data_config)
            print("Data regenerated.")

        # Training loop
        model.train()
        train_loss, train_acc = 0.0, 0.0
        num_train_batches = 0
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

            mask = (targets != -100)
            if mask.any():
                train_acc += ((predicted == targets) & mask).float().sum().item() / mask.float().sum().item()
            else:
                train_acc += 1.0
            num_train_batches += 1

        train_loss = train_loss / num_train_batches
        train_acc = train_acc / num_train_batches
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation loop
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        num_val_batches = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[EVAL] Epoch {epoch+1}/{epochs}"):
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                outputs = model(inputs).permute(0, 2, 1)
                loss = loss_fn(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)

                mask = (targets != -100)
                if mask.any():
                    val_acc += ((predicted == targets) & mask).float().sum().item() / mask.float().sum().item()
                else:
                    val_acc += 1.0
                num_val_batches += 1
        
        val_loss = val_loss / num_val_batches
        val_acc = val_acc / num_val_batches
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n")

        if wandb_enabled:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_path, run_id)
        if save_path:
            torch.save(model.state_dict(), save_path)
    
    if wandb_enabled:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the final model.')
    parser.add_argument('--no-wandb', action='store_true', help='If True, disable Weights & Biases for logging.')
    parser.add_argument('--regenerate', action='store_true', help='If True, regenerate data every epoch.')
    args = parser.parse_args()

    print("Loading configs...")
    train_config_path = os.path.join('configs', 'train.yaml')
    data_config_path = os.path.join('configs', 'data.yaml')
    model_config_path = os.path.join('configs', 'model.yaml')

    train_config = TrainConfig.from_dict(kwargs=load_yml(train_config_path))
    data_config = DataConfig.from_dict(kwargs=load_yml(data_config_path))
    model_config = ModelConfig.from_dict(kwargs=load_yml(model_config_path))
    
    # Data
    np.random.seed(data_config.seed)
    torch.manual_seed(data_config.seed)
    torch.cuda.manual_seed(data_config.seed)

    # Model
    print("Building the model...")
    model = get_model(model_config)

    # Checkpoint and model save paths
    checkpoint_path = os.path.join("logs", "checkpoint.pth")
    save_path = os.path.join("models", args.save_path) if args.save_path else None

    train(
        model=model,
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        wandb_enabled=not args.no_wandb,
        regenerate_data=args.regenerate,
    )

if __name__ == "__main__":
    main()