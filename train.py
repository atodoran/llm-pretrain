import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import argparse
import wandb
import random
import string

from model import get_model
from data import get_loaders
from config import TrainConfig, ModelConfig, DataConfig, load_yml
from utils import get_run_name_base

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

def get_latest_checkpoint(run_name_prefix):
    checkpoints = [os.path.join("logs", f) for f in os.listdir("logs") if f.endswith(".pth")]
    if run_name_prefix is not None:
        checkpoints = [
            ckpt for ckpt in checkpoints
            if os.path.basename(ckpt).startswith(run_name_prefix)
        ]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

def train(
    model,
    data_config: DataConfig,
    train_config: TrainConfig,
    model_config: ModelConfig,
    wandb_enabled=True,
    resume=None,
    name=None,
):

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    epochs = train_config.epochs
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_name_base = get_run_name_base(
        data_config=data_config,
        train_config=train_config,
        num_params=num_params,
    )

    checkpoint_path = None
    if resume:
        run_name_prefix = run_name_base + "_" + resume if isinstance(resume, str) else run_name_base
        checkpoint_path = get_latest_checkpoint(run_name_prefix)
        if checkpoint_path == None:
            raise ValueError(f"No checkpoint found for prefix: {run_name_prefix}.")
        run_name = os.path.basename(checkpoint_path)[:-4]
    else:
        if name is None:
            name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        run_name = f"{run_name_base}_{name}"
        checkpoint_path = os.path.join("logs", f"{run_name}.pth")

    save_path = os.path.join("models", f"{run_name}.pth")
    
    start_epoch, run_id = 0, None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, run_id = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Resumed from epoch {start_epoch}")

    if wandb_enabled:
        run = wandb.init(
            project="llm-pretraining",
            id=run_id,
            resume="allow",
            name=run_name,
            config={
                "task": data_config.task,
                "n_train_samples": data_config.n_train_samples,
                "n_val_samples": data_config.n_val_samples,
                "seq_length": data_config.seq_length,
                "regenerate": data_config.regenerate,
                "depth": model_config.depth,
                "dim": model_config.dim,
                "attn_heads": model_config.attn_heads,
                "learning_rate": train_config.learning_rate,
                "batch_size": data_config.batch_size,
                "num_params": num_params
            },
        )
        run_id = run.id
        wandb.watch(model, log="all")
    
    print("Building the dataset...")
    train_loader, val_loader = get_loaders(data_config)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        if data_config.regenerate and epoch > start_epoch:
            print("Regenerating data...")
            train_loader, val_loader = get_loaders(data_config)
            print("Data regenerated.")

        # Training loop
        model.train()
        train_loss, train_acc = 0.0, 0.0
        num_train_batches = 0
        for inputs, targets in tqdm(train_loader, desc=f"Training  "):
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
            for inputs, targets in tqdm(val_loader, desc=f"Validation"):
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
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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

        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, save_path, run_id)
    
    if wandb_enabled:
        wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train the model.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', nargs='?', const=True, default=None, help='Resume from checkpoint. Optionally specify a signature.')
    group.add_argument('--name', type=str, default=None, help='Signature for the run (new run only).')
    parser.add_argument('--no-wandb', action='store_true', help='If True, disable Weights & Biases for logging.')
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
    train(
        model=model,
        data_config=data_config,
        train_config=train_config,
        model_config=model_config,
        wandb_enabled=not args.no_wandb,
        resume=args.resume,
        name=args.name,
    )

if __name__ == "__main__":
    main()