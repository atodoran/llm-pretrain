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
import hydra
from omegaconf import DictConfig
import sys
import gc
import signal

from model import get_model
from data import get_loaders
from utils import get_run_name_base
from tools.prefix_patch import prefix_patch

def cleanup(signum, frame):
    wandb.finish()
    sys.exit(0)

try:
    profile
except NameError:
    def profile(func): return func


def save_checkpoint(model, optimizer, epoch, best_val_acc, path, wandb_id):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val_acc": best_val_acc,
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
    epoch = ckpt.get("epoch", 0)
    best_val_acc = ckpt.get("best_val_acc", 0)
    wandb_id = ckpt.get("wandb_id")
    return epoch, best_val_acc, wandb_id


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


@profile
def train(
    model,
    config: DictConfig,
):

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        eps=config.train.epsilon,
        weight_decay=config.train.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    epochs = config.train.epochs
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_name_base = get_run_name_base(
        config=config,
        num_params=num_params,
    )

    resume, name = config.train.resume, config.train.name
    checkpoint_path = None
    if resume is not None:
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
    
    start_epoch, best_val_acc, run_id = 0, 0, None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_val_acc, run_id = load_checkpoint(model, optimizer, checkpoint_path, device)
        print(f"Resumed from epoch {start_epoch}")

    if config.train.wandb:
        run = wandb.init(
            entity="harvardml",
            project="llm-pretraining",
            id=run_id,
            resume="allow",
            name=run_name,
            config={
                "task": config.task.name,
                "n_train_samples": config.data.n_train_samples,
                "n_val_samples": config.data.n_val_samples,
                "seq_length": config.data.seq_length,
                "regenerate": config.data.regenerate,
                "depth": config.model.depth,
                "dim": config.model.dim,
                "attn_heads": config.model.attn_heads,
                "learning_rate": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
                "beta1": config.train.beta1,
                "beta2": config.train.beta2,
                "epsilon": config.train.epsilon,
                "batch_size": config.train.batch_size,
                "num_params": num_params,
                "pretrained": config.model.pretrained
            },
        )
        run_id = run.id
        wandb.watch(model, log="gradients", log_freq=100, log_graph=False)
    
    print("Building the dataset...")
    train_loader, val_loader = get_loaders(config)

    print("Starting training...")
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        if config.data.regenerate and epoch > start_epoch:
            print("Regenerating data...")
            train_loader, val_loader = get_loaders(config)
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
            predicted = outputs.detach().argmax(1)

            with torch.no_grad():
                mask = (targets != -100)
                if mask.any():
                    train_acc += (predicted.eq(targets) & mask).float().sum().item() / mask.float().sum().item()
                else:
                    train_acc += 1.0
                num_train_batches += 1

        train_loss = train_loss / num_train_batches
        train_acc = train_acc / num_train_batches

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
                predicted = outputs.detach().argmax(1)

                mask = (targets != -100)
                if mask.any():
                    val_acc += (predicted.eq(targets) & mask).float().sum().item() / mask.float().sum().item()
                else:
                    val_acc += 1.0
                num_val_batches += 1
        
        val_loss = val_loss / num_val_batches
        val_acc = val_acc / num_val_batches

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if config.train.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch + 1, best_val_acc, checkpoint_path, run_id)

        if save_path and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch + 1, best_val_acc, save_path, run_id)
        
        if config.train.prefix_patch_epochs is not None and (epoch + 1) % config.train.prefix_patch_epochs == 0:
            print("Running prefix patch...")
            name = run_name + "_" + str(epoch + 1)
            prefix_patch(model, config, name)
            model.train()
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print("Running prefix patch...")
    name = run_name + "_" + str(epochs)
    prefix_patch(model, config, name)

    if config.train.wandb:
        wandb.finish()

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    signal.signal(signal.SIGTERM, cleanup)
    
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed(config.data.seed)

    if config.train.name is not None and config.train.resume is not None:
        raise ValueError("Cannot specify both 'name' and 'resume'.")

    print("Building the model...")
    model = get_model(config)

    if config.model.pretrained is not None and config.train.resume is None:
        pretrained_path = os.path.join("models", f"{config.model.pretrained}.pth")
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}...")
            state_dict = torch.load(pretrained_path, map_location="cpu")["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Pretrained weights not found at {pretrained_path}.")

    train(
        model=model,
        config=config,
    )

if __name__ == "__main__":
    main()