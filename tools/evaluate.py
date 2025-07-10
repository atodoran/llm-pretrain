import sys
sys.path.append('./')

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from model import get_model
from data import get_loaders

def evaluate(
    model,
    config: DictConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print("Running skewed data experiments...\n")
    skewness_values = [0, 0.3, 0.7, 1.2, 1.6, 2.0]
    losses, accs = [], []
    for skewness in skewness_values:
        config.task.skewness = skewness
        val_loader = get_loaders(config, which=("val",))

        loss, acc = 0.0, 0.0
        num_batches = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Evaluation"):
                inputs = inputs.to(device)
                targets = targets.to(device).long()
                outputs = model(inputs).permute(0, 2, 1)
                batch_loss = loss_fn(outputs, targets)

                loss += batch_loss.item()
                predicted = outputs.detach().argmax(1)

                mask = (targets != -100)
                if mask.any():
                    acc += (predicted.eq(targets) & mask).float().sum().item() / mask.float().sum().item()
                else:
                    acc += 1.0
                num_batches += 1
        
        loss = loss / num_batches
        acc = acc / num_batches

        losses.append(loss)
        accs.append(acc)
    
    print("\nEvaluation Results by Skewness:")
    print("{:<10} {:<15} {:<15}".format("Skewness", "Loss", "Accuracy"))
    print("-" * 40)
    for skew, loss, acc in zip(skewness_values, losses, accs):
        print("{:<10} {:<15.4f} {:<15.4f}".format(skew, loss, acc))
        

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed(config.data.seed)

    print("Building the model...")
    model = get_model(config)

    if config.model.pretrained is not None:
        pretrained_path = os.path.join("models", f"{config.model.pretrained}.pth")
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}...")
            state_dict = torch.load(pretrained_path, map_location="cpu")["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"Pretrained weights not found at {pretrained_path}.")
    else:
        print("Warning: No pretrained path selected. Proceeding without loading pretrained weights.")

    evaluate(
        model=model,
        config=config,
    )

if __name__ == "__main__":
    main()