import sys
sys.path.append('./')

from x_transformers import Encoder, TransformerWrapper
import torch
import numpy as np
import os
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from model import get_model
from data import get_loaders

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    print("Building model...")
    model = get_model(config)

    if config.model.pretrained:
        path = os.path.join("models", f"{config.model.pretrained}.pth")
        sd   = torch.load(path, map_location="cpu")["model_state_dict"]
        model.load_state_dict(sd, strict=False)

    config.data.n_val_samples = 1
    loader_val = get_loaders(config, which=("val",))

    batch = next(iter(loader_val))
    inputs, targets = batch
    logits, attn_maps = model(inputs, return_attn=True)
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    seq_len = attn_maps[0].shape[-1]

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(2 * num_heads, 2 * num_layers))
    if num_layers == 1:
        axes = [axes]
    if num_heads == 1:
        axes = [[ax] for ax in axes]

    for i, attn_map in enumerate(attn_maps):
        attn = attn_map[0].detach().cpu().numpy()  # (heads, seq_len, seq_len)
        for h in range(num_heads):
            ax = axes[i][h]
            im = ax.imshow(attn[h], aspect='auto', cmap='viridis', vmin=0, vmax=attn[h].max())
            ax.set_title(f'L{i} H{h}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    plt.savefig("plots/attention_maps.png")
    plt.close(fig)

if __name__ == "__main__":
    main()