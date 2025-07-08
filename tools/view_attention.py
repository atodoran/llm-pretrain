from x_transformers import Encoder, TransformerWrapper
import torch
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from data import get_loaders
from config import ModelConfig, DataConfig, load_yml

def get_model(model_config):
    attn_layers = Encoder(
        depth=model_config.depth,
        dim=model_config.dim,
        heads=model_config.attn_heads,
    )
    model = TransformerWrapper(
        attn_layers=attn_layers,
        max_seq_len=model_config.max_seq_len,
        num_tokens=model_config.vocab_size,
    )
    return model

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="View the attention maps of the model.")
    parser.add_argument('--model-path', type=str, default=None, help='Path to the model checkpoint.')
    args = parser.parse_args()

    data_config_path = os.path.join('configs', 'data.yaml')
    model_config_path = os.path.join('configs', 'model.yaml')
    data_config = DataConfig.from_dict(kwargs=load_yml(data_config_path))
    model_config = ModelConfig.from_dict(kwargs=load_yml(model_config_path))

    model = get_model(model_config)
    model = load_checkpoint(model, args.model_path)
    
    np.random.seed(data_config.seed)
    torch.manual_seed(data_config.seed)
    torch.cuda.manual_seed(data_config.seed)

    data_config.batch_size = 1
    _, loader_val = get_loaders(data_config)

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