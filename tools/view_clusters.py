import sys
sys.path.append('./')

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from x_transformers import TransformerWrapper
from model import get_model
from data import PermutationComposition, collate_fn
from torch.utils.data import DataLoader
from functools import partial
from prefix_patch import apply_block
from sympy.combinatorics.permutations import Permutation
import hydra
from omegaconf import DictConfig

def extract_last_layer_representations(model, loader, device):
    model.eval()
    reps = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            # Embedding logic
            tok    = model.token_emb(inputs)
            pos    = model.pos_emb(tok)
            hidden = tok + pos
            hidden = model.post_emb_norm(hidden)
            hidden = model.emb_dropout(hidden)
            hidden = model.project_emb(hidden)
            # Run through all blocks
            for block_spec in model.attn_layers.layers:
                hidden = apply_block(block_spec, hidden)
            # hidden: (batch, seq_len, dim)
            reps.append(hidden.cpu().numpy())
            labels.append(targets.cpu().numpy())
    reps = np.concatenate(reps, axis=0)      # (N, seq_len, dim)
    labels = np.concatenate(labels, axis=0)  # (N, seq_len)
    return reps, labels

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model from config
    model = get_model(config).to(device)
    
    # Load checkpoint
    assert config.probe.checkpoint is not None, "Please specify probe.checkpoint in your config or CLI."
    ckpt = torch.load(config.probe.checkpoint, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)

    # Prepare dataset & loader
    dataset = PermutationComposition(
        n_samples=config.probe.n_samples,
        seq_length=config.model.max_seq_len,
        n=config.task.n
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Extract last-layer representations and labels
    reps, labels = extract_last_layer_representations(model, loader, device)
    # Flatten so each token is a sample
    reps_flat = reps.reshape(-1, reps.shape[-1])      # (N * seq_len, dim)
    labels_flat = labels.reshape(-1)                  # (N * seq_len,)

    # PCA
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(reps_flat)
    explained_var = np.sum(pca.explained_variance_ratio_)

    # Plot 1: color by permutation identity (3D)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pcs[:,0], pcs[:,1], pcs[:,2], c=labels_flat, cmap='tab20', s=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA of all tokens (colored by permutation)\nExplained variance: {explained_var:.3f}")
    fig.colorbar(scatter, ax=ax, label="Permutation rank")
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/pca_permutation_3d.png")
    print("Saved plot to plots/pca_permutation_3d.png")

    # Plot 2: color by parity (3D)
    parity = np.array([Permutation.unrank_lex(config.task.n, int(rank)).parity() for rank in labels_flat])
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    for p, color, label in zip([0,1], ['blue', 'red'], ['even', 'odd']):
        idxs = np.where(parity == p)[0]
        ax.scatter(pcs[idxs,0], pcs[idxs,1], pcs[idxs,2], c=color, label=label, s=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA of all tokens (colored by parity)\nExplained variance: {explained_var:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/pca_parity_3d.png")
    print("Saved plot to plots/pca_parity_3d.png")



if __name__ == "__main__":
    main()