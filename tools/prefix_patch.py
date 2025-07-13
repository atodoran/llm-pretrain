import sys
sys.path.append('./')

import os
import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from model import get_model
from data import get_loaders

import torch.nn as nn
import torch

def _unwrap_norm(norm):
    """
    If norm is a nested ModuleList (or list/tuple) wrapping one LayerNorm,
    dig down until you hit a real nn.Module or None.
    """
    if norm is None:
        return None
    if isinstance(norm, (nn.ModuleList, list, tuple)):
        return _unwrap_norm(norm[0]) if len(norm) > 0 else None
    return norm  # real nn.Module

def apply_block(block_spec: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    """
    block_spec = ModuleList([ norms, layer_module, residual_fn ])
      norms          = ModuleList([ pre_norm, post_branch_norm, post_main_norm ])
      layer_module   = Attention or FeedForward
      residual_fn    = Residual()
    """
    norms, layer_module, residual_fn = block_spec

    # grab the three norms (some may be None, and some may be wrapped in a ModuleList)
    pre_norm         = _unwrap_norm(norms[0])
    post_branch_norm = _unwrap_norm(norms[1])
    post_main_norm   = _unwrap_norm(norms[2])

    # pre-norm branch
    branch_in = pre_norm(x) if pre_norm is not None else x
    out       = layer_module(branch_in)

    # post-branch norm (sandwich-norm style)
    if post_branch_norm is not None:
        out = post_branch_norm(out)

    # add residual
    x2 = x + out

    # final main norm
    if post_main_norm is not None:
        x2 = post_main_norm(x2)

    return x2

def run_up_to_layer(model, token_ids: torch.LongTensor, layer_idx: int) -> torch.Tensor:
    # embeddings
    tok    = model.token_emb(token_ids)
    pos    = model.pos_emb(tok)
    hidden = tok + pos
    hidden = model.post_emb_norm(hidden)
    hidden = model.emb_dropout(hidden)
    hidden = model.project_emb(hidden)

    # apply each block
    for i, block_spec in enumerate(model.attn_layers.layers):
        hidden = apply_block(block_spec, hidden)
        if i == layer_idx:
            return hidden.clone()

    raise IndexError(f"layer_idx {layer_idx} >= depth {len(model.attn_layers.layers)}")

def run_from_layer(model, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
    # finish the remaining blocks
    for i, block_spec in enumerate(model.attn_layers.layers):
        if i <= layer_idx:
            continue
        hidden = apply_block(block_spec, hidden)

    return model.to_logits(hidden)

def prefix_patch(model, config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_loader = get_loaders(config, which=("val",))
    seq_len    = config.data.seq_length
    n_layers   = config.model.depth
    prefixes   = list(range(seq_len + 1))

    def ld(logits, y_true, y_alt):
        return (logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
              - logits.gather(1, y_alt.unsqueeze(1)).squeeze(1))

    def nld(ld_p, ld_cor, ld_c):
        return (ld_p - ld_cor) / (ld_c - ld_cor + 1e-9)

    nld_table = np.zeros((n_layers, len(prefixes)))

    for li, L in enumerate(range(1, n_layers * 2, 2)):
        print(f"---- Layer {li + 1} ----")
        for pi, P in enumerate(prefixes):
            all_batch_nlds = []

            with torch.no_grad():
                for clean_ids, _ in val_loader:
                    clean_ids = clean_ids.to(device)
                    B = clean_ids.size(0)

                    # make corrupted batch
                    corrupt_ids = clean_ids.clone()
                    for i in range(B):
                        orig = clean_ids[i, 0].item()
                        cands = (clean_ids[i, 1:] != orig).nonzero(as_tuple=True)[0]
                        if cands.numel():
                            j = np.random.choice(cands.cpu().numpy())
                            corrupt_ids[i, 0] = clean_ids[i, j+1]

                    # 1) clean run up to and after layer L
                    post_clean    = run_up_to_layer(model, clean_ids, L)
                    logits_clean  = run_from_layer(model, post_clean, L)

                    # 2) corrupt baseline
                    post_corr     = run_up_to_layer(model, corrupt_ids, L)
                    logits_corr   = run_from_layer(model, post_corr, L)

                    # 3) patched: splice in the first P tokens of clean's post
                    post_patched = post_corr.clone()
                    post_patched[:, :P, :] = post_clean[:, :P, :]
                    logits_pat  = run_from_layer(model, post_patched, L)

                    # pick predictions
                    y_clean  = logits_clean.argmax(dim=-1)[:, -1]
                    y_corr   = logits_corr.argmax(dim=-1)[:, -1]
                    idxs     = torch.arange(B, device=device)

                    ld_c = ld(logits_clean[idxs, -1], y_clean, y_corr)
                    ld_k = ld(logits_corr[idxs, -1], y_clean, y_corr)
                    ld_p = ld(logits_pat[idxs, -1], y_clean, y_corr)

                    # mask trivial cases
                    valid = (y_clean != y_corr) & ((ld_c - ld_k).abs() > 1e-6)
                    if valid.any():
                        batch_nld = nld(ld_p[valid], ld_k[valid], ld_c[valid])
                        all_batch_nlds.append(batch_nld.cpu())

            if all_batch_nlds:
                all_nlds = torch.cat(all_batch_nlds)
                median_nld = float(all_nlds.median())
            else:
                median_nld = float('nan')

            nld_table[li, pi] = median_nld
            print(f"  ({li + 1},{P}) median NLD: {median_nld:.3f}")

    # print table
    print("\nMean NLD table (layers × prefixes):")
    header = "     " + " ".join(f"{p:>6}" for p in prefixes)
    print(header)
    for l, row in enumerate(nld_table):
        print("L{:<2} ".format(l) + " ".join(f"{v:6.3f}" for v in row))

    # heatmap
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.imshow(nld_table, aspect='auto', cmap='viridis')
    plt.colorbar(label='Median NLD')
    plt.xlabel('Prefix length')
    plt.ylabel('Layer index')
    n_ticks = 10
    tick_positions = np.linspace(0, len(prefixes)-1, n_ticks, dtype=int)
    tick_labels = [prefixes[i] for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    plt.yticks(np.arange(n_layers), np.arange(n_layers))
    plt.title('Prefix-Patching NLD Heatmap')
    plt.tight_layout()

    # save plot
    plot_name = config.model.pretrained
    plot_name = plot_name.rsplit('/', 1)[-1]
    plot_name = "prefix_patch_" + plot_name
    plt.savefig(f"plots/{plot_name}.png")
    print(f"plots/{plot_name}.png")

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

    prefix_patch(model, config)

if __name__ == "__main__":
    main()
