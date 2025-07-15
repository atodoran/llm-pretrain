import sys
sys.path.append('./')

import os
import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import random

from model import get_model
from data import get_loaders

import torch.nn as nn
import torch

from colorama import init as colorama_init
from termcolor import colored

colorama_init()
normalize_rows = False
annotate = True

def save_heatmap(matrix, title, xlabel, ylabel, xticks, yticks, filename,
                 cmap='viridis'):
    data = matrix.copy()

    if normalize_rows:
        row_min = np.nanmin(data, axis=1, keepdims=True)
        row_max = np.nanmax(data, axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            data = (data - row_min) / (row_max - row_min + 1e-9)

    fig, ax = plt.subplots(figsize=(20, 6))
    cax = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
    fig.colorbar(cax, label='Normalized' if normalize_rows else 'NLD')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_xticklabels(xticks, rotation=90)
    ax.set_yticklabels(yticks)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha='center', va='center',
                            color='white' if data[i, j] < 0.5 else 'black',
                            fontsize=8)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath}")


def mask_future_keys(nld_table, threshold=0.9):
    L, Q, K = nld_table.shape
    mask = np.zeros((L, Q, K), dtype=bool)
    for q in range(Q):
        mask[:, q, :q] = nld_table[:, q, :q] < threshold
    return mask

def plot_nld_for_query(nld_table, key_points, Q):
    qi = Q - 1
    slice_ = nld_table[:, qi, :]  # shape: (L, K)
    layers = [f"L{l+1}" for l in range(nld_table.shape[0])]
    keys = [str(k) for k in key_points]

    save_heatmap(slice_,
                 title=f"NLD Heatmap (Query = {Q})",
                 xlabel="Key Position",
                 ylabel="Layer",
                 xticks=keys,
                 yticks=layers,
                 filename=f"super_plot.png",
                 cmap='coolwarm')

def plot_past_influence_count(nld_table, threshold=0.9):
    mask = mask_future_keys(nld_table, threshold)
    count_map = np.sum(mask, axis=2)  # shape: (L, Q)
    layers = [f"L{l+1}" for l in range(nld_table.shape[0])]
    queries = [f"Q{q+1}" for q in range(nld_table.shape[1])]

    save_heatmap(count_map,
                 title=f"# of Keys (K < Q) with NLD < {threshold}",
                 xlabel="Query Position",
                 ylabel="Layer",
                 xticks=queries,
                 yticks=layers,
                 filename=f"super_plot.png",
                 cmap='Blues')

def plot_mean_past_nld(nld_table):
    L, Q, K = nld_table.shape
    mean_map = np.full((L, Q), np.nan)

    for l in range(L):
        for q in range(Q):
            if q > 0:
                vals = nld_table[l, q, :q]
                if not np.all(np.isnan(vals)):
                    mean_map[l, q] = np.nanmean(vals)

    layers = [f"L{l+1}" for l in range(L)]
    queries = [f"Q{q+1}" for q in range(Q)]

    save_heatmap(mean_map,
                 title="Mean NLD (K < Q)",
                 xlabel="Query Position",
                 ylabel="Layer",
                 xticks=queries,
                 yticks=layers,
                 filename="super_plot.png",
                 cmap='viridis')

def plot_first_past_influential_key(nld_table, threshold=0.9):
    L, Q, K = nld_table.shape
    earliest_map = np.full((L, Q), fill_value=np.nan)

    for l in range(L):
        for q in range(Q):
            if q == 0:
                continue
            past_keys = np.where(nld_table[l, q, :q] < threshold)[0]
            if len(past_keys) > 0:
                earliest_map[l, q] = past_keys[0]

    layers = [f"L{l+1}" for l in range(L)]
    queries = [f"Q{q+1}" for q in range(Q)]

    save_heatmap(earliest_map,
                 title=f"First Past Key (K < Q) with NLD < {threshold}",
                 xlabel="Query Position",
                 ylabel="Layer",
                 xticks=queries,
                 yticks=layers,
                 filename=f"super_plot.png",
                 cmap='plasma')


def interactive_viewer_matplotlib(nld_table, key_points, seq_len):
    while True:
        print("\nChoose visualization:")
        print("1. NLD heatmap for a specific query position Q")
        print("2. Count of keys with NLD < threshold")
        print("3. Mean NLD across keys")
        print("4. First influential key with NLD < threshold")
        print("5. Quit")

        try:
            choice = input("Enter option (1–5): ").strip()
            if choice in ("5", "q", "quit", "exit"):
                break

            if choice == "1":
                Q = int(input(f"Enter query position Q (1 to {seq_len}): "))
                plot_nld_for_query(nld_table, key_points, Q)

            elif choice == "2":
                threshold = float(input("Threshold for NLD (e.g., 0.9): "))
                plot_past_influence_count(nld_table, threshold)

            elif choice == "3":
                plot_mean_past_nld(nld_table)

            elif choice == "4":
                threshold = float(input("Threshold for NLD (e.g., 0.9): "))
                plot_first_past_influential_key(nld_table, threshold)

            else:
                print("Invalid choice.")
        except Exception as e:
            print(f"Error: {e}")

def _unwrap_norm(norm):
    if norm is None:
        return None
    if isinstance(norm, (nn.ModuleList, list, tuple)):
        return _unwrap_norm(norm[0]) if len(norm) > 0 else None
    return norm

class TransformerBlock(nn.Module):
    def __init__(self, norms, layer_module):
        super().__init__()
        self.pre_norm, self.post_branch_norm, self.post_main_norm = [
            _unwrap_norm(n) for n in norms
        ]
        self.layer = layer_module

    def forward(self, x):
        branch_in = self.pre_norm(x) if self.pre_norm else x
        out = self.layer(branch_in)
        if self.post_branch_norm:
            out = self.post_branch_norm(out)
        x2 = x + out
        return self.post_main_norm(x2) if self.post_main_norm else x2

def convert_blocks_in_place(model):
    new_layers = nn.ModuleList()
    for norms, layer_module, _ in model.attn_layers.layers:
        new_layers.append(TransformerBlock(norms, layer_module))
    model.attn_layers.layers = new_layers

def run_up_to_layer(model, token_ids: torch.LongTensor, layer_idx: int):
    h = model.token_emb(token_ids) + model.pos_emb(token_ids)
    h = model.post_emb_norm(h)
    h = model.emb_dropout(h)
    h = model.project_emb(h)

    for i, block in enumerate(model.attn_layers.layers):
        h = block(h)
        if i == layer_idx:
            return h.clone()
    raise IndexError

def run_from_layer(model, hidden: torch.Tensor, layer_idx: int):
    for i, block in enumerate(model.attn_layers.layers):
        if i <= layer_idx:
            continue
        hidden = block(hidden)
    return model.to_logits(hidden)

def super_patch(model, config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    convert_blocks_in_place(model)
    if torch.__version__ >= '2':
        model = torch.compile(model)

    seq_len    = config.data.seq_length
    n_layers   = config.model.depth
    query_points = list(range(seq_len))
    key_points = list(range(seq_len))
    name = config.model.pretrained.rsplit('/', 1)[-1]
    nld_table_path = f"nld_tables/{name}.npy"

    nld_table = np.zeros((n_layers, len(query_points), len(key_points)))

    if os.path.exists(nld_table_path):
        print(f"Loading cached NLD table from {nld_table_path}")
        nld_table = np.load(nld_table_path)
    else:
        print("Computing NLD table from scratch...")
        def ld(logits, y_true, y_alt):
            return (logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
                  - logits.gather(1, y_alt.unsqueeze(1)).squeeze(1))

        def nld(ld_p, ld_cor, ld_c):
            return (ld_p - ld_cor) / (ld_c - ld_cor + 1e-9)

        config.data.use_tqdm = False
        for li, L in enumerate(range(1, n_layers * 2, 2)):
            print(f"Layer {li + 1}")
            for qi, Q in enumerate(query_points):
                for ki, K in enumerate(key_points):
                    if Q <= K:
                        nld_table[li, qi, ki] = 0
                        print(f"  (L={li + 1}, K={K}, Q={Q}) median NLD: {0:.3f}")
                        continue
                    
                    val_loader = get_loaders(config, which=("val",))
                    all_batch_nlds = []

                    with torch.no_grad():
                        for clean_ids, _ in val_loader:
                            clean_ids = clean_ids.to(device)
                            B = clean_ids.size(0)

                            # make corrupted batch
                            orig_tok = clean_ids[:, K:K+1]              # (B,1)
                            diff_mask = clean_ids.ne(orig_tok)          # (B,S)
                            j_idx = torch.multinomial(diff_mask.float(), 1)   # (B,1)
                            corrupt_ids = clean_ids.clone()
                            src_tok = clean_ids.gather(1, j_idx)        # (B,1)
                            tgt_idx = torch.full_like(j_idx, K)         # (B,1)
                            corrupt_ids.scatter_(1, tgt_idx, src_tok)

                            # 1) clean run
                            post_clean    = run_up_to_layer(model, clean_ids, L)
                            logits_clean  = run_from_layer(model, post_clean, L)

                            # 2) corrupt baseline
                            post_corr     = run_up_to_layer(model, corrupt_ids, L)
                            logits_corr   = run_from_layer(model, post_corr, L)

                            # 3) patched
                            post_patched = post_corr.clone()
                            post_patched[:, :Q, :] = post_clean[:, :Q, :]
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

                    nld_table[li, qi, ki] = median_nld
                    print(f"  (L={li + 1}, K={K}, Q={Q}) median NLD: {median_nld:.3f}")

        # Save the computed table
        np.save(cache_path, nld_table)
        print(f"NLD table saved to {cache_path}")

    # Interactive session
    interactive_viewer_matplotlib(nld_table, key_points, seq_len)


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

    super_patch(model, config)

if __name__ == "__main__":
    main()
