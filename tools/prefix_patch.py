import os
import numpy as np
import torch
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from model import get_model
from data import get_loaders

def prefix_patch(model, config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_loader = get_loaders(config, which=("val",))

    seq_len = config.data.seq_length
    num_layers = config.model.depth
    prefix_positions = list(range(seq_len + 1))

    # Helper to hook modules
    from contextlib import contextmanager
    @contextmanager
    def layer_hook(module, fn):
        handle = module.register_forward_hook(fn)
        try:
            yield
        finally:
            handle.remove()

    def ld(logits, y_clean, y_corrupt):
        # difference in logit scores
        return (logits[..., y_clean] - logits[..., y_corrupt])

    def nld(ld_patched, ld_corrupt, ld_clean):
        return (ld_patched - ld_corrupt) / (ld_clean - ld_corrupt + 1e-9)

    # Table to store median NLDs
    nld_table = np.zeros((num_layers, len(prefix_positions)))

    for layer_idx in range(num_layers):
        # pick the residual wrapper for this layer
        block_res = model.attn_layers.layers[layer_idx][2]

        for pos_idx, prefix_len in enumerate(prefix_positions):
            scores = []
            with torch.no_grad():
                for clean_inputs, _ in val_loader:
                    clean_inputs = clean_inputs.to(device)
                    # corrupt first token
                    corrupt_inputs = clean_inputs.clone()
                    for i in range(clean_inputs.size(0)):
                        orig = clean_inputs[i, 0].item()
                        cands = (clean_inputs[i, 1:] != orig).nonzero(as_tuple=True)[0]
                        if cands.numel():
                            j = np.random.choice(cands.cpu().numpy())
                            corrupt_inputs[i, 0] = clean_inputs[i, j + 1]

                    # save post-residual for clean
                    clean_cache = {}
                    def save_post(module, inp, out):
                        clean_cache['post'] = out.detach().clone()

                    with layer_hook(block_res, save_post):
                        logits_clean = model(clean_inputs)

                    # baseline corrupt
                    logits_corrupt = model(corrupt_inputs)

                    # predictions
                    y_clean = logits_clean.argmax(dim=-1)[:, -1]
                    y_corrupt = logits_corrupt.argmax(dim=-1)[:, -1]
                    idx = torch.arange(clean_inputs.size(0), device=device)

                    # compute ld for clean and corrupt
                    ld_clean = ld(logits_clean[idx, -1], y_clean, y_corrupt)
                    ld_corrupt = ld(logits_corrupt[idx, -1], y_clean, y_corrupt)

                    # patch residual for corrupt
                    def patch_post(module, inp, out):
                        patched = out.clone()
                        patched[:, :prefix_len, :] = clean_cache['post'][:, :prefix_len, :]
                        return patched

                    with layer_hook(block_res, patch_post):
                        logits_patched = model(corrupt_inputs)

                    ld_patched = ld(logits_patched[idx, -1], y_clean, y_corrupt)

                    # filter valid
                    mask = (y_clean != y_corrupt) & ((ld_clean - ld_corrupt).abs() > 1e-6)
                    if mask.any():
                        batch_nld = nld(ld_patched[mask], ld_corrupt[mask], ld_clean[mask])
                        scores.append(batch_nld.cpu())

            if scores:
                all_scores = torch.cat(scores)
                median_nld = float(torch.median(all_scores))
                mean_nld = float(torch.mean(all_scores))
            else:
                median_nld, mean_nld = float('nan')
            nld_table[layer_idx, pos_idx] = mean_nld
            print(f"Layer {layer_idx}, prefix {prefix_len}: mean NLD = {median_nld:.3f}")

    # print table
    print("NLD Table:")
    header = "      " + " ".join(f"{p:>6}" for p in prefix_positions)
    print(header)
    for i, row in enumerate(nld_table):
        print(f"L{i:<2} " + " ".join(f"{v:6.3f}" for v in row))

    # plot
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12,6))
    plt.imshow(nld_table, aspect='auto', cmap='viridis')
    plt.colorbar(label='Mean NLD')
    plt.xlabel('Prefix')
    plt.ylabel('Layer')
    plt.xticks(np.arange(len(prefix_positions)), prefix_positions, rotation=45)
    plt.yticks(np.arange(num_layers), list(range(num_layers)))
    plt.title('Prefix Patching NLD Heatmap')
    plt.tight_layout()
    plt.savefig('plots/prefix_patch.png')

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    np.random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    print("Building model...")
    model = get_model(config)

    if config.model.pretrained:
        path = os.path.join("models", f"{config.model.pretrained}.pth")
        if os.path.exists(path):
            print(f"Loading weights from {path}...")
            sd = torch.load(path, map_location="cpu")["model_state_dict"]
            model.load_state_dict(sd, strict=False)
        else:
            raise FileNotFoundError(f"No pretrained file at {path}")
    else:
        print("No pretrained weights specified; training from scratch.")

    prefix_patch(model, config)

if __name__ == "__main__":
    main()
