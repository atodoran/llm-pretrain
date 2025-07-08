import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig

from model import get_model
from data import get_loaders

def prefix_patch(
    model,
    config: DictConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ------`---------------------------------------------------------------
    # 0.  Helpers ----------------------------------------------------------
    # ------`---------------------------------------------------------------

    @contextmanager
    def layer_hook(module, fn):
        """Register a forward hook, automatically remove it afterwards."""
        handle = module.register_forward_hook(fn, prepend=False)
        try:
            yield
        finally:
            handle.remove()

    def ld(logits, y_clean, y_corrupt):
        """Logit-difference (scalar, batch size 1 for simplicity)."""
        # logits: [1, vocab]
        return (logits[0, y_clean] - logits[0, y_corrupt]).item()

    def nld(ld_patched, ld_corrupt, ld_clean):
        """Normalized-Logit-Difference."""
        return (ld_patched - ld_corrupt) / (ld_clean - ld_corrupt + 1e-9)
    
    seq_len = config.data.seq_length

    import matplotlib.pyplot as plt

    num_layers = config.model.depth
    prefix_positions = list(range(10, seq_len + 1, 10))
    nld_table = np.zeros((num_layers, len(prefix_positions)))

    for layer_idx in range(num_layers):
        for pos_idx, PREFIX_LEN in enumerate(prefix_positions):
            val_loader = get_loaders(config, which=("val",))
            ff_pn = model.attn_layers.layers[layer_idx][1]

            with torch.no_grad():
                total_score = 0
                num_batches = 0

                for clean_inputs, targets in val_loader:
                    batch_size = clean_inputs.size(0)

                    corrupt_inputs = clean_inputs.clone()
                    for i in range(batch_size):
                        orig_token = clean_inputs[i, 0].item()
                        candidates = (clean_inputs[i, 1:] != orig_token).nonzero(as_tuple=True)[0]
                        if len(candidates) > 0:
                            rand_idx = np.random.choice(candidates.cpu().numpy())
                            new_token = clean_inputs[i, rand_idx + 1].item()
                            corrupt_inputs[i, 0] = new_token

                    clean_cache = {}

                    def save_post_residual(_, __, out):
                        if isinstance(out, tuple):
                            out = out[0]
                        clean_cache["post"] = out.detach().clone()

                    with layer_hook(ff_pn, save_post_residual):
                        logits_clean = model(clean_inputs)

                    logits_corrupt = model(corrupt_inputs)

                    y_clean   = logits_clean.argmax(dim=-1)[:, -1]
                    y_corrupt = logits_corrupt.argmax(dim=-1)[:, -1]

                    ld_clean = []
                    ld_corrupt = []
                    for i in range(batch_size):
                        ld_clean.append(ld(logits_clean[i, -1].unsqueeze(0), y_clean[i].item(), y_corrupt[i].item()))
                        ld_corrupt.append(ld(logits_corrupt[i, -1].unsqueeze(0), y_clean[i].item(), y_corrupt[i].item()))
                    ld_clean = torch.tensor(ld_clean, device=device)
                    ld_corrupt = torch.tensor(ld_corrupt, device=device)

                    def patch_post_residual(_, __, out):
                        is_tuple = isinstance(out, tuple)
                        x        = out[0] if is_tuple else out
                        x = x.clone()
                        x[:, :PREFIX_LEN, :] = clean_cache["post"][:, :PREFIX_LEN, :]
                        return (x, *out[1:]) if is_tuple else x

                    with layer_hook(ff_pn, patch_post_residual):
                        logits_patched = model(corrupt_inputs)

                    ld_patched = []
                    for i in range(batch_size):
                        ld_patched.append(ld(logits_patched[i, -1].unsqueeze(0), y_clean[i].item(), y_corrupt[i].item()))
                    ld_patched = torch.tensor(ld_patched, device=device)

                    score = nld(ld_patched, ld_corrupt, ld_clean)
                    total_score += score.sum().item()

            avg_nld = total_score / config.data.n_val_samples
            nld_table[layer_idx, pos_idx] = avg_nld
            print(f"Avg NLD for layer {layer_idx}, prefix {PREFIX_LEN}: {avg_nld:.3f}")

    # Print table
    print("\nNLD Table (layers x prefix positions):")
    print("      " + " ".join([f"{p:>6}" for p in prefix_positions]))
    for i, row in enumerate(nld_table):
        print(f"Layer {i:<2} " + " ".join([f"{v:6.3f}" for v in row]))

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(nld_table, aspect='auto', cmap='viridis')
    plt.colorbar(label='Avg NLD')
    plt.xlabel('Prefix Position')
    plt.ylabel('Layer')
    plt.xticks(ticks=np.arange(len(prefix_positions)), labels=prefix_positions, rotation=45)
    plt.yticks(ticks=np.arange(num_layers), labels=[f"{i}" for i in range(num_layers)])
    plt.title('NLD Heatmap (Layer x Prefix Position)')
    plt.tight_layout()
    plt.savefig("plots/prefix_patch.png")

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

    prefix_patch(
        model=model,
        config=config,
    )

if __name__ == "__main__":
    main()