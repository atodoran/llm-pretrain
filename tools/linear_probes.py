import sys
sys.path.append('./')

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from x_transformers import TransformerWrapper
from model import get_model
from data import PermutationComposition, collate_fn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from functools import partial
from prefix_patch import apply_block


def default_identity_state(batch_inputs, batch_targets):
    """
    Probes the model's target labels directly.
    Returns a NumPy array of shape (batch_size, seq_len).
    """
    return batch_targets.cpu().numpy()

def permutation_parity_state(batch_inputs, batch_targets, n=3):
    """
    Extracts the parity (even=0, odd=1) of each permutation token in batch_inputs.
    batch_inputs: Tensor of shape (batch_size, seq_len), each entry is a permutation rank.
    n: size of the permutation (should match the 'n' used in PermutationComposition)
    Returns: np.ndarray of shape (batch_size, seq_len) with parity values.
    """
    from sympy.combinatorics.permutations import Permutation

    batch_inputs_np = batch_inputs.cpu().numpy()
    batch_size, seq_len = batch_inputs_np.shape
    parity_array = np.zeros_like(batch_inputs_np)
    for i in range(batch_size):
        for j in range(seq_len):
            rank = batch_inputs_np[i, j]
            perm = Permutation.unrank_lex(n, rank)
            parity_array[i, j] = perm.parity()
    return parity_array


class StateExtractor:
    def __init__(self, state_fn=default_identity_state):
        """
        state_fn: (inputs: Tensor, targets: Tensor) -> np.ndarray of shape (batch_size, seq_len)
        Defines which ground-truth states to probe.
        """
        self.state_fn = state_fn

    def extract(self, inputs, targets):
        states = self.state_fn(inputs, targets)
        print(states)
        return states


class RepresentationExtractor:
    def __init__(self, model: TransformerWrapper):
        """
        Uses model(..., return_intermediates=True) to grab per-layer states.
        """
        self.model = model
        self.depth = model.attn_layers.depth # number of transformer blocks

    def extract(self, inputs: torch.Tensor) -> list:
        """
        Performs a forward pass, collecting post-residual hidden states for each block.
        Returns a list of length `depth`, each of shape (batch, seq_len, dim) as NumPy array.
        """
        # Embedding logic
        tok    = self.model.token_emb(inputs)
        pos    = self.model.pos_emb(tok)
        hidden = tok + pos
        hidden = self.model.post_emb_norm(hidden)
        hidden = self.model.emb_dropout(hidden)
        hidden = self.model.project_emb(hidden)

        # Collect post-residual hidden states after each block
        post_residual_states = []
        for i, block_spec in enumerate(self.model.attn_layers.layers):
            hidden = apply_block(block_spec, hidden)
            post_residual_states.append(hidden.detach().cpu().numpy())

        return post_residual_states

    # def extract(self, inputs: torch.Tensor) -> list:
    #     """
    #     Performs a forward pass returning post-residual hidden states for each block.
    #     Returns a list of length `depth`, each of shape (batch, seq_len, dim) as NumPy array.
    #     """
    #     logits, intermediates = self.model(inputs, return_intermediates=True)
    #     post_states = intermediates.layer_hiddens
    #     assert len(post_states) == 2 * self.depth + 1

    #     res_stream = post_states[1::2]  # take only post-residual states
    #     assert len(res_stream) == self.depth

    #     return [h.detach().cpu().numpy() for h in res_stream]


class ProbeTrainer:
    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: DataLoader,
                 state_extractor: StateExtractor,
                 repr_extractor: RepresentationExtractor,
                 device: torch.device = None):
        self.model = model.to(device or torch.device('cpu'))
        self.loader = data_loader
        self.state_ext = state_extractor
        self.repr_ext = repr_extractor
        self.device = device or torch.device('cpu')

    def collect_data(self):
        """
        Runs the model over the loader, collecting states and activations.
        Returns:
          states: np.ndarray of shape (N, seq_len)
          reps: dict[layer_idx -> np.ndarray of shape (N, seq_len, dim)]
        """
        self.model.eval()
        all_states = []
        # initialize per-layer lists
        activations = {i: [] for i in range(2 * self.repr_ext.depth)}

        with torch.no_grad():
            for batch in self.loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # extract ground-truth states
                states = self.state_ext.extract(inputs, targets)
                all_states.append(states)

                # extract per-layer representations
                layer_outputs = self.repr_ext.extract(inputs)
                for idx, arr in enumerate(layer_outputs):
                    activations[idx].append(arr)

        # concatenate all sequence batches
        states = np.concatenate(all_states, axis=0)
        reps = {idx: np.concatenate(frames, axis=0)
                for idx, frames in activations.items()}
        return states, reps

    def train_probes(self):
        """
        For each layer, trains a LogisticRegression probe.
        Returns dict[layer_idx -> accuracy]
        """
        states, reps = self.collect_data()
        results = {}
        # flatten tokens into examples
        flat_y = states.reshape(-1)
        for idx, activation in reps.items():
            N, L, D = activation.shape
            X = activation.reshape(N * L, D)
            clf = LogisticRegression(max_iter=500, n_jobs=-1)
            clf.fit(X, flat_y)
            results[idx] = clf.score(X, flat_y)
        return results


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

    # prepare dataset & loader
    dataset = PermutationComposition(
        n_samples=config.probe.n_samples,
        seq_length=config.model.max_seq_len,
        n=3
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # set up probes
    repr_ext = RepresentationExtractor(model)
    state_ext = StateExtractor(state_fn=partial(permutation_parity_state, n=config.task.n) if config.probe.state == 'parity' else default_identity_state)
    trainer = ProbeTrainer(model, loader, state_ext, repr_ext, device)

    # train and print
    results = trainer.train_probes()
    print(f"Probe results for {config.probe.state} state (layer_idx: accuracy):")
    for idx in sorted(results):
        print(f"  Layer {idx:2d}: {results[idx]:.4f}")

    results = trainer.train_probes()
    print(f"Probe results for {config.probe.state} state (layer_idx: accuracy):")
    for idx in sorted(results):
        print(f"  Layer {idx:2d}: {results[idx]:.4f}")

    # --- Save plot ---
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(4, 2.5))
    layers = list(sorted(results.keys()))
    accuracies = [results[idx] for idx in layers]
    plt.bar(layers, accuracies)
    plt.xlabel("Layer index")
    plt.ylabel("Probe accuracy")
    plt.title(f"Probe accuracy by layer ({config.probe.state} state)")
    plt.tight_layout()
    plt.savefig(f"plots/probe_{config.probe.state}.png")
    print(f"Saved plot to plots/probe_{config.probe.state}.png")


if __name__ == "__main__":
    main()
