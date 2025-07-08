import sys
sys.path.append('./')

import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from x_transformers import TransformerWrapper
from model import get_model
from data import PermutationComposition, collate_fn
from torch.utils.data import DataLoader
from config import ModelConfig, load_yml
import hydra
from omegaconf import DictConfig


def default_identity_state(batch_inputs, batch_targets):
    """
    Probes the model's target labels directly.
    Returns a NumPy array of shape (batch_size, seq_len).
    """
    return batch_targets.cpu().numpy()


class StateExtractor:
    def __init__(self, state_fn=default_identity_state):
        """
        state_fn: (inputs: Tensor, targets: Tensor) -> np.ndarray of shape (batch_size, seq_len)
        Defines which ground-truth states to probe.
        """
        self.state_fn = state_fn

    def extract(self, inputs, targets):
        return self.state_fn(inputs, targets)


class RepresentationExtractor:
    def __init__(self, model: TransformerWrapper):
        """
        Uses model(..., return_intermediates=True) to grab per-layer states.
        """
        self.model = model
        self.depth = model.attn_layers.depth # number of transformer blocks

    def extract(self, inputs: torch.Tensor) -> list:
        """
        Performs a forward pass returning post-residual hidden states for each block.
        Returns a list of length `depth`, each of shape (batch, seq_len, dim) as NumPy array.
        """
        logits, intermediates = self.model(inputs, return_intermediates=True)
        post_states = intermediates.layer_hiddens
        assert len(post_states) == 2 * self.depth + 1

        res_stream = post_states[1::2]  # take only post-residual states
        assert len(res_stream) == self.depth
        
        # convert to NumPy
        return [h.detach().cpu().numpy() for h in res_stream]


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
        activations = {i: [] for i in range(self.repr_ext.depth)}

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
            clf = LogisticRegression(max_iter=200, n_jobs=-1)
            clf.fit(X, flat_y)
            results[idx] = clf.score(X, flat_y)
        return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model from config
    model = get_model(cfg).to(device)
    
    # Load checkpoint
    assert cfg.probe.checkpoint is not None, "Please specify probe.checkpoint in your config or CLI."
    ckpt = torch.load(cfg.probe.checkpoint, map_location=device)
    sd = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)

    # prepare dataset & loader
    dataset = PermutationComposition(
        n_samples=100,
        seq_length=cfg.model.max_seq_len,
        n=3
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # set up probes
    repr_ext = RepresentationExtractor(model)
    state_ext = StateExtractor()
    trainer = ProbeTrainer(model, loader, state_ext, repr_ext, device)

    # train and print
    results = trainer.train_probes()
    print("Probe results (layer_idx: accuracy):")
    for idx in sorted(results):
        print(f"  Layer {idx:2d}: {results[idx]:.4f}")

if __name__ == "__main__":
    main()
