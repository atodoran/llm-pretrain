from x_transformers import Decoder, TransformerWrapper
from omegaconf import DictConfig

def get_model(config: DictConfig):
    attn_layers = Decoder(
        depth=config.model.depth,
        dim=config.model.dim,
        heads=config.model.attn_heads,
    )
    model = TransformerWrapper(
        attn_layers=attn_layers,
        max_seq_len=config.model.max_seq_len,
        num_tokens=config.model.vocab_size,
    )
    return model