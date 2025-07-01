from x_transformers import Decoder, TransformerWrapper

def get_model(model_config):
    attn_layers = Decoder(
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