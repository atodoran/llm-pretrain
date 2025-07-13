def get_run_name_base(config, num_params):
    def params_str(num):
        if num >= 1e9:
            return f"{num/1e9:.0f}B"
        elif num >= 1e6:
            return f"{num/1e6:.0f}M"
        elif num >= 1e3:
            return f"{num/1e3:.0f}K"
        else:
            return str(num)

    learning_rate_str = f"{config.train.learning_rate:.0e}"
    beta2_str = f"{config.train.beta2}"
    weight_decay_str = f"{config.train.weight_decay:.0e}"
    batch_size_str = f"{config.train.batch_size}"
    name = (
        f"{config.task.name}_"
        f"{params_str(num_params)}_"
        f"{learning_rate_str}_"
        f"{beta2_str}_"
        f"{weight_decay_str}_"
        f"{batch_size_str}"
    )
    pretrained_str = config.model.pretrained.rsplit('/', 1)[-1] if config.model.pretrained else None
    if pretrained_str is not None:
        name += f"_{pretrained_str}"

    return name