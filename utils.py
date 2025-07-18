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

    name = (
        f"{config.task.name}_"
        f"{params_str(num_params)}_"
        f"{config.train.learning_rate:.0e}_"
        f"{config.train.beta2}_"
        f"{config.train.weight_decay:.0e}_"
        f"{config.train.batch_size}_"
        f"{config.data.seed}"
    )
    pretrained_str = config.model.pretrained.rsplit('/', 1)[-1] if config.model.pretrained else None
    if pretrained_str is not None:
        name += f"_{pretrained_str}"

    return name