def get_run_name_base (config, num_params):
    def human_readable_params(num):
        if num >= 1e9:
            return f"{num/1e9:.0f}B"
        elif num >= 1e6:
            return f"{num/1e6:.0f}M"
        elif num >= 1e3:
            return f"{num/1e3:.0f}K"
        else:
            return str(num)

    lr = config.train.learning_rate
    lr_str = f"{lr:.0e}" if lr < 1e-3 else f"{lr:.3f}"
    run_name = (
        f"{config.data.task}_"
        f"{human_readable_params(num_params)}_"
        f"{lr_str}"
    )

    return run_name