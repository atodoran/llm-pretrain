def get_run_name_base (data_config, train_config, num_params):
    def human_readable_params(num):
        if num >= 1e9:
            return f"{num/1e9:.0f}B"
        elif num >= 1e6:
            return f"{num/1e6:.0f}M"
        elif num >= 1e3:
            return f"{num/1e3:.0f}K"
        else:
            return str(num)

    lr_str = f"{train_config.learning_rate:.0e}" if train_config.learning_rate < 1e-3 else f"{train_config.learning_rate:.3f}"
    run_name = (
        f"{data_config.task}_"
        f"{human_readable_params(num_params)}_"
        f"{lr_str}"
    )

    return run_name