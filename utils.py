def get_run_name_base (config, num_params):
    def params_str(num):
        if num >= 1e9:
            return f"{num/1e9:.0f}B"
        elif num >= 1e6:
            return f"{num/1e6:.0f}M"
        elif num >= 1e3:
            return f"{num/1e3:.0f}K"
        else:
            return str(num)

    lr_str = f"{config.train.learning_rate:.0e}"
    b2_str = f"{config.train.beta2}"
    wd_str = f"{config.train.weight_decay:.0e}"
    bs_str = f"{config.train.batch_size}"
    run_name = (
        f"{config.task.name}_"
        f"{params_str(num_params)}_"
        f"{lr_str}_"
        f"{b2_str}_"
        f"{wd_str}_"
        f"{bs_str}"
    )

    return run_name