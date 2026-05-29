DATA_CFG = dict(
    data_path   = "../t30.nc",
    input_vars  = ["q_lev0", "q_lev1", "psi_lev0", "psi_lev1"],
    target_vars = ["q_lev0", "q_lev1", "psi_lev0", "psi_lev1"],

    window_size      = 4,  # Number of past timesteps fed to the model per variable
    forecast_horizon = 4,  # rollout length of predictions

    ss_ratio         = 0.6, #Ratio of schedualed sampling during traning (Forced teahcing)
    ss_warmup_epochs = 5, #Number of epochs to warmup shedualed sampling.

#Sampling splits: 210 runs in train, 45 in val.
    train_end = 210, 
    val_end   = 255,
)

# train.py configurations
TRAIN_CFG = dict(
    default_model = "fno",
    default_loss  = "mse",
    batch_size    = 32,
    lr            = 1e-3,
    max_epochs    = 100,
    patience      = 10,
    grad_clip     = 1.0,
    device        = "auto",
    ckpt_dir      = "checkpoints",
    results_dir   = "results",
)

# Model configrations, Unet is kept fixed
MODEL_CFGS = dict( 
    unet=dict(),  #Unet used vanilla hyperparameters, with additonal temporal context and residual prediction changes fixed in the models constructor.
    fno  = dict(
        hidden_channels = 64,
        n_layers        = 4,
        n_modes_x       = 32,  # 64×64 grid supports up to 32 modes.
        n_modes_y       = 32,
    ),)


# loss function configurations. Losses without hyperparameters lambda, have an empty dict.
LOSS_PARAMS = dict(
    mse                 = dict(),
    weighted_mse        = dict(),
    mse_grad            = dict(lambda_grad=0.075),
    mse_mean_constraint = dict(lambda_phys=0.05),
    combined_physics    = dict(lambda_grad=0.075, lambda_phys=0.05, lambda_std=0.10),
)

# Evaluation configrations
EVAL_CFG = dict(
    batch_size = 32, #Number of samples per evaluation batch, same as traning batch.
    n_samples  = 5, #Number of samples used for plotting and evaluaton.
)


