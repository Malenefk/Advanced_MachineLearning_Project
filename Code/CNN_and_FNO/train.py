"""
train.py — training loop, iterative autoregressive rollout

Commands CLI for start:
    python train.py -- model_name [unet, fno] --loss function [mse, weighted_mse, loss mse_grad, mse_mean_constraint, combined_physics] --resume if continue


Outputs are written into: sweep_results/<model_name>_<loss_function>/...
--------------------------------------------------------------------------
What is saved:
    best_model.pt   — best validation checkpoint (follows: weights, norm_stats and it's config)
    last_model.pt   — last epoch checkpoint (supports --resume, for traning after interruption)
    history.csv     — epoch level traning loss, validation loss, learning rate, epoch time and
                      values for loss functions's lambda components.
    config.json     — all hyperparameters, number of parameters, best validation loss, training time, peak memory usage while training,
                      loss parameteres.

Scheduled sampling:
  Scheduled sampling (given by ss_ratio) defines percentages of how often model swaps groundruth with its own
  predictions back under tranin. The swaps happen per step, not by batch.
  * 0 = Usesonly ground truth frames.
  * 0.6= 60% of the time, model use own predictions.
  * 1 = uses only own predictions
  
Progressive ss ratio warmup:
  Makes the model start with ground truth early, and go towards the desired ss_ratio by X epochs given by ss_warmup_epochs

Gradient clipping:
  Mitigate lagre gradient spikes early in traning, gradient clipping at 1.0 used as deafult.

Possible loss functions:
    mse                 — Mean Squared Error (default)
    weighted_mse        — MSE with per channel weigthing 
    mse_grad            — MSE with spatial gradient penalty
    mse_mean_constraint — MSE with spatial mean conservation penalty
    combined_physics    — MSE with weigthing, gradient, mean constraint and standard deviation penalty
"""

import argparse
import json
import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import TRAIN_CFG, DATA_CFG, MODEL_CFGS, LOSS_PARAMS, get_device
from dataset import get_dataloaders, DATA_PATH, INPUT_VARS, TARGET_VARS, WINDOW_SIZE
from losses import LOSS_REGISTRY
from utils import slide_window, create_model

#Hyperparameters
BATCH_SIZE       = TRAIN_CFG["batch_size"]
LR               = TRAIN_CFG["lr"]
MAX_EPOCHS       = TRAIN_CFG["max_epochs"]
PATIENCE         = TRAIN_CFG["patience"]
GRAD_CLIP        = TRAIN_CFG.get("grad_clip", 1.0)
DEVICE           = get_device(TRAIN_CFG)
DEFAULT_MODEL    = TRAIN_CFG["default_model"]
DEFAULT_LOSS     = TRAIN_CFG["default_loss"]

FORECAST_HORIZON = DATA_CFG["forecast_horizon"]
SS_RATIO         = DATA_CFG["ss_ratio"]
SS_WARMUP_EPOCHS = DATA_CFG.get("ss_warmup_epochs", 0)


#Get the current ratio of scheduled sampling, based on the selected warmup
def get_current_ss_ratio(epoch: int) -> float:
    """ Compute scheduled sampling ratio for give traning epoch.
    Linearly warm up of schedueled sampling ratio. After warmup ss_ratio holds constant for rest of the traning.
    Arguments:
        epoch int: current traning epoch
    Return:
    float: schedueled sampling ratio for given epoch."""

    if SS_WARMUP_EPOCHS <= 0:
        return SS_RATIO
    return min(SS_RATIO, SS_RATIO * epoch / SS_WARMUP_EPOCHS)


# Schemeatic function for traning an epoch.

def train_one_epoch(model, loader, optimizer, device, loss_fn, loss_params,
                    window_size, forecast_horizon, ss_ratio):
    """
    Run full traning for one epoch with the autoregressive rollout
    Training Loop: Predicts one step, ss is added, computes loss, gradient clipping and back propagation

    Arguments:
        Model nn.module: Model used for traning one epoch.
        Loader : training dataloader 
        optimizer: pytorch Adam
        device: cuda if avaible otherwise cpu
        loss_fn: used loss function from losses.py
        loss_params: lambda hyperparamters for given loss function
        window_size: number of time steps in input from config.py 
        forecast_horizon: number for future steps to be predicted from config.py 
        ss_ratio: schedueld sampling from config.py 

    Output:
        avg_loss foalt : average loss per sample over all batches in the epoch.
        component_avgs dict: average values for each of the lamda loss components averaged across all batches"""
    
    model.train()
    total_loss       = 0.0
    component_totals = {}
    n_batches        = 0
    n_target_vars    = len(TARGET_VARS)

#batch traning loop, with weigths updated after each batch.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        x_cur          = x.clone()
        step_loss      = torch.tensor(0.0, device=device)
        batch_comps    = {}

#Time step training loop
        for step in range(forecast_horizon):
            pred = model(x_cur)   

            gt_frame = torch.cat(
                [y[:, v * forecast_horizon + step : v * forecast_horizon + step + 1]
                 for v in range(n_target_vars)], dim=1
            ) 

            step_weight   = 1.0 + step / max(1, forecast_horizon - 1)
            scalar, comps = loss_fn(pred, gt_frame, **loss_params)
            step_loss     = step_loss + step_weight * scalar

            
            for k, v in comps.items():
                batch_comps[k] = batch_comps.get(k, 0.0) + step_weight * v


            # Random use of scheduled sampling, based on its percerntages in current get_current_ss_ratio. 
            use_pred   = torch.rand(1).item() < ss_ratio
            next_frame = pred.detach() if use_pred else gt_frame
            x_cur      = slide_window(x_cur, next_frame, window_size, INPUT_VARS, TARGET_VARS)

        weight_sum = sum(1.0 + s / max(1, forecast_horizon - 1) for s in range(forecast_horizon))
        loss       = step_loss / weight_sum

        # Averaged components losses for the epoch.
        for k in batch_comps:
            batch_comps[k] /= weight_sum
            component_totals[k] = component_totals.get(k, 0.0) + batch_comps[k]
        n_batches += 1

        loss.backward()
        if GRAD_CLIP > 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    component_avgs = {k: v / n_batches for k, v in component_totals.items()}
    return total_loss / len(loader.dataset), component_avgs



# Validation loop
@torch.no_grad()
def validate(model, loader, device, loss_fn, loss_params, window_size, forecast_horizon):
    """Validation loop: Runs a free rollout, only validated on its own predictions.
    Evaluate the model using it own predictions only, computes loss at each step.

    Argumetns:
        model nn.Module: Model to be evalauted
        loader dataLoader: validation datalaoder.
        device str: use cuda, else cpu
        loss_fn: given loss function
        loss_params dict: hyperparamters for the loss function (lambda)
        window_size int: Number of timesteps as input
        forecast_horizon int: number of timesteps to be predicted.

    Return:
        float: average validation losses for the whole validation set samples."""
    model.eval()
    total_loss    = 0.0
    n_target_vars = len(TARGET_VARS)

#Iterate over each batch
    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        x_cur = x.clone()
        step_loss = torch.tensor(0.0, device=device)

#iterate over each step in forecast horizon
        for step in range(forecast_horizon):
            pred     = model(x_cur)
            gt_frame = torch.cat(
                [y[:, v * forecast_horizon + step : v * forecast_horizon + step + 1]
                 for v in range(n_target_vars)], dim=1
            )
            step_weight = 1.0 + step / max(1, forecast_horizon - 1)
            scalar, _   = loss_fn(pred, gt_frame, **loss_params)
            step_loss  += step_weight * scalar
            x_cur       = slide_window(x_cur, pred, window_size, INPUT_VARS, TARGET_VARS)

        weight_sum  = sum(1.0 + s / max(1, forecast_horizon - 1) for s in range(forecast_horizon))
        total_loss += (step_loss / weight_sum).item() * x.size(0)

    return total_loss / len(loader.dataset)



# #Main traning loop

def train(model_name: str = None, loss_name: str = None,
          run_dir: str = None, resume: bool = False):
    """
    Main full traning loop, whole pipeline
    Build dataloader, build mode, run traning loop, early stopping, save models and log.

    Argugments:
    model_name str : model to be trained
    loss_name str: Loss functions "mse" (deafult) , "weighted_mse", "loss mse_grad", "mse_mean_constraint" or "combined_physics"
    run_dir str: output directory for saving models and logs
    resume: if resume = True,  resume the traning from last_model.pt in given run_dir path

    Return:
    model nn.Module: best epoch model
    history dict: Logs from traing one epoch: train, val, lr and time
    train_seconds float : Total training time in seconds
    peak_mem_mb float: peak GPU memory usage in MB, or RSS for CPUs
    best_val float: best validation loss achieved
    """
    #Initialization and setup
    if model_name is None:
        model_name = DEFAULT_MODEL
    if loss_name is None:
        loss_name = DEFAULT_LOSS

    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Given loss function dont exist '{loss_name}' "
            f"Available loss functions: {sorted(LOSS_REGISTRY.keys())}"
        )
    loss_fn     = LOSS_REGISTRY[loss_name]
    loss_params = LOSS_PARAMS.get(loss_name, {})

    if run_dir is None:
        run_dir = os.path.join("sweep_results", f"{model_name}_{loss_name}")
    os.makedirs(run_dir, exist_ok=True)

    train_loader, val_loader, _, in_ch, _, norm_stats = get_dataloaders(
        data_path=DATA_PATH, input_vars=INPUT_VARS, target_vars=TARGET_VARS,
        window=WINDOW_SIZE, out_steps=FORECAST_HORIZON, batch_size=BATCH_SIZE,
    )

    model_out_ch = len(TARGET_VARS)

    model = create_model(
        model_name, in_ch, model_out_ch,
        window_size=WINDOW_SIZE,
        input_vars=list(INPUT_VARS),
        target_vars=list(TARGET_VARS),
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}  in iterative mode")
    print(f"  device          : {DEVICE}")
    print(f"  loss            : {loss_name}")
    print(f"  in_channels     : {in_ch}   model out_channels: {model_out_ch}")
    print(f"  window_size     : {WINDOW_SIZE}")
    print(f"  forecast_horizon: {FORECAST_HORIZON}")
    print(f"  ss_ratio target : {SS_RATIO} , warmed up over {SS_WARMUP_EPOCHS} epochs")
    print(f"  grad_clip       : {GRAD_CLIP}")
    print(f"  parameters      : {n_params:,}")
    print(f"  run_dir         : {run_dir}")
    print(f"{'='*60}\n")

    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Reset GPU memory stats, runs only on cuda GPU devices
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_epoch = 1
    best_val    = float("inf")
    no_improve  = 0
    history     = {"train": [], "val": [], "lr": [], "time": [], "components": []}
    train_start = time.time()

#Resume traning from last checkpoint.
    last_ckpt = os.path.join(run_dir, "last_model.pt")
    if resume and os.path.exists(last_ckpt):
        print(f"Resuming from checkpoint: {last_ckpt}")
        ckpt        = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        no_improve  = ckpt["no_improve"]
        history     = ckpt["history"]
        print(f"Resumed from epoch {ckpt['epoch']}, best_val={best_val:.6f}\n")
    elif resume:
        print(f"No checkpoint at '{last_ckpt}', starting from scratch.\n")

#Traning loop and validation for each epoch.
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0         = time.time()
        current_ss = get_current_ss_ratio(epoch)

        train_loss, train_comps = train_one_epoch(
            model, train_loader, optimizer, DEVICE, loss_fn, loss_params,
            window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON,
            ss_ratio=current_ss,
        )

        val_loss = validate(
            model, val_loader, DEVICE, loss_fn, loss_params,
            window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON,
        )
        elapsed = time.time() - t0

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["time"].append(elapsed)
        history["components"].append(train_comps)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d}/{MAX_EPOCHS}   "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"ss={current_ss:.3f}  [{elapsed:.1f}s]")

        ckpt_data = dict(
            model_name=model_name, loss_name=loss_name,
            model_state=model.state_dict(),
            in_channels=in_ch, out_channels=model_out_ch,
            norm_stats=norm_stats, window_size=WINDOW_SIZE,
            out_steps=1, input_vars=list(INPUT_VARS), target_vars=list(TARGET_VARS),
            forecast_horizon=FORECAST_HORIZON, ss_ratio=SS_RATIO,
        )
#Update best model checkpoint if best validation loss achived
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save({**ckpt_data, "epoch": epoch, "val_loss": best_val},
                       os.path.join(run_dir, "best_model.pt"))
            print(f"  *** Best model saved (val={best_val:.6f}) ***")
        else:
            no_improve += 1

        torch.save({**ckpt_data,
                    "epoch": epoch, "val_loss": val_loss,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val": best_val, "no_improve": no_improve,
                    "history": history},
                   last_ckpt)

#Early stopping
        if PATIENCE > 0 and no_improve >= PATIENCE:
            print(f"\nEarly stopping after {PATIENCE} epochs without improvement.")
            break

    train_seconds = time.time() - train_start

    if DEVICE == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else: #for CPU, get RSS
        try:
            import psutil
            peak_mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        except ImportError:
            peak_mem_mb = float("nan")

 # Reload best weights, in case of early stopping or last epoch not being the best.
    best_ckpt = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(best_ckpt["model_state"])

# Save history logs
    hist_df = pd.DataFrame({
        "epoch":      list(range(1, len(history["train"]) + 1)),
        "train_loss": history["train"],
        "val_loss":   history["val"],
        "lr":         history["lr"],
        "time_s":     history["time"],
    })
    if history["components"] and loss_params:
        #only for losses with lambda components
        comp_keys = sorted(history["components"][0].keys())
        for k in comp_keys:
            hist_df[f"comp_{k}"] = [ep.get(k, float("nan")) for ep in history["components"]]
    hist_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

# Save config into JSON
    cfg_out = dict(
        model=model_name, loss=loss_name,
        loss_params=loss_params,
        batch_size=BATCH_SIZE, lr=LR,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, grad_clip=GRAD_CLIP,
        window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON,
        ss_ratio=SS_RATIO, ss_warmup_epochs=SS_WARMUP_EPOCHS,
        input_vars=list(INPUT_VARS), target_vars=list(TARGET_VARS),
        in_channels=in_ch, out_channels=model_out_ch,
        n_params=n_params,
        best_val_loss=float(best_val),
        train_seconds=train_seconds,
        peak_mem_mb=peak_mem_mb,
    )
    if model_name in MODEL_CFGS and MODEL_CFGS[model_name]:
        cfg_out.update(MODEL_CFGS[model_name])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_out, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Training complete  —  {model_name.upper()} / {loss_name}")
    print(f"  best_val    = {best_val:.6f}")
    print(f"  train time  = {train_seconds/60:.1f} min")
    print(f"  peak memory = {peak_mem_mb:.0f} MB")
    print(f"  run_dir     = {run_dir}/")
    print(f"{'='*60}\n")
    return model, history, train_seconds, peak_mem_mb, best_val

# CLI commands
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one (model, loss) combination for ocean turbulence forecasting."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=["unet", "fno"],
        help=f"Model to train (default: '{DEFAULT_MODEL}' from TRAIN_CFG)",
    )
    parser.add_argument(
        "--loss", type=str, default=DEFAULT_LOSS, choices=sorted(LOSS_REGISTRY.keys()),
        help=f"Loss function (default: '{DEFAULT_LOSS}')",
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Output directory (default: sweep_results/model_name_loss_fn)",
    )
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last_model.pt saved in run_dir")
    args = parser.parse_args()
    train(model_name=args.model, loss_name=args.loss,
          run_dir=args.run_dir, resume=args.resume)
