import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import TRAIN_CFG, EVAL_CFG, LOSS_PARAMS
from dataset import get_dataloaders, DATA_PATH
from metrics import _rollout_all_vars, _compute_all_metrics, _denorm_var
from utils import create_model
BATCH_SIZE    = EVAL_CFG["batch_size"]
N_SAMPLES     = EVAL_CFG["n_samples"]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def save_sample_figure(sample_idx: int, x_input: np.ndarray,
                       pred: np.ndarray, target: np.ndarray,
                       out_steps: int, window_size: int,
                       save_dir: str, eval_var: str, input_vars: list):
    """Save frames of qualitative: predictions, input amnd targets (3x4 layout).

    Plot: last inputframe, whole model prediction for given sample, and corresponding ground
    truth. Predictions and ground truth share a common colour scale. Save as a PNG into seperate folder.

    Arguments:
        sample_idx int: index of the samples to be plotted.
        x_input np.ndarray: input array given sample, given in shape (n_input_vars * window_size, H, W)
        pred np.ndarray: predicted fields in physical units, with shape (out_steps, H, W)
        target np.ndarray: ground truth fields in physical units, with shape (out_steps, H, W)
        out_steps int: number of forecast horizon to be plotted.
        window_size int: number of past timesteps per variable in input.
        save_dir str: path where the figure will be saved
        eval_var str: name of plotted variable
        input_vars list: ordered list variable names, to locate the correct input channel.
        
    Returns:
       Saves figure to save_dir/sample_XXX_variablename.png"""
    

    fig, axes = plt.subplots(3, out_steps, figsize=(3 * out_steps, 7))
    if out_steps == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        f"Sample {sample_idx}  —  {eval_var}  |  Input | Prediction | Ground Truth",
        fontsize=11,
    )

    iv            = input_vars.index(eval_var) if eval_var in input_vars else 0
    last_input_ch = iv * window_size + (window_size - 1)
    last_frame    = x_input[last_input_ch]

    for t in range(out_steps):
        p    = pred[t]
        y    = target[t]
        vmin = min(float(p.min()), float(y.min()))
        vmax = max(float(p.max()), float(y.max()))
        axes[0, t].imshow(last_frame, cmap="RdBu_r", vmin=float(last_frame.min()), vmax=float(last_frame.max()))
        axes[0, t].set_title("Input (t=0)")
        axes[0, t].axis("off")
        axes[1, t].imshow(p, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1, t].set_title(f"Pred t+{t + 1}")
        axes[1, t].axis("off")
        axes[2, t].imshow(y, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[2, t].set_title(f"Truth t+{t + 1}")
        axes[2, t].axis("off")

    plt.tight_layout()
    path = os.path.join(save_dir, f"sample_{sample_idx:03d}_{eval_var}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def save_rmse_figure(metrics_phys: dict,
                     forecast_horizon: int, target_vars: list, save_dir: str):
    """ Save Residual mean square error per step for every target variable in physical units
    
    Create seperate subplot per target variable with RMSE at each step t+1 through t+T. Saved as .png.

    Arguments:
        metrics_phys dict: physical scale metrics dict from _compute_all_metrics function
        forecast_horizon int: number of forecast steps shown on the
        target_vars list: (ordered) name list of target variable 
        save_dir str: folder path where the figure are saved

    Returns:
        saves figure to  save_dir/rmse_per_step.png"""
    
    steps = list(range(1, forecast_horizon + 1))
    fig, axes = plt.subplots(1, len(target_vars),
                             figsize=(4.5 * len(target_vars), 3.5),
                             sharey=False)
    if len(target_vars) == 1:
        axes = [axes]
    for ax, vname in zip(axes, target_vars):
        model_rmse = metrics_phys["per_var"][vname]["per_step_rmse"]

        ax.plot(steps, model_rmse, marker="o", lw=2, label="model")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("RMSE in physical unit")
        ax.set_title(f"Per step RMSE — {vname}")
        ax.set_xticks(steps)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "rmse_per_step.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def save_std_ratio_figure(metrics_norm: dict,
                          forecast_horizon: int, target_vars: list, save_dir: str):
    """save amplitude collapse plot: pred_std / target_std per step for all variables.
    Create seperate subplot per target variable with referance line for collapse at 0.5, and ideal on 1.0. Saved as .png.

    Args:
        metrics_norm dict: z-score normalised metrics dict from _compute_all_metrics function
        forecast_horizon int: number of forecast steps shown on the
        target_vars list: (ordered) name list of target variable 
        save_dir str: folder path where the figure are saved

    Returns:
        saves figure to save_dir/std_ratio.png."""
    
    steps = list(range(1, forecast_horizon + 1))
    fig, axes = plt.subplots(1, len(target_vars), figsize=(4.5 * len(target_vars), 3.5), sharey=True)
    if len(target_vars) == 1:
        axes = [axes]

    for ax, vname in zip(axes, target_vars):
        model_ratio = metrics_norm["amplitude_collapse"][vname]["per_step_std_ratio"]
        
        ax.plot(steps, model_ratio, marker="o", lw=2, label="model")
        ax.axhline(1.0, color="k",       lw=1.2, ls="--", label="ideal (1.0)")
        ax.axhline(0.5, color="tab:red", lw=0.8, ls=":",  label="collapse (0.5)")
        ax.set_xlabel("Forecast step")
        ax.set_ylabel("std ratio  (pred / target)")
        ax.set_title(f"Amplitude collapse — {vname}")
        ax.set_xticks(steps)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "std_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# main evaluation 
def evaluate(checkpoint_path: str):
    """Evaluate and save metrices for saved model.
    
    Load checkpoint, reconstruct model, run autoregressive rollout for the test set.
    Compute metrices, save results into metrics.json and save plots.
    
    Argument:
        checkpoint_path str: path to the model, here "best_model.pt" saved during traing, with trian.py
    
    Returns:
        Plots and metrics.json containing normalised and physical metrics saved into the checkpoints directiory"""
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at '{checkpoint_path}'")

    run_dir = os.path.dirname(checkpoint_path)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ckpt             = torch.load(checkpoint_path, map_location=DEVICE)
    in_ch            = ckpt["in_channels"]
    out_ch           = ckpt["out_channels"]
    norm_stats       = ckpt["norm_stats"]
    window_size      = ckpt["window_size"]
    input_vars       = ckpt["input_vars"]
    target_vars      = ckpt["target_vars"]
    forecast_horizon = ckpt["forecast_horizon"]
    model_name       = ckpt.get("model_name", "unknown")
    loss_name        = ckpt.get("loss_name",  "unknown")

    if ckpt.get("out_steps", 1) != 1:
        raise ValueError(f"Expected an iterative checkpoint, with one output step at a time")

    model = create_model(
        model_name, in_ch, out_ch, window_size,
        input_vars=input_vars, target_vars=target_vars,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f" Evaluating {model_name.upper()} / {loss_name}  iterative mode")
    print(f"  checkpoint epoch : {ckpt['epoch']}   val_loss: {ckpt['val_loss']:.6f}")
    print(f"  forecast_horizon : {forecast_horizon}")
    print(f"  input_vars       : {input_vars}")
    print(f"  target_vars      : {target_vars}")
    print(f"  run_dir          : {run_dir}")

    _, _, test_loader, _, _, _ = get_dataloaders(
        data_path=DATA_PATH,
        input_vars=input_vars,
        target_vars=target_vars,
        window=window_size,
        out_steps=forecast_horizon,
        batch_size=BATCH_SIZE)
    
    print(f"Test samples: {len(test_loader.dataset)}\n")

#rollout predictions for all test samples
    preds_norm, targets_norm = _rollout_all_vars(
        model, test_loader, DEVICE, window_size, forecast_horizon) 

#collect all input batches from test loader

    all_inputs = []
    for x, _ in test_loader:
        all_inputs.append(x.numpy())
    inputs_np = np.concatenate(all_inputs, axis=0)   

    # Compute all metrics
    metrics_norm = _compute_all_metrics(preds_norm, targets_norm, norm_stats, space="norm")
    metrics_phys = _compute_all_metrics(preds_norm, targets_norm, norm_stats, space="phys")

    print("\nOverall metrics (norm / phys)")
    for k in metrics_norm["overall"]:
        print(f"  {k}: {metrics_norm['overall'][k]:.6f} / {metrics_phys['overall'][k]:.6e}")
    print("\nPer step RMSE (norm / phys)")
    for s in range(forecast_horizon):
        key = f"t+{s+1}"
        nm  = metrics_norm["per_step"][key]["rmse"]
        pm  = metrics_phys["per_step"][key]["rmse"]
        print(f"  {key}  norm={nm:.4f}  phys={pm:.4e}")
    print("\nPer variable RMSE (phys) and mean std ratio")
    for vname in target_vars:
        mv = metrics_phys["per_var"][vname]["rmse"]
        sr = metrics_norm["amplitude_collapse"][vname]["mean_std_ratio"]
        print(f"  {vname}: rmse={mv:.4e}  mean_std_ratio={sr:.3f}")
    
    loss_component_history: dict = {}
    history_csv = os.path.join(run_dir, "history.csv")
    run_loss_params = LOSS_PARAMS.get(loss_name, {})
    if run_loss_params and os.path.isfile(history_csv):
        hist_df     = pd.read_csv(history_csv)
        comp_cols   = [c for c in hist_df.columns if c.startswith("comp_")]
        if comp_cols:
            for col in comp_cols:
                key = col[len("comp_"):]                
                loss_component_history[key] = hist_df[col].tolist()
            # Record the static lambda values used
            loss_component_history["lambda_params"] = run_loss_params

    metrics_payload = {"norm": metrics_norm, "phys": metrics_phys}
    if loss_component_history:
        metrics_payload["loss_component_history"] = loss_component_history

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics.json → {run_dir}/")

    save_rmse_figure(metrics_phys, forecast_horizon, target_vars, fig_dir)
    save_std_ratio_figure(metrics_norm, forecast_horizon, target_vars, fig_dir)
    y_mean = norm_stats["y_mean"]
    y_std  = norm_stats["y_std"]

    for i in range(min(N_SAMPLES, preds_norm.shape[0])):
        for v, vname in enumerate(target_vars):
            if vname not in input_vars:
                continue
            pred_phys   = _denorm_var(preds_norm  [i, v], y_mean, y_std, v)   
            target_phys = _denorm_var(targets_norm [i, v], y_mean, y_std, v)
            save_sample_figure(
                sample_idx=i, x_input=inputs_np[i],
                pred=pred_phys, target=target_phys,
                out_steps=forecast_horizon, window_size=window_size,
                save_dir=fig_dir, eval_var=vname,
                input_vars=input_vars, )
    print(f"Done. results saved to {run_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on the test set")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best_model.pt: sweep_results/fno_combined_physics/best_model.pt",)
    args = parser.parse_args()
    evaluate(checkpoint_path=args.checkpoint)
