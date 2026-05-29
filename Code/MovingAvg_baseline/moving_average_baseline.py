import argparse
import json
import os
import numpy as np
import torch
from config import DATA_CFG, EVAL_CFG
from dataset import (
    DATA_PATH,
    INPUT_VARS,
    TARGET_VARS,
    WINDOW_SIZE,
    get_dataloaders,
)
from metrics import _compute_all_metrics

#pre-preparation to be make data compatible
def _to_jsonable(obj):
    """Convert numpy arrays inside nested dicts to be JSON safe formating in python types
    
    Arguments:
     obj: numpy array 

     Return:
        same structre, but with pure python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_jsonable(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

# update input window for next forcast, and forward slidning 
def _slide_window(x_cur, pred, window_size, input_vars, target_vars):
    """Slide the input window forward by one step after one prediction
    
    For each input variable, there is a target variable.  
    Drop oldest frame from window, append newst into the updated window. If no new frame exist, repeat the previous.
    
    Arguments:
            x_cur torch.tensor: current input tensor with shape (B, n_input_vars *  window_size, H, W)
            pred torch.tensor: The tensor predicted by model with shape (B, n_target_vars, H, W) 
            input_vars  list : ordered list of input variable names, ["q_lev0","q_lev1", "psi_lev0", "psi_lev1"]
            target_vars list : ordered list of target variable name, ["q_lev0","q_lev1", "psi_lev0", "psi_lev1"]
            window_size int: number of timesteps as input
    
    Return:
        troch.Tensor: Updated and shifted forward tensor by one step, with shape: (B, n_input_vars * window_size, H, W).
    """
    target_index = {name: i for i, name in enumerate(target_vars)}
    parts = []

    for iv, name in enumerate(input_vars):
        start = iv * window_size
        block = x_cur[:, start : start + window_size]
        if name in target_index:
            pred_ch = target_index[name]
            new_frame = pred[:, pred_ch : pred_ch + 1]
        else:
            new_frame = block[:, -1:] #Keep last as input if not updated

        parts.append(torch.cat([block[:, 1:], new_frame], dim=1))

    return torch.cat(parts, dim=1)


# create a simple moving average for prediction
def moving_average_rollout(test_loader, window_size, forecast_horizon, input_vars, target_vars, ma_window,):
    """ Run simple moving average rollout for full test set
    
    At each step, predict the next fra,e for each tartget variable as the mean of the input whole window. 
    The window is shifted forward for each new prediction.
    
    Arguments:
        test_loader dataloader: dataloader for test set
         window_size int : number of timesteps in window
        input_vars list: (ordered) list of input variables, possibilites: ["q_lev0", "q_lev1", "psi_lev0", "psi_lev1"]
        target_vars list: (ordered) list of traget variables,possibilities: ["q_lev0", "q_lev1", "psi_lev0", "psi_lev1"]
        forecast_horizon int: number of timesteps to be predicted
        ma_window int: number of current frames in input window, used for average.
    
    Returns: 
        preds_norm and predicted fields in z-score normalization unit, given in shape (N, n_target_vars, forecast_horizon, H, W)
        targets_norm ground truth (tragets) fields in z-score normalization unit, given in shape (N, n_target_vars, forecast_horizon, H, W) """
    
    all_preds = []
    all_targets = []
    n_target_vars = len(target_vars)

    if ma_window < 1 or ma_window > window_size:
        raise ValueError(f"ma_window must be between 1 and window_size={window_size}")

    for x, y in test_loader:
        x_cur = x.clone()
        step_preds = []
        step_targets = []

        for step in range(forecast_horizon):
            pred_vars = []

            for target_name in target_vars:
                if target_name not in input_vars:
                    raise ValueError(
                        f"Target variable {target_name!r} isnot a input variable"
                        "This moving average baseline needs target variables"  )
                
                iv = input_vars.index(target_name)
                start = iv * window_size
                block = x_cur[:, start : start + window_size]
                pred_var = block[:, -ma_window:].mean(dim=1, keepdim=True)
                pred_vars.append(pred_var)

            pred = torch.cat(pred_vars, dim=1)
            target = torch.cat(
                [y[:, v * forecast_horizon + step : v * forecast_horizon + step + 1]
                 for v in range(n_target_vars)], dim=1)
            
            step_preds.append(pred.numpy())
            step_targets.append(target.numpy())
            x_cur = _slide_window(x_cur, pred, window_size, input_vars, target_vars)

        all_preds.append(np.stack(step_preds, axis=2))
        all_targets.append(np.stack(step_targets, axis=2))

    preds_norm = np.concatenate(all_preds, axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)
    return preds_norm, targets_norm

#denormalization of variables
def denormalise_all_vars(arr_norm, norm_stats):
    """ Convert normalised array back to physical units.

    Converting back: by inverse z-score normalization variable wise using previous traning statics from dataset.py
     
 
    Arguments:
        arr_norm ndarray: normalized array, with shape (N, n_target_vars, forecast_horizon, H, W)
        norm_stats dict: normalisation statics containing: y_std y_mean, returned from load_and_split in dataset.py
        
    Returns:
        Denormalized ndaarry in orginal physical unit, with same shape as arr_norm"""
    

    y_mean = np.asarray(norm_stats["y_mean"]).reshape(-1)
    y_std = np.asarray(norm_stats["y_std"]).reshape(-1)

    arr_phys = np.empty_like(arr_norm)
    for v in range(arr_norm.shape[1]):
        arr_phys[:, v] = arr_norm[:, v] * y_std[v] + y_mean[v]

    return arr_phys
 
# create simple moving average model
def main():
    """ Run whole moving average baseline on test set
    load test data, run simple moving average, compute metrices, save results into the output folder.

     Argumnets:
        --out_dir str: Output folder for results. defult given by baseline_results/moving_average.

    Returns:
        Saves to outout folder
            metrics.json:  normalised and physical evaluation metrics and config
            predictions.npz:  predicted and target arrays in normalised and physical units"""
    
    
    parser = argparse.ArgumentParser(
        description="Simple moving average baseline for next 4 stepss."
    )
    parser.add_argument("--out_dir", type=str, default="baseline_results/moving_average")
    args = parser.parse_args()
    forecast_horizon = DATA_CFG["forecast_horizon"]
    batch_size = EVAL_CFG.get("batch_size", 16)
    ma_window = WINDOW_SIZE
    os.makedirs(args.out_dir, exist_ok=True)

    _, _, test_loader, _, _, norm_stats = get_dataloaders(
        data_path=DATA_PATH,
        input_vars=INPUT_VARS,
        target_vars=TARGET_VARS,
        window=WINDOW_SIZE,
        out_steps=forecast_horizon,
        batch_size=batch_size,)

    preds_norm, targets_norm = moving_average_rollout(
        test_loader=test_loader,
        window_size=WINDOW_SIZE,
        forecast_horizon=forecast_horizon,
        input_vars=list(INPUT_VARS),
        target_vars=list(TARGET_VARS),
        ma_window=ma_window,)

    metrics_norm = _compute_all_metrics(preds_norm, targets_norm, norm_stats, space="norm")
    metrics_phys = _compute_all_metrics(preds_norm, targets_norm, norm_stats, space="phys")

    preds_phys = denormalise_all_vars(preds_norm, norm_stats)
    targets_phys = denormalise_all_vars(targets_norm, norm_stats)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    predictions_path = os.path.join(args.out_dir, "predictions.npz")

#Save into JSON
    with open(metrics_path, "w") as f:
        json.dump(
            {   "baseline": "moving_average",
                "ma_window": ma_window,
                "forecast_horizon": forecast_horizon,
                "input_vars": list(INPUT_VARS),
                "target_vars": list(TARGET_VARS),
                "norm": _to_jsonable(metrics_norm),
                "phys": _to_jsonable(metrics_phys),},
            f,
            indent=2,)
#result into .npz
    np.savez_compressed(
        predictions_path,
        preds_norm=preds_norm,
        targets_norm=targets_norm,
        preds_phys=preds_phys,
        targets_phys=targets_phys,
        input_vars=np.array(list(INPUT_VARS)),
        target_vars=np.array(list(TARGET_VARS)),)

    print(f"Saved metric-> {metrics_path}")
    print(f"Saved predictions-> {predictions_path}")
    print(f"Prediction shapes-> {preds_norm.shape}")
    print(f"Overall RMSE norm-> {metrics_norm['overall']['rmse']}")
    print(f"Overall RMSE phys-> {metrics_phys['overall']['rmse']}")

if __name__ == "__main__":
    main()
