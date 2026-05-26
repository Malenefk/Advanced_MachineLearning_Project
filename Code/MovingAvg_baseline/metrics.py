from typing import Dict
import numpy as np
import torch
from dataset import TARGET_VARS
from utils import slide_window


# denormalisation from z-score to its physical space for each variable
def _denorm_var(arr: np.ndarray, y_mean, y_std, var_idx: int) -> np.ndarray:
    """Inverse z-score normalization for one target variable.
    Converting back normalized array to physical units variable wise using previou traning statistics from dataset.py.
 
    Arguments:
        arr ndarray: normalized array
        y_mean: (ordered) array with values of mean, per variable for denromalizaton
        y_std: (ordered) array with values of standard deviation, per variable for denromalizaton
        var_ind: orderd index of target variables to denormalise
        
    Returns:
        Denormalized ndaarry in orginal physical unit, with same shape as input array
       """
    mean_v = float(np.asarray(y_mean).reshape(-1)[var_idx])
    std_v  = float(np.asarray(y_std ).reshape(-1)[var_idx])
    return arr * std_v + mean_v

#metrics compuation functions: RMSE, MAE and MASE
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """ Compute Root Mean Squared Error for two arrays

    Arguments:
        a np.ndarray: predicted values
        b np.ndarray: target values 

    Returns:
        value of RSME
    """
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    """ Compute mean absolute error for two arrays

    Argument:
        a np.ndarray: predicted values
        b np.ndarray: target values 

    Returns:
        value of MAE
    """
    return float(np.mean(np.abs(a - b)))

def _mase(pred: np.ndarray, target: np.ndarray) -> float:
    """ Compute mean absolute scaled error aginst a presistence baseline

    Scale MAE of model predictions the MAE of presistence forecast model, that prediction previous state.
    If fewer then two traget timesteps, the function return simple MAE. 

    Arguments:
        pred np.ndarray: predicted values, with shape (N, n_steps, ...)
        target np.ndarray: target values, with shape (N, n_steps, ...)

    Returns:
        value of MASE or nan if baseline error is too small (uncomputable)
    """
    if target.shape[1] < 2:
        return _mae(pred, target)
    naive_err = float(np.mean(np.abs(target[:, 1:] - target[:, :-1])))
    if naive_err < 1e-12:
        return float("nan")
    return _mae(pred, target) / naive_err



#rollout
@torch.no_grad()
def _rollout_all_vars(model, test_loader, device, window_size,
                      forecast_horizon) -> tuple:
    """ Free-run autoregressive rollout over the full test set.
    For each forecast step, predict one timestep forward using own appended prediction as most recent input inside the window.
    Window is shifted forward using slide_window after each prediction.
    
    Argumetns:
        model nn.Module: trained mode to be evalauted in eval mode
        test_loader dataLoader: test dataloader
        device str: use cuda, else cpu
        window_size int: Number of timesteps as input
        forecast_horizon int: number of timesteps to be predicted.


    Returns:
    pred ndarray: predicted field with shape (N, n_target_vars, forecast_horizon, H, W) in z-score normalised scale
    targets ndarray:  ground truth (tragets) fields with shape (N, n_target_vars, forecast_horizon, H, W) in z-score normalised scale
    """
    model.eval()
    all_preds, all_targets = [], []
    n_tv = len(TARGET_VARS)

#iterate over test batches
    for x, y in test_loader:
        x, y  = x.to(device), y.to(device)
        x_cur = x.clone()
        step_preds = []  
#iterate over forecast steps
        for _ in range(forecast_horizon):
            pred = model(x_cur)   
            step_preds.append(pred.cpu().numpy())
            x_cur = slide_window(x_cur, pred, window_size, list(TARGET_VARS), list(TARGET_VARS))


        preds_np = np.stack(step_preds, axis=2)  
        targs_np = np.stack(
            [y[:, v * forecast_horizon : (v + 1) * forecast_horizon].cpu().numpy()
             for v in range(n_tv)],
            axis=1,
        )  

        all_preds.append(preds_np)
        all_targets.append(targs_np)

    preds   = np.concatenate(all_preds,   axis=0) 
    targets = np.concatenate(all_targets, axis=0)
    return preds, targets



#metrics computaion functions
def _compute_all_metrics(preds_norm: np.ndarray,
                         targets_norm: np.ndarray,
                         norm_stats: dict,
                         space: str = "norm") -> Dict:
    """ Computation of full metric table in: normalised and physical space.

        Support metrices in physical and in normalized (z-score) scale/units. 
        Computes each matrices, for eah step and monitor amplitude collapse. 

    Arguments:
        preds_norm np.ndarray: predicted fields with z-score scale, in shape (N, n_target_vars, T, H, W).
        targets_norm np.ndarray: ground truth fields with z-score scale, in shape (N, n_target_vars, T, H, W).
        norm_stats dict: normalisation statistics dict with 'y_mean' and 'y_std. Given from load_and_split function in dataset.py.
        space str: evaluation scale/unit. 'norm' evaluates in z-score scale; 'phys' denormalises the values back 
        to physical unit.
        
        Return:
          A nested dicts in Json format:
           overall: metrices for {mse, mae, mase, accum_rmse},
           per_step: per step metrices {rmse, mae},
           per_var:  per variable metrics {"q_lev0": {rmse,...}, "q_lev1"{...},psi_lev0...},
           "amplitude_collapse": per variable std ratio {"q_lev0": {per_step_std_ratio: [...], mean_std_ratio:},...}"""

    N, n_tv, T, H, W = preds_norm.shape
    y_mean = norm_stats["y_mean"]
    y_std  = norm_stats["y_std"]
#Denormalise predictions and traget to physical scale if true
    if space == "phys":
        preds   = np.stack([_denorm_var(preds_norm[:, v],   y_mean, y_std, v) for v in range(n_tv)], axis=1)
        targets = np.stack([_denorm_var(targets_norm[:, v], y_mean, y_std, v) for v in range(n_tv)], axis=1)
    else:
        preds   = preds_norm
        targets = targets_norm


# overall common metrics for the model all variables and steps togethers
    overall_rmse = _rmse(preds, targets)
    overall_mae  = _mae(preds, targets)

# flatten to (N, n_tv*T, H*W) for MASE computation
    p_flat = preds.reshape(N, n_tv * T, -1)
    t_flat = targets.reshape(N, n_tv * T, -1)
    overall_mase = _mase(p_flat, t_flat)

# accumulated RMSE over the whole forcast horizion
    per_step_mse = [
        float(np.mean((preds[:, :, s] - targets[:, :, s]) ** 2))
        for s in range(T)
    ]
    accum_rmse = float(np.sqrt(np.mean(per_step_mse)))


# per step metrices for all variables
 
    per_step = {}
    for s in range(T):
        per_step[f"t+{s+1}"] = {
            "rmse": _rmse(preds[:, :, s], targets[:, :, s]),
            "mae":  _mae (preds[:, :, s], targets[:, :, s]),
        }


    # per step and per variable metrices    
    per_var = {}
    amplitude_collapse = {}
    
    for v, vname in enumerate(TARGET_VARS):
        p_v = preds  [:, v]    # (N, T, H, W)
        t_v = targets[:, v]

        var_rmse  = _rmse(p_v, t_v)
        step_rmse = [_rmse(p_v[:, s], t_v[:, s]) for s in range(T)]

# amplitude collapse: pred_std divided on target_std 
        std_ratio = []
        for s in range(T):
            p_std = float(np.std(p_v[:, s]))
            t_std = float(np.std(t_v[:, s]))
            std_ratio.append(p_std / (t_std + 1e-12))

        per_var[vname] = {
            "rmse":          var_rmse,
            "mae":           _mae(p_v, t_v),
            "per_step_rmse": step_rmse,
        }
        amplitude_collapse[vname] = {
            "per_step_std_ratio": std_ratio,
            "mean_std_ratio":     float(np.mean(std_ratio)),
        }

# return the full metric table as dict. 
    return {
        "overall": {
            "rmse":       overall_rmse,
            "mae":        overall_mae,
            "mase":       overall_mase,
            "accum_rmse": accum_rmse,
        },
        "per_step":           per_step,
        "per_var":            per_var,
        "amplitude_collapse": amplitude_collapse,}