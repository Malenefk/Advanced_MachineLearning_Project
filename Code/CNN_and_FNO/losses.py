import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


#Bulding blocks and components
def _spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute spatial gradients of predictions and targets.
    approximate gradients in both directions height and width. Penalize directions from predictions with different spatial structures
    than target's. 
    Arguments:
        pred torch.tensor: model predictions, shape (..., H, W)
        traget torch.tensor: tensor for tragets, shape (..., H, W)
    Return:
        torch.Tensor: tensor, sum of MSE for x and y gradients.
    """
    pred_dx  = pred[..., 1:, :]   - pred[..., :-1, :]
    targ_dx  = target[..., 1:, :] - target[..., :-1, :]
    pred_dy  = pred[..., :, 1:]   - pred[..., :, :-1]
    targ_dy  = target[..., :, 1:] - target[..., :, :-1]
    return F.mse_loss(pred_dx, targ_dx) + F.mse_loss(pred_dy, targ_dy)

def _spatial_mean_constraint(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes MSE for spatial means of predictions and targets, per batch element and channel.
        Porpuse: conservation of energy/variable magnitude over the field.
        
    Arguments:
        pred torch.tensor: model predictions, shape (B, C, H, W).
        traget torch.tensor: tensor for tragets, shape (B, C, H, W).
    
    Return:
        torch.Tensor: tensor representing the mean constraint loss"""

    pred_mean   = pred.flatten(2).mean(dim=-1)    # (B, C) spatial dimentions mean and flat 
    target_mean = target.flatten(2).mean(dim=-1)  
    return F.mse_loss(pred_mean, target_mean)

def _std_anti_collapse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE spatial standard deviation between predictions and target.
    Penalises amplitude collapse, were model predicts lower spatial variability than its target. Computed per batch element and channel.

    Arguments:
       pred torch.tensor: model predictions, shape (B, C, H, W).
       traget torch.tensor: tensor for tragets, shape (B, C, H, W).
    
    Return:
        torch.Tensor: tensor representing standard deviation anti-collapse penalty.
       
       """
    pred_std = pred.flatten(2).std(dim=-1)    # (B, C) spatial dimentions std and flats
    targ_std = target.flatten(2).std(dim=-1) 
    return F.mse_loss(pred_std, targ_std)

def _channel_weights(target: torch.Tensor, C: int) -> torch.Tensor:
    """Compute inverse-variance 1/(sigma^2) based per channel weights estimated from the current batch.
        Prevent high channels with high varaince from dominating (often will be the upper layers)

    Arguments:
       traget torch.tensor: tensor for tragets, shape (B, C, H, W).
       C int: Number of output channel
    
    Return:
        torch.Tensor: tensor with normalized channels weights of shape (C,) variance scaled to preserve mean weight magntiude.
       """
    var = target.flatten(2).var(dim=-1).mean(dim=0).clamp(min=1e-8) 
     # inverse variance weighting 
    w   = 1.0 / var 
    return (w / w.sum() * C).to(target.device)


#Loss functions
def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    **kwargs,   
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Mean squared error (MSE)
    
    Args:
        pred torch.Tensor: predictions made by the model, with shape (B, C, H, W).
        target torch.Tensor: ground truth targets, with shape (B, C, H, W).
        **kwargs:not used, its accepted for interface compatibility for severel losses.

    Returns:
            loss torch.Tensor: MSE loss.
            comps dict: component log 'mse'."""
    
    
    loss = F.mse_loss(pred, target)
    return loss, {"mse": loss.item()}

def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_weights: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted MSE, inverse variance computed seperatley per channel.
    
    Arguments:
        pred torch.Tensor: predictions made by the model, with shape (B, C, H, W).
        target torch.Tensor: ground truth targets, with shape (B, C, H, W).
        channel_weights torch.Tensor: computed weights with shape (C,).
        **kwargs:not used, its accepted for interface compatibility for severel losses.

    Returns:
            loss torch.Tensor: weighted MSE loss.
            comps dict: component log 'wmse'.""" 
    C = pred.shape[1]
    w = channel_weights.to(pred.device) if channel_weights is not None \
        else _channel_weights(target, C)

    per_ch_mse = torch.stack([F.mse_loss(pred[:, c], target[:, c]) for c in range(C)])
    loss       = (w * per_ch_mse).sum()

    comps = {"wmse": loss.item()}
    for c in range(C):
        comps[f"mse_ch{c}"] = per_ch_mse[c].item()
    return loss, comps

def mse_grad(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_grad: float = 0.1,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MSE with an additional spatial gradient penalty.
    Combination of pointwise MSE and gradient loss penalty.
    
    Arguments:
        pred torch.Tensor: predictions made by the model, with shape (B, C, H, W).
        target torch.Tensor: ground truth targets, with shape (B, C, H, W).
        lambda_grad float: hyperparamter adjusting the weigth. Given by lambda_grad defined in config.py
        **kwargs:not used, its accepted for interface compatibility for severel losses.

    Returns:
            loss torch.Tensor: MSE + lambda_grad * gradient_loss
            comps dict: component log 'mse', 'grad' and 'lambda_grad'"""
    l_mse  = F.mse_loss(pred, target) 
    l_grad = _spatial_gradient_loss(pred, target)
    loss   = l_mse + lambda_grad * l_grad
    return loss, {"mse": l_mse.item(), "grad": l_grad.item(),
                  "lambda_grad": lambda_grad}

def mse_mean_constraint(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_phys: float = 0.05,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MSE with an additional spatial mean conservation constraint penalty.
    Penetalizes for differences in spatial mean of predictions, when differeing from its tragets.
    
    Arguments:
        pred torch.Tensor: predictions made by the model, with shape (B, C, H, W).
        target torch.Tensor: ground truth targets, with shape (B, C, H, W).
        lambda_phys float: hyperparamter adjusting the mean constraint term. Given by lambda_phys defined in config.py
        **kwargs:not used, its accepted for interface compatibility for severel losses.

    Returns:
            loss torch.Tensor: MSE + lambda_phys * mean_constraint
            comps dict: component log 'mse', 'mean_constraint' and 'lambda_phys'"""
    l_mse  = F.mse_loss(pred, target)
    l_phys = _spatial_mean_constraint(pred, target)
    loss   = l_mse + lambda_phys * l_phys
    return loss, {"mse": l_mse.item(), "mean_constraint": l_phys.item(),
                  "lambda_phys": lambda_phys}


def combined_physics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_grad: float = 0.075,
    lambda_phys: float = 0.05,
    lambda_std:  float = 0.10,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined physics loss with four components.
    Uses the inverse-variance weighted MSE with spatial gradient penalty, spatial mean conservation constraint and standard deviation penalty against amplitude collapsing.
    
    
    Arguments:
        pred torch.Tensor: predictions made by the model, with shape (B, C, H, W).
        target torch.Tensor: ground truth targets, with shape (B, C, H, W).
        lambda_grad float: hyperparamter adjusting the weigth. Given by lambda_grad defined in config.py
        lambda_phys float: hyperparamter adjusting the mean constraint term. Given by lambda_phys defined in config.py
        lambda_std float: hyperparameter adjusteing the standard deviation (std) ratio, between the predictions and tragets to prevent mplitude collapse.
        **kwargs:not used, its accepted for interface compatibility for severel losses.

    Returns:
            loss torch.Tensor: combined loss
            comps dict: component log 'wmse', 'grad', 'mean_constraint' and 'std'"""
    C = pred.shape[1]
    w = _channel_weights(target, C)

    per_ch_mse = torch.stack([F.mse_loss(pred[:, c], target[:, c]) for c in range(C)])
    l_wmse  = (w * per_ch_mse).sum()
    l_grad  = _spatial_gradient_loss(pred, target)
    l_phys  = _spatial_mean_constraint(pred, target)
    l_std   = _std_anti_collapse(pred, target)

    loss = l_wmse + lambda_grad * l_grad + lambda_phys * l_phys + lambda_std * l_std
    return loss, {
        "wmse": l_wmse.item(), "grad": l_grad.item(),
        "mean_constraint": l_phys.item(), "std": l_std.item(),}


#List of possible loss functions
LOSS_REGISTRY = {
    "mse":mse, "weighted_mse": weighted_mse,"mse_grad": mse_grad, "mse_mean_constraint": mse_mean_constraint,"combined_physics":combined_physics,}