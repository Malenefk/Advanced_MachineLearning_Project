import torch
import torch.nn as nn
from typing import List
from config import MODEL_CFGS


#temporal positional encoding module
class TemporalPositionalEncoding(nn.Module):
    """
    Learnable scalar bias, used channel wise to encode temporal position orders. Make the model distinguish temporal position for each frame.
    Cost: little, just one learnable parameter per channel.
    
    
    Channel layout:
        channels [v * window_size: (v+1) * window_size] = variable v, inside each block: oldest -> newest

   Arguments:
    In channel: total number of input channels (n_input_vars * window_size) given in input shape (B, in_channels, H, W)

    Return:
    Same as input, with added bias. Shape: (B, in_channels, H, W)
    """

    def __init__(self, in_channels: int):
        super().__init__()
    
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.uniform_(self.bias, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


#residual (delta) predictor Module
def _build_residual_base(
    x: torch.Tensor,
    window_size: int,
    input_vars: List[str],
    target_vars: List[str],
) -> torch.Tensor:
    """
    Extract the newest frame from each target variable from the input tensor.

    Used in delta forcasting, to make model predict the change(delta), rather then a field from scratch.
    
    Arguments:
        x : input torch.tensor, with shape (B, n_input_vars * window_size, H, W)
        window_size: Number of past frames from x
        input_vars  : (ordered) list of input variable names
        target_vars : (ordered) list of target variable names

    Return:
        torch.Tensor:containing most recent input frame for each target variable. Tensor shape (B, n_target_vars, H, W)
    """
    frames = []
    for var in target_vars:
        if var not in input_vars:
            raise ValueError(
                f"Target variable '{var}' is not in  input_vars={input_vars}")
        
        iv  = input_vars.index(var)
        ch  = iv * window_size + (window_size - 1)   
        frames.append(x[:, ch : ch + 1])   #get last time step from input current window   
    return torch.cat(frames, dim=1)                  



 #window  slides function.
def slide_window(x_cur: torch.Tensor, new_pred: torch.Tensor,
                 window_size: int,
                 input_vars: List[str], target_vars: List[str]) -> torch.Tensor:
    """
    Slide a multi-variable window forwarded by one timestep.
    Drop oldest frame from window, append newst into the updated window. If no new frame exist, repeat the previous.
    
    channel layout:
        channels [v*window_size : (v+1)*window_size]  = variable v, oldest->newest

    Argument:
    torch.Tensor:
        x_cur: Current input tensor with shape (B, n_input_vars *  window_size, H, W)
        new_pred: The tensor predicted by model with shape (B, n_target_vars, H, W) 
    int:
        window_size: number of timesteps in window
    List:
        input_vars (ordered) list of input variables, possibilites: ["q_lev0", "q_lev1", "psi_lev0", "psi_lev1"]
        target_vars (ordered) list of traget variables,possibilities: ["q_lev0", "q_lev1", "psi_lev0", "psi_lev1"]

    Return:
        troch.Tensor: Updated and shifted forward tensor by one step, with shape: (B, n_input_vars * window_size, H, W).
    """
    target_ch = {v: i for i, v in enumerate(target_vars)} #create an dictionary
    parts = []

    for iv, var in enumerate(input_vars):
        s   = iv * window_size
        blk = x_cur[:, s : s + window_size]           
        if var in target_ch:
            ti = target_ch[var]
            new_frame = new_pred[:, ti:ti+1]   # (B, 1, H, W)
            parts.append(torch.cat([blk[:, 1:], new_frame], dim=1)) #update input window, remove old
        else:
            parts.append(torch.cat([blk[:, 1:], blk[:, -1:]], dim=1)) # repeat last frame

    return torch.cat(parts, dim=1)



#build mode based on hyperparamter and model name.
def create_model(model_name: str, in_ch: int, out_ch: int,
                 window_size: int,
                 input_vars: List[str], target_vars: List[str]):
    """
    Creating a model wit hyperparameters defined in config.py file.

    Residual (delta) predictioons and Temporan positonal encoding are fixed in both models.
    Unet have noe hyperparameters, as they are fixed to its vanilla values.

    
    Arguments:
        model_name str: ["unet", "fno"]
        in_ch int: n_input_vars  * window_size
        out_ch int: n_target_vars * 1 
        window_size int: past timesteps per variable
        input_vars list[str] : (ordered) list of input variable names
        target_varslist[str] : (ordered) list of target variable names

    Return:
        nn.Module: model ready for traning.
    """
    from Unet import UNet
    from FNO import FNO2d

    if model_name == "unet":
        return UNet(
            in_channels=in_ch,
            out_channels=out_ch,
            window_size=window_size,
            input_vars=input_vars,
            target_vars=target_vars,
            residual=True,)

    elif model_name == "fno":
        cfg = MODEL_CFGS.get("fno", {})
        return FNO2d(
            in_channels=in_ch,
            out_channels=out_ch,
            window_size=window_size,
            hidden_channels=cfg.get("hidden_channels", 64),
            n_layers=cfg.get("n_layers", 4),
            n_modes_x=cfg.get("n_modes_x", 16),
            n_modes_y=cfg.get("n_modes_y", 16),
            residual=True,
            input_vars=input_vars,
            target_vars=target_vars,)
    else:
        raise ValueError(f"Unknown model: '{model_name}'. Valid choices: unet, fno")


