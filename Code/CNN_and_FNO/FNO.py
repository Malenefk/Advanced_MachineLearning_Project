"""
Fourier neural operator 2D for timestep predictions of ocean turbulence. 



Input (Batch, in_channels, Heught, Width) -> Temporal Positional Encoding -> Lifting MLP   ->  FNO Block:  (SpectralConv2d (Fourier branch)+ Conv1×1+(skip branch) Conv3×3 + (local branch)  GELU. )x Layers ->
                                                                                                                                                                                            |
                |                                                                                                                                                                           V
                V
Projection MLP  ->  (Batch, out_channels, H, W) + per-target last input frame (residual / delta prediction) -> Output (Batch, out_channels, Height, Width)


Channel layout defined in dataset.py, given by VARIABLE-MAJOR
    channels [variables * window_size : (variables+1) * window_size]  = variable v, oldest -> newest

Model's Interface:
--
    model = UNet(in_channels, out_channels, 
                window_size=4,
                 input_vars=["q_lev0", "q_lev1", psi_lev0, psi_lev1],
                 target_vars=["q_lev0", "q_lev1", psi_lev0, psi_lev1],
                 residual=True)
                 
    y_hat = model(x)   # x : (B, in_channels,  H, W)
                       # y : (B, out_channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from utils import TemporalPositionalEncoding, _build_residual_base


# Spectral Convolution Module: integral operator in Fourier space, keeps spatial resolution, capture long range dependencies and truncates high frequencies.

class SpectralConv2d(nn.Module):
    """2D Fourier integral operator:
    
    Apply learnable linear transformation to low freqiency fourier modes, Transform with 2DFFT to fourier space,
    truncates the FFT to the lowest n_modes_x * n_modes_y  frequencies, multiplies by learned weights matrix,
    and transforms back with inverse FFT. Spatial grid resolution is kept.
    
        Argument:
        in_channels int: number of input channels
        out_channels int: number of output channels
        n_modes_x int:Number of Fourier modes to keep along x. Hyperparameter given form config.py
        n_modes_y int: Number of Fourier modes to keep along y. Hyperparameter given form config.py
    
    """

    def __init__(self, in_channels, out_channels, n_modes_x, n_modes_y):
        super().__init__()
        
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes_x    = n_modes_x
        self.n_modes_y    = n_modes_y

        scale  = 1.0 / (in_channels * out_channels)
        shape  = (in_channels, out_channels, n_modes_x, n_modes_y)

        self.weights1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

#FFT -> truncate -> multiply with learned weights -> inverse FFT
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Application of spectral convolution into input tensor

        Transforms into FFT
        Trucation: multiplies low-frequencies modes with learned weight matrix.
        Converts back from FFT by iFFT
        
        Arguments
            x torch.Tensor: input tensor, with shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: output tensor, with shape (B, out_channels, H, W)    """
        
        B, C, H, W = x.shape

        x_ft   = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.n_modes_x,  :self.n_modes_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.n_modes_x, :self.n_modes_y], self.weights1)
        out_ft[:, :, -self.n_modes_x:, :self.n_modes_y] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.n_modes_x:, :self.n_modes_y], self.weights2)

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


# Single 2D FNO Module: 

class FNOBlock2d(nn.Module):
    """ single FNO block layer, with three componenets:
    Applies three operations and runs its sum into activation function.
    The tree componentss:
        spectral: SpectralConv2d, global fourier pathway
        skip: Conv 1×1, skip connection
        local: Conv 3×3, local spatial structure extractor
    
    Arguments
        channels int: number of channels
        n_modes_x int:Number of Fourier modes to keep along x. Hyperparameter given form config.py
        n_modes_y int: Number of Fourier modes to keep along y. Hyperparameter given form config.py
        """

    def __init__(self, channels, n_modes_x, n_modes_y):
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, n_modes_x, n_modes_y)
        self.skip     = nn.Conv2d(channels, channels, kernel_size=1)
        self.local    = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.skip(x) + self.local(x))


# FNO2d Model architecture 

class FNO2d(nn.Module):
    """ 2D Fourier Neural Operator with temporal positional encoding module and residual predictions.

        Standard FNO architecture( P -> FNOBlock2d -> FNOBlock2d -> FNOBlock2d -> Q)
        with additional custom Temporal positional encoding and a delta forecast

    Arguments:  
            int as input
            in_channels: input channels (n_input_vars * window_size)
            out_channels: number of output channels. iterative, hence: always 1 step at a time
            window_size: past timesteps per variable in the input channel stack.
            hidden_channels: width of a FNO block, defined as hyperparameter in config.py
            n_layers: number of FNO blocks, defined as hyperparameter in config.py
            n_modes_x : number of Fourier low-frequency modes in x direction kept, defined as hyperparameter in config.py
            n_modes_y: number of Fourier low-frequency modes in y direction kept, defined as hyperparameter in config.py
        --------
            residual boolean : if True, output = UNet_delta + last_frame_per_target_var.
            input_vars  list : ordered list of input variable names, ["q_lev0","q_lev1", "psi_lev0", "psi_lev1"]
            target_vars list : ordered list of target variable name, ["q_lev0","q_lev1", "psi_lev0", "psi_lev1"]
    """


#Model constructor
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int = 8,
        hidden_channels: int = 64,
        n_layers: int = 4,
        n_modes_x: int = 32,
        n_modes_y: int = 32,
        residual: bool = True,
        input_vars:  Optional[List[str]] = None,
        target_vars: Optional[List[str]] = None,
    ):
        super().__init__()

        if in_channels % window_size != 0:
            raise ValueError(
                f"in_channels ({in_channels}) must be divisible by the length of window_size ({window_size}). "
                f"in_channels = n_input_vars * window_size."
            )

        if residual and (input_vars is None or target_vars is None):
            raise ValueError(
                "If residual=True , both input_vars and target_vars must be provided."
                "Example : input_vars=['q_lev0','psi_lev0], then target_vars=['q_lev0','psi_lev0']"
            )

        self.window_size = window_size
        self.residual    = residual
        self.input_vars  = list(input_vars)  if input_vars  is not None else []
        self.target_vars = list(target_vars) if target_vars is not None else []

        self.temporal_pe = TemporalPositionalEncoding(in_channels)

# P operator: Lifting: in_channels -> hidden_channels
        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1),
        )

#FNo blocks
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(hidden_channels, n_modes_x, n_modes_y)
            for _ in range(n_layers)
        ])

# Q operator projection: hidden_channels -> out_channels
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 2, out_channels, kernel_size=1),
        )

#Models forward pass
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Forward pass, whole propagation path
        
        Standard FNO architecture( P -> FNOBlock2d -> FNOBlock2d -> FNOBlock2d -> Q)
        with additional custom Temporal positional encoding and a delta forecast modules.
        
        Arguments:
            x torch.tensor: input tensor with shape (B, in_channels, H, W), channel layout from dataset.py

        Returns:
            y torch.tensor: predicted output tensor with shape (B, in_channels, H, W)
        """
        if self.residual:
            base = _build_residual_base(
                x, self.window_size, self.input_vars, self.target_vars
            )

        x = self.temporal_pe(x)
        x = self.lifting(x)
        for block in self.fno_blocks: #Given the defined numver of layers in config.py
            x = block(x)
        delta = self.projection(x)   

        if self.residual:
            return delta + base

        return delta
