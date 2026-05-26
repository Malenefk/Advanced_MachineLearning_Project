import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from utils import TemporalPositionalEncoding, _build_residual_base
 
class DoubleConv(nn.Module):
    """ Double Convolution block
    Used to extract features from channels
    Includes Batch normalization and a Relu activation funciton

    Arguments:
       in_channels int:number of input channels
       out_channels int: number of output channels
       mid_channels int: intermediate channel, between the double convolutions"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Encoder module block
    Reduces spatial dimention by half with maxpool convolution, increase number of channels by 2.

    Arguments:
        in_channels int: number of input channels
        out_channels int: number of output channels"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """ Decoder module block
    Decodes: increase spatial resolution by 2 with transposed convolution, while reducing number of channels by half.
    Performs a skip connection to its corresponding encoder block
    
    Arguments:
        in_channels int: number of input channels
        out_channels int: number of output channels"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Upsample x1 align spatial dimentions with x2, and combine channels with the skip connection x2
    
    Arguments:
        x1 torch.Tensor: input from the previous decoder step, shape (B, in_channels, H, W)
        x2 torch.Tensor: skip connection with corresponding encoder block, shape (B, in_channels//2, H*2, W*2)
        
    Returns:
        torch.Tensor: output tensor, with shape (B, out_channels, H*2, W*2)"""
        x1    = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1    = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

#final output convolution module: to get the desired number of output channels.
class OutConv(nn.Module):
    """ Final 1x1 convolution, mapping to the last feature channel to the number given by output channels
    
     Arguments:
        in_channels int: number of input hidden channels from previous decoder
        out_channels int: number of input channels = n_traget_vars"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# main model architecture
class UNet(nn.Module):
    """U-Net with Temporal Positional Encoding, Delta Forecasting modules.
    Follow standard unet architecture with additional Temporal Positional Encoding, Delta Forecasting modules.:
    
    1. Temporal Positional Encoding        
    2. Encoder: Double conv -> Down x4
    3. Decoder: Up x4 with skip connection -> OutConv
    4. Delta Forecasting modules
            
            
    Arguments:
        int as input
            in_channels: input channels (n_input_vars * window_size)
            out_channels: number of output channels. iterative, hence: always 1 step at a time
            window_size: past timesteps per variable in the input channel stack.
        -------
            residual boolean : if True, output = UNet_delta + last_frame_per_target_var.
            input_vars  list : ordered list of input variable names, ["q_lev0","q_lev1", "psi_lev0", "psi_lev1"]
            target_vars list : ordered list of target variable name, ["q_lev0","q_lev1", "psi_lev0", "psi_lev1"]"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int,
        input_vars:  Optional[List[str]] = None,
        target_vars: Optional[List[str]] = None,
        residual:    bool = True,
    ):
        super().__init__()
        self.window_size = window_size
        self.residual    = residual and (input_vars is not None) and (target_vars is not None)
        self.input_vars  = list(input_vars)  if input_vars  is not None else []
        self.target_vars = list(target_vars) if target_vars is not None else []

        self.temporal_pe = TemporalPositionalEncoding(in_channels)

    # Encoder
        self.inc   = DoubleConv(in_channels, 64, mid_channels=32)
        self.down1 = Down(64,   128)
        self.down2 = Down(128,  256)
        self.down3 = Down(256,  512)
        self.down4 = Down(512, 1024)

    # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512,  256)
        self.up3 = Up(256,  128)
        self.up4 = Up(128,   64)

        self.outc = OutConv(64, out_channels)

#models forward pass
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if self.residual:
            base = _build_residual_base(
                x, self.window_size, self.input_vars, self.target_vars
            ) 

        x = self.temporal_pe(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_t = self.up1(x5, x4)
        x_t = self.up2(x_t, x3)
        x_t = self.up3(x_t, x2)
        x_t = self.up4(x_t, x1)

        delta = self.outc(x_t)  

        if self.residual:            
            return delta + base

        return delta
