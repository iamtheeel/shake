###
# Footfal
# Joshua Mehlman
# MIC Lab
# Fall, 2025
###
# Models Complex Utils
# Tools to help with complex valued models
###
import torch #torch
from torch import nn

# Our own implementation of complex layers
# From ChatGPT
# Average Pooling Layer
class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor):
        return torch.complex(
            self.pool(x.real),
            self.pool(x.imag)
        )
    
# From ChatGPT
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, **bn_kwargs):
        super().__init__()
        # mirror BN params for real and imag
        self.bn_r = nn.BatchNorm2d(num_features, **bn_kwargs)
        self.bn_i = nn.BatchNorm2d(num_features, **bn_kwargs)

    def forward(self, x: torch.Tensor):
        # x: (N, C, H, W), complex64
        xr, xi = x.real, x.imag
        yr = self.bn_r(xr)
        yi = self.bn_i(xi)
        return torch.complex(yr, yi)

# From ChatGPT
class ComplexGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, **gn_kwargs):
        super().__init__()
        # mirror GN params for real and imag
        self.gn_r = nn.GroupNorm(num_groups, num_channels, **gn_kwargs)
        self.gn_i = nn.GroupNorm(num_groups, num_channels, **gn_kwargs)

    def forward(self, x: torch.Tensor):
        """
        x: (N, C, H, W), dtype=torch.complex64/complex128
        """
        xr, xi = x.real, x.imag
        yr = self.gn_r(xr)
        yi = self.gn_i(xi)
        return torch.complex(yr, yi)
    