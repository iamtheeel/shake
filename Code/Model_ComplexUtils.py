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
import complextorch as cplx_torch

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

class ComplexGroupNorm(nn.Module): # From ChatGPT
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
    
class ComplexHardtanh(nn.Module): # From ChatGPT
    def __init__(self, min_val=0.0, max_val=6.0):
        super().__init__()
        self.min_val, self.max_val = min_val, max_val
    def forward(self, x):
        xr, xi = x.real, x.imag
        return torch.complex(xr.clamp(self.min_val, self.max_val),
                             xi.clamp(self.min_val, self.max_val))
    

def model2Complex(layers, debug=False):
    for name, module in layers.named_children():
        if debug: print(f"Converting layer: {name} | Type: {type(module)}")
        if isinstance(module, nn.Conv2d):
            # nn.Conv2d --> set: dtype=torch.cfloat
            if debug: print(f" - Converting Conv2d to complex")
            module.to(dtype=torch.cfloat)  # Change to complex float is all that is nessisary 

            #model2Complex(module) # Recursive call
        ### Normalization layers
        elif isinstance(module, nn.GroupNorm):
            # nn.BatchNorm2d --> ComplexBatchNorm2d
            if debug: print(f" - Converting GroupNorm to ComplexGroupNorm")
            # Mirror GN config  --- From ChatGPT
            new = ComplexGroupNorm(
                num_groups=module.num_groups,
                num_channels=module.num_channels,
                eps=module.eps,
                affine=module.affine,
            )
            # Copy affine params to both real/imag norms --- From ChatGPT
            if module.affine:
                with torch.no_grad():
                    new.gn_r.weight.copy_(module.weight)
                    new.gn_r.bias.copy_(module.bias)
                    new.gn_i.weight.copy_(module.weight)
                    new.gn_i.bias.copy_(module.bias)

            layers._modules[name] = new                 # <- replace on the PARENT
            #model2Complex(module) # Recursive call will loop forever.
        elif isinstance(module, nn.BatchNorm2d):
            if debug: print(f" - **** Converting BatchNorm2d to ComplexBatchNorm2d   *****   NOT IMPLEMENTED YET")

        ### The activation functions
        elif isinstance(module, nn.ReLU):
            # nn.ReLU --> complextorch.CReLU
            if debug: print(f" - Converting ReLU to ComplexReLU")
            layers._modules[name] = cplx_torch.nn.CReLU(inplace=False)
            continue  # IMPORTANT: don't recurse into the replacement

        elif isinstance(module, (nn.ReLU, nn.ReLU6)):
            if debug: print(" - Replacing ReLU/ReLU6 -> CReLU(inplace=False)")
            layers._modules[name] = cplx_torch.nn.CReLU(inplace=False)
            continue  # IMPORTANT: don't recurse into the replacement

        # Optional safety, in case any model uses nn.Hardtanh directly:
        elif isinstance(module, nn.Hardtanh):
            if debug: print(" - Replacing Hardtanh -> ComplexHardtanh")
            layers._modules[name] =  ComplexHardtanh(min_val=module.min_val, max_val=module.max_val)
            continue

        else:
            model2Complex(module, debug=False) # Recursive call
            #model2Complex(module, debug=debug) # Recursive call