import math
import os
import torch
import torch.nn as nn
from .common import load_config

ACTS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}

class ForwardMLP(nn.Module):
    def __init__(self, widths=(128, 128, 128, 128), activation="silu", 
                 bounds=((1e-4, 1.0), (1e-2, 20.0))):
        super().__init__()
        layers, d = [], 2
        for w in widths:
            layers += [nn.Linear(d, w), ACTS[activation]()]
            d = w
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
        
        # Bounds for input normalization (log space)
        lo = torch.tensor([math.log(bounds[0][0]), math.log(bounds[1][0])])
        hi = torch.tensor([math.log(bounds[0][1]), math.log(bounds[1][1])])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(self, mua, musp):
        """mua, musp -> log(Rd). Returns Rd."""
        x = torch.stack([mua, musp], dim=-1)
        x_log = torch.log(x.clamp_min(1e-8))
        
        # Normalize inputs to approximately [-1, 1] based on expected bounds
        x_norm = 2 * (x_log - self.lo) / (self.hi - self.lo) - 1
        
        z = self.net(x_norm)
        # Network outputs log(Rd). Exponentiate to get Rd.
        return torch.exp(z)

_MLP_INSTANCE = None

def get_forward_mlp(device="cpu"):
    global _MLP_INSTANCE
    if _MLP_INSTANCE is None:
        cfg = load_config()
        # Default bounds that comfortably cover the LUT
        b = ((1e-4, 1.0), (1e-2, 20.0)) 
        net = ForwardMLP(bounds=b)
        path = os.path.join(cfg["results_dir"], "forward_mlp.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Please run scripts/00b_train_forward_mlp.py first.")
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.to(device)
        net.eval()
        _MLP_INSTANCE = net
    
    # In case we switch devices
    _MLP_INSTANCE.to(device)
    return _MLP_INSTANCE
