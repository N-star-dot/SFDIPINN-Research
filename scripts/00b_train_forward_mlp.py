import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfdi.common import load_config, load_lut
from sfdi.forward_mlp import ForwardMLP

def train():
    cfg = load_config()
    lut = load_lut(cfg)
    
    # 1. Prepare training data from LUT
    M, S = np.meshgrid(lut.mua, lut.musp, indexing="ij")
    freqs = cfg["use_freqs"]
    R_lut = np.stack([lut.grid(f) for f in freqs], -1) # (n_mua, n_musp, 2)
    
    # Subsample to speed up training
    M = torch.tensor(M[::5, ::5].ravel(), dtype=torch.float32)
    S = torch.tensor(S[::5, ::5].ravel(), dtype=torch.float32)
    R = torch.tensor(R_lut[::5, ::5].reshape(-1, 2), dtype=torch.float32)
    
    # Remove any NaN or <=0 values just in case
    valid = (R > 0).all(dim=-1) & (M > 0) & (S > 0)
    M, S, R = M[valid], S[valid], R[valid]
    
    # Target is log(Rd)
    target = torch.log(R)
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = ((1e-4, 1.0), (1e-2, 20.0))
    net = ForwardMLP(widths=(128, 128, 128, 128), bounds=b).to(dev)
    
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1000)
    
    X_m = M.to(dev)
    X_s = S.to(dev)
    Y = target.to(dev)
    
    print(f"Training ForwardMLP on {len(X_m)} LUT points...")
    
    epochs = 3000
    for ep in range(1, epochs + 1):
        out_Rd = net(X_m, X_s)
        out_log = torch.log(out_Rd.clamp_min(1e-8))
        
        loss = nn.functional.mse_loss(out_log, Y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step(loss)
        
        if ep % 1000 == 0 or ep == epochs:
            print(f"Epoch {ep:4d}, Loss (Log MSE): {loss.item():.6g}")
            
    out_path = os.path.join(cfg["results_dir"], "forward_mlp.pt")
    torch.save(net.state_dict(), out_path)
    print(f"Saved ForwardMLP to {out_path}")

if __name__ == "__main__":
    train()
