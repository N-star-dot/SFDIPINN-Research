"""
The self-supervised PINN from slide 17.

    [Rd f=0, Rd f=0.1] -> MLP -> (mu_a, mu_s') -> Cuccia or PCBC -> [Rd^ f=0, Rd^ f=0.1]
    loss = 1/2 * sum_f Huber_0.03( log(Rd^_f / Rd_f) )        (slide 17)
       or  mean_f ( log(Rd^_f / Rd_f) )^2                      ("Log MSE", ablation panel)

Settings copied from the ablation panel: widths 128,128,128, SiLU, AdamW,
lr 0.003, decay 0.9997 per step, batch 32768, seed 42, uniform sampling over
all selected pixels, about 10,000 steps, evaluate every 100.
The LUT never enters training. It is only used to grade the answers.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from .physics import forward

ACTS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class PINN(nn.Module):
    def __init__(self, widths=(128, 128, 128), activation="silu",
                 bounds=((5e-4, 0.5), (0.02, 10.0))):
        super().__init__()
        layers, d = [], 2
        for w in widths:
            layers += [nn.Linear(d, w), ACTS[activation]()]
            d = w
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
        lo = torch.tensor([math.log(bounds[0][0]), math.log(bounds[1][0])])
        hi = torch.tensor([math.log(bounds[0][1]), math.log(bounds[1][1])])
        self.register_buffer("lo", lo); self.register_buffer("hi", hi)

    def forward(self, R):
        """R: (batch, 2) measured Rd. Returns mu_a, mu_s' inside the bounds."""
        z = self.net(torch.log(R.clamp_min(1e-6)))            # log input
        logp = self.lo + (self.hi - self.lo) * torch.sigmoid(z)
        p = torch.exp(logp)
        return p[:, 0], p[:, 1]


def measurement_loss(R_pred, R_meas, kind="huber_log", delta=0.03):
    r = torch.log(R_pred.clamp_min(1e-8)) - torch.log(R_meas.clamp_min(1e-8))
    if kind == "huber_log":
        a = r.abs()
        h = torch.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))
        return 0.5 * h.sum(-1).mean()
    if kind == "log_mse":
        return (r * r).mean()
    raise ValueError(kind)


def train(R_train, cfg_pinn, model_name, A, freqs, evaluate=None, log=print):
    """R_train: (n_pix, 2) float array of measured Rd, all training pixels pooled."""
    torch.manual_seed(cfg_pinn["seed"]); np.random.seed(cfg_pinn["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = cfg_pinn["bounds"]
    net = PINN(cfg_pinn["widths"], cfg_pinn["activation"], (tuple(b["mua"]), tuple(b["musp"]))).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg_pinn["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg_pinn["lr_decay_per_step"])
    X = torch.as_tensor(R_train, dtype=torch.float32, device=dev)
    n = X.shape[0]; bs = min(cfg_pinn["batch_size"], n)
    hist = []
    for step in range(1, cfg_pinn["steps"] + 1):
        idx = torch.randint(0, n, (bs,), device=dev)          # uniform over all selected pixels
        Rm = X[idx]
        mua, musp = net(Rm)
        Rp = forward(model_name, mua, musp, freqs, A)
        loss = measurement_loss(Rp, Rm, cfg_pinn["loss"], cfg_pinn["huber_delta"])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % cfg_pinn["eval_every"] == 0 or step == cfg_pinn["steps"]:
            rec = {"step": step, "loss": loss.item()}
            if evaluate is not None:
                rec.update(evaluate(net))
            hist.append(rec)
            if step % (cfg_pinn["eval_every"] * 10) == 0 or step == cfg_pinn["steps"]:
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    return net, hist


@torch.no_grad()
def predict(net, R, chunk=1 << 18):
    dev = next(net.parameters()).device
    out_a, out_s = [], []
    for i in range(0, len(R), chunk):
        a, s = net(torch.as_tensor(R[i:i + chunk], dtype=torch.float32, device=dev))
        out_a.append(a.cpu().numpy()); out_s.append(s.cpu().numpy())
    return np.concatenate(out_a), np.concatenate(out_s)
