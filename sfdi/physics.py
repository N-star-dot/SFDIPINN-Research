"""
Forward models: (mu_a, mu_s', f) -> diffuse reflectance Rd.

Both written in the POST et al. (2023) convention for the boundary term,
A = (1 + Reff) / (1 - Reff). Cuccia's own paper uses (1 - Reff)/(2(1 + Reff)),
which is 1/(2A). Same model, different symbol. Do not mix them.

Shorthand used by both:
    mu_tr = mu_a + mu_s'            a' = mu_s' / mu_tr
    x     = sqrt(3 mu_a mu_tr + (2 pi f)^2) / mu_tr
    u     = (4A/3) x
    Cuccia 2009 (Eq. 10):  Rd = a' / [ (1 + x)(1 + u/2) ]
    PCBC  2023 (Eq. 10):   Rd = a' (1 - e^-u) / [ u (1 + x) ]

The slides' PCBC uses a "previously calibrated boundary parameter B". It is
not in the Post paper. Until you learn what B is, pass the lab's number as
A_value in config.yaml, or leave the Fresnel A (Post Eq. 8).
"""
import math
import numpy as np
import torch

PI = math.pi


def reff_poly(n):
    """Groenhuis polynomial for Reff (often paired with Cuccia, not in his paper)."""
    return 0.0636 * n + 0.668 + 0.710 / n - 1.440 / n ** 2


def A_poly(n):
    R = reff_poly(n)
    return (1 + R) / (1 - R)


def A_fresnel(n, n_out=1.0, steps=20000):
    """Post Eq. 8, A from Fresnel integrals. 2.515 at n = 1.33, 2.949 at 1.40."""
    t = (np.arange(steps) + 0.5) * (PI / 2) / steps
    dt = (PI / 2) / steps
    st = n * np.sin(t) / n_out
    rf = np.ones_like(t)
    ok = st < 1
    ci, ct = np.cos(t[ok]), np.sqrt(1 - st[ok] ** 2)
    rs = ((n * ci - n_out * ct) / (n * ci + n_out * ct)) ** 2
    rp = ((n * ct - n_out * ci) / (n * ct + n_out * ci)) ** 2
    rf[ok] = 0.5 * (rs + rp)
    num = np.sum(rf * np.cos(t) ** 2 * np.sin(t)) * dt
    den = np.sum(rf * np.cos(t) * np.sin(t)) * dt
    return (1 + 3 * num) / (1 - 2 * den)


def resolve_A(cfg_forward):
    """A from config: 'fresnel', 'poly', or an explicit value (e.g. the lab's B)."""
    src = cfg_forward.get("A_source", "fresnel")
    n = cfg_forward.get("n", 1.4)
    if src == "value":
        return float(cfg_forward["A_value"])
    return A_fresnel(n) if src == "fresnel" else A_poly(n)


def _prep(mua, musp, f):
    mua = torch.as_tensor(mua)
    musp = torch.as_tensor(musp, dtype=mua.dtype)
    f = torch.as_tensor(f, dtype=mua.dtype)
    mutr = mua + musp
    a = musp / mutr
    x = torch.sqrt(3 * mua * mutr + (2 * PI * f) ** 2) / mutr
    return a, x


def rd_cuccia(mua, musp, f, A):
    a, x = _prep(mua, musp, f)
    u = (4 * A / 3) * x
    return a / ((1 + x) * (1 + u / 2))


def rd_pcbc(mua, musp, f, A):
    a, x = _prep(mua, musp, f)
    u = (4 * A / 3) * x
    frac = torch.where(u > 1e-8, -torch.expm1(-u) / u.clamp_min(1e-8), 1 - u / 2)
    return a * frac / (1 + x)


MODELS = {"cuccia": rd_cuccia, "pcbc": rd_pcbc}


def forward(model, mua, musp, freqs, A):
    """Rd at several frequencies, stacked on the last axis."""
    if model == "mlp":
        from .forward_mlp import get_forward_mlp
        net = get_forward_mlp(device=mua.device)
        return net(mua, musp)  # Outputs shape (..., 2)
        
    fn = MODELS[model]
    return torch.stack([fn(mua, musp, f, A) for f in freqs], dim=-1)
