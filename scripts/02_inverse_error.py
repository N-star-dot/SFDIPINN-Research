"""
Slides 10 to 16: Inverse Model Error Analysis.

Take each LUT grid point's Rd pair, then ask each formula which (mu_a, mu_s')
would produce that same Rd pair. Compare with the LUT's own answer.
  fig 02a  LUT inverse and Cuccia inverse maps, in the (Rd f=0, Rd f=0.1) plane (slide 10)
  fig 02b  relative error maps for PCBC and Cuccia inverses (slides 11 and 12)
  table    pixel-wise mean |relative error| for one subject (slides 13 to 16)
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, load_lut, DIVERGING, diverging_norm, savefig
from sfdi.physics import forward, resolve_A
from sfdi.data import Cell

torch.set_default_dtype(torch.float64)
cfg = load_config(); lut = load_lut(cfg); A = resolve_A(cfg["forward"])
F = cfg["use_freqs"]; WL = cfg["wavelengths"]


def invert(model, R, init, iters=60):
    """Solve model(mu_a, mu_s') = R for many pixels at once (damped Newton in log space)."""
    R = torch.as_tensor(R); th = torch.log(torch.as_tensor(init)).clone()
    tgt = torch.log(R)
    lo = torch.log(torch.tensor([1e-5, 1e-3])); hi = torch.log(torch.tensor([5.0, 50.0]))
    h = 1e-6
    for _ in range(iters):
        f0 = torch.log(forward(model, th[:, 0].exp(), th[:, 1].exp(), F, A)) - tgt
        J = torch.empty(len(th), 2, 2)
        for j in range(2):
            tp = th.clone(); tp[:, j] += h
            J[:, :, j] = (torch.log(forward(model, tp[:, 0].exp(), tp[:, 1].exp(), F, A)) - tgt - f0) / h
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        d0 = (J[:, 1, 1] * f0[:, 0] - J[:, 0, 1] * f0[:, 1]) / det
        d1 = (-J[:, 1, 0] * f0[:, 0] + J[:, 0, 0] * f0[:, 1]) / det
        step = torch.stack([d0, d1], -1).nan_to_num(0.0).clamp(-0.5, 0.5)
        th = torch.minimum(torch.maximum(th - step, lo), hi)
    res = (torch.log(forward(model, th[:, 0].exp(), th[:, 1].exp(), F, A)) - tgt).abs().max(-1).values
    out = th.exp().numpy()
    out[(res > 1e-5).numpy()] = np.nan                    # no solution: outside the model's reach
    return out


M, S = np.meshgrid(lut.mua, lut.musp, indexing="ij")
Rg = np.stack([lut.grid(f).ravel() for f in F], -1)
truth = np.stack([M.ravel(), S.ravel()], -1)
inv = {m: invert(m, Rg, truth) for m in ("cuccia", "pcbc")}
tri = mtri.Triangulation(Rg[:, 0], Rg[:, 1])
bx, by = lut.boundary(*F)


def plane(ax, vals, title, cmap, norm=None, vmin=None, vmax=None, cb=""):
    ok = np.isfinite(vals)
    mask = ~ok[tri.triangles].all(1)
    t = mtri.Triangulation(Rg[:, 0], Rg[:, 1], tri.triangles, mask=mask)
    h = ax.tripcolor(t, np.nan_to_num(vals), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="gouraud")
    ax.plot(bx, by, "k--", lw=0.8, label="LUT boundary")
    ax.set_xlabel(f"Rd, f = {F[0]}"); ax.set_ylabel(f"Rd, f = {F[1]}"); ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); plt.colorbar(h, ax=ax, label=cb)


# ---- slide 10
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
for j, (lab, col) in enumerate((("μa", 0), ("μs′", 1))):
    top = 0.4 if col == 0 else 10.0
    plane(axs[j, 0], truth[:, col], f"LUT {lab}", "Blues", vmin=0, vmax=top, cb=lab)
    plane(axs[j, 1], inv["cuccia"][:, col], f"Cuccia inverse {lab}", "Blues", vmin=0, vmax=top, cb=lab)
    lim = 0.01 if col == 0 else 0.4
    plane(axs[j, 2], inv["cuccia"][:, col] - truth[:, col], f"Cuccia - LUT, {lab}", DIVERGING, diverging_norm(lim), cb=lab)
fig.suptitle("Inverse model error: Cuccia vs LUT (slide 10)")
savefig(fig, cfg, "02a_inverse_cuccia.png")

# ---- slides 11 and 12
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
for j, (lab, col) in enumerate((("μa", 0), ("μs′", 1))):
    for i, m in enumerate(("pcbc", "cuccia")):
        r = 100 * (inv[m][:, col] - truth[:, col]) / truth[:, col]
        plane(axs[j, i], np.clip(r, -20, 20), f"{m}: 100 ({lab} - LUT) / LUT", DIVERGING, diverging_norm(20), cb="%")
fig.suptitle("Inverse model relative error (slides 11 and 12)")
savefig(fig, cfg, "02b_inverse_relative.png")

# ---- slides 13 to 16
for subject, site, side in cfg["table_cells"]:
    cell = Cell(cfg, subject, site, side, lut)
    rows, pool = [], {f"{m} {p}": [] for m in ("cuccia", "pcbc") for p in ("μa", "μs′")}
    for w, wl in enumerate(WL):
        R, ref, _ = cell.pixels(w)
        if ref is None or len(R) == 0:
            continue
        row = {"wavelength": wl}
        for m in ("cuccia", "pcbc"):
            est = invert(m, R, ref)
            for c, p in enumerate(("μa", "μs′")):
                r = np.abs(100 * (est[:, c] - ref[:, c]) / ref[:, c]); r = r[np.isfinite(r)]
                row[f"{m} {p}"] = r.mean(); pool[f"{m} {p}"].append(r)
        rows.append(row)
    rows.append({"wavelength": "all", **{k: np.concatenate(v).mean() for k, v in pool.items()}})
    table = pd.DataFrame(rows)
    name = f"{subject}-{site}-{side}"
    table.to_csv(os.path.join(cfg["results_dir"], f"02c_table_{name}.csv"), index=False, float_format="%.3f")
    print(f"\n  Pixel-wise mean |relative error| (%) of inverted μa, μs′ vs LUT, {name}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
