"""
Slides 2 to 9: Forward Model Error Analysis.

(mu_a, mu_s') on the LUT grid -> Rd from the LUT, from Cuccia, from PCBC.
  fig 01a  LUT | Cuccia | Cuccia - LUT        (slides 2 and 3, log axes)
  fig 01b  PCBC - LUT beside Cuccia - LUT      (slide 4)
  fig 01c  100 (model - LUT) / LUT             (slide 5)
  fig 01d  same maps with one subject's pixels on top, and the table of
           pixel-wise mean absolute relative difference   (slides 6 to 9)
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, load_lut, DIVERGING, WL_COLORS, diverging_norm, savefig
from sfdi.physics import forward, resolve_A
from sfdi.data import Cell

torch.set_default_dtype(torch.float64)
cfg = load_config(); lut = load_lut(cfg); A = resolve_A(cfg["forward"])
F = cfg["use_freqs"]; WL = cfg["wavelengths"]
print(f"A (Post convention) = {A:.4f}   LUT: {lut.meta or cfg['lut_path']}")

M, S = np.meshgrid(lut.mua, lut.musp, indexing="ij")
R_lut = np.stack([lut.grid(f) for f in F], -1)                      # (n_mua, n_musp, 2)
R_mod = {m: forward(m, torch.tensor(M), torch.tensor(S), F, A).numpy() for m in ("cuccia", "pcbc")}
diff = {m: R_mod[m] - R_lut for m in R_mod}
rel = {m: 100 * diff[m] / R_lut for m in R_mod}


def panel(ax, Z, title, cmap, norm=None, vmin=None, vmax=None, cb=""):
    h = ax.pcolormesh(lut.mua, lut.musp, Z.T, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title, fontsize=10)
    ax.set_xlabel("μa (1/mm)"); ax.set_ylabel("μs′ (1/mm)")
    plt.colorbar(h, ax=ax, label=cb)


# ---- slides 2 and 3
fig, axs = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
for k, f in enumerate(F):
    panel(axs[k, 0], R_lut[..., k], f"LUT Rd, f = {f}", "Blues", vmin=0, vmax=1, cb="Rd")
    panel(axs[k, 1], R_mod["cuccia"][..., k], f"Cuccia Rd, f = {f}", "Blues", vmin=0, vmax=1, cb="Rd")
    panel(axs[k, 2], diff["cuccia"][..., k], f"Cuccia - LUT, f = {f}", DIVERGING, diverging_norm(0.03), cb="ΔRd")
fig.suptitle("Forward model error: Cuccia vs LUT (slides 2 and 3)")
savefig(fig, cfg, "01a_cuccia_vs_lut.png")

# ---- slide 4
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
for k, f in enumerate(F):
    panel(axs[k, 0], diff["pcbc"][..., k], f"PCBC - LUT, f = {f}", DIVERGING, diverging_norm(0.03), cb="ΔRd")
    panel(axs[k, 1], diff["cuccia"][..., k], f"Cuccia - LUT, f = {f}", DIVERGING, diverging_norm(0.03), cb="ΔRd")
fig.suptitle(f"PCBC vs Cuccia, both against the LUT (slide 4), A = {A:.3f}")
savefig(fig, cfg, "01b_pcbc_and_cuccia_minus_lut.png")

# ---- slide 5
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
for k, f in enumerate(F):
    for j, m in enumerate(("pcbc", "cuccia")):
        panel(axs[k, j], np.clip(rel[m][..., k], -50, 50), f"100 ({m} - LUT) / LUT, f = {f}",
              DIVERGING, diverging_norm(50), cb="%")
fig.suptitle("Relative difference (slide 5)")
savefig(fig, cfg, "01c_relative_difference.png")

# ---- slides 6 to 9: one subject's pixels on the maps, and the table
for subject, site, side in cfg["table_cells"]:
    cell = Cell(cfg, subject, site, side, lut)
    rows, pts = [], []
    for w, wl in enumerate(WL):
        _, ref, _ = cell.pixels(w)
        if ref is None or len(ref) == 0:
            continue
        Rl = lut.forward(ref[:, 0], ref[:, 1], F)
        row = {"wavelength": wl}
        for m in ("cuccia", "pcbc"):
            Rm = forward(m, torch.tensor(ref[:, 0]), torch.tensor(ref[:, 1]), F, A).numpy()
            r = np.abs(100 * (Rm - Rl) / Rl)
            for k, f in enumerate(F):
                ok = np.isfinite(r[:, k])
                row[f"{m} f={f}"] = r[ok, k].mean()
                row[f"_{m}_{k}"] = r[ok, k]
        rows.append(row); pts.append((wl, ref))
    allrow = {"wavelength": "all"}
    for c in rows[0]:
        if c.startswith("_"):
            m, k = c[1:].rsplit("_", 1)
            allrow[f"{m} f={F[int(k)]}"] = np.concatenate([r[c] for r in rows]).mean()
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows] + [allrow])
    cols = ["wavelength"] + [f"{m} f={f}" for m in ("cuccia", "pcbc") for f in F]
    table = table[cols]
    name = f"{subject}-{site}-{side}"
    table.to_csv(os.path.join(cfg["results_dir"], f"01d_table_{name}.csv"), index=False, float_format="%.3f")
    print(f"\n  Pixel-wise mean |relative difference| (%), {name}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for k, f in enumerate(F):
        for j, m in enumerate(("pcbc", "cuccia")):
            ax = axs[k, j]
            panel(ax, np.clip(rel[m][..., k], -50, 50), f"{m}, f = {f}", DIVERGING, diverging_norm(50), cb="%")
            for (wl, ref) in pts:
                sel = ref[np.random.default_rng(0).choice(len(ref), min(1500, len(ref)), replace=False)]
                ax.scatter(sel[:, 0], sel[:, 1], s=1, alpha=0.35, color=WL_COLORS[WL.index(wl)], label=f"{wl} nm")
    axs[0, 0].legend(markerscale=8, fontsize=7, loc="upper left")
    fig.suptitle(f"{name} pixels on the relative-difference maps (slides 6 to 9)")
    savefig(fig, cfg, f"01d_pixels_{name}.png")
