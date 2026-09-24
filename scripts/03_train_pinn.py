"""
Slide 17 to 25: train the self-supervised PINN on C1 (all four sites, all
wavelengths), predict C2 and C3, grade against the LUT per wavelength.

  python scripts/03_train_pinn.py                 # both forward models, full settings
  python scripts/03_train_pinn.py --quick         # 400 steps, smaller batch, to check it runs
"""
import argparse, os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, load_lut, DIVERGING, diverging_norm, savefig
from sfdi.physics import resolve_A
from sfdi.pinn import train, predict
from sfdi.experiment import CellCache, gather, grade, pooled

p = argparse.ArgumentParser()
p.add_argument("--models", default="cuccia,pcbc")
p.add_argument("--quick", action="store_true")
p.add_argument("--steps", type=int); p.add_argument("--batch", type=int)
a = p.parse_args()
cfg = load_config(); lut = load_lut(cfg); A = resolve_A(cfg["forward"])
pc = dict(cfg["pinn"])
if a.quick:
    pc.update(steps=400, batch_size=8192)
if a.steps: pc["steps"] = a.steps
if a.batch: pc["batch_size"] = a.batch
F, WL = cfg["use_freqs"], cfg["wavelengths"]

cache = CellCache(cfg, lut)
train_cells, train_w = cache.select(cfg["train_on"])
test_cells, _ = cache.select(cfg["test_on"])
all_w = list(range(len(WL)))
R_train = gather(train_cells, train_w)
print(f"training pixels: {len(R_train):,} from {[c.name for c in train_cells]}")

results = {}
for model in a.models.split(","):
    print(f"\n=== PINN with {model} forward model (A = {A:.3f}) ===")
    ev = lambda net: {"test_mua_%": pooled(grade(net, test_cells, all_w, cap=4000), "_ea")}
    net, hist = train(R_train, pc, model, A, F, evaluate=ev)
    torch.save(net.state_dict(), os.path.join(cfg["results_dir"], f"03_pinn_{model}.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(cfg["results_dir"], f"03_history_{model}.csv"), index=False)
    rows = grade(net, test_cells, all_w)
    per_wl = []
    for w, wl in enumerate(WL):
        rw = [r for r in rows if r["w"] == w]
        per_wl.append({"wavelength": wl, "mua_err_%": pooled(rw, "_ea"), "musp_err_%": pooled(rw, "_es")})
    results[model] = (net, pd.DataFrame(per_wl))
    print(results[model][1].to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
    results[model][1].to_csv(os.path.join(cfg["results_dir"], f"03_per_wavelength_{model}.csv"), index=False)

# ---- slide 17, right side
fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
colors = {"cuccia": "#2a78d6", "pcbc": "#eb6834"}
for model, (_, df) in results.items():
    for ax, col in zip(axs, ("mua_err_%", "musp_err_%")):
        ax.plot(df["wavelength"], df[col], "-o", color=colors.get(model), label=model)
        for x, y in zip(df["wavelength"], df[col]):
            ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
axs[0].set_ylabel("|μa PINN - μa LUT| / μa LUT (%)"); axs[1].set_ylabel("|μs′ PINN - μs′ LUT| / μs′ LUT (%)")
axs[1].set_xlabel("wavelength (nm)"); axs[0].legend()
fig.suptitle("Train on " + ", ".join(cfg["train_on"]["subjects"]) + ", predict " + ", ".join(cfg["test_on"]["subjects"]) + " (slide 17)")
savefig(fig, cfg, "03a_error_vs_wavelength.png")

# ---- slides 18 to 25: error maps for the first test cell
cell = test_cells[0]
for pi, pname in enumerate(("mua", "musp")):
    fig, axs = plt.subplots(len(results), len(WL), figsize=(2.1 * len(WL), 2.0 * len(results) + 0.6), squeeze=False)
    for i, (model, (net, _)) in enumerate(results.items()):
        for w, wl in enumerate(WL):
            R, ref, v = cell.pixels(w)
            img = np.full(v.shape, np.nan)
            if len(R):
                est = predict(net, R)[pi]
                img[v] = 100 * (est - ref[:, pi]) / ref[:, pi]
            h = axs[i, w].imshow(img, cmap=DIVERGING, norm=diverging_norm(25))
            axs[i, w].set_xticks([]); axs[i, w].set_yticks([])
            if i == 0: axs[i, w].set_title(f"{wl} nm", fontsize=9)
            if w == 0: axs[i, w].set_ylabel(model, fontsize=9)
    fig.colorbar(h, ax=axs, shrink=0.8, label=f"100 ({pname} PINN - LUT) / LUT (%)")
    fig.suptitle(f"{cell.name}, {pname} relative error (slides 18 to 25)")
    savefig(fig, cfg, f"03b_maps_{pname}_{cell.name}.png")
