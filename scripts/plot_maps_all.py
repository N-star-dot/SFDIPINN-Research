import argparse, os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, load_lut, DIVERGING, diverging_norm, savefig
from sfdi.physics import resolve_A
from sfdi.pinn import PINN, predict
from sfdi.experiment import CellCache

cfg = load_config()
lut = load_lut(cfg)
cache = CellCache(cfg, lut)
test_cells, _ = cache.select(cfg["test_on"])
WL = cfg["wavelengths"]
A = resolve_A(cfg["forward"])

models = ["cuccia", "pcbc", "mlp"]
results = {}
for model in models:
    # Load model
    b = cfg["pinn"]["bounds"]
    net = PINN(cfg["pinn"]["widths"], cfg["pinn"]["activation"], (tuple(b["mua"]), tuple(b["musp"])))
    path = os.path.join(cfg["results_dir"], f"03_pinn_{model}.pt")
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    results[model] = (net, None)

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
    fig.suptitle(f"{cell.name}, {pname} relative error (ALL MODELS)")
    savefig(fig, cfg, f"03b_maps_{pname}_{cell.name}_all.png")
    print(f"Saved 03b_maps_{pname}_{cell.name}_all.png")
