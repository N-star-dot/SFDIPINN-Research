"""
Slides 26 to 38: ablation runs. Each entry under `ablations:` in config.yaml
changes the training set (wavelengths, sites, subject) or the forward model.
Every run is graded on every cell, like the dashboard table.

  python scripts/04_ablation.py --quick            # check it runs
  python scripts/04_ablation.py --only wl_811_851   # one run
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, load_lut, savefig
from sfdi.physics import resolve_A
from sfdi.pinn import train
from sfdi.experiment import CellCache, gather, grade, pooled

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true"); p.add_argument("--only")
p.add_argument("--steps", type=int); p.add_argument("--batch", type=int)
a = p.parse_args()
cfg = load_config(); lut = load_lut(cfg); A = resolve_A(cfg["forward"])
pc = dict(cfg["pinn"])
if a.quick: pc.update(steps=300, batch_size=8192)
if a.steps: pc["steps"] = a.steps
if a.batch: pc["batch_size"] = a.batch
WL = cfg["wavelengths"]; all_w = list(range(len(WL)))
cache = CellCache(cfg, lut)
every = cache.select({})[0]
summary = []

for abl in cfg["ablations"]:
    if a.only and abl["name"] != a.only:
        continue
    spec = {**cfg["train_on"], **abl.get("train_on", {})}
    model = abl.get("model", "pcbc")
    tcells, tw = cache.select(spec)
    train_names = {c.name for c in tcells}
    print(f"\n=== {abl['name']}: {model}, train on {sorted(train_names)}, wavelengths {[WL[i] for i in tw]} ===")
    net, _ = train(gather(tcells, tw), pc, model, A, cfg["use_freqs"])
    rows = grade(net, every, all_w)
    cols = [c.name + ("*" if c.name in train_names else "") for c in every]
    tab = {}
    for key, lab in (("err_mua", "mua"), ("err_musp", "musp")):
        M = np.full((len(WL), len(every)), np.nan)
        for r in rows:
            M[r["w"], [c.name for c in every].index(r["cell"])] = r[key]
        tab[lab] = pd.DataFrame(M, index=[f"{lab} {w}" for w in WL], columns=cols)
    out = pd.concat([tab["mua"], tab["musp"]])
    held = [r for r in rows if r["cell"] not in train_names]
    out.loc["held-out mean mua"] = pooled(held, "_ea"); out.loc["held-out mean musp"] = pooled(held, "_es")
    out.to_csv(os.path.join(cfg["results_dir"], f"04_{abl['name']}.csv"), float_format="%.2f")
    summary.append({"run": abl["name"], "model": model, "train": ",".join(sorted(train_names)),
                    "wavelengths": ",".join(str(WL[i]) for i in tw),
                    "held_out_mua_%": pooled(held, "_ea"), "held_out_musp_%": pooled(held, "_es")})
    print(f"  held-out mean error: mua {summary[-1]['held_out_mua_%']:.2f}%, musp {summary[-1]['held_out_musp_%']:.2f}%")

    fig, axs = plt.subplots(2, 1, figsize=(1.0 * len(every) + 2, 9))
    for ax, lab in zip(axs, ("mua", "musp")):
        D = tab[lab].values
        ax.imshow(np.clip(D, 0, 20), cmap="Oranges", vmin=0, vmax=20, aspect="auto")
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if np.isfinite(D[i, j]):
                    ax.text(j, i, f"{D[i, j]:.2g}", ha="center", va="center", fontsize=7,
                            color="white" if D[i, j] > 12 else "black")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, fontsize=7)
        ax.set_yticks(range(len(WL))); ax.set_yticklabels([f"{lab} {w}" for w in WL], fontsize=8)
    fig.suptitle(f"{abl['name']} ({model}), * = training cell, % error vs LUT")
    savefig(fig, cfg, f"04_{abl['name']}.png")

if summary:
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(cfg["results_dir"], "04_summary.csv"), index=False, float_format="%.2f")
    print("\n" + s.to_string(index=False, float_format=lambda v: f"{v:6.2f}"))
