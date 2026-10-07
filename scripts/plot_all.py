import argparse, os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, savefig

cfg = load_config()
models = ["cuccia", "pcbc", "mlp"]
results = {}
for model in models:
    df = pd.read_csv(os.path.join(cfg["results_dir"], f"03_per_wavelength_{model}.csv"))
    results[model] = (None, df)

# ---- slide 17, right side
fig, axs = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
colors = {"cuccia": "#2a78d6", "pcbc": "#eb6834", "mlp": "#2ca02c"}
for model, (_, df) in results.items():
    for ax, col in zip(axs, ("mua_err_%", "musp_err_%")):
        ax.plot(df["wavelength"], df[col], "-o", color=colors.get(model), label=model)
        for x, y in zip(df["wavelength"], df[col]):
            ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
axs[0].set_ylabel("|μa PINN - μa LUT| / μa LUT (%)")
axs[1].set_ylabel("|μs′ PINN - μs′ LUT| / μs′ LUT (%)")
axs[1].set_xlabel("wavelength (nm)")
axs[0].legend()
fig.suptitle("Train on C1, predict C2 and C3 (slide 17) - ALL MODELS")
savefig(fig, cfg, "03a_error_vs_wavelength_all.png")
print("Saved 03a_error_vs_wavelength_all.png")
