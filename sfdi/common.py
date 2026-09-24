import os
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# blue = model below LUT, orange = model above LUT, as in the slides
DIVERGING = LinearSegmentedColormap.from_list("bo", ["#2f5fa7", "#9bb5da", "#f7f7f7", "#e8b184", "#c8742f"])
WL_COLORS = ["#1f77b4", "#17becf", "#2ca02c", "#bcbd22", "#ff7f0e", "#d62728", "#e377c2", "#9467bd"]


def load_config(path=None):
    path = path or os.path.join(HERE, "config.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in ("data_root", "lut_path", "results_dir"):
        if k in cfg and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(HERE, cfg[k])
    os.makedirs(cfg["results_dir"], exist_ok=True)
    return cfg


def load_lut(cfg):
    from .lut import LUT
    if cfg.get("lut_format", "npz") == "mat":
        return LUT.from_mat(cfg["lut_path"], cfg["lut_mat_keys"])
    return LUT.from_npz(cfg["lut_path"])


def diverging_norm(lim):
    return TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)


def savefig(fig, cfg, name):
    p = os.path.join(cfg["results_dir"], name)
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  wrote", os.path.relpath(p, HERE))
