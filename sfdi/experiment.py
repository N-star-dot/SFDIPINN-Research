"""Shared pieces for training and ablation runs: pick cells and pixels, grade
predictions against the LUT (mean absolute relative error, %, as in the slides)."""
import numpy as np
from .data import Cell
from .pinn import predict


def spec_values(cfg, spec):
    subjects = spec.get("subjects", cfg["subjects"])
    sites = spec.get("sites", cfg["sites"])
    sides = spec.get("sides", cfg["sides"])
    wls = spec.get("wavelengths", "all")
    wls = cfg["wavelengths"] if wls == "all" else wls
    return subjects, sites, sides, [cfg["wavelengths"].index(w) for w in wls]


class CellCache:
    def __init__(self, cfg, lut):
        self.cfg, self.lut, self.cells = cfg, lut, {}

    def get(self, subject, site, side):
        k = (subject, site, side)
        if k not in self.cells:
            self.cells[k] = Cell(self.cfg, subject, site, side, self.lut)
        return self.cells[k]

    def select(self, spec):
        subjects, sites, sides, wi = spec_values(self.cfg, spec)
        return [self.get(a, b, c) for a in subjects for b in sites for c in sides], wi


def gather(cells, wl_idx):
    return np.concatenate([c.pixels(w)[0] for c in cells for w in wl_idx], 0)


def grade(net, cells, wl_idx, cap=None, seed=0):
    """Rows of (cell, wavelength index, mu_a error %, mu_s' error %, n pixels)."""
    rng = np.random.default_rng(seed); rows = []
    for c in cells:
        for w in wl_idx:
            R, ref, _ = c.pixels(w)
            if len(R) == 0:
                continue
            if cap and len(R) > cap:
                sel = rng.choice(len(R), cap, replace=False); R, ref = R[sel], ref[sel]
            a, s = predict(net, R)
            ea = np.abs(a - ref[:, 0]) / ref[:, 0] * 100
            es = np.abs(s - ref[:, 1]) / ref[:, 1] * 100
            rows.append({"cell": c.name, "subject": c.subject, "site": c.site, "side": c.side,
                         "w": w, "err_mua": ea.mean(), "err_musp": es.mean(), "n": len(R),
                         "_ea": ea, "_es": es})
    return rows


def pooled(rows, key):
    """Pixel-pooled mean of a per-pixel error array across rows."""
    arr = np.concatenate([r[key] for r in rows]) if rows else np.array([np.nan])
    return float(arr.mean())
