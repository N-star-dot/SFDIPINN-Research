"""Fake subjects in the same on-disk format as the lab data, so every script
runs end to end today. Delete data/synthetic once you have real files.

Optical properties follow typical forearm values (high mu_a in blue, low in
the near infrared), with smooth patches and a few "veins"."""
import os, sys
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, load_lut, HERE

cfg = load_config(); lut = load_lut(cfg)
H, W = 130, 174                         # real frames are 1040 x 1392
MUA0 = np.array([0.20, 0.13, 0.055, 0.022, 0.013, 0.012, 0.014, 0.015])
MUSP0 = np.array([2.00, 1.85, 1.65, 1.55, 1.35, 1.25, 1.15, 1.10])
yy, xx = np.mgrid[:H, :W]


def smooth(rng, s):
    f = gaussian_filter(rng.standard_normal((H, W)), s)
    return f / f.std()


def mask_for(site, rng):
    if site == "Arm":
        c = H / 2 + 6 * np.sin(xx / W * np.pi); half = 30 + 6 * xx / W
        return (np.abs(yy - c) < half) & (xx > 8) & (xx < W - 8)
    palm = (xx > 70) & (xx < 150) & (yy > 30) & (yy < 105)
    fingers = np.zeros_like(palm)
    for k in range(4):
        y0 = 32 + 18 * k
        fingers |= (xx > 18) & (xx <= 70) & (yy > y0) & (yy < y0 + 11)
    return palm | fingers


for si, subject in enumerate(cfg["subjects"]):
    os.makedirs(os.path.join(cfg["data_root"], subject), exist_ok=True)
    for sj, site in enumerate(cfg["sites"]):
        for sk, side in enumerate(cfg["sides"]):
            rng = np.random.default_rng(1000 * si + 10 * sj + sk)
            m = mask_for(site, rng)
            patch_a, patch_s = smooth(rng, 12), smooth(rng, 18)
            veins = np.zeros((H, W))
            for _ in range(3):
                y0, slope = rng.uniform(20, H - 20), rng.uniform(-0.3, 0.3)
                veins += np.exp(-((yy - (y0 + slope * (xx - W / 2))) / 2.5) ** 2)
            subj_scale = 1.0 + 0.15 * si                       # subjects differ a bit
            mua = MUA0[:, None, None] * subj_scale * np.exp(0.25 * patch_a) * (1 + 0.8 * veins)
            musp = MUSP0[:, None, None] * np.exp(0.08 * patch_s)
            R = lut.forward(mua, musp, cfg["use_freqs"])      # (n_wl, H, W, 2)
            R = R * (1 + 0.01 * rng.standard_normal(R.shape))  # 1% camera noise
            stack = np.zeros((len(cfg["stack_freqs"]), len(MUA0), H, W), np.float32)
            for k, f in enumerate(cfg["use_freqs"]):
                stack[cfg["stack_freqs"].index(f)] = R[..., k]
            fmt = dict(subject=subject, site=site, side=side)
            tifffile.imwrite(os.path.join(cfg["data_root"], cfg["file_pattern"].format(**fmt)),
                             stack, imagej=True, metadata={"axes": "ZCYX"})
            np.save(os.path.join(cfg["data_root"], cfg["mask_pattern"].format(**fmt)), m)
            ra, rs = lut.inverse(R[..., 0], R[..., 1], *cfg["use_freqs"])  # what the lab "supplies"
            np.savez_compressed(os.path.join(cfg["data_root"], cfg["ref_pattern"].format(**fmt)),
                                mua=ra.astype(np.float32), musp=rs.astype(np.float32))
            np.savez_compressed(os.path.join(cfg["data_root"], subject, f"truth-{site}-{side}.npz"),
                                mua=mua.astype(np.float32), musp=musp.astype(np.float32))
    print("wrote", subject)
print("synthetic data in", os.path.relpath(cfg["data_root"], HERE))
