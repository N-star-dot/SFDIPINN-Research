"""
Run this first whenever you change config.yaml (especially after switching to
the lab's real files). It checks every path and shape and tells you what is
wrong before a long training run does.

    python scripts/check_setup.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tifffile
from sfdi.common import load_config, load_lut, HERE
from sfdi.physics import resolve_A, A_fresnel, A_poly
from sfdi.data import Cell, hyperstack_labels

problems = []
def ok(msg): print("  [ok]  ", msg)
def bad(msg): print("  [FIX] ", msg); problems.append(msg)

cfg = load_config()
print("\nLUT")
try:
    lut = load_lut(cfg)
    ok(f"{os.path.relpath(cfg['lut_path'], HERE)}  meta={lut.meta}")
    ok(f"mu_a {lut.mua.min():.4g} to {lut.mua.max():.4g} ({len(lut.mua)} pts), "
       f"mu_s' {lut.musp.min():.4g} to {lut.musp.max():.4g} ({len(lut.musp)} pts), freqs {list(lut.freqs)}")
    for f in cfg["use_freqs"]:
        try: lut.fidx(f)
        except AssertionError: bad(f"LUT has no frequency {f}. use_freqs must be in the LUT")
    if not (np.nanmin(lut.Rd) >= 0 and np.nanmax(lut.Rd) <= 1.0001):
        bad(f"LUT Rd outside [0, 1]: {np.nanmin(lut.Rd):.3g} to {np.nanmax(lut.Rd):.3g}. Check Rd_axes and units")
    else:
        ok("LUT Rd values inside [0, 1]")
except Exception as e:
    bad(f"cannot load LUT: {e}"); lut = None

print("\nForward model")
A = resolve_A(cfg["forward"])
ok(f"n = {cfg['forward']['n']}, A_source = {cfg['forward']['A_source']}, A used = {A:.4f} "
   f"(Fresnel {A_fresnel(cfg['forward']['n']):.4f}, polynomial {A_poly(cfg['forward']['n']):.4f})")
if cfg["forward"]["A_source"] != "value":
    print("          note: the slides' PCBC uses a calibrated B. Put it in A_value with A_source: value once known")

print("\nData files")
first = True
for subject in cfg["subjects"]:
    for site in cfg["sites"]:
        for side in cfg["sides"]:
            fmt = dict(subject=subject, site=site, side=side)
            path = os.path.join(cfg["data_root"], cfg["file_pattern"].format(**fmt))
            name = f"{subject}-{site}-{side}"
            if not os.path.exists(path):
                bad(f"{name}: missing {os.path.relpath(path, HERE)}"); continue
            with tifffile.TiffFile(path) as tf:
                s = tf.series[0]; axes, shape = s.axes, s.shape
            if first:
                print(f"  first file axes={axes} shape={shape}")
                labels = hyperstack_labels(path)
                if labels:
                    print("  ImageJ slice labels (use these to fill stack_freqs):")
                    for i, l in enumerate(labels[:16]): print(f"     {i:2d}: {l}")
                first = False
            nz = shape[axes.index("Z")] if "Z" in axes else 1
            nc = shape[axes.index("C")] if "C" in axes else 1
            if nz != len(cfg["stack_freqs"]): bad(f"{name}: {nz} Z slices but stack_freqs has {len(cfg['stack_freqs'])}")
            if nc != len(cfg["wavelengths"]): bad(f"{name}: {nc} channels but wavelengths has {len(cfg['wavelengths'])}")
            try:
                c = Cell(cfg, subject, site, side, lut)
            except Exception as e:
                bad(f"{name}: {e}"); continue
            roi = c.mask.mean() * 100
            valid = [c.valid(w).sum() / max(c.mask.sum(), 1) * 100 for w in range(len(cfg["wavelengths"]))]
            r0 = c.R[0][:, c.mask]; r1 = c.R[1][:, c.mask]
            msg = (f"{name}: ROI {roi:.0f}% of frame, Rd f0 {np.nanmedian(r0):.3f}, f0.1 {np.nanmedian(r1):.3f} (medians), "
                   f"ref = {c.ref_source}, valid pixels per wavelength % = " + " ".join(f"{v:.0f}" for v in valid))
            if roi < 1: bad(msg + "  <- ROI almost empty")
            elif np.nanmedian(r1) >= np.nanmedian(r0): bad(msg + "  <- Rd at f=0.1 should be below f=0. Check stack_freqs order")
            else: ok(msg)

print("\n" + ("All checks passed." if not problems else f"{len(problems)} thing(s) to fix above."))
