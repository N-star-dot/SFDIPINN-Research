"""Build a stand-in Monte Carlo LUT (white MC) so everything runs before you
have the lab's LUT. About 3 to 6 minutes on a laptop CPU with the defaults."""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.common import load_config, HERE
from sfdi.mc_lut import simulate, build_lut
from sfdi.lut import LUT

p = argparse.ArgumentParser()
p.add_argument("--photons", type=int, default=200_000)
p.add_argument("--g", type=float, default=0.8)
p.add_argument("--n", type=float, default=None, help="tissue index, default from config")
p.add_argument("--grid", type=int, default=70)
p.add_argument("--out", default=None)
a = p.parse_args()
cfg = load_config()
n = a.n or cfg["forward"]["n"]
t = time.time()
rho, L, N = simulate(a.photons, n_tissue=n, g=a.g, seed=cfg["pinn"]["seed"])
print(f"simulated {N} photons in {time.time()-t:.0f}s, {len(rho)/N:.3f} escaped")
mua = np.logspace(np.log10(5e-4), np.log10(0.5), a.grid)
musp = np.logspace(np.log10(0.02), np.log10(10.0), a.grid)
Rd = build_lut(rho, L, N, mua, musp, cfg["use_freqs"])
lut = LUT(mua, musp, cfg["use_freqs"], Rd,
          {"kind": "white Monte Carlo stand-in", "photons": N, "g": a.g, "n": n})
out = a.out or cfg["lut_path"]
os.makedirs(os.path.dirname(out), exist_ok=True)
lut.save_npz(out)
print("wrote", os.path.relpath(out, HERE), f"({time.time()-t:.0f}s total)")
