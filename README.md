# SFDI Research

A research kit for **Spatial Frequency Domain Imaging (SFDI)** that reconstructs
tissue optical properties (absorption μₐ and reduced scattering μₛ′) from diffuse
reflectance, and trains a **Physics-Informed Neural Network (PINN)** to do the
inversion more accurately than the classical analytic models.

It reproduces the Tian Lab PINN study one step at a time. Out of the box it runs
end-to-end on synthetic stand-in data; point `config.yaml` at the lab's real files
to get the real numbers.

## What it does

SFDI projects patterned light at several spatial frequencies onto tissue and
measures the diffuse reflectance `Rd`. From that reflectance you can recover the
optical properties, but the standard closed-form models are only approximate. This
project:

1. **Benchmarks the forward models** — compares the Cuccia diffusion approximation
   and the PCBC (Post convention) model against a Monte Carlo look-up table (LUT).
2. **Benchmarks the inverse** — recovers μₐ / μₛ′ from `Rd` and measures the error
   in the `(Rd@f=0, Rd@f=0.1)` plane.
3. **Trains a PINN** — an MLP that maps reflectance to optical properties, trained
   on one subject and graded on held-out subjects per wavelength.
4. **Runs ablations** — over wavelength, anatomical site, subject, and forward model.

## Setup

```
pip install -r requirements.txt
```

Requires Python 3, with NumPy, SciPy, PyTorch, Matplotlib, pandas, PyYAML, and
tifffile (plus `roifile` if your masks are ImageJ `.roi` files).

## Run order

Each script writes to `results/`. Steps run in order; steps `00` are only needed
when you don't yet have the lab's real LUT and data.

| Step | Script | Output |
|---|---|---|
| 0 | `scripts/00_build_standin_lut.py` | stand-in Monte Carlo LUT (skip with the lab LUT) |
| 0 | `scripts/00_make_synthetic_data.py` | fake C1–C3 subjects in the lab's file format (skip with real data) |
| check | `scripts/check_setup.py` | checks paths, shapes, frequency order, ROI and LUT coverage. Run after every config change |
| 1 | `scripts/01_forward_error.py` | Cuccia and PCBC vs LUT maps, per-pixel tables |
| 2 | `scripts/02_inverse_error.py` | inverse maps in the `(Rd f=0, Rd f=0.1)` plane, tables |
| 3 | `scripts/03_train_pinn.py` | PINN trained on C1, graded on C2/C3 per wavelength, error maps |
| 4 | `scripts/04_ablation.py` | wavelength, site, subject and forward-model ablations |

Add `--quick` to steps 3 and 4 for a fast smoke test (a few hundred steps). Full
settings are 10,000 steps at batch 32,768 — roughly 20 minutes per run on a 2-core
CPU, much faster on a GPU.

`example_results/` shows what every script produced on the synthetic data.

## Switching to real data

Change only `config.yaml`, then run `python scripts/check_setup.py` until it says "All checks passed".

Edit `config.yaml` (fields that only the project lead can supply are marked `ASK PL`):

1. **Rd files** — set `data_root` and `file_pattern`. Hyperstacks are ImageJ files
   named like `R-Arm-Left-WithoutCorrection.tif`: 8 wavelengths × 8 spatial
   frequencies, 1392×1040, 32-bit.
2. **Frequencies** — fill `stack_freqs` with the real Z-slice frequencies. Inspect a
   file with `python -c "from sfdi.data import hyperstack_labels as h; print(h('path/to/file.tif'))"`.
3. **LUT** — set `lut_path`, `lut_format: mat`, and `lut_mat_keys` to the variable
   names in their `.mat` file; `Rd_axes` describes the array ordering.
4. **Masks** — ROI polygons from ImageJ (`.roi`), or PNG/NPY masks. Set `mask_pattern`.
5. **Supplied optical properties** — the per-pixel μₐ, μₛ′ maps. Set `ref_pattern`.
   Expected: `.npz` or `.mat` with arrays `mua` and `musp`, each shaped (8 wavelengths, 1040, 1392).
   If their maps are TIFFs, convert once:
   `np.savez("OP-Arm-Left.npz", mua=tifffile.imread("mua.tif"), musp=tifffile.imread("musp.tif"))`.
   If the file is missing, the LUT inverse of `Rd` is used instead (the checker says which).
6. **Forward model** — set `forward.n`, and put the calibrated boundary parameter B
   in `A_value` with `A_source: value`.

## Code map

| File | What it does |
|---|---|
| `sfdi/physics.py` | Cuccia and PCBC forward formulas (Post convention A), Fresnel A |
| `sfdi/lut.py` | LUT loading, forward interpolation, inverse lookup, LUT boundary |
| `sfdi/mc_lut.py` | White Monte Carlo used to build the stand-in LUT |
| `sfdi/data.py` | Hyperstack, mask and supplied-property loading; one "cell" per subject/site/side |
| `sfdi/pinn.py` | MLP, Huber-log and Log-MSE losses, training loop |
| `sfdi/experiment.py` | Selecting training cells and grading against the LUT |
| `config.yaml` | Every setting, with the unknowns marked `ASK PL` |

## Notes on reproducibility

With stand-in data the figures take the same *shape* as the reference study — PCBC
beats Cuccia, and both are worst at 471 and 526 nm — but the absolute numbers will
not match, because the stand-in LUT (white Monte Carlo, g = 0.8, n = 1.4, flat
tissue) is not the lab's LUT and the subjects are synthetic. One effect (training on
811/851 nm improving held-out error) only reproduces with the real LUT.
