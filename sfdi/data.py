"""
Loading the lab's SFDI data.

From slide 1 the files are ImageJ hyperstacks like
    R-Arm-Left-WithoutCorrection.tif
    8 channels (C = wavelength, 471 ... 851 nm) x 8 slices (Z = spatial frequency),
    1392 x 1040 pixels, 32-bit float, about 353 MB each.
"R" is reflectance (Rd), "WithoutCorrection" means no surface-profile correction.

One "cell" = one subject + site + side (C1 Arm Left, ...), as in the ablation
table. Everything is configured by the patterns in config.yaml.
"""
import os
import numpy as np
import tifffile
from matplotlib.path import Path


def read_hyperstack(path):
    """Return Rd as an array (freq, wavelength, y, x)."""
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr = s.asarray()
        axes = s.axes                      # e.g. 'ZCYX' for ImageJ hyperstacks
    want = "ZCYX"
    for a in want:
        if a not in axes:                  # single-frequency or single-channel file
            arr = arr[np.newaxis]; axes = a + axes
    arr = np.transpose(arr, [axes.index(a) for a in want])
    return arr.astype(np.float32)


def hyperstack_labels(path):
    """ImageJ slice labels, if present, to check which Z is which frequency."""
    with tifffile.TiffFile(path) as tf:
        md = tf.imagej_metadata or {}
    return md.get("Labels") or md.get("labels")


def load_mask(path, shape):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(bool)
    if ext in (".png", ".tif", ".tiff"):
        from PIL import Image
        return np.array(Image.open(path)) > 0
    if ext == ".roi":                      # ImageJ ROI saved from the yellow polygons
        import roifile                     # pip install roifile
        xy = roifile.roiread(path).coordinates()
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        return Path(xy).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(shape)
    raise ValueError(f"unknown mask type {path}")


class Cell:
    """Rd at the chosen frequencies for one subject/site/side, plus mask and
    the lab-supplied (LUT) optical properties if available."""

    def __init__(self, cfg, subject, site, side, lut=None):
        self.subject, self.site, self.side = subject, site, side
        root = cfg["data_root"]
        fmt = dict(subject=subject, site=site, side=side)
        stack = read_hyperstack(os.path.join(root, cfg["file_pattern"].format(**fmt)))
        zi = [cfg["stack_freqs"].index(f) for f in cfg["use_freqs"]]
        self.R = stack[zi]                                     # (2, n_wl, H, W)
        self.mask = load_mask(os.path.join(root, cfg["mask_pattern"].format(**fmt)), self.R.shape[-2:])
        ref = os.path.join(root, cfg["ref_pattern"].format(**fmt)) if cfg.get("ref_pattern") else None
        if ref and os.path.exists(ref):
            d = np.load(ref)
            self.mua_ref, self.musp_ref = d["mua"], d["musp"]  # (n_wl, H, W)
        elif lut is not None:                                   # fall back: invert with the LUT
            self.mua_ref, self.musp_ref = lut.inverse(self.R[0], self.R[1], *cfg["use_freqs"])
        else:
            self.mua_ref = self.musp_ref = None

    @property
    def name(self):
        return f"{self.subject}-{self.site}-{self.side}"

    def valid(self, w):
        """Pixels inside the ROI with finite Rd and a finite LUT answer."""
        v = self.mask & np.isfinite(self.R[:, w]).all(0) & (self.R[:, w] > 0).all(0)
        if self.mua_ref is not None:
            v &= np.isfinite(self.mua_ref[w]) & np.isfinite(self.musp_ref[w])
        return v

    def pixels(self, w):
        v = self.valid(w)
        R = self.R[:, w][:, v].T                                # (n_pix, 2)
        ref = None
        if self.mua_ref is not None:
            ref = np.stack([self.mua_ref[w][v], self.musp_ref[w][v]], -1)
        return R, ref, v


def load_cells(cfg, subjects, sites, sides, lut=None):
    return [Cell(cfg, s, si, sd, lut) for s in subjects for si in sites for sd in sides]
