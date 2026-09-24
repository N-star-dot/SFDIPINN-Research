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
from scipy.io import loadmat


def read_hyperstack(path, z_select=None):
    """Return Rd as (freq, wavelength, y, x). With z_select, read only those
    frequency slices from disk (a real file is ~353 MB, we need 2 of 8 slices)."""
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        axes, shape = s.axes, s.shape
        lead_axes, lead_shape = axes[:-2], shape[:-2]            # everything except Y, X
        arr = None
        if z_select is not None and "Z" in lead_axes and len(tf.pages) == int(np.prod(lead_shape)):
            # read page by page: only the frequency slices we need
            zpos = lead_axes.index("Z")
            keys, sub_shape = [], list(lead_shape)
            sub_shape[zpos] = len(z_select)
            for idx in np.ndindex(*sub_shape):
                full = list(idx); full[zpos] = z_select[idx[zpos]]
                keys.append(int(np.ravel_multi_index(full, lead_shape)))
            arr = tf.asarray(key=keys).reshape(*sub_shape, *shape[-2:])
            z_select = None                                     # already applied
        if arr is None:
            arr = s.asarray()
    for a in "ZCYX":
        if a not in axes:                                       # single-frequency or single-channel file
            arr = arr[np.newaxis]; axes = a + axes
    arr = np.transpose(arr, [axes.index(a) for a in "ZCYX"])
    if z_select is not None:
        arr = arr[list(z_select)]
    return np.ascontiguousarray(arr, dtype=np.float32)


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
    if ext in (".roi", ".zip"):            # ImageJ ROI (or RoiSet.zip) from the yellow polygons
        import roifile                     # pip install roifile
        rois = roifile.roiread(path)
        rois = rois if isinstance(rois, list) else [rois]
        yy, xx = np.mgrid[:shape[0], :shape[1]]
        pts = np.c_[xx.ravel(), yy.ravel()]
        m = np.zeros(shape[0] * shape[1], bool)
        for r in rois:                     # union of all polygons in the file
            m |= Path(r.coordinates()).contains_points(pts)
        return m.reshape(shape)
    raise ValueError(f"unknown mask type {path}")


class Cell:
    """Rd at the chosen frequencies for one subject/site/side, plus mask and
    the lab-supplied (LUT) optical properties if available."""

    def __init__(self, cfg, subject, site, side, lut=None):
        self.subject, self.site, self.side = subject, site, side
        root = cfg["data_root"]
        fmt = dict(subject=subject, site=site, side=side)
        zi = [cfg["stack_freqs"].index(f) for f in cfg["use_freqs"]]
        self.R = read_hyperstack(os.path.join(root, cfg["file_pattern"].format(**fmt)), zi)  # (2, n_wl, H, W)
        self.mask = load_mask(os.path.join(root, cfg["mask_pattern"].format(**fmt)), self.R.shape[-2:])
        ref = os.path.join(root, cfg["ref_pattern"].format(**fmt)) if cfg.get("ref_pattern") else None
        self.ref_source = None
        if ref and os.path.exists(ref):
            d = np.load(ref) if ref.endswith(".npz") else loadmat(ref)
            self.mua_ref = np.asarray(d["mua"], np.float32)     # (n_wl, H, W)
            self.musp_ref = np.asarray(d["musp"], np.float32)
            self.ref_source = "supplied"
        elif lut is not None:                                   # fall back: invert with the LUT, ROI only
            self.mua_ref = np.full(self.R.shape[1:], np.nan, np.float32)
            self.musp_ref = np.full(self.R.shape[1:], np.nan, np.float32)
            for w in range(self.R.shape[1]):
                a, s_ = lut.inverse(self.R[0, w][self.mask], self.R[1, w][self.mask], *cfg["use_freqs"])
                self.mua_ref[w][self.mask], self.musp_ref[w][self.mask] = a, s_
            self.ref_source = "LUT inverse of Rd"
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
