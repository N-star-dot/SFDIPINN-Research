"""
Loading the lab's SFDI data.

From slide 1 the files are ImageJ hyperstacks like
    R-Arm-Left-WithoutCorrection.tif
    8 channels (C = wavelength, 471 ... 851 nm) x 8 slices (Z = spatial frequency),
    1392 x 1040 pixels, 32-bit float, about 353 MB each.
"R" is reflectance (Rd), "WithoutCorrection" means no surface-profile correction.

One "cell" = one subject + site + side (C1 Arm Left, ...), as in the ablation
table. Everything is configured by the patterns in config.yaml. If config.yaml
has a processed_root (built by scripts/ingest_data.py), Cell loads the small
cropped .npz instead of the ~450 MB of raw TIFFs -- see sfdi/ingest.py.
"""
import os
import re
import json
import hashlib
import numpy as np
import tifffile
from matplotlib.path import Path
from scipy.io import loadmat

_QTY_RE = re.compile(r"Z:\s*(mu_a|mu_s_prime)")

# Bump whenever ingest.py's .npz contents or semantics change, so an old cache
# is treated as stale even if ref_caps/use_freqs/wavelengths/stack_freqs match.
_INGEST_FORMAT_VERSION = 2


def ingest_signature(cfg):
    """Hash of the settings that determine what scripts/ingest_data.py's .npz
    cache should contain. sfdi/ingest.py embeds this in each .npz and in
    manifest.json; Cell compares it against the *current* config before
    trusting a cache, so a config change (e.g. ref_caps) can't silently use
    stale data."""
    key = {"use_freqs": cfg["use_freqs"], "wavelengths": cfg["wavelengths"],
           "stack_freqs": cfg["stack_freqs"], "ref_caps": cfg.get("ref_caps", {}),
           "format_version": _INGEST_FORMAT_VERSION}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


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


def load_mask(path, shape=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path).astype(bool)
    if ext in (".tif", ".tiff"):
        return tifffile.imread(path) > 0
    if ext == ".png":
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


def _plateau(x, min_count=100, min_frac=1e-4):
    """The value a LUT inversion got clamped to, or None. A clamp shows up as
    the array's maximum repeated on many pixels; a genuine maximum is ~unique."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    top = float(x.max())
    n = int(np.count_nonzero(np.abs(x - top) <= 1e-6 * abs(top)))
    return top if n >= max(min_count, min_frac * x.size) else None


def resolve_caps(mua, musp, ref_caps):
    """Turn ref_caps into numbers. 'auto' detects the clamp per file, since
    subjects were not all inverted with the same bound (C3 sits at 0.1955,
    the others at 0.2). Detection uses the whole frame, not the ROI, so the
    raw and processed loading paths always agree."""
    out = {}
    for k, arr in (("mua", mua), ("musp", musp)):
        v = ref_caps.get(k)
        out[k] = _plateau(np.asarray(arr)) if v == "auto" else v
    return out


def apply_ref_caps(mua, musp, ref_caps, roi=None):
    """The lab's LUT inversion hits a wall at its bounds (mu_a clamped on many
    471 nm pixels) and sometimes fails outright (NaN); both are "no answer",
    not real readings, so turn capped pixels into NaN too -- downstream code
    (Cell.valid) already drops NaNs. Caps are numbers or 'auto' (see
    resolve_caps). Returns per-wavelength counts of (capped, already-NaN)
    pixels, counted within `roi` if given else over the whole frame; that's
    what ingest.py puts in the manifest."""
    mua = np.array(mua, dtype=np.float32, copy=True)
    musp = np.array(musp, dtype=np.float32, copy=True)
    n_wl = len(mua)
    region = roi if roi is not None else np.ones(mua.shape[-2:], bool)
    caps = resolve_caps(mua, musp, ref_caps)
    cap_a, cap_s = caps["mua"], caps["musp"]
    capped, nan = [0] * n_wl, [0] * n_wl
    for w in range(n_wl):
        bad = np.zeros(mua[w].shape, bool)
        if cap_a is not None: bad |= mua[w] >= cap_a
        if cap_s is not None: bad |= musp[w] >= cap_s
        nan[w] = int(np.count_nonzero((np.isnan(mua[w]) | np.isnan(musp[w])) & ~bad & region))
        capped[w] = int(np.count_nonzero(bad & region))
        mua[w] = np.where(bad, np.nan, mua[w])
        musp[w] = np.where(bad, np.nan, musp[w])
    return mua, musp, capped, nan


def _check_ref_label_order(labels, n_wl):
    """Catch an M-file whose Z slices are not (mu_a, mu_s') before it silently
    swaps every pixel's absorption and scattering."""
    if len(labels) < 2 * n_wl:
        return
    m0, m1 = _QTY_RE.search(labels[0]), _QTY_RE.search(labels[n_wl])
    order = [m.group(1) if m else None for m in (m0, m1)]
    if order != ["mu_a", "mu_s_prime"]:
        raise ValueError(f"unexpected M-file quantity order {order}, expected ['mu_a', 'mu_s_prime']")


class Cell:
    """Rd at the chosen frequencies for one subject/site/side, plus mask and
    the lab-supplied (LUT) optical properties if available."""

    def __init__(self, cfg, subject, site, side, lut=None):
        self.subject, self.site, self.side = subject, site, side
        proc = cfg.get("processed_root")
        cand = os.path.join(proc, f"{self.name}.npz") if proc else None
        if cand and os.path.exists(cand):
            self._load_processed(cfg, cand)
        else:
            self._load_raw(cfg, subject, site, side, lut)

    def _load_processed(self, cfg, path):
        d = np.load(path)
        # A cache built under different settings (ref_caps, use_freqs, ...) is
        # silently wrong (wrong columns, wrong units, wrong capped pixels), not
        # just stale -- fail loudly rather than guess. The signature check
        # catches everything ingest_signature hashes (including ref_caps);
        # the shape checks are a defense-in-depth fallback for old caches
        # written before the "sig" field existed.
        sig_mismatch = "sig" in d and d["sig"].item() != ingest_signature(cfg)
        shape_mismatch = list(d["wavelengths"]) != list(cfg["wavelengths"]) or list(d["freqs"]) != list(cfg["use_freqs"])
        if sig_mismatch or shape_mismatch:
            raise ValueError(f"{self.name}: processed cache {path} was built under different ingest "
                              f"settings than the current config -- rerun scripts/ingest_data.py")
        self.R = d["R"]
        self.mask = d["mask"].astype(bool)
        self.mua_ref = d["mua"]; self.musp_ref = d["musp"]
        self.ref_source = "supplied"
        self.bbox = tuple(int(v) for v in d["bbox"])
        self.frame_shape = tuple(int(v) for v in d["frame_shape"])
        self.source = "processed"

    def _load_raw(self, cfg, subject, site, side, lut):
        root = cfg["data_root"]
        fmt = dict(subject=subject, site=site, side=side)
        zi = [cfg["stack_freqs"].index(f) for f in cfg["use_freqs"]]
        self.R = read_hyperstack(os.path.join(root, cfg["file_pattern"].format(**fmt)), zi)  # (2, n_wl, H, W)
        self.mask = load_mask(os.path.join(root, cfg["mask_pattern"].format(**fmt)), self.R.shape[-2:])
        self.bbox = (0, self.R.shape[-2], 0, self.R.shape[-1])
        self.frame_shape = tuple(self.R.shape[-2:])
        ref = os.path.join(root, cfg["ref_pattern"].format(**fmt)) if cfg.get("ref_pattern") else None
        self.ref_source = None
        if ref and os.path.exists(ref):
            if ref.lower().endswith((".tif", ".tiff")):          # lab's M-*.tif: Z = (mu_a, mu_s')
                arr = read_hyperstack(ref)                        # (2, n_wl, H, W)
                labels = hyperstack_labels(ref)
                if labels:
                    _check_ref_label_order(labels, arr.shape[1])
                mua, musp = arr[0], arr[1]
            elif ref.endswith(".npz"):
                d = np.load(ref)
                mua, musp = np.asarray(d["mua"], np.float32), np.asarray(d["musp"], np.float32)
            else:
                d = loadmat(ref)
                mua, musp = np.asarray(d["mua"], np.float32), np.asarray(d["musp"], np.float32)
            self.mua_ref, self.musp_ref, _, _ = apply_ref_caps(mua, musp, cfg.get("ref_caps", {}))
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
        self.source = "raw"

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
