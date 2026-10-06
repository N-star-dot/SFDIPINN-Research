"""
Ingest the lab's real hyperstacks into a small per-cell .npz cache, so later
scripts don't each pay ~450 MB of TIFF reads for every cell. Run via
scripts/ingest_data.py (python -m or `python scripts/ingest_data.py`), not
usually called directly.

One cell = one subject/site/side (e.g. C1-Arm-Left). For each cell we: check
the raw files exist and their ImageJ labels match config.yaml, read only the
2 Z (frequency) slices use_freqs needs out of the 8 in the R file, read the
small 2-slice M file (mu_a, mu_s'), cap/NaN the LUT-bound failures, crop
everything to the mask's bounding box, and write one .npz. A manifest.json
records provenance (source file size/mtime) so a rerun only re-ingests cells
whose sources or pipeline settings actually changed.
"""
import os
import re
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile

from .data import read_hyperstack, hyperstack_labels, load_mask, apply_ref_caps, resolve_caps, ingest_signature

_CHUNK = 1 << 20   # 1 MB
# A duplicate-folder candidate is the expected name plus a macOS copy suffix
# ("C1-Arm 2", "C1-Arm copy"), not just any folder that happens to start with
# the expected name (e.g. "C1-Arm2", "C1-Arm-old" are real, different cells).
_DUP_SUFFIX = r"( \d+| copy)?$"

_R_RE = re.compile(r"C:\s*wavelength\s*=\s*(\d+)\s*nm\s*\|\s*Z:\s*frequency\s*=\s*([\d.]+)\s*mm\^-1")
_M_RE = re.compile(r"C:\s*wavelength\s*=\s*(\d+)\s*nm\s*\|\s*Z:\s*(mu_a|mu_s_prime)")


class IngestError(Exception):
    pass


def parse_labels(labels):
    """ImageJ 'Labels' metadata says which C/Z slice is which wavelength and
    frequency (R files) or quantity (M files). Returns first-seen, order-
    preserving unique values, so they line up positionally with cfg's
    wavelengths/stack_freqs or ['mu_a', 'mu_s_prime']."""
    wl, freqs, quantities = [], [], []
    for lab in labels:
        m = _R_RE.match(lab)
        if m:
            w, f = int(m.group(1)), float(m.group(2))
            if w not in wl: wl.append(w)
            if f not in freqs: freqs.append(f)
            continue
        m = _M_RE.match(lab)
        if m:
            w, q = int(m.group(1)), m.group(2)
            if w not in wl: wl.append(w)
            if q not in quantities: quantities.append(q)
            continue
        raise IngestError(f"unrecognised ImageJ label: {lab!r}")
    return {"wavelengths": wl, "freqs": freqs} if freqs else {"wavelengths": wl, "quantities": quantities}


def _fingerprint(path):
    """Fast heuristic for "are these two files the same": size plus a hash of
    the first and last 1 MB, not a full hash of a ~350 MB hyperstack."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(_CHUNK))
        if size > _CHUNK:
            f.seek(max(size - _CHUNK, 0))
            h.update(f.read(_CHUNK))
    return size, h.hexdigest()


def _expected_dirs(cfg):
    """Top-level folder name(s) the configured file_pattern implies, e.g.
    '{subject}-{site}' -> 'C1-Arm'."""
    dirs = set()
    for subject in cfg["subjects"]:
        for site in cfg["sites"]:
            for side in cfg["sides"]:
                rel = cfg["file_pattern"].format(subject=subject, site=site, side=side)
                dirs.add(rel.replace("\\", "/").split("/")[0])
    return dirs


def _find_duplicates(cfg):
    """data_root sometimes has accidental copies like 'P1-Arm 2' alongside
    'P1-Arm'. A byte-identical copy (by _fingerprint) is skipped and reported;
    a same-named-but-different copy is a data problem we refuse to guess
    about. Hidden files (.DS_Store etc.) are ignored outright; anything else
    unrecognised is reported, not silently dropped."""
    root = cfg["data_root"]
    expected = _expected_dirs(cfg)
    skipped = []
    for name in sorted(os.listdir(root)):
        if name in expected or name.startswith("."):
            continue
        full = os.path.join(root, name)
        src = next((e for e in expected if re.match(re.escape(e) + _DUP_SUFFIX, name)), None)
        if not os.path.isdir(full) or src is None:
            skipped.append({"path": name, "reason": "unrecognised"})
            continue
        src_full = os.path.join(root, src)
        a_files, b_files = sorted(os.listdir(full)), sorted(os.listdir(src_full))
        same = a_files == b_files and all(
            _fingerprint(os.path.join(full, f)) == _fingerprint(os.path.join(src_full, f)) for f in a_files)
        if same:
            skipped.append({"path": name, "reason": f"duplicate of {src}"})
        else:
            raise IngestError(f"{name} looks like a duplicate of {src} but its contents differ -- resolve by hand")
    return skipped


def _source_paths(cfg, subject, site, side):
    fmt = dict(subject=subject, site=site, side=side)
    return {
        "R": cfg["file_pattern"].format(**fmt),
        "Mask": cfg["mask_pattern"].format(**fmt),
        "M": cfg["ref_pattern"].format(**fmt),
    }


def _sources_unchanged(cfg, prev):
    root = cfg["data_root"]
    for rel, (size, mtime_ns) in prev.get("sources", {}).items():
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            return False
        st = os.stat(full)
        if st.st_size != size or st.st_mtime_ns != mtime_ns:
            return False
    return True


def _ingest_one(cfg, subject, site, side, sig):
    root, out = cfg["data_root"], cfg["processed_root"]
    name = f"{subject}-{site}-{side}"
    paths = _source_paths(cfg, subject, site, side)
    r_path = os.path.join(root, paths["R"])
    mask_path = os.path.join(root, paths["Mask"])
    ref_path = os.path.join(root, paths["M"])

    r_labels = hyperstack_labels(r_path)
    if r_labels:
        parsed = parse_labels(r_labels)
        if parsed["wavelengths"] != cfg["wavelengths"]:
            raise IngestError(f"{name}: R file wavelengths {parsed['wavelengths']} "
                               f"!= config wavelengths {cfg['wavelengths']}")
        if parsed["freqs"] != cfg["stack_freqs"]:
            raise IngestError(f"{name}: R file frequencies {parsed['freqs']} "
                               f"!= config stack_freqs {cfg['stack_freqs']}")

    zi = [cfg["stack_freqs"].index(f) for f in cfg["use_freqs"]]
    R = read_hyperstack(r_path, zi)                              # only the 2 of 8 Z slices we need

    m_labels = hyperstack_labels(ref_path)
    if m_labels:
        parsed = parse_labels(m_labels)
        if parsed["wavelengths"] != cfg["wavelengths"]:
            raise IngestError(f"{name}: M file wavelengths {parsed['wavelengths']} "
                               f"!= config wavelengths {cfg['wavelengths']}")
        if parsed.get("quantities") != ["mu_a", "mu_s_prime"]:
            raise IngestError(f"{name}: M file quantities {parsed.get('quantities')} "
                               f"!= ['mu_a', 'mu_s_prime']")
    M = read_hyperstack(ref_path)                                # small: only 2 Z slices total
    mua_full, musp_full = M[0], M[1]

    raw_mask = tifffile.imread(mask_path)
    vals = set(np.unique(raw_mask).tolist())
    if not (vals <= {0, 1} or vals <= {0, 255}):
        raise IngestError(f"{name}: mask is not binary (values {sorted(vals)})")
    mask = load_mask(mask_path, R.shape[-2:])
    if not mask.any():
        raise IngestError(f"{name}: mask is empty")

    frame_shape = mask.shape
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1   # half-open bbox

    mua, musp, capped, nan = apply_ref_caps(mua_full, musp_full, cfg.get("ref_caps", {}), roi=mask)
    caps_used = resolve_caps(mua_full, musp_full, cfg.get("ref_caps", {}))
    roi_pixels = int(mask.sum())

    warnings, bad_wl = [], []
    for w in range(R.shape[1]):
        r0, r1 = R[0, w][mask], R[1, w][mask]
        if np.nanmedian(r1) >= np.nanmedian(r0):                 # Rd should fall as frequency rises
            bad_wl.append(cfg["wavelengths"][w])
    if bad_wl:
        warnings.append(f"Rd does not decrease with frequency at wavelengths {bad_wl} (check stack_freqs order)")

    R = R[:, :, y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    mua = mua[:, y0:y1, x0:x1]
    musp = musp[:, y0:y1, x0:x1]

    os.makedirs(out, exist_ok=True)
    fname = f"{name}.npz"
    final = os.path.join(out, fname)
    tmp = final + ".tmp"
    with open(tmp, "wb") as fh:                                  # write to tmp, then atomic rename
        np.savez(fh, R=R, mask=mask, mua=mua, musp=musp,
                  bbox=np.array([y0, y1, x0, x1]), frame_shape=np.array(frame_shape),
                  wavelengths=np.array(cfg["wavelengths"]), freqs=np.array(cfg["use_freqs"]), sig=sig)
    os.replace(tmp, final)

    sources = {}
    for rel in paths.values():
        st = os.stat(os.path.join(root, rel))
        sources[rel] = [st.st_size, st.st_mtime_ns]

    return {"file": fname, "sources": sources, "roi_pixels": roi_pixels,
            "ref_capped_pixels": capped, "ref_nan_pixels": nan, "ref_caps_used": caps_used,
            "warnings": warnings, "signature": sig}


def run(cfg, force=False):
    root, out = cfg["data_root"], cfg["processed_root"]
    os.makedirs(out, exist_ok=True)
    manifest_path = os.path.join(out, "manifest.json")
    old_cells = {}
    if not force and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            old_cells = json.load(f).get("cells", {})

    skipped = _find_duplicates(cfg)

    cell_specs = [(s, si, sd) for s in cfg["subjects"] for si in cfg["sites"] for sd in cfg["sides"]]
    # fail fast, before spending time reading any pixel data: every source file must exist
    for subject, site, side in cell_specs:
        name = f"{subject}-{site}-{side}"
        for label, rel in _source_paths(cfg, subject, site, side).items():
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                raise IngestError(f"{name}: missing {label} file {path}")

    sig = ingest_signature(cfg)
    cells, stats, todo = {}, {"ingested": 0, "up_to_date": 0}, []
    for subject, site, side in cell_specs:
        name = f"{subject}-{site}-{side}"
        prev = old_cells.get(name)
        if (prev and prev.get("signature") == sig and _sources_unchanged(cfg, prev)
                and os.path.exists(os.path.join(out, prev.get("file", "")))):
            cells[name] = prev
            stats["up_to_date"] += 1
        else:
            todo.append((subject, site, side, name))

    if todo:
        with ThreadPoolExecutor(max_workers=min(4, len(todo))) as ex:
            futs = {ex.submit(_ingest_one, cfg, s, si, sd, sig): name for s, si, sd, name in todo}
            for fut, name in futs.items():
                cells[name] = fut.result()
                stats["ingested"] += 1

    manifest = {"cells": cells, "skipped": skipped, "stats": stats}
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)
    return manifest
