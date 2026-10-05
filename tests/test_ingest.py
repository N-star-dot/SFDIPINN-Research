"""Test contract for the real-data ingestion pipeline (sfdi/ingest.py).

Fixtures are tiny TIFFs written in exactly the lab's on-disk format:
  {subject}-{site}/R-{site}-{side}-WithoutCorrection.tif     ZCYX float32, Z = 8 frequencies
  {subject}-{site}/M-{site}-{side}-WithoutCorrection.tif     ZCYX float32, Z = (mu_a, mu_s')
  {subject}-{site}/Mask-{site}-{side}-WithoutCorrection.tif  YX uint8 0/255
with ImageJ slice labels like 'C: wavelength = 471 nm | Z: frequency = 0.05 mm^-1'.
"""
import os, sys, json, shutil
import numpy as np
import pytest
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi import ingest
from sfdi.data import Cell

WL = [471, 526, 591, 621, 691, 731, 811, 851]
FREQS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
H, W = 24, 32
CAP = 0.2


def fmt_f(f):
    return ("%g" % f)


def r_labels(wls=WL, freqs=FREQS):
    return [f"C: wavelength = {w} nm | Z: frequency = {fmt_f(f)} mm^-1" for f in freqs for w in wls]


def m_labels(wls=WL):
    return [f"C: wavelength = {w} nm | Z: {q}" for q in ("mu_a", "mu_s_prime") for w in wls]


def write_stack(path, arr, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, arr.astype(np.float32), imagej=True,
                     metadata={"axes": "ZCYX", "Labels": labels})


def make_cell(root, subject, site, side, seed=0, wls=WL, freqs=FREQS):
    rng = np.random.default_rng(seed)
    d = os.path.join(root, f"{subject}-{site}")
    # Rd decreasing with frequency, 0 < Rd < 1
    base = rng.uniform(0.2, 0.6, (len(wls), H, W))
    R = np.stack([base * np.exp(-3 * f) for f in freqs])                 # (Z, C, H, W)
    mua = rng.uniform(0.005, 0.1, (len(wls), H, W))
    mua[0, :4, :] = CAP                                                     # LUT-capped rows at 471 nm
    mua[1, 5, 10] = np.nan                                                  # failed inversion (in ROI)
    musp = rng.uniform(0.5, 2.0, (len(wls), H, W))
    M = np.stack([mua, musp])
    mask = np.zeros((H, W), np.uint8); mask[3:20, 6:28] = 255               # bbox y 3..19, x 6..27
    write_stack(os.path.join(d, f"R-{site}-{side}-WithoutCorrection.tif"), R, r_labels(wls, freqs))
    write_stack(os.path.join(d, f"M-{site}-{side}-WithoutCorrection.tif"), M, m_labels(wls))
    tifffile.imwrite(os.path.join(d, f"Mask-{site}-{side}-WithoutCorrection.tif"), mask, imagej=True)
    return dict(R=R.astype(np.float32), mua=mua.astype(np.float32), musp=musp.astype(np.float32), mask=mask > 0)


def make_cfg(raw, out, subjects=("C1",), sites=("Arm",), sides=("Left", "Right")):
    return {
        "data_root": str(raw),
        "processed_root": str(out),
        "file_pattern": "{subject}-{site}/R-{site}-{side}-WithoutCorrection.tif",
        "mask_pattern": "{subject}-{site}/Mask-{site}-{side}-WithoutCorrection.tif",
        "ref_pattern": "{subject}-{site}/M-{site}-{side}-WithoutCorrection.tif",
        "wavelengths": list(WL), "stack_freqs": list(FREQS), "use_freqs": [0.0, 0.1],
        "ref_caps": {"mua": CAP},
        "subjects": list(subjects), "sites": list(sites), "sides": list(sides),
    }


@pytest.fixture
def tree(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    truth = {sd: make_cell(str(raw), "C1", "Arm", sd, seed=i) for i, sd in enumerate(("Left", "Right"))}
    return raw, out, truth


# t1 -------------------------------------------------------------- label parsing
def test_t1_parse_labels_reads_axes_from_imagej_labels():
    r = ingest.parse_labels(r_labels())
    assert r["wavelengths"] == WL and r["freqs"] == FREQS
    m = ingest.parse_labels(m_labels())
    assert m["wavelengths"] == WL and m["quantities"] == ["mu_a", "mu_s_prime"]


def test_t1_config_mismatch_is_a_clear_error(tree):
    raw, out, _ = tree
    cfg = make_cfg(raw, out); cfg["stack_freqs"] = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    with pytest.raises(ingest.IngestError, match="stack_freqs"):
        ingest.run(cfg)


# t2 -------------------------------------------------------------- discovery
def test_t2_identical_duplicate_folder_is_skipped_and_reported(tree):
    raw, out, _ = tree
    shutil.copytree(raw / "C1-Arm", raw / "C1-Arm 2")
    man = ingest.run(make_cfg(raw, out))
    assert len(man["cells"]) == 2
    dup = [s for s in man["skipped"] if "C1-Arm 2" in s["path"]]
    assert dup and all("duplicate" in s["reason"] for s in dup)


def test_t2_conflicting_duplicate_folder_raises(tree):
    raw, out, _ = tree
    make_cell(str(raw).replace("raw", "raw2"), "C1", "Arm", "Left", seed=99)
    shutil.copytree(str(raw).replace("raw", "raw2") + "/C1-Arm", raw / "C1-Arm 2")
    with pytest.raises(ingest.IngestError, match="C1-Arm 2"):
        ingest.run(make_cfg(raw, out))


def test_t2_missing_file_is_reported_per_cell(tree):
    raw, out, _ = tree
    os.remove(raw / "C1-Arm" / "M-Arm-Right-WithoutCorrection.tif")
    with pytest.raises(ingest.IngestError, match="C1-Arm-Right"):
        ingest.run(make_cfg(raw, out))


# t3 -------------------------------------------------------------- processed output
def test_t3_processed_cell_is_cropped_and_lossless(tree):
    raw, out, truth = tree
    man = ingest.run(make_cfg(raw, out))
    path = os.path.join(out, man["cells"]["C1-Arm-Left"]["file"])
    d = np.load(path)
    y0, y1, x0, x1 = d["bbox"]
    assert (y0, y1, x0, x1) == (3, 20, 6, 28)
    assert tuple(d["frame_shape"]) == (H, W)
    assert d["R"].shape == (2, 8, 17, 22) and d["R"].dtype == np.float32
    t = truth["Left"]
    np.testing.assert_array_equal(d["R"][0], t["R"][0][:, 3:20, 6:28])        # f = 0
    np.testing.assert_array_equal(d["R"][1], t["R"][2][:, 3:20, 6:28])        # f = 0.1
    np.testing.assert_array_equal(d["mask"], t["mask"][3:20, 6:28])
    assert d["mask"].dtype == bool
    assert list(d["wavelengths"]) == WL and list(d["freqs"]) == [0.0, 0.1]


def test_t3_rerun_is_incremental_and_detects_source_change(tree):
    raw, out, _ = tree
    cfg = make_cfg(raw, out)
    ingest.run(cfg)
    man2 = ingest.run(cfg)
    assert man2["stats"]["ingested"] == 0 and man2["stats"]["up_to_date"] == 2
    make_cell(str(raw), "C1", "Arm", "Left", seed=7)                         # source changed
    man3 = ingest.run(cfg)
    assert man3["stats"]["ingested"] == 1
    cfg["use_freqs"] = [0.0, 0.05]                                           # pipeline settings changed
    assert ingest.run(cfg)["stats"]["ingested"] == 2


def test_t3_manifest_is_json_with_provenance(tree):
    raw, out, _ = tree
    ingest.run(make_cfg(raw, out))
    with open(os.path.join(out, "manifest.json")) as f:
        man = json.load(f)
    c = man["cells"]["C1-Arm-Right"]
    for k in ("file", "sources", "roi_pixels", "ref_capped_pixels", "ref_nan_pixels"):
        assert k in c


# t4 -------------------------------------------------------------- validation
def test_t4_capped_and_nan_reference_pixels_are_excluded(tree):
    raw, out, _ = tree
    cfg = make_cfg(raw, out)
    man = ingest.run(cfg)
    rep = man["cells"]["C1-Arm-Left"]
    assert rep["ref_capped_pixels"][0] == 22                                 # row y=3 x 22 ROI columns
    assert rep["ref_nan_pixels"][1] == 1
    c = Cell(cfg, "C1", "Arm", "Left")
    v = c.valid(0)
    assert not v[:1, :].any()                                                # rows y=3 (capped) gone
    assert v[1:, :].sum() > 0
    assert not c.valid(1)[5 - 3, 10 - 6]                                     # nan pixel excluded
    assert np.isnan(c.mua_ref[0][0]).all()


def test_t4_rd_order_violation_is_reported(tree):
    raw, out, _ = tree
    p = raw / "C1-Arm" / "R-Arm-Left-WithoutCorrection.tif"
    R = tifffile.imread(p); R[[0, 2]] = R[[2, 0]]                           # swap f=0 and f=0.1
    write_stack(str(p), R, r_labels())
    man = ingest.run(make_cfg(raw, out))
    assert any("Rd" in w for w in man["cells"]["C1-Arm-Left"]["warnings"])


# t5 -------------------------------------------------------------- Cell equivalence
def test_t5_cell_from_processed_matches_cell_from_raw(tree):
    raw, out, _ = tree
    cfg = make_cfg(raw, out)
    ingest.run(cfg)
    fast = Cell(cfg, "C1", "Arm", "Right")
    assert fast.source == "processed"
    slow_cfg = dict(cfg); slow_cfg.pop("processed_root")
    slow = Cell(slow_cfg, "C1", "Arm", "Right")
    assert slow.source == "raw" and slow.ref_source == fast.ref_source == "supplied"
    for w in range(len(WL)):
        Rf, reff, _ = fast.pixels(w); Rs, refs, _ = slow.pixels(w)
        np.testing.assert_array_equal(Rf, Rs)
        np.testing.assert_array_equal(reff, refs)


# t8 -------------------------------------------------------------- stale cache (review round 1)
def test_t8_processed_cache_built_with_other_ref_caps_is_refused(tree):
    raw, out, _ = tree
    cfg = make_cfg(raw, out)
    ingest.run(cfg)
    cfg2 = dict(cfg); cfg2["ref_caps"] = {}
    with pytest.raises(Exception, match="ingest_data"):
        Cell(cfg2, "C1", "Arm", "Left")


# t9 -------------------------------------------------------------- folder name matching (review round 1)
def test_t9_similarly_named_folder_is_unrecognised_not_fatal(tree):
    raw, out, _ = tree
    make_cell(str(raw).replace("raw", "raw2"), "C1", "Arm", "Left", seed=99)
    for name in ("C1-Arm2", "C1-Arm-old"):
        shutil.copytree(str(raw).replace("raw", "raw2") + "/C1-Arm", raw / name)
    man = ingest.run(make_cfg(raw, out))
    reasons = {os.path.basename(s["path"]): s["reason"] for s in man["skipped"]}
    assert "unrecognised" in reasons["C1-Arm2"] and "unrecognised" in reasons["C1-Arm-old"]
