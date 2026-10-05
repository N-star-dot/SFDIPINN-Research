"""The lab's LUT is a MATLAB struct: LUT.Mua / LUT.Musp are meshgrids (rows =
mu_s', cols = mu_a) and LUT.M1, LUT.M2 hold Rd at the two frequencies."""
import os, sys
import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sfdi.lut import LUT

KEYS = {"struct": "LUT", "mua": "Mua", "musp": "Musp", "Rd": ["M1", "M2"], "freqs": [0.0, 0.1]}


def write_struct_lut(path):
    mua = np.linspace(0.001, 0.4, 7); musp = np.linspace(0.1, 10, 5)
    A, S = np.meshgrid(mua, musp)                         # (n_musp, n_mua), like MATLAB meshgrid
    M1 = S / (S + 10 * A); M2 = 0.5 * M1                  # any monotone stand-in
    savemat(path, {"LUT": {"Mua": A, "Musp": S, "M1": M1, "M2": M2}})
    return mua, musp, M1, M2


def test_struct_lut_loads_with_correct_axes(tmp_path):
    p = str(tmp_path / "lut.mat")
    mua, musp, M1, M2 = write_struct_lut(p)
    lut = LUT.from_mat(p, KEYS)
    np.testing.assert_allclose(lut.mua, mua); np.testing.assert_allclose(lut.musp, musp)
    assert list(lut.freqs) == [0.0, 0.1] and lut.Rd.shape == (2, 7, 5)
    np.testing.assert_allclose(lut.grid(0.0), M1.T); np.testing.assert_allclose(lut.grid(0.1), M2.T)
    # forward at a grid node returns that node's Rd
    np.testing.assert_allclose(lut.forward([mua[3]], [musp[2]], [0.0, 0.1])[0], [M1[2, 3], M2[2, 3]], rtol=1e-9)


def test_struct_lut_rejects_freq_count_mismatch(tmp_path):
    p = str(tmp_path / "lut.mat"); write_struct_lut(p)
    with pytest.raises(ValueError, match="freqs"):
        LUT.from_mat(p, dict(KEYS, freqs=[0.0]))
