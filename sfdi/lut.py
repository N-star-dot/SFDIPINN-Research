"""
Monte Carlo lookup table: a grid of Rd[f, mu_a, mu_s'].

forward(): (mu_a, mu_s') -> Rd, bilinear in (log mu_a, log mu_s').
inverse(): (Rd at f0, Rd at f1) -> (mu_a, mu_s'), linear interpolation over the
           grid's own Rd pairs. NaN outside the LUT boundary, which is what
           makes some 471/526 nm pixels "unavailable" in the slides.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from scipy.io import loadmat


class LUT:
    def __init__(self, mua, musp, freqs, Rd, meta=None):
        self.mua = np.asarray(mua, float).ravel()
        self.musp = np.asarray(musp, float).ravel()
        self.freqs = np.asarray(freqs, float).ravel()
        self.Rd = np.asarray(Rd, float)
        assert self.Rd.shape == (len(self.freqs), len(self.mua), len(self.musp)), \
            f"Rd must be (n_freq, n_mua, n_musp), got {self.Rd.shape}"
        self.meta = meta or {}
        self._fwd = [RegularGridInterpolator((np.log(self.mua), np.log(self.musp)), self.Rd[k],
                                             bounds_error=False, fill_value=np.nan)
                     for k in range(len(self.freqs))]
        self._inv = {}

    # ---------------------------------------------------------------- io
    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        meta = d["meta"].item() if "meta" in d else {}
        return cls(d["mua"], d["musp"], d["freqs"], d["Rd"], meta)

    @classmethod
    def from_mat(cls, path, keys):
        """keys maps our names to the .mat variable names, e.g.
        {'mua': 'mua', 'musp': 'musp', 'freqs': 'fx', 'Rd': 'Rd'}. If the lab's
        Rd array is ordered differently, fix it with 'Rd_axes' like 'mua,musp,f'."""
        d = loadmat(path)
        Rd = np.asarray(d[keys["Rd"]], float)
        order = keys.get("Rd_axes", "f,mua,musp").split(",")
        Rd = np.transpose(Rd, [order.index(a) for a in ("f", "mua", "musp")])
        return cls(d[keys["mua"]], d[keys["musp"]], d[keys["freqs"]], Rd, {"source": str(path)})

    def save_npz(self, path):
        np.savez_compressed(path, mua=self.mua, musp=self.musp, freqs=self.freqs, Rd=self.Rd,
                            meta=np.array(self.meta, dtype=object))

    # ---------------------------------------------------------------- use
    def fidx(self, f):
        k = int(np.argmin(np.abs(self.freqs - f)))
        assert abs(self.freqs[k] - f) < 1e-6, f"LUT has no frequency {f}, has {self.freqs}"
        return k

    def forward(self, mua, musp, freqs):
        mua = np.asarray(mua, float); musp = np.asarray(musp, float)
        pts = np.stack([np.log(mua).ravel(), np.log(musp).ravel()], -1)
        out = [self._fwd[self.fidx(f)](pts).reshape(mua.shape) for f in freqs]
        return np.stack(out, -1)

    def grid(self, f):
        """Rd on the stored grid at frequency f, shape (n_mua, n_musp)."""
        return self.Rd[self.fidx(f)]

    def inverse(self, R0, R1, f0=0.0, f1=0.1):
        key = (f0, f1)
        if key not in self._inv:
            A, S = np.meshgrid(self.mua, self.musp, indexing="ij")
            pts = np.stack([self.grid(f0).ravel(), self.grid(f1).ravel()], -1)
            vals = np.stack([np.log(A).ravel(), np.log(S).ravel()], -1)
            self._inv[key] = LinearNDInterpolator(pts, vals)
        R0 = np.asarray(R0, float); R1 = np.asarray(R1, float)
        out = self._inv[key](np.stack([R0.ravel(), R1.ravel()], -1))
        return np.exp(out[:, 0]).reshape(R0.shape), np.exp(out[:, 1]).reshape(R0.shape)

    def boundary(self, f0=0.0, f1=0.1):
        """Outline of the LUT domain in the (Rd f0, Rd f1) plane, for plotting."""
        R0, R1 = self.grid(f0), self.grid(f1)
        edge = [(R0[:, 0], R1[:, 0]), (R0[-1, :], R1[-1, :]), (R0[::-1, -1], R1[::-1, -1]), (R0[0, ::-1], R1[0, ::-1])]
        return np.concatenate([e[0] for e in edge]), np.concatenate([e[1] for e in edge])
