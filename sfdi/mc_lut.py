"""
Stand-in Monte Carlo lookup table, built with the "white Monte Carlo" trick.

WHY: the slides grade everything against the lab's Monte Carlo LUT. Until you
have that file, this builds a comparable one so every script runs today.
Swap in the lab LUT the moment you get it (see lut.py, LUT.from_mat).

HOW (white MC, the same idea the Roblyer lab's public LUT code uses):
  1. Simulate photons once in a medium with mu_s' = 1 and mu_a = 0 (units of
     transport mean free paths). Record, for every photon that escapes, where
     it left the surface (radius rho') and how far it travelled (L').
  2. Any real tissue is the same random walk stretched by 1/mu_s':
         rho = rho' / mu_s',   L = L' / mu_s'
  3. Absorption just down-weights each photon by exp(-mu_a L).
  4. Stripes at frequency f: weight each exit point by J0(2 pi f rho) (the
     Hankel transform of a pencil beam, as in Post 2023 and Cuccia 2009).
         Rd(mu_a, mu_s', f) = mean over photons of exp(-mu_a L) J0(2 pi f rho)

Simplifications (say so if you show these numbers):
  * Henyey-Greenstein scattering with anisotropy g (default 0.8), mu_s' = 1
    means mu_s = 1/(1-g) in simulation units. g changes high-f reflectance,
    so match the lab LUT's g once you know it.
  * flat, uniform, semi-infinite tissue
  * Fresnel reflection at the skin surface for tissue index n
  * specular reflection at entry is not counted (SFDI rejects it)
"""
import numpy as np
from scipy.special import j0


def fresnel_R(cos_i, n1, n2=1.0):
    """Unpolarized Fresnel reflectance, light going from n1 into n2."""
    sin_t = n1 / n2 * np.sqrt(np.clip(1 - cos_i ** 2, 0, 1))
    R = np.ones_like(cos_i)
    ok = sin_t < 1
    ci, st = cos_i[ok], sin_t[ok]
    ct = np.sqrt(1 - st ** 2)
    rs = ((n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)) ** 2
    rp = ((n1 * ct - n2 * ci) / (n1 * ct + n2 * ci)) ** 2
    R[ok] = 0.5 * (rs + rp)
    return R


def hg_cos(g, u):
    """Sample cos(theta) from Henyey-Greenstein with anisotropy g."""
    if abs(g) < 1e-6:
        return 2 * u - 1
    t = (1 - g * g) / (1 - g + 2 * g * u)
    return (1 + g * g - t * t) / (2 * g)


def simulate(n_photons=200_000, n_tissue=1.4, g=0.8, max_len=3e4, seed=0):
    """White MC. Returns exit radius rho' and path length L' of escaped photons
    (in units of 1/mu_s'), plus the number launched (the rest count as zero)."""
    rng = np.random.default_rng(seed)
    mus = 1.0 / (1.0 - g)                                # so that mu_s' = 1
    N = n_photons
    x = np.zeros(N); y = np.zeros(N); z = np.zeros(N); L = np.zeros(N)
    ux = np.zeros(N); uy = np.zeros(N); uz = np.ones(N)
    alive = np.arange(N)
    out_rho, out_L = [], []
    while alive.size:
        s = -np.log(rng.random(alive.size)) / mus        # step, mu_t = mu_s
        zi = z[alive] + s * uz[alive]
        crossing = zi < 0
        # photons that reach the surface: move to it, then Fresnel test
        if crossing.any():
            a = alive[crossing]
            t = z[a] / (-uz[a])
            x[a] += t * ux[a]; y[a] += t * uy[a]; z[a] = 0.0; L[a] += t
            refl = rng.random(a.size) < fresnel_R(-uz[a], n_tissue)
            esc = a[~refl]
            out_rho.append(np.hypot(x[esc], y[esc])); out_L.append(L[esc].copy())
            # reflected ones: mirror direction, finish the leftover step inside
            b = a[refl]
            uz[b] = -uz[b]
            left = s[crossing][refl] - t[refl]
            x[b] += left * ux[b]; y[b] += left * uy[b]; z[b] += left * uz[b]; L[b] += left
        stay = alive[~crossing]
        ss = s[~crossing]
        x[stay] += ss * ux[stay]; y[stay] += ss * uy[stay]; z[stay] += ss * uz[stay]; L[stay] += ss
        # scatter everyone still inside (HG deflection, standard MCML rotation)
        inside = np.concatenate([stay, alive[crossing][refl]]) if crossing.any() else stay
        cth = hg_cos(g, rng.random(inside.size))
        sth = np.sqrt(np.clip(1 - cth ** 2, 0, 1))
        phi = 2 * np.pi * rng.random(inside.size)
        cp, sp = np.cos(phi), np.sin(phi)
        a, b, c = ux[inside], uy[inside], uz[inside]
        near_pole = np.abs(c) > 0.99999
        den = np.sqrt(np.clip(1 - c * c, 1e-12, None))
        nx = sth * (a * c * cp - b * sp) / den + a * cth
        ny = sth * (b * c * cp + a * sp) / den + b * cth
        nz = -sth * cp * den + c * cth
        nx = np.where(near_pole, sth * cp, nx)
        ny = np.where(near_pole, sth * sp, ny)
        nz = np.where(near_pole, np.sign(c) * cth, nz)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz)
        ux[inside] = nx / norm; uy[inside] = ny / norm; uz[inside] = nz / norm
        alive = inside[L[inside] < max_len]
    return np.concatenate(out_rho), np.concatenate(out_L), N


def build_lut(rho, L, n_launched, mua, musp, freqs):
    """Rd[f, i_mua, j_musp] on a grid, from one white-MC run."""
    mua = np.asarray(mua, float); musp = np.asarray(musp, float)
    Rd = np.zeros((len(freqs), len(mua), len(musp)))
    for j, ms in enumerate(musp):
        Lphys = L / ms                                  # mm
        W = np.exp(-np.outer(mua, Lphys))               # (n_mua, n_photons)
        for k, f in enumerate(freqs):
            J = j0(2 * np.pi * f * rho / ms)
            Rd[k, :, j] = W @ J / n_launched
    return Rd
