"""
filter_conv.py
==============
Differentiable SED → 10-band AB magnitude module.

Given rest-frame log₁₀ L_ν (from the Speculator), redshift z, and log stellar
mass logmass, this module:
  1. Redshifts the wavelength grid:  λ_obs = λ_rest × (1 + z)
  2. Converts L_ν → F_ν [maggies] using luminosity distance from a WMAP9
     look-up table (differentiable via linear interpolation).
  3. Evaluates each filter's transmission T(λ_obs) via differentiable linear
     interpolation.
  4. Computes the AB filter-averaged flux:
         f_ν_filt = ∫ F_ν(λ) T(λ) dλ/λ  /  ∫ T(λ) dλ/λ
  5. Returns m_AB = -2.5 log₁₀(f_ν_filt)  for each of the 10 filters.

All operations are differentiable w.r.t. z (via wl_obs) and logmass (via
the flux scale). Filter weights and the d_L table are frozen buffers.

Filter ordering (must match catalog column ordering):
  0  LSST u   (lsst_baseline_u)
  1  LSST g   (lsst_baseline_g)
  2  LSST r   (lsst_baseline_r)
  3  LSST i   (lsst_baseline_i)
  4  LSST z   (lsst_baseline_z)
  5  LSST y   (lsst_baseline_y)
  6  Euclid VIS
  7  Euclid Y
  8  Euclid J
  9  Euclid H

Usage
-----
    fc = FilterConv(filter_dir, wl_rest)
    m_ab = fc(log_spec, z, logmass)   # → (batch, 10) AB mags
"""

import os
import numpy as np
import torch
import torch.nn as nn

# Physical / unit constants (CGS)
_LSUN_CGS   = 3.828e33    # erg/s
_MAGGIE_CGS = 3.631e-20   # erg/s/cm²/Hz  (= 3631 Jy × 1e-23)

# Euclid filter files (filename, display name)
_EUCLID_FILES = [
    ("Euclid_VIS.vis.dat", "Euclid VIS"),
    ("Euclid_NISP.Y.dat",  "Euclid Y"),
    ("Euclid_NISP.J.dat",  "Euclid J"),
    ("Euclid_NISP.H.dat",  "Euclid H"),
]

# LSST filter names for sedpy
_LSST_SEDPY = [f"lsst_baseline_{b}" for b in "ugrizy"]


def _load_filters(filter_dir: str):
    """
    Load 10 filters.  Returns list of (wave_aa, transmission) np.ndarray tuples,
    in the canonical ordering u g r i z y VIS Y J H.
    """
    try:
        import sedpy.observate as sedobs
        lsst = sedobs.load_filters(_LSST_SEDPY)
        lsst_curves = [(np.asarray(f.wavelength, dtype=np.float32),
                        np.asarray(f.transmission, dtype=np.float32)) for f in lsst]
    except Exception as e:
        raise RuntimeError(f"Could not load LSST filters via sedpy: {e}")

    euclid_curves = []
    for fname, dname in _EUCLID_FILES:
        path = os.path.join(filter_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Euclid filter not found: {path}")
        wave, trans = np.loadtxt(path, unpack=True)
        euclid_curves.append((wave.astype(np.float32), trans.astype(np.float32)))

    return lsst_curves + euclid_curves   # length 10


def _build_dl_table(z_min: float = 0.001, z_max: float = 7.0, n: int = 2000):
    """Pre-compute WMAP9 luminosity distance [cm] on a fine z grid."""
    from astropy.cosmology import WMAP9
    import astropy.units as u
    z_grid = np.linspace(z_min, z_max, n).astype(np.float32)
    dl_cm  = WMAP9.luminosity_distance(z_grid).to(u.cm).value.astype(np.float32)
    return z_grid, dl_cm


# ─────────────────────────────────────────────────────────────────────────────
# Helper: differentiable 1-D linear interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _linear_interp(x_query: torch.Tensor,
                   x_grid: torch.Tensor,
                   y_grid: torch.Tensor,
                   extrapolate_zero: bool = True) -> torch.Tensor:
    """
    Differentiable piecewise-linear interpolation.

    Parameters
    ----------
    x_query : any shape, e.g. (batch, n_wave)
    x_grid  : (N,) sorted 1-D tensor
    y_grid  : (N,) tensor
    extrapolate_zero : if True, points outside x_grid range → 0.0

    Returns
    -------
    y_query : same shape as x_query
    """
    x_clamp = x_query.clamp(x_grid[0], x_grid[-1])
    # searchsorted returns indices in [0, N]; clamp to [1, N-1] so idx-1 ≥ 0
    idx = torch.searchsorted(x_grid.contiguous(), x_clamp.contiguous())
    idx = idx.clamp(1, len(x_grid) - 1)

    x0 = x_grid[idx - 1];  x1 = x_grid[idx]
    y0 = y_grid[idx - 1];  y1 = y_grid[idx]
    t  = (x_clamp - x0) / (x1 - x0 + 1e-30)
    t  = t.clamp(0.0, 1.0)
    y  = y0 + t * (y1 - y0)

    if extrapolate_zero:
        in_range = (x_query >= x_grid[0]) & (x_query <= x_grid[-1])
        y = y * in_range.to(y.dtype)

    return y


def _trapz(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Trapezoidal integration over the last dimension. Compatible with all PyTorch versions."""
    dx = x[..., 1:] - x[..., :-1]                 # (... , N-1)
    return (0.5 * (y[..., :-1] + y[..., 1:]) * dx).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class FilterConv(nn.Module):
    """
    Differentiable SED → 10-band AB photometry.

    Parameters
    ----------
    filter_dir : str   path to directory containing Euclid_*.dat files
    wl_rest    : Tensor (n_wave,)  rest-frame wavelength grid [Å] from Speculator
    """

    def __init__(self, filter_dir: str, wl_rest: torch.Tensor):
        super().__init__()

        self.register_buffer("wl_rest", wl_rest.float())

        # ── Load filters ──────────────────────────────────────────────────
        filter_curves = _load_filters(filter_dir)
        self.n_filt = len(filter_curves)   # 10

        for j, (wl, trans) in enumerate(filter_curves):
            self.register_buffer(f"filt_wl_{j}",   torch.from_numpy(wl))
            self.register_buffer(f"filt_T_{j}",    torch.from_numpy(trans))

        # ── Luminosity distance look-up table ─────────────────────────────
        z_grid, dl_cm = _build_dl_table()
        self.register_buffer("z_grid",  torch.from_numpy(z_grid))
        self.register_buffer("dl_grid", torch.from_numpy(dl_cm))

        # ── Unit scale factor (precomputed, constant) ─────────────────────
        # log10( L_sun_cgs / (4π × maggie_cgs) )  — everything except (1+z)/d_L²
        self._log_unit = float(np.log10(_LSUN_CGS / (4.0 * np.pi * _MAGGIE_CGS)))

    # ── Private helpers ───────────────────────────────────────────────────

    def _dl(self, z: torch.Tensor) -> torch.Tensor:
        """Differentiable d_L(z) [cm] via linear interpolation. z: (batch,)"""
        return _linear_interp(z.unsqueeze(1), self.z_grid, self.dl_grid,
                               extrapolate_zero=False).squeeze(1)

    def _filter_transmission(self, wl_obs: torch.Tensor, j: int) -> torch.Tensor:
        """Evaluate filter j at observed wavelengths. wl_obs: (batch, n_wave) → (batch, n_wave)"""
        filt_wl = getattr(self, f"filt_wl_{j}")
        filt_T  = getattr(self, f"filt_T_{j}")
        return _linear_interp(wl_obs, filt_wl, filt_T, extrapolate_zero=True)

    # ── Main forward ──────────────────────────────────────────────────────

    def forward(self,
                log_spec:  torch.Tensor,
                z:         torch.Tensor,
                logmass:   torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        log_spec : (batch, n_wave)  log₁₀ L_ν [L_sun/Hz per 1 M_sun formed]
        z        : (batch,)         redshift (clamped to d_L table range)
        logmass  : (batch,)         log₁₀ M_formed [M_sun]

        Returns
        -------
        m_ab : (batch, 10)  AB magnitudes (one per filter)
               Bands with near-zero flux return a large (faint) value.
        """
        batch  = z.shape[0]
        z_safe = z.clamp(self.z_grid[0], self.z_grid[-1])

        # 1. Observed-frame wavelength grid  (batch, n_wave)
        wl_obs = self.wl_rest.unsqueeze(0) * (1.0 + z_safe.unsqueeze(1))

        # 2. Flux density F_ν [maggies]
        #    F_ν = 10^(log_spec + logmass) × L_sun × (1+z) / (4π d_L²) / maggie_cgs
        #    log₁₀ F_ν = log_spec + logmass + log_unit + log₁₀((1+z)/d_L²)
        dl_cm       = self._dl(z_safe)                         # (batch,)
        log_flux_scale = (
            logmass
            + self._log_unit
            + torch.log10((1.0 + z_safe).clamp(min=1e-6))
            - 2.0 * torch.log10(dl_cm.clamp(min=1.0))
        )                                                       # (batch,)
        log_fnu = log_spec + log_flux_scale.unsqueeze(1)       # (batch, n_wave)
        fnu     = torch.pow(10.0, log_fnu.clamp(min=-60.0, max=20.0))  # (batch, n_wave) maggies

        # 3. Filter integration: f_filt = ∫ F_ν T(λ) dλ/λ  /  ∫ T(λ) dλ/λ
        inv_wl  = 1.0 / wl_obs.clamp(min=1.0)                # (batch, n_wave)
        mags    = []

        for j in range(self.n_filt):
            T_obs = self._filter_transmission(wl_obs, j)      # (batch, n_wave)

            weight = T_obs * inv_wl                            # T(λ)/λ  (batch, n_wave)
            numer  = _trapz(fnu * weight, wl_obs)              # (batch,)
            denom  = _trapz(weight,       wl_obs)              # (batch,)

            f_filt = numer / denom.clamp(min=1e-30)            # (batch,) maggies
            m_ab   = -2.5 * torch.log10(f_filt.clamp(min=1e-30))
            mags.append(m_ab)

        return torch.stack(mags, dim=1)   # (batch, 10)
