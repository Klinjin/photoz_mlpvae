"""
photoz_mlpvae.py
================
PhotozMLPVAE: shared trunk -> [z MLP head] -> [SPS VAE head, conditioned on z]
-> speculator -> filter_conv. Redshift gets its own supervised 1-dim latent
with a dedicated MLP head; the remaining 15 SPS parameters keep a VAE latent
conditioned on the redshift estimate. See README.md for architecture,
training-stage, and gradient-flow details; PLAN.md for design history.

Usage
-----
    model = PhotozMLPVAE(speculator_dir, filter_dir)
    mags_recon, z_pred, mu_z, log_var_z, mu_15, log_var_15 = model(x_phot)
    losses = model.loss(x_phot, z_spec)
    z_samples, z_mean, z_std = model.predict_z(x_phot, n_samples=500)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from photoz_vae.model.vae_encoder import _ResBlock
from photoz_vae.model.speculator_torch import SpeculatorInoueIGM
from photoz_vae.model.filter_conv import FilterConv

PARAM_NAMES = [
    "zred", "logmass", "logzsol", "dust2", "dust1_fraction", "dust_index",
    "gas_logz", "fagn", "agn_tau", "igm_scale",
    "logsfr_ratios_0", "logsfr_ratios_1", "logsfr_ratios_2",
    "logsfr_ratios_3", "logsfr_ratios_4", "logsfr_ratios_5",
]
PARAM_NAMES_15 = PARAM_NAMES[1:]

N_PARAMS_15   = 15
N_PARAMS_FULL = 16

ZRED_IDX_FULL    = 0
LOGMASS_IDX_FULL = 1
LOGZSOL_IDX_FULL = 2

_LATENT_CLAMP = 15.0

# softplus(0) = ln(2) ~= 0.693, not 1 -- so plain z_pred = z_m_i*softplus(z_raw)
# does NOT center on z_m_i when the z_head has learned nothing yet (raw~0);
# it centers ~31% low. _SOFTPLUS_CENTER_SHIFT solves softplus(c)=1 so that
# _z_scale(., centered=True) recenters raw=0 -> scale=1.0 -> z_pred=z_m_i,
# without changing softplus's bounded-linear-growth/exponential-decay shape
# (which is what keeps z_pred sane under the +/-_LATENT_CLAMP clamp on
# z_raw, unlike a plain exp(z_raw) reparameterization). Off by default
# (existing checkpoints' z_head biases are calibrated against the
# uncentered function); see MLPVAEEncoder's centered_z_scale flag.
_SOFTPLUS_CENTER_SHIFT = math.log(math.e - 1.0)


def _z_scale(z_raw: torch.Tensor, centered: bool = False) -> torch.Tensor:
    """softplus(z_raw), optionally shifted so raw=0 -> scale=1.0 exactly."""
    if centered:
        z_raw = z_raw + _SOFTPLUS_CENTER_SHIFT
    return F.softplus(z_raw)


def _z_scale_np(z_raw: np.ndarray, centered: bool = False) -> np.ndarray:
    """Numpy mirror of _z_scale, for sample_theta_posterior's numpy-only path."""
    if centered:
        z_raw = z_raw + _SOFTPLUS_CENTER_SHIFT
    return np.maximum(z_raw, 0.0) + np.log1p(np.exp(-np.abs(z_raw)))


def constrain_params_15(z_raw_15: torch.Tensor) -> torch.Tensor:
    """Map unconstrained 15-dim latent -> physical SPS params (zred excluded)."""
    parts = [
        torch.sigmoid(z_raw_15[:, 0:1])  * 6.5 + 6.5,
        torch.tanh(z_raw_15[:, 1:2])     * 1.25 - 0.95,
        torch.sigmoid(z_raw_15[:, 2:3])  * 5.0,
        torch.sigmoid(z_raw_15[:, 3:4])  * 2.5,
        torch.tanh(z_raw_15[:, 4:5])     * 0.85 - 0.35,
        torch.tanh(z_raw_15[:, 5:6])     * 1.4 - 0.8,
        torch.pow(10.0, torch.sigmoid(z_raw_15[:, 6:7]) * 6.0 - 5.0),
        torch.pow(10.0, torch.sigmoid(z_raw_15[:, 7:8]) * 1.7 + 0.602),
        torch.sigmoid(z_raw_15[:, 8:9])  * 2.5,
        torch.tanh(z_raw_15[:, 9:])      * 6.0,
    ]
    return torch.cat(parts, dim=1)


_ZRED_SIGMA_FLOOR = 1e-2


def _zred_sigma_phys(mu_z: torch.Tensor, log_var_z: torch.Tensor,
                     z_m: torch.Tensor, centered: bool = False) -> torch.Tensor:
    """Physical-space sigma_z, delta-method push-forward through z_m(i)*_z_scale(.)."""
    arg = mu_z + _SOFTPLUS_CENTER_SHIFT if centered else mu_z
    jac = z_m * torch.sigmoid(arg)
    sigma_raw = torch.exp(0.5 * log_var_z)
    return (sigma_raw * jac).clamp(min=_ZRED_SIGMA_FLOOR)


_I_BAND_IDX = 3   # position of the i-band in every mag_cols variant


def _i_mag_from_x(x_phot: torch.Tensor, use_colors: bool,
                  feat_mean: torch.Tensor = None,
                  feat_scale: torch.Tensor = None) -> torch.Tensor:
    """Extract (and optionally un-standardize) the i-band magnitude column from x_phot."""
    n_bands = x_phot.shape[1] // 4
    idx = 2 * (n_bands - 1) if use_colors else _I_BAND_IDX
    col = x_phot[:, idx]
    if feat_mean is not None:
        col = col * feat_scale[idx] + feat_mean[idx]
    return col


def _unstandardize_scaled(x_phot: torch.Tensor, feat_mean: torch.Tensor,
                          feat_scale: torch.Tensor) -> torch.Tensor:
    """Recover raw-mag-space values for x_phot's scaled portion (first 2*n_bands columns)."""
    if feat_mean is None or feat_scale is None:
        return x_phot
    n_scaled = feat_mean.shape[0]
    scaled   = x_phot[:, :n_scaled] * feat_scale + feat_mean
    return torch.cat([scaled, x_phot[:, n_scaled:]], dim=1)


def _resolve_feat_scaler(cfg: dict, n_in: int, use_colors: bool):
    """Build (feat_mean, feat_scale) for PhotozMLPVAE.load(), migrating older checkpoint formats."""
    if "feat_mean" in cfg and "feat_scale" in cfg:
        return cfg["feat_mean"], cfg["feat_scale"]
    n_bands  = n_in // 4
    n_scaled = 2 * n_bands
    feat_mean, feat_scale = [0.0] * n_scaled, [1.0] * n_scaled
    if "i_ref_mean" in cfg or "i_ref_std" in cfg:
        idx = 2 * (n_bands - 1) if use_colors else _I_BAND_IDX
        feat_mean[idx]  = cfg.get("i_ref_mean", 0.0)
        feat_scale[idx] = cfg.get("i_ref_std", 1.0)
    return feat_mean, feat_scale


_ZPRIOR_INIT = dict(z0=0.18, km=0.02, pm=0.18, alpha=2.77, beta=1.71)


def log_p_bpz(z: torch.Tensor, i_mag: torch.Tensor,
              alpha: torch.Tensor, beta: torch.Tensor,
              z0: torch.Tensor, km: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
    """BPZ-style (Benitez 2000) redshift-prior log-density, closed-form normalized.

    P(z|T,i) ∝ z^alpha * exp[-(z/z_m)^alpha],  z_m = z0 + km*(i-16) + pm*(i-16)_+^beta

    `alpha` is the ONE density shape parameter (same exponent in both the
    power-law prefactor and the exponential -- this is the actual Benitez
    2000 form, not a free-standing second exponent). `beta` never appears in
    the density itself; it only shapes how the turnover z_m bends with
    magnitude, via the z_m(i) formula above. Mixing `beta` into the
    exponential/normalization (as earlier versions of this function did) is
    NOT this prior -- it silently changes the density family and breaks the
    closed-form normalization if only some of the three `beta` occurrences
    below are swapped for `alpha`. See README.md's "z prior" section for the
    normalization derivation.
    """
    d  = i_mag - 16.0
    zm = (z0 + km * d + pm * d.clamp(min=0.0).pow(beta)).clamp(min=1e-3)
    return (
        torch.log(alpha) - (alpha + 1.0) * torch.log(zm)
        - torch.lgamma((alpha + 1.0) / alpha)
        + alpha * torch.log(z.clamp(min=1e-6))
        - (z / zm) ** alpha
    )


def fit_zprior(z_spec: np.ndarray, i_mag: np.ndarray,
               init: dict = None,
               faint_mag_threshold: float = 23.0,
               faint_z_threshold: float = 2.0,
               faint_weight: float = 20.0,
               alpha_min: float = 1.89) -> dict:

    from scipy.optimize import minimize
    from scipy.special import gammaln

    init = init or _ZPRIOR_INIT
    z_spec = np.asarray(z_spec, dtype=np.float64)
    i_mag  = np.asarray(i_mag, dtype=np.float64)
    valid  = np.isfinite(z_spec) & np.isfinite(i_mag) & (z_spec > 0) & (i_mag > -1.0)
    n_sentinel = int(((i_mag <= -1.0) & np.isfinite(i_mag)).sum())
    z_spec, i_mag = z_spec[valid], i_mag[valid]
    if z_spec.size == 0:
        raise ValueError("fit_zprior: no valid (z_spec>0, finite) galaxies to fit against")
    if n_sentinel:
        print(f"  fit_zprior: dropped {n_sentinel:,} row(s) with missing "
              f"(sentinel) i-band magnitude before fitting")

    boost_mask = (i_mag >= faint_mag_threshold) & (z_spec >= faint_z_threshold)
    weights = np.where(boost_mask, faint_weight, 1.0)
    print(f"  fit_zprior: up-weighting {int(boost_mask.sum()):,}/{z_spec.size:,} galaxies with "
          f"i_mag>={faint_mag_threshold:g} AND z_spec>={faint_z_threshold:g} by "
          f"{faint_weight:g}x (alpha floored at {alpha_min:g})")

    def _neg_log_lik(params):
        z0, km, pm, alpha, beta = params
        d      = i_mag - 16.0
        zm     = np.clip(z0 + km * d + pm * np.clip(d, 0.0, None) ** beta, 1e-3, None)
        log_p  = (np.log(alpha) - (alpha + 1.0) * np.log(zm)
                  - gammaln((alpha + 1.0) / alpha)
                  + alpha * np.log(z_spec) - (z_spec / zm) ** alpha)
        return -np.average(log_p, weights=weights)

    x0     = np.array([init["z0"], init["km"], init["pm"], init["alpha"], init["beta"]])
    bounds = [(1e-4, None), (0.0, None), (0.0, None), (alpha_min, None), (1e-2, None)]
    res    = minimize(_neg_log_lik, x0, method="L-BFGS-B", bounds=bounds)
    z0, km, pm, alpha, beta = res.x
    return dict(z0=float(z0), km=float(km), pm=float(pm),
               alpha=float(alpha), beta=float(beta))


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc point estimate from an already-computed p(z) grid
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_z_map(z_grid: torch.Tensor, mixture_pdf: torch.Tensor,
                   prior_pdf: torch.Tensor = None, sample_method: str = "normal",
                   gamma: float = 0.15):
    """MAP redshift + a local-curvature sigma from an already-evaluated
    p(z) grid (e.g. `predict_z_pdf()`'s `mixture_pdf`), optionally times a
    prior density on the same grid. Shared here (moved out of the
    `_fixedprior` recovery module, which keeps its own frozen copy) so any
    model version's `z_point()`/diagnostics can reuse one canonical
    point-estimate convention instead of reimplementing MAP+curvature
    logic per model file.

    sample_method="risk" picks the Bayes point estimate under the
    Lorentzian loss L(dz) = 1 - 1/(1+(dz/gamma)^2), i.e. the z_hat (swept
    over z_grid itself) minimizing the posterior-weighted risk
        R(z_hat) = integral p(z|x) * L(z_hat - z) dz
    evaluated over z_grid via trapz. Unlike the bare MAP, this is smooth
    in the posterior shape and down-weights (via the (dz/gamma)^2
    saturation) mass sitting in a far secondary mode, at the cost of one
    extra (n_grid, n_grid) trapz per batch; gamma=0.15 matches the usual
    |dz|>0.15 catastrophic-outlier tolerance used elsewhere in this repo's
    metrics. Its companion "z_std" output is NOT a sigma here but a
    BPZ-style confidence C(z_pred) = integral_{z_pred-0.03}^{z_pred+0.03}
    p(z|x) dz -- a probability in [0, 1] (higher = more confident), the
    opposite sense of every other sample_method's z_std.
    """
    z_grid      = torch.as_tensor(z_grid)
    mixture_pdf = torch.as_tensor(mixture_pdf, dtype=z_grid.dtype)
    posterior   = mixture_pdf if prior_pdf is None else mixture_pdf * torch.as_tensor(prior_pdf, dtype=z_grid.dtype)

    # Guard against a degenerate row (all-NaN/all-zero mixture_pdf or
    # prior_pdf for some galaxy -- e.g. a masked/corrupted entry, or an
    # extreme z_m/mu combination that underflows every grid point to 0 in
    # float32) BEFORE normalizing: trapz/argmax/multinomial all silently
    # misbehave or hard-crash (RuntimeError: invalid multinomial
    # distribution) on a row that sums to <=0. Falling back to uniform
    # over the grid for just that row is a safe "no information" default
    # that keeps the whole batch vectorized instead of crashing it.
    posterior = posterior.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    degenerate = posterior.sum(dim=-1, keepdim=True) <= 0
    if degenerate.any():
        posterior = torch.where(degenerate, torch.ones_like(posterior), posterior)

    # Product of 2 normalized densities isn't itself normalized; mixture_pdf
    # alone already is (predict_z_pdf's own construction), but renormalizing
    # unconditionally is harmless and keeps this correct either way.
    norm      = torch.trapz(posterior, z_grid, dim=-1).clamp(min=1e-300)
    posterior = posterior / norm.unsqueeze(-1)

    idx      = posterior.argmax(dim=-1)          # (batch,)  highest-probability grid point
    z_mode   = z_grid[idx]

    n_grid  = z_grid.shape[-1]
    h       = (z_grid[1] - z_grid[0]).clamp(min=1e-12)
    idx_lo  = (idx - 1).clamp(min=0)
    idx_hi  = (idx + 1).clamp(max=n_grid - 1)

    log_p     = torch.log(posterior.clamp(min=1e-300))
    batch_idx = torch.arange(posterior.shape[0], device=posterior.device)
    curvature = (log_p[batch_idx, idx_hi] - 2.0 * log_p[batch_idx, idx]
                + log_p[batch_idx, idx_lo]) / (h ** 2)

    var   = -1.0 / curvature.nan_to_num(nan=-1e-6).clamp(max=-1e-6)
    z_std = var.sqrt()

    if sample_method == "deterministic":
        return z_mode, z_std
    elif sample_method == "normal":
        # One vectorized sample per galaxy -- loc=z_mode/scale=z_std are both
        # (batch,) tensors, so this single .sample() call draws the entire
        # batch's worth of independent per-galaxy Normals jointly, not one
        # galaxy at a time.
        z_pred = torch.distributions.Normal(loc=z_mode, scale=z_std).sample()
    elif sample_method == "sample":
        # posterior is already finite/non-negative (guarded above), so no
        # need to re-clean it here.
        idx    = torch.multinomial(posterior, num_samples=100, replacement=False)  # (batch, 100) grid idx
        sample = z_grid[idx]                                                       # (batch, 100) z values
        z_std  = sample.std(dim=1)
        z_pred = sample.mean(dim=1)
        # z_pred = torch.median(sample, dim=1).values  # (batch,) median of 100 draws from the multinomial
        #DEPRECATED:no improvement z_pred = z_grid[torch.multinomial(posterior, num_samples=1, replacement=False) ].squeeze(dim=1)  # (batch,) single draw from the multinomial
    elif sample_method == "risk":
        # Bayes point estimate under the Lorentzian loss, minimized over
        # candidate z_hat swept across z_grid itself:
        #   R(z_hat) = integral_z p(z|x) * L(z_hat - z) dz
        #   L(dz)    = 1 - 1/(1+(dz/gamma)^2)
        #   z_pred   = argmin_{z_hat in z_grid} R(z_hat)
        # R(z_hat) is linear in `posterior` for fixed z_hat, so fold the
        # trapezoid quadrature weights into the loss kernel once and reduce
        # via a single (batch, n_grid) @ (n_grid, n_cand) matmul -- NOT an
        # elementwise (batch, n_cand, n_grid) broadcast (368,177 x 500 x 500
        # ~= 368GB in float32, OOMs immediately at real dataset scale).
        # Matmul keeps memory at O(n_grid^2) + O(batch*n_grid) instead.
        h      = torch.diff(z_grid)
        w      = torch.zeros_like(z_grid)
        w[:-1] += h / 2
        w[1:]  += h / 2                                                  # trapezoid weights, (n_grid,)
        dz     = z_grid.view(-1, 1) - z_grid.view(1, -1)                 # (n_cand, n_grid): z_hat - z
        loss_w = (1.0 - 1.0 / (1.0 + (dz / gamma) ** 2)) * w.view(1, -1) # (n_cand, n_grid), weights folded in
        R      = posterior @ loss_w.T                                    # (batch, n_grid) @ (n_grid, n_cand) -> (batch, n_cand)
        z_pred = z_grid[R.argmin(dim=-1)]                                 # (batch,)
        window    = 0.03
        in_window = (z_grid.unsqueeze(0) - z_pred.unsqueeze(-1)).abs() <= window  # (batch, n_grid)
        z_std     = torch.trapz(torch.where(in_window, posterior, torch.zeros_like(posterior)),
                                 z_grid, dim=-1)                                  # (batch,) confidence in [0, 1]
    else:
        raise ValueError(f"predict_z_map: unknown sample_method {sample_method!r} "
                         f"(expected 'deterministic', 'normal', 'sample', or 'risk')")

    return z_pred, z_std


class MLPVAEEncoder(nn.Module):
    """Shared ResBlock trunk feeding a dedicated z head and a z-conditioned SPS VAE head."""

    def __init__(self, n_in: int = 40, width: int = 512,
                 n_latent: int = N_PARAMS_15, dropout: float = 0.1,
                 detach_z_pred: bool = True, use_colors: bool = True,
                 zprior_params: dict = None,
                 feat_mean=None, feat_scale=None,
                 centered_z_scale: bool = False):
        super().__init__()
        self.n_latent = n_latent
        self.detach_z_pred = detach_z_pred
        self.use_colors = use_colors
        # See _SOFTPLUS_CENTER_SHIFT's docstring: off by default so existing
        # checkpoints (z_head bias calibrated against plain softplus) load
        # and behave unchanged; opt in for new/from-scratch runs.
        self.centered_z_scale = centered_z_scale
        n_scaled = 2 * (n_in // 4)
        feat_mean_t  = torch.zeros(n_scaled) if feat_mean  is None else torch.as_tensor(feat_mean,  dtype=torch.float32)
        feat_scale_t = torch.ones(n_scaled)  if feat_scale is None else torch.as_tensor(feat_scale, dtype=torch.float32)
        if feat_mean_t.shape != (n_scaled,) or feat_scale_t.shape != (n_scaled,):
            raise ValueError(
                f"feat_mean/feat_scale must have shape ({n_scaled},) to match "
                f"x_phot's scaled portion (n_in={n_in} -> n_bands={n_in // 4}); "
                f"got {tuple(feat_mean_t.shape)}/{tuple(feat_scale_t.shape)}")
        self.register_buffer("feat_mean", feat_mean_t)
        self.register_buffer("feat_scale", feat_scale_t)

        zprior_params = zprior_params or _ZPRIOR_INIT
        for name, val in zprior_params.items():
            self.register_buffer(f"zprior_{name}", torch.tensor(float(val), dtype=torch.float32))

        self.input_proj = nn.Sequential(
            nn.Linear(n_in, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock(width, dropout),
            _ResBlock(width, dropout),
        )
        self.z_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.vae_head = nn.Linear(width + 1, n_latent * 2)

        nn.init.xavier_uniform_(self.z_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.z_head[-1].bias)
        nn.init.xavier_uniform_(self.vae_head.weight, gain=0.1)
        nn.init.zeros_(self.vae_head.bias)

    def prior_params(self):
        """(alpha, beta, z0, km, pm) for the i-mag redshift prior, floor-clamped."""
        alpha = self.zprior_alpha.clamp(min=1e-3)
        beta  = self.zprior_beta.clamp(min=1e-3)
        z0    = self.zprior_z0.clamp(min=1e-4)
        km    = self.zprior_km.clamp(min=0.0)
        pm    = self.zprior_pm.clamp(min=0.0)
        return alpha, beta, z0, km, pm

    def z_m(self, i_mag: torch.Tensor) -> torch.Tensor:
        """Learned characteristic (turnover) redshift z_m(i)."""
        _, beta, z0, km, pm = self.prior_params()
        d  = i_mag - 16.0
        zm = z0 + km * d + pm * d.clamp(min=0.0).pow(beta)
        return zm.clamp(min=1e-3)

    def forward(self, x: torch.Tensor):
        """Returns z_pred, mu_z, log_var_z, mu_15, log_var_15, z_m."""
        h = self.input_proj(x)
        h = self.res_blocks(h)

        z_out     = self.z_head(h)
        mu_z      = z_out[:, 0].clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        log_var_z = z_out[:, 1].clamp(-7.0, 2.0)
        z_raw     = self.reparameterize(mu_z, log_var_z).clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        i_mag     = _i_mag_from_x(x, self.use_colors, self.feat_mean, self.feat_scale)
        z_m_i     = self.z_m(i_mag)
        z_pred    = z_m_i * _z_scale(z_raw, self.centered_z_scale)

        z_for_vae  = z_pred.detach() if self.detach_z_pred else z_pred
        cond       = torch.cat([h.detach(), z_for_vae.unsqueeze(1)], dim=1)
        vae_out    = self.vae_head(cond)
        mu_15      = vae_out[:, :self.n_latent].clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        log_var_15 = vae_out[:, self.n_latent:].clamp(-7.0, 2.0)

        return z_pred, mu_z, log_var_z, mu_15, log_var_15, z_m_i

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = mu + eps * exp(0.5 * log_var)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def freeze_vae_head(self):
        """Freeze the 15-param SPS head (stage 1 / z-supervised warmup)."""
        for p in self.vae_head.parameters():
            p.requires_grad_(False)

    def unfreeze_vae_head(self):
        """Unfreeze the 15-param SPS head (stage 2 / joint fine-tuning)."""
        for p in self.vae_head.parameters():
            p.requires_grad_(True)


class PhotozMLPVAE(nn.Module):
    """Photometric-redshift MLP-VAE: z gets its own supervised latent + MLP head."""

    def __init__(self,
                 speculator_dir: str,
                 filter_dir: str,
                 encoder_width: int = 512,
                 encoder_dropout: float = 0.1,
                 encoder_n_in: int = 40,
                 detach_z_pred: bool = True,
                 use_colors: bool = True,
                 double_precision: bool = False,
                 zprior_params: dict = None,
                 feat_mean=None,
                 feat_scale=None,
                 scaler=None,
                 centered_z_scale: bool = False):
        super().__init__()
        self.detach_z_pred = detach_z_pred
        self.use_colors = use_colors
        self._double_precision = double_precision

        if scaler is not None:
            feat_mean, feat_scale = scaler.mean_, scaler.scale_

        self.encoder = MLPVAEEncoder(
            n_in=encoder_n_in, width=encoder_width,
            n_latent=N_PARAMS_15, dropout=encoder_dropout,
            detach_z_pred=detach_z_pred, use_colors=use_colors,
            zprior_params=zprior_params,
            feat_mean=feat_mean, feat_scale=feat_scale,
            centered_z_scale=centered_z_scale,
        )

        self.speculator  = SpeculatorInoueIGM(speculator_dir)
        self.filter_conv = FilterConv(filter_dir, self.speculator.wl_rest)

        for p in self.speculator.parameters():
            p.requires_grad_(False)
        for p in self.filter_conv.parameters():
            p.requires_grad_(False)

        if double_precision:
            self.double()

    def encode(self, x_phot: torch.Tensor):
        """Returns z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15, z_m."""
        z_pred, mu_z, log_var_z, mu_15, log_var_15, z_m = self.encoder(x_phot)
        z_raw_15 = self.encoder.reparameterize(mu_15, log_var_15).clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        return z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15, z_m

    def decode(self, z_pred: torch.Tensor, z_raw_15: torch.Tensor) -> torch.Tensor:
        """Returns m_ab, (batch, 10) reconstructed AB magnitudes."""
        theta_15 = constrain_params_15(z_raw_15)
        zred_raw   = z_pred.detach() if self.detach_z_pred else z_pred
        theta_full = torch.cat([zred_raw.unsqueeze(1), theta_15], dim=1)

        log_spec, _ = self.speculator(theta_full)
        m_ab = self.filter_conv(log_spec,
                                theta_full[:, ZRED_IDX_FULL],
                                theta_full[:, LOGMASS_IDX_FULL])
        return m_ab

    def forward(self, x_phot: torch.Tensor):
        """Returns m_ab_recon, z_pred, mu_z, log_var_z, mu_15, log_var_15."""
        z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15, _ = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)
        return m_ab_recon, z_pred, mu_z, log_var_z, mu_15, log_var_15

    def encode_theta(self, x: torch.Tensor):
        """Standard diagnostic interface: returns z_pred, log_sigma_z, theta_full."""
        _, mu_z, log_var_z, mu_15, _, z_m = self.encoder(x)
        centered    = self.encoder.centered_z_scale
        z_pred      = z_m * _z_scale(mu_z, centered)
        log_sigma_z = torch.log(_zred_sigma_phys(mu_z, log_var_z, z_m, centered))
        theta_15    = constrain_params_15(mu_15)
        theta_full  = torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)
        return z_pred, log_sigma_z, theta_full

    @torch.no_grad()
    def sample_theta_posterior(self, x, n_samples: int, device: str, rng):
        """Draw posterior samples per galaxy; returns (N_gal, n_samples, 16) numpy float32."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(device=device, dtype=next(self.parameters()).dtype)
        _, mu_z, log_var_z, mu_15, log_var_15, z_m = self.encoder(x)
        mu_z_np   = mu_z.cpu().numpy()
        std_z_np  = np.exp(0.5 * log_var_z.cpu().numpy())
        z_m_np    = z_m.cpu().numpy()
        mu_15_np  = mu_15.cpu().numpy()
        std_15_np = np.exp(0.5 * log_var_15.cpu().numpy())

        n_gal = x.shape[0]
        all_samples = np.empty((n_gal, n_samples, N_PARAMS_FULL), dtype=np.float32)
        for i in range(n_gal):
            eps_z   = rng.standard_normal(n_samples).astype(np.float32)
            z_raw_s = mu_z_np[i] + eps_z * std_z_np[i]
            z_s     = z_m_np[i] * _z_scale_np(z_raw_s, self.encoder.centered_z_scale)
            z_s     = np.clip(z_s, 0.0, None)

            eps_15   = rng.standard_normal((n_samples, 15)).astype(np.float32)
            z_raw_15 = mu_15_np[i] + eps_15 * std_15_np[i]
            with torch.no_grad():
                theta_15 = constrain_params_15(
                    torch.from_numpy(z_raw_15).to(device)
                ).cpu().numpy()
            all_samples[i] = np.concatenate([z_s[:, None], theta_15], axis=1)
        return all_samples

    def loss(self,
             x_phot:    torch.Tensor,
             z_spec:    torch.Tensor,
             lam_z:     float = 20.0,
             lam_r:     float = 1.0,
             beta:      float = 0.0,
             lam_prior: float = 1.0,
             sigma_floor: float = 0.3,
             z_weights: torch.Tensor = None,
             use_nll_z: bool = True,
             decouple_z_nll: bool = False) -> dict:
        """Compute the 3-term MLPVAE loss (z-NLL + i-mag prior, reconstruction, KL)."""
        z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15, z_m = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)

        sq_err  = (z_pred - z_spec) ** 2
        sigma_z = _zred_sigma_phys(mu_z, log_var_z, z_m, self.encoder.centered_z_scale)
        if use_nll_z:
            inv_var = 1.0 / (sigma_z ** 2)
            if decouple_z_nll:
                calibration = 0.5 * sq_err.detach() * inv_var + torch.log(sigma_z)
                per_gal     = sq_err + calibration
            else:
                nll     = 0.5 * sq_err * inv_var + torch.log(sigma_z)
                per_gal = nll
        else:
            per_gal = sq_err

        alpha, beta_prior, z0, km, pm = self.encoder.prior_params()
        i_mag = _i_mag_from_x(x_phot, self.use_colors, self.encoder.feat_mean, self.encoder.feat_scale)
        log_prior = log_p_bpz(z_pred, i_mag, alpha, beta_prior, z0, km, pm)
        per_gal   = per_gal - lam_prior * log_prior

        if z_weights is not None:
            loss_z = (z_weights * per_gal).mean()
        else:
            loss_z = per_gal.mean()

        n_bands, n_c = x_phot.shape[1] // 4, x_phot.shape[1] // 4 - 1
        x_real = _unstandardize_scaled(x_phot, self.encoder.feat_mean, self.encoder.feat_scale)
        if self.use_colors:
            target     = torch.cat([x_real[:, :n_c], x_real[:, 2 * n_c:2 * n_c + 1]], dim=1)
            target_err = torch.cat([x_real[:, n_c:2 * n_c], x_real[:, 2 * n_c + 1:2 * n_c + 2]], dim=1)
            valid      = 1.0 - torch.cat([x_real[:, 2 * n_c + 2:3 * n_c + 2],
                                          x_real[:, 4 * n_c + 2:4 * n_c + 3]], dim=1)
            recon      = torch.cat([m_ab_recon[:, :n_c] - m_ab_recon[:, 1:n_bands],
                                    m_ab_recon[:, _I_BAND_IDX:_I_BAND_IDX + 1]], dim=1)
        else:
            target, target_err = x_real[:, :n_bands], x_real[:, n_bands:2 * n_bands]
            valid  = 1.0 - x_real[:, 2 * n_bands:3 * n_bands]
            recon  = m_ab_recon[:, :n_bands]

        _zero = torch.zeros(1, device=x_phot.device).squeeze()
        if lam_r > 0.0:
            sig2    = target_err ** 2 + sigma_floor ** 2
            chi2    = valid * (recon - target) ** 2 / sig2
            n_valid = valid.sum(dim=1).clamp(min=1.0)
            loss_r  = (chi2.sum(dim=1) / n_valid).mean()
            loss_r  = torch.nan_to_num(loss_r, nan=0.0, posinf=0.0)
        else:
            loss_r = _zero

        if beta > 0.0:
            loss_kl = 0.5 * (
                log_var_15.exp() + mu_15 ** 2 - 1.0 - log_var_15
            ).sum(dim=1).mean()
        else:
            loss_kl = _zero

        total = lam_z * loss_z + lam_r * loss_r + beta * loss_kl

        return {
            "total":       total,
            "z_sup":       loss_z,
            "recon":       loss_r,
            "kl":          loss_kl,
            "log_sigma_z": torch.log(sigma_z).mean().detach(),
            "z_m_mean":    z_m.mean().detach(),
        }

    @torch.no_grad()
    def predict_z(self, x_phot: torch.Tensor, n_samples: int = 500):
        """Returns z_samples, z_mean, z_std from the z-head posterior."""
        _, mu_z, log_var_z, _, _, z_m = self.encoder(x_phot)
        centered = self.encoder.centered_z_scale
        z_mean = z_m * _z_scale(mu_z, centered)
        z_std  = _zred_sigma_phys(mu_z, log_var_z, z_m, centered)

        std     = torch.exp(0.5 * log_var_z)
        z_raw_s = mu_z.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
            mu_z.shape[0], n_samples, device=mu_z.device)
        z_samples = (z_m.unsqueeze(1) * _z_scale(z_raw_s, centered)).clamp(min=0.0)

        return z_samples, z_mean, z_std

    @torch.no_grad()
    def predict_params(self, x_phot: torch.Tensor) -> torch.Tensor:
        """Returns theta_full, (batch, 16) posterior means for all physical SPS parameters."""
        _, mu_z, _, mu_15, _, z_m = self.encoder(x_phot)
        z_mean     = z_m * _z_scale(mu_z, self.encoder.centered_z_scale)
        theta_15   = constrain_params_15(mu_15)
        theta_full = torch.cat([z_mean.unsqueeze(1), theta_15], dim=1)
        return theta_full

    def save(self, path: str, scaler=None, col_medians=None):
        """Save model weights + optional preprocessor state."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        payload = {
            "model_state": self.state_dict(),
            "encoder_config": {
                "n_in":          self.encoder.input_proj[0].in_features,
                "width":         self.encoder.input_proj[0].out_features,
                "n_latent":      self.encoder.n_latent,
                "detach_z_pred": self.detach_z_pred,
                "use_colors":    self.use_colors,
                "double_precision": self._double_precision,
                "centered_z_scale": self.encoder.centered_z_scale,
                "zprior_params": {
                    "z0": self.encoder.zprior_z0.item(),
                    "km": self.encoder.zprior_km.item(),
                    "pm": self.encoder.zprior_pm.item(),
                    "alpha": self.encoder.zprior_alpha.item(),
                    "beta": self.encoder.zprior_beta.item(),
                },
                "feat_mean":  self.encoder.feat_mean.tolist(),
                "feat_scale": self.encoder.feat_scale.tolist(),
            },
            "scaler":      scaler,
            "col_medians": col_medians,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, speculator_dir: str, filter_dir: str,
             device: str = "cpu"):
        """Load model from a saved checkpoint, migrating older config formats."""
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg     = payload["encoder_config"]
        n_in       = cfg.get("n_in", 40)
        use_colors = cfg.get("use_colors", True)
        feat_mean, feat_scale = _resolve_feat_scaler(cfg, n_in, use_colors)
        model   = cls(speculator_dir, filter_dir,
                      encoder_width=cfg["width"],
                      encoder_n_in=n_in,
                      detach_z_pred=cfg.get("detach_z_pred", True),
                      use_colors=use_colors,
                      double_precision=cfg.get("double_precision", False),
                      zprior_params=cfg.get("zprior_params"),
                      feat_mean=feat_mean, feat_scale=feat_scale,
                      centered_z_scale=cfg.get("centered_z_scale", False))

        state = payload["model_state"]
        n_fixed = 0
        for name, saved in state.items():
            mod_path, _, attr = name.rpartition(".")
            try:
                mod = model.get_submodule(mod_path)
            except AttributeError:
                continue
            if attr in mod._buffers and mod._buffers[attr] is not None \
                    and mod._buffers[attr].shape != saved.shape:
                mod._buffers[attr] = saved.clone().to(mod._buffers[attr].device)
                n_fixed += 1
        if n_fixed:
            print(f"NOTE: {n_fixed} frozen-decoder buffers on disk "
                  f"({speculator_dir}) have different shapes than this "
                  f"checkpoint was trained with -- using the checkpoint's "
                  f"own decoder buffers.")

        missing, unexpected = model.load_state_dict(state, strict=False)
        zprior_missing = [k for k in missing if "zprior_" in k]
        feat_migrated  = [k for k in missing    if k.endswith((".feat_mean", ".feat_scale"))]
        feat_old       = [k for k in unexpected if k.endswith((".i_ref_mean", ".i_ref_std"))]
        other_missing  = [k for k in missing    if "zprior_" not in k and k not in feat_migrated]
        unexpected     = [k for k in unexpected if k not in feat_old]
        if zprior_missing:
            print(f"NOTE: checkpoint predates the learned i-mag redshift "
                  f"prior -- {len(zprior_missing)} zprior_* buffers left at "
                  f"their _ZPRIOR_INIT default (this checkpoint was never "
                  f"trained with a zprior_params= fit).")
        if feat_migrated or feat_old:
            print(f"NOTE: checkpoint predates the full feat_mean/feat_scale "
                  f"un-standardization vectors (2026-08-26) -- migrated from "
                  f"its old scalar i_ref_mean/i_ref_std (or defaulted to "
                  f"raw/identity if it had neither), see _resolve_feat_scaler.")
        if other_missing or unexpected:
            import warnings
            warnings.warn(
                f"PhotozMLPVAE.load: {len(other_missing)} unexpected-missing "
                f"keys, {len(unexpected)} unexpected keys (architecture "
                f"mismatch beyond the known zprior_* case -- partial load; "
                f"missing={other_missing}, unexpected={unexpected})."
            )
        model.to(device)
        model.eval()
        return model, payload.get("scaler"), payload.get("col_medians")
