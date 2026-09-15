"""
photoz_mlpvae_mdn.py
====================
PhotozMLPVAE with a mixture-density (MDN) redshift head: the 1-dim Gaussian
z latent is replaced by a K-component Gaussian mixture so degenerate
photometry (e.g. Balmer/Lyman-break confusion) can keep multiple plausible
redshift modes instead of collapsing to one. See README.md for architecture
and gradient-flow details shared with photoz_mlpvae.py; PLAN.md for design
history.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from photoz_vae.model.vae_encoder import _ResBlock
from photoz_vae.model.speculator_torch import SpeculatorInoueIGM
from photoz_vae.model.filter_conv import FilterConv

from photoz_mlpvae.model.photoz_mlpvae import (
    PARAM_NAMES, PARAM_NAMES_15, N_PARAMS_15, N_PARAMS_FULL,
    ZRED_IDX_FULL, LOGMASS_IDX_FULL,
    _ZRED_SIGMA_FLOOR, _LATENT_CLAMP,
    constrain_params_15, _I_BAND_IDX,
    _i_mag_from_x, _unstandardize_scaled, _resolve_feat_scaler,
    _ZPRIOR_INIT, fit_zprior, log_p_bpz, predict_z_map)


_LOG_VAR_CLAMP    = (-8.0, 4.0)
_LOG_VAR_15_CLAMP = (-6.0, 4.0)

# ratio = z/z_m inside _mixture_component_log_pdf(). raw = ratio +
# log1p(-exp(-ratio)) itself stays finite for any ratio (that's the whole
# point of the log1p form), but d(raw)/d(ratio) ~ 1/ratio blows up as
# ratio->0 (near the old min=1e-8 floor that gave gradients ~1e8 from this
# term ALONE, before any real signal) and raw's own magnitude grows
# unbounded as ratio->inf, which can push (raw-mu)/sigma_raw to the
# 1e2-1e3 range against an untrained head's near-zero mu/tiny sigma_raw --
# exactly what produced 2026-09's v14 epoch-1 NaN (gnorm 349->1287 then
# NaN) under the new exact closed-form z-NLL (every galaxy hits this
# every step now, not just whichever component got sampled/picked, so a
# single degenerate galaxy in a batch is enough). 100.0 is well above any
# realistic z/z_m (max real data: z<=8.5 over z_m>=~0.086-0.1 at the
# bright end, so ratio<=~99) so genuine high-z/low-z_m galaxies keep
# their real gradient untouched -- deliberately loose, not tight: see
# PLAN.md's decoupled-NLL revert (attenuating genuine outlier signal here
# made σ_NMAD 4x worse) for why this only clamps the numerically
# degenerate tail, not real data.
_MDN_RATIO_CLAMP = (1e-4, 100.0)


_ZRED_SIGMA_FLOOR_FRAC = 0.05


def _component_sigma_phys(mu_k: torch.Tensor, log_var_k: torch.Tensor,
                           z_m: torch.Tensor,
                           sigma_floor: float = _ZRED_SIGMA_FLOOR,
                           sigma_floor_frac: float = _ZRED_SIGMA_FLOOR_FRAC) -> torch.Tensor:
    """Delta-method physical sigma per mixture component (any trailing shape)."""
    jac = z_m * torch.sigmoid(mu_k)
    sigma_raw = torch.exp(0.5 * log_var_k)
    sigma_phys = sigma_raw * jac
    eff_floor  = torch.clamp(sigma_floor_frac * z_m, min=sigma_floor)
    return sigma_phys.clamp(min=eff_floor)


def _mixture_component_log_pdf(z: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor,
                               z_m: torch.Tensor,
                               sigma_floor: float = None,
                               sigma_floor_frac: float = None) -> torch.Tensor:
    """Per-component log-density in z-space under z = z_m * softplus(raw),
    raw ~ N(mu, exp(log_var)) -- the exact (Jacobian-corrected) change of
    variables `predict_z_pdf()` plots, factored out here so the training
    loss, `z_point()`'s posterior sampling, and `predict_z_pdf()`'s
    diagnostics all evaluate the SAME density and can't silently drift out
    of sync (see PLAN.md's Failure 19 zprior-density-bug for what happens
    when a density formula gets reimplemented twice with a subtle
    mismatch).

    `z`/`mu`/`log_var`/`z_m` broadcast together -- callers use this two
    ways: (batch,1) vs (batch,K) [z_spec against every component, `loss()`]
    or (1,1,G) vs (batch,K,1) [a shared grid, `z_point()`/`predict_z_pdf()`].

    If `sigma_floor` is given, the RAW-space sigma is floored so the
    IMPLIED PHYSICAL sigma AT EACH COMPONENT'S OWN MEAN (mu, not the query
    z) never drops below `sigma_floor` -- reuses `_component_sigma_phys`'s
    existing floor exactly, just feeding its result into this density
    instead of `_component_sigma_phys`'s own local single-component
    return. Guards against the same 2026-08-17 MDN-overconfidence-NLL-spike
    failure mode this floor was originally added to fix, now that every
    component (not just whichever one got sampled) contributes to the
    loss every step.
    """
    # Both ends of _MDN_RATIO_CLAMP guard a distinct failure mode, not just
    # "ratio must be positive": d(raw)/d(ratio) ~ 1/ratio explodes as
    # ratio->0 (the old min=1e-8 alone let this term's OWN derivative hit
    # ~1e8 before any real signal), and raw's magnitude itself grows
    # unboundedly as ratio->inf, which can push (raw-mu)/sigma_raw into
    # the 1e2-1e3 range against an untrained head's near-zero mu/tiny
    # sigma_raw. Both ends stay far outside any realistic z/z_m (see
    # _MDN_RATIO_CLAMP's own comment) so real data's gradient is
    # untouched; this only clamps the numerically degenerate tail.
    ratio = (z / z_m).clamp(min=_MDN_RATIO_CLAMP[0], max=_MDN_RATIO_CLAMP[1])
    # log(expm1(ratio)) = ratio + log1p(-exp(-ratio)) -- an exact algebraic
    # identity (expm1(x) = exp(x)*(1-exp(-x))), but this form never
    # computes exp(+ratio), so it can't overflow for large ratio (e.g. a
    # degenerate/near-zero z_m early in training, or z_spec far outside
    # z_m's current scale) the way the naive torch.log(torch.expm1(ratio))
    # does (found via this session's smoke test: NaN loss from exactly
    # this on a synthetic z_m~1e-3 batch). Correctly reduces to log(ratio)
    # as ratio->0, same as the naive form there.
    raw   = ratio + torch.log1p(-torch.exp(-ratio))
    sigma_raw = torch.exp(0.5 * log_var)
    if sigma_floor is not None:
        sigma_phys_floored = _component_sigma_phys(mu, log_var, z_m, sigma_floor, sigma_floor_frac)
        jac_at_mu = (z_m * torch.sigmoid(mu)).clamp(min=1e-8)
        sigma_raw = torch.maximum(sigma_raw, sigma_phys_floored / jac_at_mu)
    log_pdf_raw = (-0.5 * ((raw - mu) / sigma_raw) ** 2
                   - torch.log(sigma_raw) - 0.5 * math.log(2.0 * math.pi))
    jac     = (z_m * torch.sigmoid(raw)).clamp(min=1e-12)
    log_jac = torch.log(jac)
    return log_pdf_raw - log_jac


class MLPVAEEncoderMDN(nn.Module):
    """Shared ResBlock trunk feeding an MDN z head and the z-conditioned SPS head."""

    def __init__(self, n_in: int = 40, width: int = 512,
                 n_latent: int = N_PARAMS_15, n_mix: int = 3,
                 dropout: float = 0.1,
                 zred_sigma_floor: float = _ZRED_SIGMA_FLOOR,
                 zred_sigma_floor_frac: float = _ZRED_SIGMA_FLOOR_FRAC,
                 detach_trunk_for_vae: bool = True,
                 use_colors: bool = True,
                 use_z_prior_sampling: bool = False,
                 z_sample_method: str = "normal",
                 zprior_params: dict = None,
                 feat_mean=None, feat_scale=None):
        super().__init__()
        self.n_latent = n_latent
        self.n_mix    = n_mix
        self.zred_sigma_floor = zred_sigma_floor
        self.zred_sigma_floor_frac = zred_sigma_floor_frac
        self.detach_trunk_for_vae = detach_trunk_for_vae
        self.use_colors = use_colors
        # z_point()'s posterior sample/point-estimate: whether the learned
        # i-mag prior p(z|i) (log_p_bpz) is folded into the mixture
        # (posterior ∝ likelihood × prior) before predict_z_map() picks a
        # point/draws a sample, or the raw MDN likelihood alone is used.
        # Default off -- matches z_point()'s pre-redesign behavior, which
        # never touched the prior.
        self.use_z_prior_sampling = use_z_prior_sampling
        # Which of predict_z_map()'s stochastic modes z_point() draws
        # during TRAINING (eval always uses "deterministic", the MAP --
        # unchanged reporting convention): "normal" approximates the
        # posterior as a single Gaussian around its MAP (fast, but loses
        # multimodality); "sample" draws directly from the discretized
        # posterior over all K components (genuinely multimodal, no
        # single-mode assumption); "risk" is deterministic even at train
        # time (no actual sampling) -- the Bayes point estimate minimizing
        # posterior-weighted Lorentzian risk (gamma=0.15, matching the
        # repo's |dz|>0.15 outlier tolerance), which down-weights mass in
        # a far secondary mode instead of committing to whichever mode the
        # bare MAP happens to land on. See predict_z_map()'s own docstring
        # for the exact formula and for why TorchPosterior couldn't be
        # used for "normal"/"sample" directly.
        if z_sample_method not in ("normal", "sample", "risk"):
            raise ValueError(f"z_sample_method must be 'normal', 'sample', or 'risk', got {z_sample_method!r}")
        self.z_sample_method = z_sample_method
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
            nn.Linear(128, 3 * n_mix),
        )
        self.vae_head = nn.Linear(width + 1, n_latent * 2)
        # Dedicated catastrophic-outlier classifier head -- see loss()'s
        # lam_outlier/predict_outlier_prob() docstrings. Always reads
        # h.detach() (not gated by detach_trunk_for_vae -- unlike vae_head,
        # there's no hypothesis under test for letting this one's gradient
        # reach the shared trunk, and every gradflow-to-trunk A/B test this
        # project has run for other auxiliary heads came back negative), so
        # it's a pure probe: training it can't perturb the z_point/mixture
        # training this file's loss() otherwise optimizes.
        self.outlier_head = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        nn.init.xavier_uniform_(self.z_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.z_head[-1].bias)
        with torch.no_grad():
            spread = torch.linspace(-1.5, 1.5, n_mix)
            self.z_head[-1].bias[n_mix:2 * n_mix] = spread
        nn.init.xavier_uniform_(self.vae_head.weight, gain=0.1)
        nn.init.zeros_(self.vae_head.bias)
        nn.init.xavier_uniform_(self.outlier_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.outlier_head[-1].bias)

    def z_mixture(self, h: torch.Tensor):
        """Trunk features -> (log_pi, mu, log_var), each (batch, K)."""
        out = self.z_head(h)
        K = self.n_mix
        logits   = out[:, :K]
        mu       = out[:, K:2 * K].clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        log_var  = out[:, 2 * K:].clamp(*_LOG_VAR_CLAMP)
        log_pi   = torch.log_softmax(logits, dim=1)
        return log_pi, mu, log_var

    def prior_params(self):
        """(alpha, beta, z0, km, pm) for the i-mag redshift prior, floor-clamped."""
        alpha = self.zprior_alpha.clamp(min=1e-3)
        beta  = self.zprior_beta.clamp(min=1e-3)
        z0    = self.zprior_z0.clamp(min=1e-4)
        km    = self.zprior_km.clamp(min=0.0)
        pm    = self.zprior_pm.clamp(min=0.0)
        return alpha, beta, z0, km, pm

    def z_m(self, i_mag: torch.Tensor) -> torch.Tensor:
        """Characteristic (turnover) redshift z_m(i)."""
        _, beta, z0, km, pm = self.prior_params()
        d  = i_mag - 16.0
        zm = z0 + km * d + pm * d.clamp(min=0.0).pow(beta)
        return zm.clamp(min=1e-3)

    def z_point(self, log_pi, mu, log_var, z_m_i, i_mag=None,
                n_grid: int = 500, z_max: float = 8.5):
        z_grid = torch.linspace(1e-3, z_max, n_grid, device=mu.device, dtype=mu.dtype)

        log_comp = _mixture_component_log_pdf(
            z_grid.view(1, 1, -1), mu.unsqueeze(-1), log_var.unsqueeze(-1),
            z_m_i.view(-1, 1, 1), self.zred_sigma_floor, self.zred_sigma_floor_frac)  # (batch, K, G)
        weights     = log_pi.exp()
        mixture_pdf = (weights.unsqueeze(-1) * log_comp.exp()).sum(dim=1)             # (batch, G)

        prior_pdf = None
        if self.use_z_prior_sampling:
            if i_mag is None:
                raise ValueError("z_point(): use_z_prior_sampling=True needs i_mag")
            alpha, beta, z0, km, pm = self.prior_params()
            prior_pdf = log_p_bpz(z_grid.view(1, -1), i_mag.view(-1, 1),
                                  alpha, beta, z0, km, pm).exp()

        method = self.z_sample_method if self.training else "deterministic"
        z_pred, sigma_z = predict_z_map(z_grid, mixture_pdf, prior_pdf, sample_method=method)
        z_pred = z_pred.clamp(min=1e-3)

        mu_k    = (weights * mu).sum(dim=1)
        log_v_k = (weights * log_var).sum(dim=1)
        return z_pred, sigma_z, mu_k, log_v_k

    def forward(self, x: torch.Tensor):
        """Returns z_pred, sigma_z, log_pi, mu_mix, log_var_mix, mu_15, log_var_15, z_m, mu_k, log_v_k, outlier_logit."""
        h = self.input_proj(x)
        h = self.res_blocks(h)

        log_pi, mu_mix, log_var_mix = self.z_mixture(h)
        i_mag = _i_mag_from_x(x, self.use_colors, self.feat_mean, self.feat_scale)
        z_m_i = self.z_m(i_mag)
        z_pred, sigma_z, mu_k, log_v_k = self.z_point(log_pi, mu_mix, log_var_mix, z_m_i, i_mag)

        h_for_vae  = h.detach() if self.detach_trunk_for_vae else h
        cond       = torch.cat([h_for_vae, z_pred.detach().unsqueeze(1)], dim=1)
        vae_out    = self.vae_head(cond)
        mu_15      = vae_out[:, :self.n_latent].clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        log_var_15 = vae_out[:, self.n_latent:].clamp(*_LOG_VAR_15_CLAMP)

        outlier_logit = self.outlier_head(h.detach()).squeeze(-1)

        return (z_pred, sigma_z, log_pi, mu_mix, log_var_mix, mu_15, log_var_15,
                z_m_i, mu_k, log_v_k, outlier_logit)

    def reparameterize(self, mu, log_var):
        """Reparameterisation trick: z = mu + eps * exp(0.5 * log_var)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
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
    """MDN-z variant of the z-separated PhotozMLPVAE. Same external API."""

    def __init__(self,
                 speculator_dir: str,
                 filter_dir: str,
                 encoder_width: int = 512,
                 encoder_dropout: float = 0.1,
                 encoder_n_in: int = 40,
                 n_mix: int = 3,
                 zred_sigma_floor: float = _ZRED_SIGMA_FLOOR,
                 zred_sigma_floor_frac: float = _ZRED_SIGMA_FLOOR_FRAC,
                 detach_trunk_for_vae: bool = True,
                 use_colors: bool = True,
                 use_z_prior_sampling: bool = False,
                 z_sample_method: str = "normal",
                 double_precision: bool = False,
                 zprior_params: dict = None,
                 feat_mean=None,
                 feat_scale=None,
                 scaler=None):
        super().__init__()
        self.use_colors = use_colors
        if scaler is not None:
            feat_mean, feat_scale = scaler.mean_, scaler.scale_
        self._double_precision = double_precision

        self.encoder = MLPVAEEncoderMDN(
            n_in=encoder_n_in, width=encoder_width,
            n_latent=N_PARAMS_15, n_mix=n_mix, dropout=encoder_dropout,
            zred_sigma_floor=zred_sigma_floor,
            zred_sigma_floor_frac=zred_sigma_floor_frac,
            detach_trunk_for_vae=detach_trunk_for_vae,
            use_colors=use_colors,
            use_z_prior_sampling=use_z_prior_sampling,
            z_sample_method=z_sample_method,
            zprior_params=zprior_params,
            feat_mean=feat_mean,
            feat_scale=feat_scale,
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
        """Returns z_pred, sigma_z, log_pi, mu_mix, log_var_mix, mu_15, log_var_15, z_raw_15, z_m, mu_k, log_v_k, outlier_logit."""
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_m, mu_k, log_v_k, outlier_logit) = self.encoder(x_phot)
        z_raw_15 = self.encoder.reparameterize(mu_15, log_var_15) \
            .clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        return (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
                mu_15, log_var_15, z_raw_15, z_m, mu_k, log_v_k, outlier_logit)

    def decode(self, z_pred: torch.Tensor, z_raw_15: torch.Tensor) -> torch.Tensor:
        """Returns m_ab, (batch, 10) reconstructed AB magnitudes."""
        theta_15   = constrain_params_15(z_raw_15)
        zred_dec   = z_pred.detach()
        theta_full = torch.cat([zred_dec.unsqueeze(1), theta_15], dim=1)
        log_spec, _ = self.speculator(theta_full)
        return self.filter_conv(log_spec,
                                theta_full[:, ZRED_IDX_FULL],
                                theta_full[:, LOGMASS_IDX_FULL])

    def forward(self, x_phot: torch.Tensor):
        """Returns m_ab_recon, z_pred, sigma_z, mu_15, log_var_15."""
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_raw_15, _z_m, _mu_k, _log_v_k, _outlier_logit) = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)
        return m_ab_recon, z_pred, sigma_z, mu_15, log_var_15

    def encode_theta(self, x: torch.Tensor):
        """Standard diagnostic interface: returns z_pred, log_sigma_z, theta_full."""
        (z_pred, sigma_z, _, _, _, mu_15, _, _, _, _, _) = self.encoder(x)
        log_sigma_z = torch.log(sigma_z)
        theta_15    = constrain_params_15(mu_15)
        theta_full  = torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)
        return z_pred, log_sigma_z, theta_full

    @torch.no_grad()
    def sample_theta_posterior(self, x, n_samples: int, device: str, rng):
        """Draw posterior samples per galaxy (z from the full mixture); returns (N_gal, n_samples, 16) numpy float32."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(device=device, dtype=next(self.parameters()).dtype)
        (_, _, log_pi, mu_mix, log_var_mix, mu_15, log_var_15, z_m, _, _, _) = self.encoder(x)
        pi_np     = log_pi.exp().cpu().numpy()
        mu_np     = mu_mix.cpu().numpy()
        std_np    = np.exp(0.5 * log_var_mix.cpu().numpy())
        mu_15_np  = mu_15.cpu().numpy()
        std_15_np = np.exp(0.5 * log_var_15.cpu().numpy())
        z_m_np    = z_m.cpu().numpy()

        n_gal = x.shape[0]
        all_samples = np.empty((n_gal, n_samples, N_PARAMS_FULL), dtype=np.float32)
        for i in range(n_gal):
            k       = rng.choice(self.encoder.n_mix, size=n_samples, p=pi_np[i] / pi_np[i].sum())
            eps_z   = rng.standard_normal(n_samples).astype(np.float32)
            z_raw_s = mu_np[i, k] + eps_z * std_np[i, k]
            z_s     = z_m_np[i] * np.logaddexp(0.0, z_raw_s)

            eps_15   = rng.standard_normal((n_samples, N_PARAMS_15)).astype(np.float32)
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
             lam_outlier: float = 0.0,
             outlier_threshold: float = 0.15,
             outlier_pos_weight: float = None) -> dict:
        """Compute the 3-term MDN loss (closed-form mixture z-NLL + i-mag
        prior, reconstruction, KL), plus an optional 4th BCE term for the
        dedicated outlier_head.

        z-supervision (`use_nll_z=True`, the default) is now the EXACT
        closed-form mixture log-likelihood -log[Σ_k π_k · p_k(z_spec)],
        via `_mixture_component_log_pdf` (the same density `z_point()`/
        `predict_z_pdf()` use) -- NOT the old sampled-/picked-component
        NLL. Every component gets an exact analytic gradient every step,
        proportional to its own posterior responsibility (softmax over
        `log_pi + log p_k(z_spec)`), including `log_pi` itself. See
        `z_point()`'s docstring for why the earlier per-step component
        pick (`Categorical.sample()`, then a Gumbel-Softmax hard pick)
        left `log_pi` stuck at a uniform 1/K and never fixed it.

        `lam_outlier` (default 0.0, off): weight on outlier_head's BCE loss
        against the online label `|z_pred.detach() - z_spec|/(1+z_spec) >
        outlier_threshold`. The label is defined relative to THIS model's
        own current z_pred, not a fixed external truth, so it's noisy in
        early epochs and sharpens as z_pred converges -- same bootstrapping
        character as any auxiliary head trained jointly with the task it
        reads its label from. outlier_head only ever sees h.detach() (see
        its definition), so this term's gradient can never reach the shared
        trunk/z_head/mixture regardless of lam_outlier's value.
        """
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_raw_15, z_m, mu_k, log_v_k, outlier_logit) = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)

        if use_nll_z:
            log_comp = _mixture_component_log_pdf(
                z_spec.unsqueeze(1), mu_mix, log_var_mix, z_m.unsqueeze(1),
                self.encoder.zred_sigma_floor, self.encoder.zred_sigma_floor_frac)  # (batch, K)
            log_mix  = torch.logsumexp(log_pi + log_comp, dim=1)                    # (batch,)
            per_gal  = -log_mix
        else:
            per_gal = (z_pred - z_spec) ** 2

        # NOTE: z_pred now comes from predict_z_map() (@torch.no_grad() --
        # see z_point()), so this log_p_bpz(z_pred, ...) term carries no
        # gradient any more regardless of lam_prior; z-supervision's own
        # gradient to zprior_* flows through z_m (inside log_comp) instead.
        # Kept for the loss dict's reported value / lam_prior=0 no-ops
        # unchanged; revisit if lam_prior>0 is ever reused deliberately.
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

        if lam_outlier > 0.0:
            # z_pred.detach(): this label is the online, evolving-as-z_pred-
            # improves target described in the docstring above, not a fixed
            # external one -- detaching just keeps it from being a second,
            # redundant gradient path into z_pred (loss_z already supervises
            # z_pred's mixture directly).
            dz_now        = (z_pred.detach() - z_spec) / (1.0 + z_spec)
            outlier_label = (dz_now.abs() > outlier_threshold).to(outlier_logit.dtype)
            pos_weight_t  = (torch.as_tensor(outlier_pos_weight, device=x_phot.device,
                                              dtype=outlier_logit.dtype)
                              if outlier_pos_weight is not None else None)
            loss_outlier = F.binary_cross_entropy_with_logits(
                outlier_logit, outlier_label, pos_weight=pos_weight_t)
        else:
            loss_outlier = _zero

        total = lam_z * loss_z + lam_r * loss_r + beta * loss_kl + lam_outlier * loss_outlier

        return {
            "total":       total,
            "z_sup":       loss_z,
            "recon":       loss_r,
            "kl":          loss_kl,
            "outlier":     loss_outlier,
            "log_sigma_z": torch.log(sigma_z).mean().detach(),
            "z_m_mean":    z_m.mean().detach(),
        }

    @torch.no_grad()
    def predict_z(self, x_phot: torch.Tensor, n_samples: int = 500):
        """Returns z_samples (full mixture), z_mean/z_std (the model's own
        point estimate). `z_mean`/`z_std` are read straight off
        `encoder.z_point()`'s output -- the same call `forward()`/`loss()`
        use internally -- rather than re-drawing a separate component pick
        here, so this can't silently drift out of sync with what training
        actually optimizes (e.g. if `z_point()`'s sampling scheme changes).
        """
        (z_mean, z_std, log_pi, mu_mix, log_var_mix,
         _, _, z_m, _, _, _) = self.encoder(x_phot)

        ks      = torch.distributions.Categorical(logits=log_pi).sample((n_samples,)).T
        mu_s    = mu_mix.gather(1, ks)
        std_s   = torch.exp(0.5 * log_var_mix.gather(1, ks))
        z_raw_s = mu_s + std_s * torch.randn_like(std_s)
        z_samples = z_m.unsqueeze(1) * F.softplus(z_raw_s)

        return z_samples, z_mean, z_std

    @torch.no_grad()
    def predict_z_pdf(self, x_phot: torch.Tensor, z_grid: torch.Tensor = None,
                       n_grid: int = 500, z_max: float = 8.5):
        (_, _, log_pi, mu_mix, log_var_mix, _, _, z_m, _, _, _) = self.encoder(x_phot)

        if z_grid is None:
            z_grid = torch.linspace(1e-3, z_max, n_grid, device=x_phot.device)
        z_grid = z_grid.to(device=x_phot.device, dtype=mu_mix.dtype)

        weights   = log_pi.exp()                     # (batch, K)
        sigma_raw = torch.exp(0.5 * log_var_mix)      # (batch, K)

        z   = z_grid.view(1, 1, -1)                   # (1,     1, G)
        zm  = z_m.view(-1, 1, 1)                       # (batch, 1, 1)
        mu  = mu_mix.unsqueeze(-1)                     # (batch, K, 1)
        sig = sigma_raw.unsqueeze(-1)                  # (batch, K, 1)

        # raw = softplus^{-1}(z / z_m); clamp keeps log(expm1(.)) finite at z->0.
        # log(expm1(ratio)) = ratio + log1p(-exp(-ratio)) -- exact identity,
        # never computes exp(+ratio) so it can't overflow for large ratio
        # (see _mixture_component_log_pdf's docstring for the same fix,
        # applied there after this session's smoke test hit a NaN loss
        # from the naive form on a degenerate z_m).
        ratio = (z / zm).clamp(min=1e-8)
        raw   = ratio + torch.log1p(-torch.exp(-ratio))
        log_pdf_raw = (-0.5 * ((raw - mu) / sig) ** 2
                       - torch.log(sig) - 0.5 * math.log(2.0 * math.pi))
        log_jac       = torch.log(zm) + F.logsigmoid(raw)   # log(dz/draw) = log(z_m * sigmoid(raw))
        component_pdf = torch.exp(log_pdf_raw - log_jac)    # (batch, K, G)

        mixture_pdf = (weights.unsqueeze(-1) * component_pdf).sum(dim=1)  # (batch, G)

        return z_grid, weights, component_pdf, mixture_pdf

    @torch.no_grad()
    def predict_outlier_risk(self, x_phot: torch.Tensor, threshold: float = 0.15,
                              z_grid: torch.Tensor = None, n_grid: int = 500,
                              z_max: float = 8.5) -> torch.Tensor:
        """Self-assessed catastrophic-outlier risk, read straight off the
        model's own posterior -- NOT a trained classifier, so it can't drift
        out of sync with whatever z_point()/predict_z_pdf() actually
        represent.

        Defined as the posterior probability mass (from `predict_z_pdf`)
        falling outside the |z - z_pred|/(1+z_pred) <= threshold band around
        the model's own point estimate z_pred -- i.e. how much of its own
        mixture the model itself places in the region that would count as a
        catastrophic outlier relative to z_pred.

        Empirically (mlpvae_mdn_dp2_v18, see
        project_mdn_outlier_risk_score memory) this beats z_std alone at
        flagging actual |dz|>threshold outliers on held-out test data
        (ROC-AUC 0.90 vs. 0.79, AP 0.53 vs. 0.31 at a 10% outlier rate) --
        collapsing the full mixture to a single Gaussian sigma_z throws away
        most of the catastrophic-outlier signal that the posterior's shape
        (multimodality, spread) still carries. A combined logistic
        regression over sigma_z/mixture-entropy/weight-margin/this score
        added nothing over this score alone, so it's exposed standalone
        rather than as a fitted classifier.

        Returns
        -------
        risk : (batch,) probability mass outside the ±threshold band, in
               [0, 1]. Higher = more likely to be a catastrophic outlier.
        """
        _, z_pred, _ = self.predict_z(x_phot, n_samples=1)
        zg, _, _, mixture_pdf = self.predict_z_pdf(
            x_phot, z_grid=z_grid, n_grid=n_grid, z_max=z_max)

        band_halfwidth = threshold * (1.0 + z_pred).unsqueeze(1)          # (batch, 1)
        in_band = (zg.view(1, -1) - z_pred.unsqueeze(1)).abs() <= band_halfwidth
        dz_grid = zg[1] - zg[0]
        total_mass   = mixture_pdf.sum(dim=1) * dz_grid
        in_band_mass = (mixture_pdf * in_band.float()).sum(dim=1) * dz_grid
        return 1.0 - (in_band_mass / total_mass.clamp_min(1e-12))

    @torch.no_grad()
    def predict_z_prior_pdf(self, x_phot: torch.Tensor, z_grid: torch.Tensor = None,
                             n_grid: int = 500, z_max: float = 6.5):
        """Learned i-mag-conditioned BPZ-style prior p(z|i), on a shared z_grid.

        This is NOT the MDN posterior `predict_z_pdf` returns -- it's the
        smooth demographic prior p(z|i_mag) (Benitez 2000 form; see
        `log_p_bpz`'s docstring) that `z_m(i)` is built from and that
        `loss()`'s `-lam_prior * log_prior` term trains `self.encoder.zprior_*`
        against. Plotting it next to `predict_z_pdf`'s posterior mixture shows
        whether a bad prediction is a posterior failure (the prior alone
        would have favored a more plausible z) or a genuinely hard,
        prior-consistent case.

        Parameters mirror `predict_z_pdf`'s -- pass the same `z_grid` to both
        so the two curves share an x-axis.

        Returns
        -------
        z_grid    : (n_grid,)       z values the density is evaluated on.
        prior_pdf : (batch, n_grid) p(z | i_mag) for each galaxy in x_phot.
        """
        i_mag = _i_mag_from_x(x_phot, self.use_colors,
                              self.encoder.feat_mean, self.encoder.feat_scale)
        alpha, beta_prior, z0, km, pm = self.encoder.prior_params()

        if z_grid is None:
            z_grid = torch.linspace(1e-3, z_max, n_grid, device=x_phot.device)
        z_grid = z_grid.to(device=x_phot.device, dtype=i_mag.dtype)

        log_prior = log_p_bpz(z_grid.view(1, -1), i_mag.view(-1, 1),
                              alpha, beta_prior, z0, km, pm)
        return z_grid, log_prior.exp()

    @torch.no_grad()
    def predict_params(self, x_phot: torch.Tensor) -> torch.Tensor:
        """Returns theta_full, (batch, 16) posterior means for all physical SPS parameters."""
        (z_pred, _, _, _, _, mu_15, _, _, _, _, _) = self.encoder(x_phot)
        theta_15   = constrain_params_15(mu_15)
        return torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)

    @torch.no_grad()
    def predict_outlier_prob(self, x_phot: torch.Tensor) -> torch.Tensor:
        """Dedicated outlier_head's own learned P(catastrophic outlier),
        i.e. sigmoid(outlier_logit) -- trained via loss()'s lam_outlier BCE
        term (see its docstring) against the online |dz|>threshold label,
        on h.detach() so it can't perturb the shared trunk. Only
        meaningfully trained when a run passes --lam-outlier>0; otherwise
        this is the head's untrained random init and carries no signal.
        Complements the closed-form, zero-training `predict_outlier_risk()`
        (posterior mass outside the band around z_pred) -- see that
        method's docstring / project_mdn_outlier_risk_score memory for how
        the two compare once this head has actually been trained.
        """
        *_, outlier_logit = self.encoder(x_phot)
        return torch.sigmoid(outlier_logit)

    def save(self, path: str, scaler=None, col_medians=None):
        """Save model weights + optional preprocessor state."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        payload = {
            "model_state": self.state_dict(),
            "encoder_config": {
                "n_in":     self.encoder.input_proj[0].in_features,
                "width":    self.encoder.input_proj[0].out_features,
                "n_latent": self.encoder.n_latent,
                "n_mix":    self.encoder.n_mix,
                "detach_trunk_for_vae": self.encoder.detach_trunk_for_vae,
                "use_colors": self.use_colors,
                "use_z_prior_sampling": self.encoder.use_z_prior_sampling,
                "z_sample_method": self.encoder.z_sample_method,
                "double_precision": self._double_precision,
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
                      n_mix=cfg.get("n_mix", 3),
                      detach_trunk_for_vae=cfg.get("detach_trunk_for_vae", True),
                      use_colors=use_colors,
                      use_z_prior_sampling=cfg.get("use_z_prior_sampling", False),
                      z_sample_method=cfg.get("z_sample_method", "normal"),
                      double_precision=cfg.get("double_precision", False),
                      zprior_params=cfg.get("zprior_params"),
                      feat_mean=feat_mean, feat_scale=feat_scale)
        model.load_state_dict(payload["model_state"], strict=False)
        model.to(device)
        model.eval()
        return model, payload.get("scaler"), payload.get("col_medians")
