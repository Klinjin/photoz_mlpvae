"""
photoz_mlpvae.py
================
PhotozMLPVAE: shared trunk → [z MLP head] → [SPS VAE head, conditioned on z] →
speculator → filter_conv.

This mimics photoz_vae.py's PhotozVAE (same trunk, same transforms, same
3-term loss, same reparameterize/NLL/KL machinery) with one structural change:
redshift is pulled out of the 16-dim joint latent into its own 1-dim latent
with a dedicated MLP head, while the remaining 15 SPS parameters keep their
own VAE head — sampled conditional on the redshift estimate. Everything else
(transforms, decoder, loss shape) is copied from photoz_vae.py unchanged.

Architecture
------------
  Shared trunk : Linear(40→512) + BN + ReLU + 2×ResBlock(512)   (identical
                 topology to PhotozEncoder in photoz_vae.py)
  z_head       : Linear(512→128) + ReLU + Linear(128→2) → (mu_z, log_var_z)
                 A dedicated 2-layer MLP (vs. the reference's single Linear
                 slice) — this is the "extra layer" for z and its σ_z, giving
                 redshift its own capacity instead of sharing one output row
                 with 15 other parameters.
  vae_head     : Linear(512+1→30) → (mu_15, log_var_15)
                 Same single-Linear simplicity as the reference model, with
                 one tweak: +1 input channel for the redshift estimate, so
                 the 15-param posterior is conditioned on z. Reads
                 h.detach() — see "Gradient flow" below.

  Frozen decoder (identical to PhotozVAE):
    theta_full = cat([z_pred.detach(), constrain_params_15(z_raw_15)], dim=1)
    SpeculatorInoueIGM(theta_full) → log_spec
    FilterConv(log_spec, z=theta_full[:,0], logmass=theta_full[:,1]) → AB mags

Redshift latent (mimics photoz_vae.py's treatment of zred exactly, just
isolated to its own 1-dim Gaussian instead of column 0 of a 16-dim one):
  z_raw ~ N(mu_z, exp(log_var_z))            reparameterized sample
  z_pred = sigmoid(z_raw) × 5.5               physical redshift
  σ_z_phys = delta-method push-forward of σ_z_raw through sigmoid(·)×5.5
             (see `_zred_sigma_phys`) — no separate sigma head; log_var_z
             plays the same role here that log_var[:,0] plays in PhotozVAE.
  The supervised NLL is evaluated at z_pred (the sample), for the same
  reason given in photoz_vae.py: it gives log_var_z a direct calibration
  signal from the label instead of shaping it only through β·KL.

Loss (three terms, same structure as PhotozVAE.loss)
-----------------------------------------------------
  L_total = λ_z × NLL_z(z_pred, z_spec, σ_z)            z-head, always on
          + λ_r × Σ mask_i (m̂_i − m_i)² / (σ_i²+σ_f²)   reconstruction
          + β   × D_KL[ N(μ_15,σ_15²) ‖ N(0, I) ]        KL, 15-param latent only

Redshift carries no KL term — it is a supervised latent, not an
unsupervised one; its only pressure is the NLL against z_spec. The KL term
covers exactly the 15 SPS parameters, matching the request to add the KL
loss "with the rest of 15 SPS parameters" in stage 2.

Training stages
---------------
  Stage 1 (warmup): vae_head frozen, λ_r=0, β=0, use_nll_z can be False→True
      Trunk + z_head train from z-supervised NLL only. vae_head still runs
      forward (SPS sampling already depends on z_pred), but it is frozen and
      unweighted, so it draws no gradient yet.
  Stage 2 (joint): vae_head unfrozen, λ_r ramped, β annealed 0→β_max.
      vae_head now trains from reconstruction + KL. z_head keeps training
      from its NLL term throughout — nothing freezes it in stage 2.

Gradient flow
-------------
  z_pred, σ_z ← z NLL only (trunk + z_head)
  mu_15, log_var_15 ← reconstruction + KL only (vae_head)
  vae_head reads [h.detach(), z_pred.detach()]: reconstruction/KL gradient
  cannot reach the trunk or the z branch. This mirrors the fix found in the
  obs_v6 run of the old 3-head MLPVAE (recon gradient leaking back through
  vae_head → trunk measurably degraded z_pred quality once λ_r ramped up).
  Here it is baked in from the start rather than patched in later: the
  z branch is trained to convergence on its own signal, and the SPS/SED
  branch is trained to converge in the parameter space *given* that
  redshift, never the other way around.

SPS parameter ordering in theta_full (16 dims) — identical to PhotozVAE,
zred transform identical, 15-param transforms identical (just reindexed):
  0   zred               sigmoid(·) × 5.5           [0,   5.5]
  1   logmass            sigmoid(·) × 5.5 + 7.0     [7,   12.5]
  2   logzsol            tanh(·) × 1.085 − 0.895    [−1.98, 0.19]
  3   dust2              sigmoid(·) × 4.0           [0,   4.0]
  4   dust1_fraction     sigmoid(·) × 2.0           [0,   2.0]
  5   dust_index         tanh(·) × 0.7 − 0.3        [−1.0, 0.4]
  6   gas_logz           tanh(·) × 1.25 − 0.75      [−2.0, 0.5]
  7   fagn               10^(sigmoid(·)×5.477 − 5)  [1e-5, ~3]
  8   agn_tau            10^(sigmoid(·)×1.477+0.699)[5,   150]
  9   igm_scale          sigmoid(·) × 2.0           [0,   2.0]
  10-15 logsfr_ratios_0..5  tanh(·) × 5.0           [−5,  5]

Usage
-----
    model = PhotozMLPVAE(speculator_dir, filter_dir)
    mags_recon, z_pred, mu_z, log_var_z, mu_15, log_var_15 = model(x_phot)
    losses = model.loss(x_phot, mags_obs, mag_errs, mask, z_spec)
    z_samples, z_mean, z_std = model.predict_z(x_phot, n_samples=500)
"""

import os
import numpy as np
import torch
import torch.nn as nn

from photoz_vae.model.vae_encoder import _ResBlock
from photoz_vae.model.speculator_torch import SpeculatorInoueIGM
from photoz_vae.model.filter_conv import FilterConv

# ─────────────────────────────────────────────────────────────────────────────
# Parameter map
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = [
    "zred",             # 0  — from the dedicated z MLP head, not the VAE latent
    "logmass",          # 1
    "logzsol",          # 2
    "dust2",            # 3
    "dust1_fraction",   # 4
    "dust_index",       # 5
    "gas_logz",         # 6
    "fagn",             # 7
    "agn_tau",          # 8
    "igm_scale",        # 9
    "logsfr_ratios_0",  # 10
    "logsfr_ratios_1",  # 11
    "logsfr_ratios_2",  # 12
    "logsfr_ratios_3",  # 13
    "logsfr_ratios_4",  # 14
    "logsfr_ratios_5",  # 15
]
PARAM_NAMES_15 = PARAM_NAMES[1:]   # the 15 VAE-encoded params (excludes zred)

N_PARAMS_15   = 15
N_PARAMS_FULL = 16

ZRED_IDX_FULL    = 0   # column index in theta_full
LOGMASS_IDX_FULL = 1   # column index in theta_full
LOGZSOL_IDX_FULL = 2   # column index in theta_full

# Speculator is trained on zred ∈ [0, 5.5]; clamp decoder input to this range
_ZRED_MAX_SPEC = 6.5


# ─────────────────────────────────────────────────────────────────────────────
# Bounded parameter transforms  (unconstrained → physical)
# ─────────────────────────────────────────────────────────────────────────────

def constrain_params_15(z_raw_15: torch.Tensor) -> torch.Tensor:
    """
    Map unconstrained 15-dim latent → physical SPS params (zred excluded).
    Identical transforms to PhotozVAE.constrain_params, indices 1-15.

    Parameters
    ----------
    z_raw_15 : (batch, 15)  unconstrained latent

    Returns
    -------
    theta_15 : (batch, 15)  physical SPS parameters (no zred)
    """
    parts = [
        torch.sigmoid(z_raw_15[:, 0:1])  * 5.5 + 7.0,                       # logmass
        torch.tanh(z_raw_15[:, 1:2])     * 1.085 - 0.895,                   # logzsol
        torch.sigmoid(z_raw_15[:, 2:3])  * 4.0,                            # dust2
        torch.sigmoid(z_raw_15[:, 3:4])  * 2.0,                            # dust1_fraction
        torch.tanh(z_raw_15[:, 4:5])     * 0.7 - 0.3,                       # dust_index
        torch.tanh(z_raw_15[:, 5:6])     * 1.25 - 0.75,                     # gas_logz
        torch.pow(10.0, torch.sigmoid(z_raw_15[:, 6:7]) * 5.477 - 5.0),     # fagn
        torch.pow(10.0, torch.sigmoid(z_raw_15[:, 7:8]) * 1.477 + 0.699),   # agn_tau
        torch.sigmoid(z_raw_15[:, 8:9])  * 2.0,                            # igm_scale
        torch.tanh(z_raw_15[:, 9:])      * 5.0,                             # logsfr_ratios_0..5
    ]
    return torch.cat(parts, dim=1)   # (batch, 15)


# Numerical floor on the delta-method physical σ_z — prevents 1/σ² blowup
# where the sigmoid Jacobian is tiny (z near 0 or 5.5).
_ZRED_SIGMA_FLOOR = 1e-3


def _zred_sigma_phys(mu_z: torch.Tensor, log_var_z: torch.Tensor) -> torch.Tensor:
    """
    Physical-space σ_z, propagated from the raw-space posterior
    N(mu_z, var_z) through the sigmoid(·)×5.5 transform via a first-order
    (delta-method) Jacobian evaluated at the mean:

        σ_z_phys ≈ σ_z_raw × d(sigmoid(z_raw)×5.5)/dz_raw |_{z_raw=mu_z}
                 = σ_z_raw × 5.5 × s(1-s),   s = sigmoid(mu_z)

    Identical formula to PhotozVAE._zred_sigma_phys, applied to the
    standalone z latent instead of column 0 of the joint 16-dim one.
    """
    s   = torch.sigmoid(mu_z)
    jac = 5.5 * s * (1.0 - s)
    sigma_raw = torch.exp(0.5 * log_var_z)
    return (sigma_raw * jac).clamp(min=_ZRED_SIGMA_FLOOR)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: shared trunk + z head + SPS (VAE) head
# ─────────────────────────────────────────────────────────────────────────────

class MLPVAEEncoder(nn.Module):
    """
    Shared ResBlock trunk (identical to PhotozEncoder) feeding two heads:
      - z_head   : 2-layer MLP → (mu_z, log_var_z), 1-dim redshift latent
      - vae_head : single Linear → (mu_15, log_var_15), 15-dim SPS latent,
                   conditioned on the z estimate

    Parameters
    ----------
    n_in    : input dimension (default 40)
    width   : hidden width (default 512)
    n_latent: SPS latent dimension (default 15, excludes zred)
    dropout : dropout probability in ResBlocks (default 0.1)
    """

    def __init__(self, n_in: int = 40, width: int = 512,
                 n_latent: int = N_PARAMS_15, dropout: float = 0.1):
        super().__init__()
        self.n_latent = n_latent

        # ── Shared trunk (identical topology to PhotozEncoder) ────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_in, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock(width, dropout),
            _ResBlock(width, dropout),
        )

        # ── z head: dedicated 2-layer MLP for (mu_z, log_var_z) ───────────
        self.z_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        # ── SPS (VAE) head: single Linear, conditioned on z_pred ──────────
        # +1 input channel is the only structural tweak vs. the reference
        # model's Linear(width, n_latent*2). Reads h.detach() so that
        # reconstruction/KL gradient never reaches the shared trunk (see
        # module docstring, "Gradient flow").
        self.vae_head = nn.Linear(width + 1, n_latent * 2)

        # Small-weight init → stable KL / z predictions at epoch 0
        nn.init.xavier_uniform_(self.z_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.z_head[-1].bias)
        nn.init.xavier_uniform_(self.vae_head.weight, gain=0.1)
        nn.init.zeros_(self.vae_head.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, n_in)

        Returns
        -------
        z_pred     : (batch,)          physical redshift (reparameterized
                                        sample during training, mean at eval)
        mu_z       : (batch,)          raw-space z posterior mean
        log_var_z  : (batch,)          raw-space z posterior log-variance
        mu_15      : (batch, n_latent) SPS posterior means, conditioned on z
        log_var_15 : (batch, n_latent) SPS posterior log-variances
        """
        h = self.input_proj(x)
        h = self.res_blocks(h)

        z_out     = self.z_head(h)
        mu_z      = z_out[:, 0]
        log_var_z = z_out[:, 1]#.clamp(-6.0, 4.0)
        z_raw     = self.reparameterize(mu_z, log_var_z)
        z_pred    = torch.sigmoid(z_raw) * _ZRED_MAX_SPEC

        # SPS sampling depends on the z estimate: vae_head reads
        # [h.detach(), z_pred.detach()] so reconstruction/KL gradient stays
        # confined to vae_head's own weights.
        cond       = torch.cat([h.detach(), z_pred.detach().unsqueeze(1)], dim=1)
        vae_out    = self.vae_head(cond)
        mu_15      = vae_out[:, :self.n_latent]
        log_var_15 = vae_out[:, self.n_latent:]#.clamp(-6.0, 4.0)

        return z_pred, mu_z, log_var_z, mu_15, log_var_15

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: z = μ + ε × exp(0.5 × log σ²)."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic mean at eval time

    def freeze_vae_head(self):
        """Freeze the 15-param SPS head (stage 1 / z-supervised warmup)."""
        for p in self.vae_head.parameters():
            p.requires_grad_(False)

    def unfreeze_vae_head(self):
        """Unfreeze the 15-param SPS head (stage 2 / joint fine-tuning)."""
        for p in self.vae_head.parameters():
            p.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class PhotozMLPVAE(nn.Module):
    """
    Photometric-redshift MLP-VAE: z gets its own supervised latent + MLP
    head; the remaining 15 SPS parameters keep a VAE latent conditioned on z.

    Parameters
    ----------
    speculator_dir : str   path to speculator/trained/Inoue_IGM
    filter_dir     : str   path to obs_catalog/filters (Euclid .dat files)
    encoder_width  : int   hidden layer width (default 512)
    encoder_dropout: float dropout probability (default 0.1)
    encoder_n_in   : int   input feature dimension (default 40)
    """

    def __init__(self,
                 speculator_dir: str,
                 filter_dir: str,
                 encoder_width: int = 512,
                 encoder_dropout: float = 0.1,
                 encoder_n_in: int = 40):
        super().__init__()

        # ── Encoder (trainable) ───────────────────────────────────────────
        self.encoder = MLPVAEEncoder(
            n_in=encoder_n_in, width=encoder_width,
            n_latent=N_PARAMS_15, dropout=encoder_dropout,
        )

        # ── Frozen decoder ────────────────────────────────────────────────
        self.speculator  = SpeculatorInoueIGM(speculator_dir)
        self.filter_conv = FilterConv(filter_dir, self.speculator.wl_rest)

        for p in self.speculator.parameters():
            p.requires_grad_(False)
        for p in self.filter_conv.parameters():
            p.requires_grad_(False)

    # ── Encode ────────────────────────────────────────────────────────────

    def encode(self, x_phot: torch.Tensor):
        """
        x_phot : (batch, n_in) scaled photometric features
        Returns z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15
        """
        z_pred, mu_z, log_var_z, mu_15, log_var_15 = self.encoder(x_phot)
        z_raw_15 = self.encoder.reparameterize(mu_15, log_var_15)
        return z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15

    # ── Decode ────────────────────────────────────────────────────────────

    def decode(self, z_pred: torch.Tensor, z_raw_15: torch.Tensor) -> torch.Tensor:
        """
        z_pred   : (batch,)    physical redshift ∈ [0, 5.5] (sample or mean)
        z_raw_15 : (batch, 15) unconstrained SPS latent
        Returns m_ab : (batch, 10) reconstructed AB magnitudes
        """
        theta_15 = constrain_params_15(z_raw_15)

        # Detach z_pred: reconstruction trains theta_15 (SED shape) only;
        # z_pred receives gradients exclusively from the supervised z loss.
        zred_dec   = z_pred.detach().clamp(max=_ZRED_MAX_SPEC)
        theta_full = torch.cat([zred_dec.unsqueeze(1), theta_15], dim=1)

        log_spec, _ = self.speculator(theta_full)
        m_ab = self.filter_conv(log_spec,
                                theta_full[:, ZRED_IDX_FULL],
                                theta_full[:, LOGMASS_IDX_FULL])
        return m_ab

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x_phot: torch.Tensor):
        """
        Returns
        -------
        m_ab_recon : (batch, 10)
        z_pred     : (batch,)
        mu_z       : (batch,)
        log_var_z  : (batch,)
        mu_15      : (batch, 15)
        log_var_15 : (batch, 15)
        """
        z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15 = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)
        return m_ab_recon, z_pred, mu_z, log_var_z, mu_15, log_var_15

    # ── Diagnostic interface (shared with photoz_utils) ───────────────────

    def encode_theta(self, x: torch.Tensor):
        """
        Standard diagnostic interface.

        Returns
        -------
        z_pred      : (batch,)    physical redshift  ∈ [0, 5.5]
        log_sigma_z : (batch,)    log σ_z, derived from log_var_z
        theta_full  : (batch, 16) physical SPS parameters
        """
        _, mu_z, log_var_z, mu_15, _ = self.encoder(x)
        z_pred      = torch.sigmoid(mu_z) * _ZRED_MAX_SPEC
        log_sigma_z = torch.log(_zred_sigma_phys(mu_z, log_var_z))
        theta_15    = constrain_params_15(mu_15)
        theta_full  = torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)
        return z_pred, log_sigma_z, theta_full

    @torch.no_grad()
    def sample_theta_posterior(self, x, n_samples: int, device: str, rng):
        """
        Draw posterior samples for each galaxy.

        z is sampled from its own raw-space Gaussian and pushed through
        sigmoid(·)×5.5; the 15 SPS params are sampled independently from
        the VAE posterior (mu_15, log_var_15), which was itself conditioned
        on a single z point during the forward pass — same approximation
        used by the reference model's posterior sampling (mu_15 is not
        re-derived per z sample).

        Returns
        -------
        samples : (N_gal, n_samples, 16)  numpy float32
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.astype(np.float32)).to(device)
        _, mu_z, log_var_z, mu_15, log_var_15 = self.encoder(x)
        mu_z_np   = mu_z.cpu().numpy()
        std_z_np  = np.exp(0.5 * log_var_z.cpu().numpy())
        mu_15_np  = mu_15.cpu().numpy()
        std_15_np = np.exp(0.5 * log_var_15.cpu().numpy())

        n_gal = x.shape[0]
        all_samples = np.empty((n_gal, n_samples, N_PARAMS_FULL), dtype=np.float32)
        for i in range(n_gal):
            eps_z   = rng.standard_normal(n_samples).astype(np.float32)
            z_raw_s = mu_z_np[i] + eps_z * std_z_np[i]
            z_s     = 1.0 / (1.0 + np.exp(-z_raw_s)) * _ZRED_MAX_SPEC
            z_s     = np.clip(z_s, 0.0, _ZRED_MAX_SPEC)

            eps_15   = rng.standard_normal((n_samples, 15)).astype(np.float32)
            z_raw_15 = mu_15_np[i] + eps_15 * std_15_np[i]
            with torch.no_grad():
                theta_15 = constrain_params_15(
                    torch.from_numpy(z_raw_15).to(device)
                ).cpu().numpy()
            all_samples[i] = np.concatenate([z_s[:, None], theta_15], axis=1)
        return all_samples

    # ── Loss ──────────────────────────────────────────────────────────────

    def loss(self,
             x_phot:    torch.Tensor,
             mags_obs:  torch.Tensor,
             mag_errs:  torch.Tensor,
             mask:      torch.Tensor,
             z_spec:    torch.Tensor,
             lam_z:     float = 20.0,
             lam_r:     float = 1.0,
             beta:      float = 0.0,
             sigma_floor: float = 0.3,
             z_weights: torch.Tensor = None,
             use_nll_z: bool = True) -> dict:
        """
        Compute MLPVAE loss (same 3-term structure as PhotozVAE.loss).

        Parameters
        ----------
        x_phot    : (batch, n_in)
        mags_obs  : (batch, 10)  observed AB mags
        mag_errs  : (batch, 10)  observed mag errors
        mask      : (batch, 10)  1 = band present
        z_spec    : (batch,)     spectroscopic redshift
        lam_z     : weight for the supervised redshift loss
        lam_r     : weight for the reconstruction loss
        beta      : KL weight (annealed 0→β_max in stage 2), 15-param latent only
        z_weights : (batch,) per-galaxy loss weights; None = uniform
        use_nll_z : True → heteroscedastic NLL; False → plain MSE (warmup)

        Returns
        -------
        dict: total, z_sup, recon, kl, log_sigma_z
        """
        z_pred, mu_z, log_var_z, mu_15, log_var_15, z_raw_15 = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)

        # ── 1. Supervised redshift loss ───────────────────────────────────
        # Point estimate uses the REPARAMETERIZED sample z_pred, for the
        # same reason as PhotozVAE: this gives log_var_z a direct
        # calibration signal from z_spec instead of shaping it only via KL
        # (which redshift doesn't even have here — see term 3).
        sq_err  = (z_pred - z_spec) ** 2
        sigma_z = _zred_sigma_phys(mu_z, log_var_z)     # always, for logging
        if use_nll_z:
            inv_var = 1.0 / (sigma_z ** 2)
            nll     = 0.5 * sq_err * inv_var + torch.log(sigma_z)
            per_gal = nll
        else:
            per_gal = sq_err
        if z_weights is not None:
            loss_z = (z_weights * per_gal).mean()
        else:
            loss_z = per_gal.mean()

        # ── 2. Noise-weighted reconstruction loss ─────────────────────────
        m_ab_recon = m_ab_recon[:, :mags_obs.shape[1]]
        _zero = torch.zeros(1, device=x_phot.device).squeeze()
        if lam_r > 0.0:
            sig2    = mag_errs ** 2 + sigma_floor ** 2
            chi2    = mask * (m_ab_recon - mags_obs) ** 2 / sig2
            n_valid = mask.sum(dim=1).clamp(min=1.0)
            loss_r  = (chi2.sum(dim=1) / n_valid).mean()
            loss_r  = torch.nan_to_num(loss_r, nan=0.0, posinf=0.0)
        else:
            loss_r = _zero

        # ── 3. KL divergence on the 15-param SPS latent only ──────────────
        # Redshift carries no KL term — it is purely supervised via term 1.
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
        }

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_z(self, x_phot: torch.Tensor, n_samples: int = 500):
        """
        Predict redshift from the z-head posterior.

        Returns
        -------
        z_samples : (batch, n_samples)  redshift posterior samples
        z_mean    : (batch,)            posterior mean (deterministic)
        z_std     : (batch,)            posterior std, from log_var_z via
                                         the delta-method (_zred_sigma_phys)
        """
        _, mu_z, log_var_z, _, _ = self.encoder(x_phot)

        z_mean = torch.sigmoid(mu_z) * _ZRED_MAX_SPEC
        z_std  = _zred_sigma_phys(mu_z, log_var_z)

        std     = torch.exp(0.5 * log_var_z)
        z_raw_s = mu_z.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
            mu_z.shape[0], n_samples, device=mu_z.device)
        z_samples = (torch.sigmoid(z_raw_s) * _ZRED_MAX_SPEC).clamp(0.0, _ZRED_MAX_SPEC)

        return z_samples, z_mean, z_std

    @torch.no_grad()
    def predict_params(self, x_phot: torch.Tensor) -> torch.Tensor:
        """
        Return posterior means for all 16 physical SPS parameters.

        Returns
        -------
        theta_full : (batch, 16)  [z_mean | constrain_params_15(mu_15)]
        """
        _, mu_z, _, mu_15, _ = self.encoder(x_phot)
        z_mean     = torch.sigmoid(mu_z) * _ZRED_MAX_SPEC
        theta_15   = constrain_params_15(mu_15)
        theta_full = torch.cat([z_mean.unsqueeze(1), theta_15], dim=1)
        return theta_full

    # ── Serialisation ─────────────────────────────────────────────────────

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
            },
            "scaler":      scaler,
            "col_medians": col_medians,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, speculator_dir: str, filter_dir: str,
             device: str = "cpu"):
        """Load model from saved checkpoint."""
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg     = payload["encoder_config"]
        model   = cls(speculator_dir, filter_dir,
                      encoder_width=cfg["width"],
                      encoder_n_in=cfg.get("n_in", 40))
        model.load_state_dict(payload["model_state"])
        model.to(device)
        model.eval()
        return model, payload.get("scaler"), payload.get("col_medians")
