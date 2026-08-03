"""
photoz_mlpvae.py
================
2-Stage MLP-VAE: MLP z-head + 15-param VAE on a shared ResBlock trunk.

Architecture
------------
  Shared trunk : Linear(40→512) + BN + ReLU + 2×ResBlock(512)
  Split heads  :
    z_head       : Linear(512→1) → sigmoid(·)×6.0  →  z_pred  (scalar)
    sigma_z_head : Linear(512+15→128) → ReLU → Linear(128→1)  →  log σ_z
                   Input: [trunk_features | mu_15.detach()]
                   SED posterior informs z uncertainty; mu_15 weights are
                   zero-initialized so phase 1 (frozen vae_head) is unaffected.
    vae_head     : Linear(512+1→30) → mu_15 / log_var_15  (15-param posterior)
                   Conditioned on z_pred.detach() to keep z_head gradient clean.

  Frozen decoder (same as PhotozVAE):
    theta_full = cat([z_pred.detach(), constrain_params_15(z_raw_15)], dim=1)
    SpeculatorInoueIGM(theta_full) → log_spec
    FilterConv(log_spec, z=theta_full[:,0], logmass=theta_full[:,1]) → 10 AB mags
    z_pred is detached in decode() so reconstruction trains only theta_15;
    z_pred receives gradients exclusively from the supervised z loss.

Loss (three terms, same structure as PhotozVAE)
-----------------------------------------------
  L_total = λ_z  × NLL_z(z_pred, z_spec, σ_z)          MLP z head
          + λ_r  × Σ mask_i (m̂_i − m_i)² / (σ_i²+σ_f²) reconstruction
          + β    × D_KL[ N(μ_15,σ_15²) ‖ N(0,I) ]       KL on 15-param latent

Gradient flows (phase 2)
------------------------
  z_pred   ← z_supervised NLL only  (h detached before vae_head; no recon→trunk path)
  theta_15 ← reconstruction + KL   (SED shape learning; reads h.detach() read-only)
  σ_z      ← NLL calibration via sigma_z_head([h, mu_15.detach()])
              recon/KL shapes mu_15 → sigma_z_head reads updated mu_15 each step
              → SED reconstruction quality informs z uncertainty

Training stages
---------------
  Phase 1 (warmup): vae_head frozen, λ_r=0, β=0, use_nll_z=False
      Trunk + z_head + sigma_z_head learn from z-supervised MSE.
      sigma_z_head receives random mu_15 but ignores it (zero-initialized weights).
  Phase 2 (joint):  vae_head unfrozen via add_param_group() (preserves trunk/z_head
      Adam momentum); LR scaled down by phase2_lr_frac; λ_r ramped, β annealed.
      sigma_z_head gradually learns to use the now-informative mu_15 for better
      uncertainty calibration — this is the SED→photo-z information pathway.

SPS parameter ordering in theta_full (16 dims)
----------------------------------------------
  Φ = standard-normal CDF; latent ~ N(0,I) → Uniform over each range.
  0   zred               z_pred (from MLP z-head, sigmoid×6.0)
  1   logmass            Φ(·)×6.0 + 7.0             [7,   13.0]
  2   logzsol            Φ(·)×2.17 − 1.98           [−1.98, 0.19]
  3   dust2              Φ(·)×4.0                   [0,   4.0]
  4   dust1_fraction     Φ(·)×2.0                   [0,   2.0]
  5   dust_index         Φ(·)×1.4 − 1.0             [−1.0, 0.4]
  6   gas_logz           Φ(·)×2.5 − 2.0             [−2.0, 0.5]
  7   fagn               10^(Φ(·)×5.477 − 5)        [1e-5, ~3]
  8   agn_tau            10^(Φ(·)×1.477 + 0.699)    [5,   150]
  9   igm_scale          Φ(·)×2.0                   [0,   2.0]
  10-15 logsfr_ratios_0..5  Φ(·)×10.0 − 5.0         [−5,  5]
"""

import os
import numpy as np
import torch
import torch.nn as nn

from photoz_vae.model.vae_encoder import _ResBlock
from photoz_vae.model.speculator_torch import SpeculatorInoueIGM
from photoz_vae.model.filter_conv import FilterConv

# ─────────────────────────────────────────────────────────────────────────────
# Parameter names
# ─────────────────────────────────────────────────────────────────────────────

PARAM_NAMES = [
    "zred",             # 0  — from MLP z-head, not VAE latent
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
N_PARAMS_15    = 15
N_PARAMS_FULL  = 16

ZRED_IDX_FULL    = 0   # column index in theta_full
LOGMASS_IDX_FULL = 1   # column index in theta_full

_ZRED_MAX_SPEC   = 6.0  # speculator training range

_SQRT2 = 2.0 ** 0.5

def _Phi(z: torch.Tensor) -> torch.Tensor:
    """Standard-normal CDF: maps N(0,1) → Uniform(0,1) marginally."""
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


# ─────────────────────────────────────────────────────────────────────────────
# Constrain 15 SPS params (all except zred)
# ─────────────────────────────────────────────────────────────────────────────

def constrain_params_15(z_raw_15: torch.Tensor) -> torch.Tensor:
    """
    Map unconstrained 15-dim latent → physical SPS params (zred excluded).

    Input/output order matches PARAM_NAMES_15 (logmass … logsfr_ratios_5).

    Parameters
    ----------
    z_raw_15 : (batch, 15)  unconstrained latent

    Returns
    -------
    theta_15 : (batch, 15)  physical SPS parameters (no zred)
    """
    p = _Phi(z_raw_15)   # (batch, 15)  element-wise ∈ (0, 1)
    parts = [
        p[:, 0:1]  * 6.0   + 7.0,                        # logmass        [7,   13.0]
        p[:, 1:2]  * 2.17  - 1.98,                       # logzsol        [-1.98, 0.19]
        p[:, 2:3]  * 4.0,                                 # dust2          [0,   4.0]
        p[:, 3:4]  * 2.0,                                 # dust1_fraction [0,   2.0]
        p[:, 4:5]  * 1.4   - 1.0,                        # dust_index     [-1.0, 0.4]
        p[:, 5:6]  * 2.5   - 2.0,                        # gas_logz       [-2.0, 0.5]
        torch.pow(10.0, p[:, 6:7]  * 5.477 - 5.0),       # fagn           [1e-5, ~3]
        torch.pow(10.0, p[:, 7:8]  * 1.477 + 0.699),     # agn_tau        [5,   150]
        p[:, 8:9]  * 2.0,                                 # igm_scale      [0,   2.0]
        p[:, 9:]   * 10.0  - 5.0,                        # logsfr_ratios  [-5,  5]
    ]
    return torch.cat(parts, dim=1)   # (batch, 15)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: shared trunk + 3 split heads
# ─────────────────────────────────────────────────────────────────────────────

class MLPVAEEncoder(nn.Module):
    """
    Shared ResBlock trunk with three split heads:
      - z_head       : Linear(width→1)  → sigmoid(·)×5.5  →  z_pred
      - sigma_z_head : Linear(width→1)                    →  log σ_z
      - vae_head     : Linear(width→30) → mu_15 / log_var_15

    Parameters
    ----------
    n_in    : input dimension (default 40)
    width   : hidden width (default 512)
    n_latent: VAE latent dimension = 15 SPS params (excludes zred)
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

        # ── Split heads ───────────────────────────────────────────────────
        self.z_head   = nn.Linear(width, 1)

        # SED-informed uncertainty head: reads [trunk | SED_posterior.detach()].
        # The mu_15 input weights are zero-initialized so that phase 1 behaviour
        # (frozen vae_head producing random mu_15) is identical to the old Linear(width,1).
        # During phase 2, as vae_head learns real SED posteriors, sigma_z_head
        # gradually activates the mu_15 weights via the NLL calibration gradient.
        self.sigma_z_head = nn.Sequential(
            nn.Linear(width + n_latent, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # vae_head takes [trunk_features | z_pred] → width+1 inputs so that
        # SPS parameter inference is explicitly conditioned on the predicted
        # redshift (z_pred is detached to keep the z_head gradient path clean).
        self.vae_head = nn.Linear(width + 1, n_latent * 2)

        # Small-weight initialisation → stable KL and z predictions at epoch 0
        for head in [self.z_head, self.vae_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

        # sigma_z_head: init trunk portion with small xavier, mu_15 portion with zeros
        nn.init.xavier_uniform_(self.sigma_z_head[0].weight[:, :width], gain=0.1)
        nn.init.zeros_(self.sigma_z_head[0].weight[:, width:])
        nn.init.zeros_(self.sigma_z_head[0].bias)
        nn.init.xavier_uniform_(self.sigma_z_head[2].weight, gain=0.1)
        nn.init.constant_(self.sigma_z_head[2].bias, -1.0)  # σ_z ≈ 0.37 at init

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, n_in)

        Returns
        -------
        z_pred      : (batch,)         MLP redshift prediction  ∈ [0, 5.5]
        log_sigma_z : (batch,)         log aleatoric z uncertainty
        mu_15       : (batch, n_latent) VAE posterior means
        log_var_15  : (batch, n_latent) VAE posterior log-variances
        """
        h = self.input_proj(x)
        h = self.res_blocks(h)

        z_pred = torch.sigmoid(self.z_head(h).squeeze(1)) * _ZRED_MAX_SPEC

        # VAE posterior: both h and z_pred are detached so reconstruction gradient
        # cannot reach the trunk.  Trunk trains solely from z supervision + NLL.
        z_h        = torch.cat([h.detach(), z_pred.detach().unsqueeze(1)], dim=1)  # (B, width+1)
        vae_out    = self.vae_head(z_h)
        mu_15      = vae_out[:, :self.n_latent]
        log_var_15 = vae_out[:, self.n_latent:].clamp(-6.0, 4.0)

        # SED-informed σ_z: sigma_z_head reads [trunk | SED_posterior.detach()].
        # mu_15.detach() keeps vae_head's gradient path clean (recon+KL only);
        # sigma_z_head still reads the mu_15 values, so SED information flows
        # one-way into σ_z every forward pass.
        sigma_in    = torch.cat([h, mu_15.detach()], dim=1)
        log_sigma_z = self.sigma_z_head(sigma_in).squeeze(1).clamp(-5.0, 0.5)

        return z_pred, log_sigma_z, mu_15, log_var_15

    def reparameterize(self, mu_15: torch.Tensor,
                       log_var_15: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick for the 15-param VAE head."""
        if self.training:
            std = torch.exp(0.5 * log_var_15)
            return mu_15 + torch.randn_like(std) * std
        return mu_15

    def freeze_vae_head(self):
        """Freeze the 15-param VAE head (Phase 1 / z-supervised warmup)."""
        for p in self.vae_head.parameters():
            p.requires_grad_(False)

    def unfreeze_vae_head(self):
        """Unfreeze the 15-param VAE head (Phase 2 / joint fine-tuning)."""
        for p in self.vae_head.parameters():
            p.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class PhotozMLPVAE(nn.Module):
    """
    2-Stage MLP-VAE for photometric redshift estimation.

    Parameters
    ----------
    speculator_dir : str   path to speculator/trained/Inoue_IGM
    filter_dir     : str   path to obs_catalog/filters (Euclid .dat files)
    encoder_width  : int   hidden layer width (default 512)
    encoder_dropout: float dropout probability (default 0.1)
    """

    def __init__(self,
                 speculator_dir: str,
                 filter_dir: str,
                 encoder_width: int = 512,
                 encoder_dropout: float = 0.1,
                 encoder_n_in: int = 40):
        super().__init__()

        # ── Trainable encoder ─────────────────────────────────────────────
        self.encoder = MLPVAEEncoder(
            n_in=encoder_n_in, width=encoder_width,
            n_latent=N_PARAMS_15, dropout=encoder_dropout,
        )

        # ── Frozen decoder (same as PhotozVAE) ───────────────────────────
        self.speculator  = SpeculatorInoueIGM(speculator_dir)
        self.filter_conv = FilterConv(filter_dir, self.speculator.wl_rest)

        for p in self.speculator.parameters():
            p.requires_grad_(False)
        for p in self.filter_conv.parameters():
            p.requires_grad_(False)

    # ── Encode ────────────────────────────────────────────────────────────

    def encode(self, x_phot: torch.Tensor):
        """
        x_phot : (batch, 40)

        Returns
        -------
        z_pred      : (batch,)
        log_sigma_z : (batch,)
        mu_15       : (batch, 15)
        log_var_15  : (batch, 15)
        z_raw_15    : (batch, 15)  reparameterized sample
        """
        z_pred, log_sigma_z, mu_15, log_var_15 = self.encoder(x_phot)
        z_raw_15 = self.encoder.reparameterize(mu_15, log_var_15)
        return z_pred, log_sigma_z, mu_15, log_var_15, z_raw_15

    # ── Decode ────────────────────────────────────────────────────────────

    def decode(self, z_pred: torch.Tensor,
               z_raw_15: torch.Tensor) -> torch.Tensor:
        """
        Assemble full 16-dim SPS theta and run frozen decoder.

        Parameters
        ----------
        z_pred   : (batch,)   redshift from MLP head  ∈ [0, 5.5]
        z_raw_15 : (batch, 15) reparameterized 15-param latent

        Returns
        -------
        m_ab : (batch, 10)  reconstructed AB magnitudes
        """
        theta_15   = constrain_params_15(z_raw_15)                   # (batch, 15)
        # Detach z_pred: reconstruction trains theta_15 (SED shape) only.
        # z_pred receives gradients exclusively from the supervised z loss.
        zred_dec   = z_pred.detach().clamp(max=_ZRED_MAX_SPEC)
        theta_full = torch.cat([zred_dec.unsqueeze(1), theta_15], dim=1)  # (batch, 16)

        log_spec, _ = self.speculator(theta_full)
        m_ab = self.filter_conv(log_spec,
                                theta_full[:, ZRED_IDX_FULL],
                                theta_full[:, LOGMASS_IDX_FULL])      # (batch, 10)
        return m_ab

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x_phot: torch.Tensor):
        """
        Full forward pass.

        Returns
        -------
        m_ab_recon  : (batch, 10)
        z_pred      : (batch,)
        mu_15       : (batch, 15)
        log_var_15  : (batch, 15)
        log_sigma_z : (batch,)
        """
        z_pred, log_sigma_z, mu_15, log_var_15, z_raw_15 = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)
        return m_ab_recon, z_pred, mu_15, log_var_15, log_sigma_z

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
        x_phot    : (batch, 40)
        mags_obs  : (batch, 10)  observed AB mags
        mag_errs  : (batch, 10)  observed mag errors
        mask      : (batch, 10)  1 = band present
        z_spec    : (batch,)     spectroscopic redshift
        lam_z     : supervised z weight
        lam_r     : reconstruction weight
        beta      : KL weight (annealed)
        sigma_floor: floor on mag error for speculator mismatch
        z_weights : (batch,) per-galaxy weights; None = uniform
        use_nll_z : if True, use heteroscedastic NLL; if False, plain MSE

        Returns
        -------
        dict: total, z_sup, recon, kl, log_sigma_z
        """
        z_pred, log_sigma_z, mu_15, log_var_15, z_raw_15 = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)

        # ── 1. Supervised z loss ──────────────────────────────────────────
        # z_pred is ALWAYS trained with plain MSE so the gradient is never
        # diluted by a learnable σ_z (which would create a degenerate NLL
        # fixed-point at z_pred = mean(z_true), σ_z large).
        # When use_nll_z=True, σ_z is calibrated via a detached NLL term
        # (sq_err.detach() breaks the gradient path to z_pred).
        sq_err = (z_pred - z_spec) ** 2
        if use_nll_z:
            # calibration only – no gradient to z_pred from this term
            inv_var     = torch.exp(-2.0 * log_sigma_z)
            calibration = 0.5 * sq_err.detach() * inv_var + log_sigma_z
            per_gal = sq_err + calibration
        else:
            per_gal = sq_err
        if z_weights is not None:
            loss_z = (z_weights * per_gal).mean()
        else:
            loss_z = per_gal.mean()

        # ── 2. Noise-weighted reconstruction loss ─────────────────────────
        # Slice decoder output to the observed band count (6 LSST or 10 LSST+Euclid).
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

        # ── 3. KL on 15-param VAE latent ──────────────────────────────────
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
            "log_sigma_z": log_sigma_z.mean().detach(),
        }

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_z(self, x_phot: torch.Tensor, n_samples: int = 500):
        """
        Predict redshift posterior.

        The deterministic point estimate is z_pred from the MLP z-head.
        The predictive distribution samples z ~ N(z_pred, σ_z) using the
        calibrated aleatoric uncertainty from sigma_z_head.

        Returns
        -------
        z_samples : (batch, n_samples)  predictive samples ∈ [0, 5.5]
        z_mean    : (batch,)            point estimate (z_pred)
        z_std     : (batch,)            aleatoric σ_z
        """
        z_pred, log_sigma_z, _, _ = self.encoder(x_phot)
        z_std = torch.exp(log_sigma_z).clamp(min=1e-4)

        eps       = torch.randn(z_pred.shape[0], n_samples, device=z_pred.device)
        z_samples = (z_pred.unsqueeze(1) + eps * z_std.unsqueeze(1)).clamp(0.0, _ZRED_MAX_SPEC)

        return z_samples, z_pred, z_std

    @torch.no_grad()
    def predict_params(self, x_phot: torch.Tensor) -> torch.Tensor:
        """
        Return posterior means for all 16 physical SPS parameters.

        Returns
        -------
        theta_full : (batch, 16)  [z_pred | constrain_params_15(mu_15)]
        """
        z_pred, _, mu_15, _ = self.encoder(x_phot)
        theta_15   = constrain_params_15(mu_15)
        theta_full = torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)
        return theta_full

    # ── Diagnostic interface (shared with photoz_utils) ───────────────────

    def encode_theta(self, x: torch.Tensor):
        """
        Standard diagnostic interface.

        Returns
        -------
        z_pred      : (batch,)    physical redshift  ∈ [0, 5.5]
        log_sigma_z : (batch,)    log aleatoric z uncertainty
        theta_full  : (batch, 16) physical SPS parameters
        """
        z_pred, log_sigma_z, mu_15, _ = self.encoder(x)
        theta_15   = constrain_params_15(mu_15)
        theta_full = torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)
        return z_pred, log_sigma_z, theta_full

    def sample_theta_posterior(self, x, n_samples: int, device: str, rng):
        """
        Draw posterior samples for each galaxy.

        For PhotozMLPVAE, z is sampled from the NLL head (N(z_pred, σ_z²))
        and the 15 SPS params are sampled from the VAE posterior.

        Parameters
        ----------
        x        : (N_gal, 40)  numpy array or torch.Tensor  (scaled features)
        n_samples: int
        device   : str
        rng      : numpy Generator (e.g. np.random.default_rng(0))

        Returns
        -------
        samples : (N_gal, n_samples, 16)  numpy float32
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.astype(np.float32)).to(device)
        z_pred, log_sigma_z, mu_15, log_var_15 = self.encoder(x)
        z_pred_np  = z_pred.cpu().numpy()
        z_std_np   = np.exp(log_sigma_z.cpu().numpy())
        mu_15_np   = mu_15.cpu().numpy()
        std_15_np  = np.exp(0.5 * log_var_15.cpu().numpy())
        n_gal = x.shape[0]
        all_samples = np.empty((n_gal, n_samples, 16), dtype=np.float32)
        for i in range(n_gal):
            eps_z  = rng.standard_normal(n_samples).astype(np.float32)
            z_samp = np.clip(z_pred_np[i] + eps_z * z_std_np[i], 0.0, _ZRED_MAX_SPEC)
            eps_15   = rng.standard_normal((n_samples, 15)).astype(np.float32)
            z_raw_15 = mu_15_np[i] + eps_15 * std_15_np[i]
            with torch.no_grad():
                theta_15 = constrain_params_15(
                    torch.from_numpy(z_raw_15).to(device)
                ).cpu().numpy()
            all_samples[i] = np.concatenate([z_samp[:, None], theta_15], axis=1)
        return all_samples

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
        missing, unexpected = model.load_state_dict(
            payload["model_state"], strict=False)
        if missing or unexpected:
            import warnings
            warnings.warn(
                f"PhotozMLPVAE.load: {len(missing)} missing keys, "
                f"{len(unexpected)} unexpected keys (architecture mismatch — "
                "partial load; new heads use random init)."
            )
        model.to(device)
        model.eval()
        return model, payload.get("scaler"), payload.get("col_medians")
