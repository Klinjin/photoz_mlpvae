"""
photoz_mlpvae_mdn.py
====================
PhotozMLPVAE with a mixture-density (MDN) redshift head.

Motivation (2026-08-04 bias decomposition, see PLAN.md): the z-separated
model's mean-dz bias is entirely a catastrophic-outlier phenomenon — ~12% of
z_true<0.5 galaxies are flung to high z_pred by the Balmer↔Lyman-break
color degeneracy, contributing more than the whole net bias. A unimodal
Gaussian posterior must place one mode somewhere; for degenerate photometry
it either picks the wrong branch or an in-between compromise. This variant
replaces the 1-dim Gaussian z latent with a K-component Gaussian mixture so
both branches can be represented, and uses the DOMINANT component's mean as
the point estimate — an ambiguous galaxy keeps a secondary high-z mode in
its posterior without the point estimate being dragged there.

Differences vs photoz_mlpvae.py (everything else identical):
  z_head    : Linear(512→128)+ReLU+Linear(128→3K) → per-component
              (logit_k, mu_k, log_var_k) in raw (pre-sigmoid) space.
              Component mu biases are initialized spread across raw space
              ([-1.5, 0, +1.5] for K=3 → z ≈ 1.0 / 2.75 / 4.5) so modes
              start separated.
  z NLL     : standard MDN negative log-likelihood in PHYSICAL space,
              -log Σ_k π_k N(z_spec | zc_k, σ_k²), with each component's
              physical σ_k from the same delta-method push-forward as the
              base model. No reparameterization needed — the mixture
              density gives (π, mu, log_var) direct gradients.
  z_pred    : training → component sampled ~ Categorical(π), reparameterized
              within it (so vae_head/decoder see both branches);
              eval → dominant-component mean (argmax π_k).
  σ_z       : dominant component's physical σ.
  predict_z : z_samples drawn from the full mixture; z_mean/z_std from the
              dominant component.

The vae_head still reads [h.detach(), z_pred.detach()] — gradient isolation
is unchanged.
"""

import os
import numpy as np
import torch
import torch.nn as nn

from photoz_vae.model.vae_encoder import _ResBlock
from photoz_vae.model.speculator_torch import SpeculatorInoueIGM
from photoz_vae.model.filter_conv import FilterConv

from photoz_mlpvae.model.photoz_mlpvae import (
    PARAM_NAMES, PARAM_NAMES_15, N_PARAMS_15, N_PARAMS_FULL,
    ZRED_IDX_FULL, LOGMASS_IDX_FULL,
    _ZRED_MAX_SPEC, _ZRED_SIGMA_FLOOR,
    constrain_params_15,
)

_LOG_VAR_CLAMP = (-8.0, 4.0)
_LOG_2PI = float(np.log(2.0 * np.pi))


def _component_sigma_phys(mu_k: torch.Tensor, log_var_k: torch.Tensor) -> torch.Tensor:
    """Delta-method physical σ per mixture component (any trailing shape)."""
    s   = torch.sigmoid(mu_k)
    jac = _ZRED_MAX_SPEC * s * (1.0 - s)
    sigma_raw = torch.exp(0.5 * log_var_k)
    return (sigma_raw * jac).clamp(min=_ZRED_SIGMA_FLOOR)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: shared trunk + MDN z head + SPS (VAE) head
# ─────────────────────────────────────────────────────────────────────────────

class MLPVAEEncoderMDN(nn.Module):
    """
    Shared ResBlock trunk feeding an MDN z head and the z-conditioned SPS head.

    Parameters
    ----------
    n_in    : input dimension (default 40)
    width   : hidden width (default 512)
    n_latent: SPS latent dimension (default 15)
    n_mix   : number of z mixture components (default 3)
    dropout : dropout probability in ResBlocks (default 0.1)
    """

    def __init__(self, n_in: int = 40, width: int = 512,
                 n_latent: int = N_PARAMS_15, n_mix: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.n_latent = n_latent
        self.n_mix    = n_mix

        self.input_proj = nn.Sequential(
            nn.Linear(n_in, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock(width, dropout),
            _ResBlock(width, dropout),
        )

        # z head → [logits(K) | mu(K) | log_var(K)]
        self.z_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * n_mix),
        )

        self.vae_head = nn.Linear(width + 1, n_latent * 2)

        nn.init.xavier_uniform_(self.z_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.z_head[-1].bias)
        # Spread component means across raw space so modes start separated
        with torch.no_grad():
            spread = torch.linspace(-1.5, 1.5, n_mix)
            self.z_head[-1].bias[n_mix:2 * n_mix] = spread
        nn.init.xavier_uniform_(self.vae_head.weight, gain=0.1)
        nn.init.zeros_(self.vae_head.bias)

    def z_mixture(self, h: torch.Tensor):
        """Trunk features → (log_pi, mu, log_var), each (batch, K)."""
        out = self.z_head(h)
        K = self.n_mix
        logits   = out[:, :K]
        mu       = out[:, K:2 * K]
        log_var  = out[:, 2 * K:].clamp(*_LOG_VAR_CLAMP)
        log_pi   = torch.log_softmax(logits, dim=1)
        return log_pi, mu, log_var

    def z_point(self, log_pi, mu, log_var):
        """
        Point estimate + its σ: training → sample a component by π and
        reparameterize within it; eval → dominant-component mean.
        """
        if self.training:
            k = torch.distributions.Categorical(logits=log_pi).sample()   # (B,)
        else:
            k = log_pi.argmax(dim=1)
        idx      = k.unsqueeze(1)
        mu_k     = mu.gather(1, idx).squeeze(1)
        log_v_k  = log_var.gather(1, idx).squeeze(1)
        if self.training:
            std   = torch.exp(0.5 * log_v_k)
            z_raw = mu_k + std * torch.randn_like(std)
        else:
            z_raw = mu_k
        z_pred  = torch.sigmoid(z_raw) * _ZRED_MAX_SPEC
        sigma_z = _component_sigma_phys(mu_k, log_v_k)
        return z_pred, sigma_z

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        z_pred     : (batch,)   physical z point estimate (see z_point)
        sigma_z    : (batch,)   physical σ of the selected component
        log_pi     : (batch,K)  mixture log-weights
        mu_mix     : (batch,K)  raw-space component means
        log_var_mix: (batch,K)  raw-space component log-variances
        mu_15      : (batch,15) SPS posterior means, conditioned on z
        log_var_15 : (batch,15) SPS posterior log-variances
        """
        h = self.input_proj(x)
        h = self.res_blocks(h)

        log_pi, mu_mix, log_var_mix = self.z_mixture(h)
        z_pred, sigma_z = self.z_point(log_pi, mu_mix, log_var_mix)

        cond       = torch.cat([h.detach(), z_pred.detach().unsqueeze(1)], dim=1)
        vae_out    = self.vae_head(cond)
        mu_15      = vae_out[:, :self.n_latent]
        log_var_15 = vae_out[:, self.n_latent:]

        return z_pred, sigma_z, log_pi, mu_mix, log_var_mix, mu_15, log_var_15

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def freeze_vae_head(self):
        for p in self.vae_head.parameters():
            p.requires_grad_(False)

    def unfreeze_vae_head(self):
        for p in self.vae_head.parameters():
            p.requires_grad_(True)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class PhotozMLPVAE(nn.Module):
    """MDN-z variant of the z-separated PhotozMLPVAE. Same external API."""

    def __init__(self,
                 speculator_dir: str,
                 filter_dir: str,
                 encoder_width: int = 512,
                 encoder_dropout: float = 0.1,
                 encoder_n_in: int = 40,
                 n_mix: int = 3):
        super().__init__()

        self.encoder = MLPVAEEncoderMDN(
            n_in=encoder_n_in, width=encoder_width,
            n_latent=N_PARAMS_15, n_mix=n_mix, dropout=encoder_dropout,
        )

        self.speculator  = SpeculatorInoueIGM(speculator_dir)
        self.filter_conv = FilterConv(filter_dir, self.speculator.wl_rest)

        for p in self.speculator.parameters():
            p.requires_grad_(False)
        for p in self.filter_conv.parameters():
            p.requires_grad_(False)

    # ── Encode / decode ────────────────────────────────────────────────────

    def encode(self, x_phot: torch.Tensor):
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15) = self.encoder(x_phot)
        z_raw_15 = self.encoder.reparameterize(mu_15, log_var_15)
        return (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
                mu_15, log_var_15, z_raw_15)

    def decode(self, z_pred: torch.Tensor, z_raw_15: torch.Tensor) -> torch.Tensor:
        theta_15   = constrain_params_15(z_raw_15)
        zred_dec   = z_pred.detach().clamp(max=_ZRED_MAX_SPEC)
        theta_full = torch.cat([zred_dec.unsqueeze(1), theta_15], dim=1)
        log_spec, _ = self.speculator(theta_full)
        return self.filter_conv(log_spec,
                                theta_full[:, ZRED_IDX_FULL],
                                theta_full[:, LOGMASS_IDX_FULL])

    def forward(self, x_phot: torch.Tensor):
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_raw_15) = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)
        return m_ab_recon, z_pred, sigma_z, mu_15, log_var_15

    # ── Diagnostic interface ───────────────────────────────────────────────

    def encode_theta(self, x: torch.Tensor):
        (z_pred, sigma_z, _, _, _, mu_15, _) = self.encoder(x)
        log_sigma_z = torch.log(sigma_z)
        theta_15    = constrain_params_15(mu_15)
        theta_full  = torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)
        return z_pred, log_sigma_z, theta_full

    @torch.no_grad()
    def sample_theta_posterior(self, x, n_samples: int, device: str, rng):
        """Posterior samples; z drawn from the full mixture."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x.astype(np.float32)).to(device)
        (_, _, log_pi, mu_mix, log_var_mix, mu_15, log_var_15) = self.encoder(x)
        pi_np     = log_pi.exp().cpu().numpy()
        mu_np     = mu_mix.cpu().numpy()
        std_np    = np.exp(0.5 * log_var_mix.cpu().numpy())
        mu_15_np  = mu_15.cpu().numpy()
        std_15_np = np.exp(0.5 * log_var_15.cpu().numpy())

        n_gal = x.shape[0]
        all_samples = np.empty((n_gal, n_samples, N_PARAMS_FULL), dtype=np.float32)
        for i in range(n_gal):
            k       = rng.choice(self.encoder.n_mix, size=n_samples, p=pi_np[i] / pi_np[i].sum())
            eps_z   = rng.standard_normal(n_samples).astype(np.float32)
            z_raw_s = mu_np[i, k] + eps_z * std_np[i, k]
            z_s     = 1.0 / (1.0 + np.exp(-z_raw_s)) * _ZRED_MAX_SPEC
            z_s     = np.clip(z_s, 0.0, _ZRED_MAX_SPEC)

            eps_15   = rng.standard_normal((n_samples, N_PARAMS_15)).astype(np.float32)
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
        """Same 3-term structure; term 1 is the MDN mixture NLL."""
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_raw_15) = self.encode(x_phot)
        m_ab_recon = self.decode(z_pred, z_raw_15)

        # ── 1. Supervised redshift loss: mixture NLL in physical space ────
        zc_k    = torch.sigmoid(mu_mix) * _ZRED_MAX_SPEC          # (B, K)
        sig_k   = _component_sigma_phys(mu_mix, log_var_mix)      # (B, K)
        if use_nll_z:
            log_comp = (log_pi
                        - 0.5 * ((z_spec.unsqueeze(1) - zc_k) / sig_k) ** 2
                        - torch.log(sig_k) - 0.5 * _LOG_2PI)      # (B, K)
            per_gal = -torch.logsumexp(log_comp, dim=1)           # (B,)
        else:
            per_gal = (z_pred - z_spec) ** 2
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
        Returns
        -------
        z_samples : (batch, n_samples)  samples from the full mixture
        z_mean    : (batch,)            dominant-component mean (point estimate)
        z_std     : (batch,)            dominant-component physical σ
        """
        (_, _, log_pi, mu_mix, log_var_mix, _, _) = self.encoder(x_phot)

        k       = log_pi.argmax(dim=1)
        idx     = k.unsqueeze(1)
        mu_k    = mu_mix.gather(1, idx).squeeze(1)
        log_v_k = log_var_mix.gather(1, idx).squeeze(1)
        z_mean  = torch.sigmoid(mu_k) * _ZRED_MAX_SPEC
        z_std   = _component_sigma_phys(mu_k, log_v_k)

        ks      = torch.distributions.Categorical(logits=log_pi).sample((n_samples,)).T  # (B, S)
        mu_s    = mu_mix.gather(1, ks)
        std_s   = torch.exp(0.5 * log_var_mix.gather(1, ks))
        z_raw_s = mu_s + std_s * torch.randn_like(std_s)
        z_samples = (torch.sigmoid(z_raw_s) * _ZRED_MAX_SPEC).clamp(0.0, _ZRED_MAX_SPEC)

        return z_samples, z_mean, z_std

    @torch.no_grad()
    def predict_params(self, x_phot: torch.Tensor) -> torch.Tensor:
        (z_pred, _, _, _, _, mu_15, _) = self.encoder(x_phot)
        theta_15   = constrain_params_15(mu_15)
        return torch.cat([z_pred.unsqueeze(1), theta_15], dim=1)

    # ── Serialisation ─────────────────────────────────────────────────────

    def save(self, path: str, scaler=None, col_medians=None):
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
            },
            "scaler":      scaler,
            "col_medians": col_medians,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, speculator_dir: str, filter_dir: str,
             device: str = "cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg     = payload["encoder_config"]
        model   = cls(speculator_dir, filter_dir,
                      encoder_width=cfg["width"],
                      encoder_n_in=cfg.get("n_in", 40),
                      n_mix=cfg.get("n_mix", 3))
        model.load_state_dict(payload["model_state"])
        model.to(device)
        model.eval()
        return model, payload.get("scaler"), payload.get("col_medians")
