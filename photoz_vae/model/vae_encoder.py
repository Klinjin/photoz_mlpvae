"""
vae_encoder.py
==============
Photometric encoder for the PhotozVAE.

Input
-----
Pre-processed 40-dim feature vector:
  [10 scaled AB mags | 10 scaled AB mag errors | 20 binary missingness flags]

Architecture
------------
  Linear(40 → 512) → BN → ReLU              (input projection)
  ResBlock(512, dropout=0.1) × 2             (residual hidden layers)
  Linear(512 → 32)                           (output: μ and log σ²)

Residual blocks preserve the photometric color signal through depth and
allow the network to learn additive corrections rather than full transforms.

Output
------
  mu      : (batch, 16)  posterior means in unconstrained space
  log_var : (batch, 16)  posterior log-variances in unconstrained space

Redshift is treated as one of the 16 SPS latent parameters, not a separate
prediction task: its point estimate is constrain_params(mu)[0]
(sigmoid(mu[:,0]) * 5.5) and its uncertainty is derived from log_var[:,0]
(no dedicated MDN/sigma head for either).
"""

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Two-layer residual block with BatchNorm and Dropout."""
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class PhotozEncoder(nn.Module):
    """
    Residual MLP encoder: photometry (40-dim) → posterior parameters (μ, log σ²) × 16.

    Parameters
    ----------
    n_in    : input dimension (default 40: 10 mags + 10 errs + 20 miss flags)
    width   : hidden layer width (default 512)
    n_latent: latent dimension = number of SPS parameters (default 16)
    dropout : dropout probability (default 0.1)
    """

    def __init__(self, n_in: int = 40, width: int = 512,
                 n_latent: int = 16, dropout: float = 0.1):
        super().__init__()
        self.n_latent = n_latent

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_in, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )

        # Residual body
        self.res_blocks = nn.Sequential(
            _ResBlock(width, dropout),
            _ResBlock(width, dropout),
        )

        # Output head: → μ and log σ² concatenated (VAE latent, used by decoder)
        self.head = nn.Linear(width, n_latent * 2)

        # Initialise output layers with small weights to avoid large initial KL
        nn.init.xavier_uniform_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (batch, n_in)  scaled photometric features

        Returns
        -------
        mu      : (batch, n_latent)  posterior means
        log_var : (batch, n_latent)  posterior log-variances (clamped for stability)

        Redshift has no dedicated uncertainty head: dimension 0 of (mu, log_var)
        *is* its posterior, exactly like the other 15 SPS parameters. See
        photoz_vae.py's `_zred_sigma_phys` for how log_var[:,0] is propagated
        into a physical-space σ_z.
        """
        h       = self.input_proj(x)
        h       = self.res_blocks(h)
        out     = self.head(h)
        mu      = out[:, :self.n_latent]
        log_var = out[:, self.n_latent:].clamp(-6.0, 4.0)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterisation trick: z = μ + ε × exp(0.5 × log σ²)

        During inference, set model.eval() to use the mean (ε=0).
        During training, ε ~ N(0, I) enables gradient flow.

        Returns
        -------
        z : (batch, n_latent)  sample from q(z|x)
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # deterministic mean at eval time

