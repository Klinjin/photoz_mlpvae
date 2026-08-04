"""
speculator_torch.py
===================
Pure-PyTorch port of the Inoue_IGM Speculator emulator.

The Speculator architecture (per wavelength band):
  - Inputs: 16 normalised SPS parameters
  - 3 hidden layers of width 256 with learnable SWISH-like activation:
        x_post = (β + (1 - β) * σ(α ⊙ x_pre)) ⊙ x_pre
  - 1 linear output layer → n_pca PCA coefficients
  - Inverse PCA + un-normalisation → log₁₀ L_ν spectrum (rest-frame)

All weights are registered as non-trainable buffers (frozen decoder).

Three wavelength bands:
  30_100  : 300 –  999 Å  (523 wavelength points, 30 PCAs)
  100_400 : 1001 – 3997 Å (602 wavelength points, 20 PCAs)
  400_3000: 4006 – 29931 Å (874 wavelength points, 20 PCAs)
Total: 1999 wavelength points

Usage
-----
    spec = SpeculatorInoueIGM(run_dir="/astro/users/lindajin/speculator/trained/Inoue_IGM")
    log_spec, wl_rest = spec(params)   # params: (batch, 16) → (batch, 1999), (1999,)
"""

import os
import numpy as np
import torch
import torch.nn as nn

_BANDS = ["30_100", "100_400", "400_3000"]


class SpeculatorBand(nn.Module):
    """Single-band Speculator NN: (batch, 16) → (batch, n_wave_band) log₁₀ L_ν."""

    def __init__(self, model_path: str, pca_path: str):
        super().__init__()

        m = np.load(model_path)
        p = np.load(pca_path)

        self.n_act: int = int(m["n_act"])   # 3 hidden layers

        # ── NN weights (frozen buffers) ───────────────────────────────────
        for i in range(self.n_act):
            self.register_buffer(f"W_{i}",    torch.from_numpy(m[f"W_{i}"].astype(np.float32)))
            self.register_buffer(f"b_{i}",    torch.from_numpy(m[f"b_{i}"].astype(np.float32)))
            self.register_buffer(f"alpha_{i}", torch.from_numpy(m[f"alpha_{i}"].astype(np.float32)))
            self.register_buffer(f"beta_{i}",  torch.from_numpy(m[f"beta_{i}"].astype(np.float32)))

        n_act = self.n_act
        self.register_buffer(f"W_{n_act}", torch.from_numpy(m[f"W_{n_act}"].astype(np.float32)))
        self.register_buffer(f"b_{n_act}", torch.from_numpy(m[f"b_{n_act}"].astype(np.float32)))

        # ── PCA normalisation (frozen buffers) ────────────────────────────
        self.register_buffer("param_shift",       torch.from_numpy(p["parameter_shift"].astype(np.float32)))
        self.register_buffer("param_scale",       torch.from_numpy(p["parameter_scale"].astype(np.float32)))
        self.register_buffer("pca_shift",         torch.from_numpy(p["pca_shift"].astype(np.float32)))
        self.register_buffer("pca_scale",         torch.from_numpy(p["pca_scale"].astype(np.float32)))
        self.register_buffer("pca_matrix",        torch.from_numpy(p["pca_transform_matrix"].astype(np.float32)))
        self.register_buffer("log_spec_scale",    torch.from_numpy(p["log_spectrum_scale"].astype(np.float32)))
        self.register_buffer("log_spec_shift",    torch.from_numpy(p["log_spectrum_shift"].astype(np.float32)))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        params : (batch, 16)  physical SPS parameters (un-normalised)

        Returns
        -------
        log_spec : (batch, n_wave_band)  log₁₀ L_ν  [L_sun/Hz per 1 M_sun formed]
        """
        # Normalise inputs
        x = (params - self.param_shift) / self.param_scale  # (batch, 16)

        # Hidden layers with learnable activation
        for i in range(self.n_act):
            W     = getattr(self, f"W_{i}")      # (in, 256)
            b     = getattr(self, f"b_{i}")      # (256,)
            alpha = getattr(self, f"alpha_{i}")  # (256,)
            beta  = getattr(self, f"beta_{i}")   # (256,)
            x_pre = x @ W + b
            x = (beta + (1.0 - beta) * torch.sigmoid(alpha * x_pre)) * x_pre

        # Linear output layer → normalised PCA coefficients
        n = self.n_act
        x = x @ getattr(self, f"W_{n}") + getattr(self, f"b_{n}")   # (batch, n_pca)

        # Un-normalise PCA coefficients
        pca_coeffs = x * self.pca_scale + self.pca_shift             # (batch, n_pca)

        # Inverse PCA → normalised log-spectrum, then un-normalise
        log_spec = (pca_coeffs @ self.pca_matrix) * self.log_spec_scale + self.log_spec_shift
        return log_spec   # (batch, n_wave_band)


class SpeculatorInoueIGM(nn.Module):
    """
    Full Inoue_IGM Speculator: concatenates the 3 wavelength bands.

    Forward pass:
        log_spec, wl_rest = model(params)

    Parameters
    ----------
    run_dir : str
        Path to the trained Inoue_IGM run directory, e.g.:
        /astro/users/lindajin/speculator/trained/Inoue_IGM
    """

    def __init__(self, run_dir: str):
        super().__init__()

        self.bands = nn.ModuleList()
        wl_parts = []

        for band_name in _BANDS:
            model_path = os.path.join(run_dir, band_name, "model.npz")
            pca_path   = os.path.join(run_dir, band_name, "pca_basis.npz")
            wl_path    = os.path.join(run_dir, "npy", f"val_{band_name}_wl.npy")

            for p in (model_path, pca_path, wl_path):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Missing Speculator file: {p}")

            self.bands.append(SpeculatorBand(model_path, pca_path))
            wl_parts.append(torch.from_numpy(np.load(wl_path).astype(np.float32)))

        # wl_rest: full wavelength grid (Å, rest-frame), shape (1999,)
        wl_cat = torch.cat(wl_parts, dim=0)
        self.register_buffer("wl_rest", wl_cat)

        # Keep parameters frozen
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def n_wave(self) -> int:
        return int(self.wl_rest.shape[0])

    def forward(self, params: torch.Tensor):
        """
        Parameters
        ----------
        params : (batch, 16)  physical SPS parameters

        Returns
        -------
        log_spec : (batch, n_wave)   log₁₀ L_ν [L_sun/Hz per 1 M_sun formed]
        wl_rest  : (n_wave,)         rest-frame wavelengths [Å]
        """
        parts = [band(params) for band in self.bands]
        log_spec = torch.cat(parts, dim=1)      # (batch, 1999)
        return log_spec, self.wl_rest
