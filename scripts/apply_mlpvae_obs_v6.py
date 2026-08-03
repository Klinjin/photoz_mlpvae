#!/usr/bin/env python3
"""
apply_mlpvae_obs_v6.py
======================
Apply a trained PhotozMLPVAE checkpoint to dp1_matched_v4_test.hdf5 and
produce three diagnostic figures using the shared photoz_utils helpers:

  Fig 1  – estimated vs true redshift density scatter  (plot_scatter_density)
  Fig 2  – bias / σ / outlier rate vs z_spec           (plot_metrics_binned_3sig)
  Fig 3  – bias / σ / outlier rate vs i-band magnitude (plot_metrics_binned_3sig)

The model's use_colors / use_euclid settings are read from config.yaml in
the model directory, so any mlpvae checkpoint can be evaluated with this script.

Usage
-----
    # default: mlpvae_obs_v6
    /astro/users/lindajin/miniforge3/envs/WL_ML_Challenge/bin/python \\
        photoz_mlpvae/scripts/apply_mlpvae_obs_v6.py

    # LSST-only model
    /astro/users/lindajin/miniforge3/envs/WL_ML_Challenge/bin/python \\
        photoz_mlpvae/scripts/apply_mlpvae_obs_v6.py --model-name mlpvae_v1_lsst
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)

from obs_catalog.dataloader import build_features
from photoz_mlpvae.model.photoz_mlpvae_old import PhotozMLPVAE
from photoz_utils import (
    plot_scatter_density,
    plot_metrics_binned_3sig,
)

# ── Constants ──────────────────────────────────────────────────────────────────

TEST_HDF5 = os.path.join(
    _BASE, "obs_catalog", "data",
    "84_test_v4_match_ecdfs_sitcomtn154_1ec71937ff3e4a0a8283535193764ec7",
    "dp1_matched_v4_test.hdf5",
)
SPECULATOR_DIR = os.path.join(_BASE, "speculator", "trained", "Inoue_IGM")
FILTER_DIR     = os.path.join(_BASE, "obs_catalog", "filters")
OUT_BASE       = os.path.join(_BASE, "photoz_mlpvae", "trained")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

REF_MAG_COL = "i_cModelMag"


def main():
    parser = argparse.ArgumentParser(
        description="Apply a PhotozMLPVAE checkpoint to the DP1 test set.")
    parser.add_argument("--model-name", default="mlpvae_obs_v6",
                        help="Subdirectory name under photoz_mlpvae/trained/")
    args = parser.parse_args()

    out_dir   = os.path.join(OUT_BASE, args.model_name)
    ckpt_path = os.path.join(out_dir, "best.pt")

    # ── Read training config to get feature settings ───────────────────────
    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    use_colors = bool(cfg.get("use_colors", True))
    use_euclid = bool(cfg.get("use_euclid", True))
    print(f"Model: {args.model_name}  |  use_colors={use_colors}"
          f"  use_euclid={use_euclid}")

    # ── Load test data ─────────────────────────────────────────────────────
    print("Loading test HDF5 ...")
    with h5py.File(TEST_HDF5, "r") as f:
        te_df = pd.DataFrame({k: f[k][:] for k in f.keys()})
    print(f"  {len(te_df):,} galaxies")

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading checkpoint ...")
    model, scaler, col_medians = PhotozMLPVAE.load(
        ckpt_path, SPECULATOR_DIR, FILTER_DIR, device=DEVICE)
    model.eval()

    # ── Build features ─────────────────────────────────────────────────────
    print("Building features ...")
    X_te, _, _, _, z_te, _, _ = build_features(
        te_df, use_colors=use_colors, use_euclid=use_euclid,
        scaler=scaler, col_medians=col_medians, fit=False,
    )

    # ── Inference ──────────────────────────────────────────────────────────
    print("Running inference ...")
    with torch.no_grad():
        _, z_pred_t, sigma_z_t = model.predict_z(
            torch.from_numpy(X_te).to(DEVICE), n_samples=1)
    z_pred  = z_pred_t.cpu().numpy()
    sigma_z = sigma_z_t.cpu().numpy() if sigma_z_t is not None else None
    dz      = (z_pred - z_te) / (1.0 + z_te)

    finite   = np.isfinite(dz) & np.isfinite(z_te)
    dz_fin   = dz[finite]

    nmad_all = 1.4826 * np.median(np.abs(dz_fin - np.median(dz_fin)))
    print(f"\nGlobal metrics (N={finite.sum():,}):")
    print(f"  Δz             = {np.mean(dz_fin):.4f}")
    print(f"  σz (NMAD)      = {nmad_all:.4f}")
    print(f"  Outlier (>3σ)  = {np.mean(np.abs(dz_fin) > 3*nmad_all):.4f}")
    print(f"  Outlier (>0.2) = {np.mean(np.abs(dz_fin) > 0.2):.4f}")

    if sigma_z is not None:
        sigma_fin = sigma_z[finite]
        pull      = dz_fin / (sigma_fin + 1e-8)
        print(f"\nUncertainty metrics:")
        print(f"  median σ_z     = {np.median(sigma_fin):.4f}")
        print(f"  pull mean      = {np.mean(pull):.4f}")
        print(f"  pull std       = {np.std(pull):.4f}  (ideal = 1.0)")

    # ── Figure 1: scatter density ──────────────────────────────────────────
    print("\nFig 1: scatter density ...")
    plot_scatter_density(
        z_te, z_pred,
        save_path=os.path.join(out_dir, "dp1_v4_test_scatter.png"),
    )

    # ── Figure 2: metrics vs z_spec ────────────────────────────────────────
    print("Fig 2: metrics vs z_spec ...")
    plot_metrics_binned_3sig(
        z_te, dz,
        bins=np.arange(0.0, 3.2, 0.2),
        xlabel=r"$z_{\rm spec}$",
        save_path=os.path.join(out_dir, "dp1_v4_test_metrics_vs_z.png"),
    )

    # ── Figure 3: metrics vs magnitude ─────────────────────────────────────
    print("Fig 3: metrics vs magnitude ...")
    plot_metrics_binned_3sig(
        te_df[REF_MAG_COL].values, dz,
        bins=np.arange(18.0, 25.5, 0.5),
        xlabel="Magnitude",
        save_path=os.path.join(out_dir, "dp1_v4_test_metrics_vs_mag.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
