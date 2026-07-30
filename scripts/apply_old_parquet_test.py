#!/usr/bin/env python3
"""
apply_old_parquet_test.py
==========================
Apply a trained PhotozMLPVAE checkpoint to the *old* parquet-catalog test
split (load_catalog(CATALOG_DEFAULT) → te_df, the pre-hdf5_presplit test
set) and save a test scatter plot. Useful for checking how a model trained
on the newer hdf5-presplit train/test files generalizes to the older
catalog-derived test split.

The model's use_colors / use_euclid / use_gaap settings are read from
config.yaml in the model directory.

Usage
-----
    conda run -n WL_ML_Challenge python \\
        photoz_mlpvae/scripts/apply_old_parquet_test.py \\
        --model-name mlpvae_v2_euclid_gaap1p0
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)

from obs_catalog.dataloader import load_catalog, build_features
from photoz_mlpvae.model.photoz_mlpvae import PhotozMLPVAE
from photoz_utils import plot_scatter

SPECULATOR_DIR = os.path.join(_BASE, "speculator", "trained", "Inoue_IGM")
FILTER_DIR     = os.path.join(_BASE, "obs_catalog", "filters")
OUT_BASE       = os.path.join(_BASE, "photoz_mlpvae", "trained")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(
        description="Apply a PhotozMLPVAE checkpoint to the old parquet-catalog test split.")
    parser.add_argument("--model-name", default="mlpvae_v2_euclid_gaap1p0",
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
    use_gaap   = bool(cfg.get("use_gaap", False))
    print(f"Model: {args.model_name}  |  use_colors={use_colors}  "
          f"use_euclid={use_euclid}  use_gaap={use_gaap}")

    # ── Load old parquet-catalog test split ─────────────────────────────────
    print("Loading old parquet catalog test split ...")
    _, _, te_df = load_catalog()
    print(f"  {len(te_df):,} galaxies")

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading checkpoint ...")
    model, scaler, col_medians = PhotozMLPVAE.load(
        ckpt_path, SPECULATOR_DIR, FILTER_DIR, device=DEVICE)
    model.eval()

    # ── Build features ───────────────────────────────────────────────────────
    print("Building features ...")
    X_te, _, _, _, z_te, _, _ = build_features(
        te_df, use_colors=use_colors, use_euclid=use_euclid, use_gaap=use_gaap,
        scaler=scaler, col_medians=col_medians, fit=False,
    )

    # ── Inference ────────────────────────────────────────────────────────────
    print("Running inference ...")
    with torch.no_grad():
        _, z_pred_t, _ = model.predict_z(
            torch.from_numpy(X_te).to(DEVICE), n_samples=1)
    z_pred = z_pred_t.cpu().numpy()

    # ── Scatter plot ─────────────────────────────────────────────────────────
    save_path = os.path.join(out_dir, "test_scatter_old_parquet.png")
    bias, nmad, fout, rms = plot_scatter(
        z_te, z_pred, save_path,
        title=f"{args.model_name}  old-parquet test",
    )
    print(f"\nOld-parquet test metrics (N={len(z_te):,}):")
    print(f"  Bias    : {bias:.4f}")
    print(f"  σ_NMAD  : {nmad:.4f}")
    print(f"  Outliers: {fout*100:.2f}%")
    print(f"  RMS     : {rms:.4f}")


if __name__ == "__main__":
    main()
