#!/usr/bin/env python3
"""
apply_mlpvae_v2_dropout.py
===========================
Apply the trained mlpvae_v2_lsst_gaap1p0 checkpoint (LSST-only, GAAP 1.0" mags)
to g-dropout and r-dropout colour-selected galaxies from the full DP1 ECDFS
crossmatched catalog, producing photo-z point estimates with aleatoric
uncertainty (sigma_z) for each selected galaxy.

Selection cuts (GAAP 1.0" magnitudes, matching redshift.ipynb) require only
photometry + morphology -- NOT a known spectroscopic/photometric redshift --
so the model is applied to genuine dropout candidates, including those with
no reference redshift.

  g-dropout (z ~ 3.5-4.2):
    (g-r > 1.0) & (r-i < 1.0) & (g-r > 0.8 + 1.5*(r-i)) & snr_i > 5 & extended

  r-dropout (z ~ 4.5-5.5):
    (r-i > 1.2) & (i-z < 0.7) & (r-i > 1.0 + 1.5*(i-z)) & extended

Usage
-----
    /astro/users/lindajin/miniforge3/envs/WL_ML_Challenge/bin/python \\
        photoz_mlpvae/scripts/apply_mlpvae_v2_dropout.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from obs_catalog.dataloader import build_features
from photoz_mlpvae.model.photoz_mlpvae import PhotozMLPVAE
from photoz_utils import plot_scatter_density

CATALOG = os.path.join(_BASE, "obs_catalog", "data",
                       "dp1_ecdfs_crossmatched_catalog.parquet")
MODEL_NAME     = "mlpvae_v2_lsst_gaap1p0"
MODEL_DIR      = os.path.join(_BASE, "photoz_mlpvae", "trained", MODEL_NAME)
CKPT_PATH      = os.path.join(MODEL_DIR, "best.pt")
SPECULATOR_DIR = os.path.join(_BASE, "speculator", "trained", "Inoue_IGM")
FILTER_DIR     = os.path.join(_BASE, "obs_catalog", "filters")
OUT_DIR        = os.path.join(MODEL_DIR, "dropout_application")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def select_dropouts(df):
    """GAAP 1.0" colour-selected g-dropout / r-dropout masks (no z requirement)."""
    g_col, r_col, i_col, z_col = (
        df["g_gaap1p0Mag"], df["r_gaap1p0Mag"], df["i_gaap1p0Mag"], df["z_gaap1p0Mag"])
    i_err = df["i_gaap1p0MagErr"]

    g_r = g_col - r_col
    r_i = r_col - i_col
    i_z = i_col - z_col
    snr_i = 1.0 / i_err.replace(0, np.nan)
    is_extended = df["refExtendedness"] == 1

    g_dropout = (
        (g_r > 1.0) & (r_i < 1.0) & (g_r > 0.8 + 1.5 * r_i) &
        (snr_i > 5) & is_extended &
        np.isfinite(g_r) & np.isfinite(r_i)
    )
    r_dropout = (
        (r_i > 1.2) & (i_z < 0.7) & (r_i > 1.0 + 1.5 * i_z) &
        is_extended &
        np.isfinite(r_i) & np.isfinite(i_z)
    )
    return g_dropout, r_dropout


def run_inference(model, sub_df, scaler, col_medians):
    X, _, _, _, z_true, _, _ = build_features(
        sub_df, use_colors=True, use_euclid=False, use_gaap=True,
        scaler=scaler, col_medians=col_medians, fit=False,
    )
    with torch.no_grad():
        _, z_pred_t, sigma_z_t = model.predict_z(
            torch.from_numpy(X).to(DEVICE), n_samples=1)
    return z_pred_t.cpu().numpy(), sigma_z_t.cpu().numpy(), z_true


def plot_dropout_zdist(name, z_pred, z_true, save_path):
    """Histogram of z_pred for all selected candidates, with the reference-z
    subset (spec/photo/grism) overlaid where available."""
    has_z = np.isfinite(z_true) & (z_true > 0)
    zmax = np.ceil(max(6.0, float(np.nanmax(z_pred))) / 0.5) * 0.5
    bins = np.arange(0, zmax + 0.2, 0.2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(z_pred, bins=bins, histtype="step", lw=2, color="tomato",
           label=fr"$z_{{\rm pred}}$ (all {len(z_pred):,} candidates)")
    if has_z.sum() > 0:
        ax.hist(z_true[has_z], bins=bins, histtype="step", lw=2, color="steelblue",
               linestyle="--", label=fr"$z_{{\rm ref}}$ ({has_z.sum():,} with known z)")
    ax.set_xlabel("Redshift", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(name)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved redshift histogram -> {save_path}")


def summarize(name, z_pred, sigma_z, z_true):
    has_z = np.isfinite(z_true) & (z_true > 0)
    print(f"\n{name}: {len(z_pred):,} galaxies selected "
          f"({has_z.sum():,} with a reference redshift)")
    print(f"  median z_pred = {np.median(z_pred):.3f}   "
          f"median sigma_z = {np.median(sigma_z):.3f}")
    if has_z.sum() > 0:
        dz = (z_pred[has_z] - z_true[has_z]) / (1.0 + z_true[has_z])
        med = np.median(dz)
        nmad = 1.4826 * np.median(np.abs(dz - med))
        pull = dz / (sigma_z[has_z] + 1e-8)
        print(f"  [vs reference z]  bias={np.mean(dz):.4f}  "
              f"sigma_NMAD={nmad:.4f}  outlier(>0.15)={np.mean(np.abs(dz) > 0.15):.4f}")
        print(f"  [pull]            mean={np.mean(pull):.3f}  std={np.std(pull):.3f} (ideal=1.0)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading catalog {CATALOG} ...")
    df = pd.read_parquet(CATALOG)
    print(f"  {len(df):,} objects")

    g_mask, r_mask = select_dropouts(df)
    print(f"g-dropout candidates: {g_mask.sum():,}")
    print(f"r-dropout candidates: {r_mask.sum():,}")
    print(f"(overlap: {(g_mask & r_mask).sum():,})")

    print(f"\nLoading checkpoint {MODEL_NAME} ...")
    model, scaler, col_medians = PhotozMLPVAE.load(
        CKPT_PATH, SPECULATOR_DIR, FILTER_DIR, device=DEVICE)
    model.eval()

    results = []
    for name, mask in [("g_dropout", g_mask), ("r_dropout", r_mask)]:
        sub_df = df[mask].reset_index(drop=True)
        z_pred, sigma_z, z_true = run_inference(model, sub_df, scaler, col_medians)
        summarize(name, z_pred, sigma_z, z_true)

        out = pd.DataFrame({
            "objectId":       sub_df["objectId"],
            "coord_ra":       sub_df["coord_ra"],
            "coord_dec":      sub_df["coord_dec"],
            "dropout_type":   name,
            "z_pred":         z_pred,
            "sigma_z_pred":   sigma_z,
            "z_ref":          sub_df["redshift"],
            "z_ref_type":     sub_df["type"],
        })
        results.append(out)

        has_z = np.isfinite(z_true) & (z_true > 0)
        if has_z.sum() > 10:
            plot_scatter_density(
                z_true[has_z], z_pred[has_z], zmax=6.0,
                save_path=os.path.join(OUT_DIR, f"{name}_scatter.png"))
        plot_dropout_zdist(
            name, z_pred, z_true,
            save_path=os.path.join(OUT_DIR, f"{name}_zdist.png"))

    out_df = pd.concat(results, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "dropout_z_predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(out_df):,} predictions -> {out_csv}")


if __name__ == "__main__":
    main()
