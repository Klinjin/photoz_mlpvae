"""
obs_catalog/dataloader.py
=========================
Shared photometric-redshift dataloader for all photoz_* models.

Encoder input dimension depends on use_colors and use_euclid:

  use_euclid=True  (default, 10 bands: 6 LSST + 4 Euclid)
    Magnitude mode (use_colors=False):
      [10 scaled AB mags | 10 scaled AB mag errors |
       10 mag-missingness flags | 10 err-missingness flags]  → 40-dim
    Colour mode (use_colors=True):
      [9 colours | 9 colour-errors | i_ref | i_ref_err |
       20 missingness flags]                                  → 40-dim

  use_euclid=False (6 LSST bands only)
    Magnitude mode:
      [6 scaled AB mags | 6 scaled AB mag errors |
       6 mag-missingness flags | 6 err-missingness flags]    → 24-dim
    Colour mode:
      [5 colours | 5 colour-errors | i_ref | i_ref_err |
       12 missingness flags]                                  → 24-dim

Public API
----------
  load_catalog(path=CATALOG_DEFAULT, seed=42, val_frac=0.08, test_frac=0.20)
      Returns (tr_df, val_df, te_df).
      val_df is an empty DataFrame if val_frac=0.

  build_features(df, use_colors=True, use_euclid=True, scaler=None,
                 col_medians=None, fit=False)
      Returns (X, mags_obs, mag_errs, mask, z_spec, scaler, col_medians).
        X           (N, D) float32   encoder input; D=40 with Euclid, 24 without
        mags_obs    (N, B) float32   raw AB mags  (imputed, for reconstruction)
        mag_errs    (N, B) float32   raw mag errs (imputed, clipped ≥ 0.005)
        mask        (N, B) float32   band-presence mask; B=10 with Euclid, 6 without
        z_spec      (N,)   float32   spectroscopic redshift
        scaler      fitted StandardScaler
        col_medians imputation medians

  compute_z_weights(z_spec, n_bins=25, z_max=5.5)
      Inverse-frequency per-galaxy weights (mean = 1).

  make_dataloader(X, mags, errs, mask, z, weights, batch_size, shuffle,
                  num_workers=2)
      PyTorch DataLoader; each batch is (x_b, mags_b, errs_b, mask_b, z_b, zw_b).

Constants (re-exported for convenience)
----------------------------------------
  MAG_COLS, ERR_COLS, TARGET_COL, N_BANDS, N_INPUT
  LSST_MAG_COLS, LSST_ERR_COLS
  GAAP_MAG_COLS, GAAP_ERR_COLS, LSST_GAAP_MAG_COLS, LSST_GAAP_ERR_COLS
  PSF_MAG_COLS, PSF_ERR_COLS, LSST_PSF_MAG_COLS, LSST_PSF_ERR_COLS
  COLOR_PAIRS, REF_MAG, REF_ERR
  GAAP_COLOR_PAIRS, GAAP_REF_MAG, GAAP_REF_ERR
  PSF_COLOR_PAIRS, PSF_REF_MAG, PSF_REF_ERR, CATALOG_DEFAULT

Example – photoz_ae / photoz_mlpvae (72/8/20 split with val)
-------------------------------------------------------------
    from obs_catalog.dataloader import (
        load_catalog, build_features, compute_z_weights, make_dataloader,
    )
    tr_df, val_df, te_df = load_catalog(CATALOG)
    X_tr, mags_tr, errs_tr, mask_tr, z_tr, scaler, col_med = \\
        build_features(tr_df, use_colors=True, use_euclid=True, fit=True)
    X_val, mags_val, errs_val, mask_val, z_val, _, _ = \\
        build_features(val_df, use_colors=True, use_euclid=True,
                       scaler=scaler, col_medians=col_med)
    tr_loader = make_dataloader(X_tr, mags_tr, errs_tr, mask_tr, z_tr,
                                compute_z_weights(z_tr), batch_size=256, shuffle=True)

Example – photoz_bnn / photoz_mlp (80/20 split, no val)
---------------------------------------------------------
    tr_df, _, te_df = load_catalog(CATALOG, val_frac=0.0, test_frac=0.20)
    X_tr, _, _, _, y_tr, scaler, col_med = build_features(tr_df, use_colors=True, fit=True)
    X_te, _, _, _, y_te, _, _            = build_features(te_df, use_colors=True,
                                                           scaler=scaler, col_medians=col_med)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────────────────────────────────────

MAG_COLS = [
    "u_cModelMag",        # 0  LSST u
    "g_cModelMag",        # 1  LSST g
    "r_cModelMag",        # 2  LSST r
    "i_cModelMag",        # 3  LSST i
    "z_cModelMag",        # 4  LSST z
    "y_cModelMag",        # 5  LSST y
    "euclid_vis_psfMag",  # 6  Euclid VIS
    "euclid_y_unifMag",   # 7  Euclid Y
    "euclid_j_unifMag",   # 8  Euclid J
    "euclid_h_unifMag",   # 9  Euclid H
]
ERR_COLS = [
    "u_cModelMagErr",
    "g_cModelMagErr",
    "r_cModelMagErr",
    "i_cModelMagErr",
    "z_cModelMagErr",
    "y_cModelMagErr",
    "euclid_vis_psfMagErr",
    "euclid_y_unifMagErr",
    "euclid_j_unifMagErr",
    "euclid_h_unifMagErr",
]
TARGET_COL = "redshift"
N_BANDS    = len(MAG_COLS)   # 10  (with Euclid)
N_INPUT    = 40              # encoder input dimension with Euclid

# LSST-only subsets (use_euclid=False) — 6 bands, 24-dim encoder input
LSST_MAG_COLS = MAG_COLS[:6]
LSST_ERR_COLS = ERR_COLS[:6]

# GAAP 1.0-arcsec aperture magnitudes (LSST bands only; Euclid unchanged)
GAAP_MAG_COLS = [
    "u_gaap1p0Mag",       # 0  LSST u
    "g_gaap1p0Mag",       # 1  LSST g
    "r_gaap1p0Mag",       # 2  LSST r
    "i_gaap1p0Mag",       # 3  LSST i
    "z_gaap1p0Mag",       # 4  LSST z
    "y_gaap1p0Mag",       # 5  LSST y
    "euclid_vis_psfMag",  # 6  Euclid VIS
    "euclid_y_unifMag",   # 7  Euclid Y
    "euclid_j_unifMag",   # 8  Euclid J
    "euclid_h_unifMag",   # 9  Euclid H
]
GAAP_ERR_COLS = [
    "u_gaap1p0MagErr",
    "g_gaap1p0MagErr",
    "r_gaap1p0MagErr",
    "i_gaap1p0MagErr",
    "z_gaap1p0MagErr",
    "y_gaap1p0MagErr",
    "euclid_vis_psfMagErr",
    "euclid_y_unifMagErr",
    "euclid_j_unifMagErr",
    "euclid_h_unifMagErr",
]
LSST_GAAP_MAG_COLS = GAAP_MAG_COLS[:6]
LSST_GAAP_ERR_COLS = GAAP_ERR_COLS[:6]

# PSF magnitudes (LSST bands only; Euclid unchanged)
PSF_MAG_COLS = [
    "u_psfMag",           # 0  LSST u
    "g_psfMag",           # 1  LSST g
    "r_psfMag",           # 2  LSST r
    "i_psfMag",           # 3  LSST i
    "z_psfMag",           # 4  LSST z
    "y_psfMag",           # 5  LSST y
    "euclid_vis_psfMag",  # 6  Euclid VIS
    "euclid_y_unifMag",   # 7  Euclid Y
    "euclid_j_unifMag",   # 8  Euclid J
    "euclid_h_unifMag",   # 9  Euclid H
]
PSF_ERR_COLS = [
    "u_psfMagErr",
    "g_psfMagErr",
    "r_psfMagErr",
    "i_psfMagErr",
    "z_psfMagErr",
    "y_psfMagErr",
    "euclid_vis_psfMagErr",
    "euclid_y_unifMagErr",
    "euclid_j_unifMagErr",
    "euclid_h_unifMagErr",
]
LSST_PSF_MAG_COLS = PSF_MAG_COLS[:6]
LSST_PSF_ERR_COLS = PSF_ERR_COLS[:6]

# Adjacent-band colour pairs: (mag1, err1, mag2, err2, colour_name)
COLOR_PAIRS = [
    ("u_cModelMag",       "u_cModelMagErr",       "g_cModelMag",       "g_cModelMagErr",       "u-g"),
    ("g_cModelMag",       "g_cModelMagErr",       "r_cModelMag",       "r_cModelMagErr",       "g-r"),
    ("r_cModelMag",       "r_cModelMagErr",       "i_cModelMag",       "i_cModelMagErr",       "r-i"),
    ("i_cModelMag",       "i_cModelMagErr",       "z_cModelMag",       "z_cModelMagErr",       "i-z"),
    ("z_cModelMag",       "z_cModelMagErr",       "y_cModelMag",       "y_cModelMagErr",       "z-y"),
    ("y_cModelMag",       "y_cModelMagErr",       "euclid_vis_psfMag", "euclid_vis_psfMagErr", "y-vis"),
    ("euclid_vis_psfMag", "euclid_vis_psfMagErr", "euclid_y_unifMag",  "euclid_y_unifMagErr",  "vis-ey"),
    ("euclid_y_unifMag",  "euclid_y_unifMagErr",  "euclid_j_unifMag",  "euclid_j_unifMagErr",  "ey-ej"),
    ("euclid_j_unifMag",  "euclid_j_unifMagErr",  "euclid_h_unifMag",  "euclid_h_unifMagErr",  "ej-eh"),
]
GAAP_COLOR_PAIRS = [
    ("u_gaap1p0Mag",      "u_gaap1p0MagErr",      "g_gaap1p0Mag",      "g_gaap1p0MagErr",      "u-g"),
    ("g_gaap1p0Mag",      "g_gaap1p0MagErr",       "r_gaap1p0Mag",      "r_gaap1p0MagErr",      "g-r"),
    ("r_gaap1p0Mag",      "r_gaap1p0MagErr",       "i_gaap1p0Mag",      "i_gaap1p0MagErr",      "r-i"),
    ("i_gaap1p0Mag",      "i_gaap1p0MagErr",       "z_gaap1p0Mag",      "z_gaap1p0MagErr",      "i-z"),
    ("z_gaap1p0Mag",      "z_gaap1p0MagErr",       "y_gaap1p0Mag",      "y_gaap1p0MagErr",      "z-y"),
    ("y_gaap1p0Mag",      "y_gaap1p0MagErr",       "euclid_vis_psfMag", "euclid_vis_psfMagErr", "y-vis"),
    ("euclid_vis_psfMag", "euclid_vis_psfMagErr",  "euclid_y_unifMag",  "euclid_y_unifMagErr",  "vis-ey"),
    ("euclid_y_unifMag",  "euclid_y_unifMagErr",   "euclid_j_unifMag",  "euclid_j_unifMagErr",  "ey-ej"),
    ("euclid_j_unifMag",  "euclid_j_unifMagErr",   "euclid_h_unifMag",  "euclid_h_unifMagErr",  "ej-eh"),
]
PSF_COLOR_PAIRS = [
    ("u_psfMag",          "u_psfMagErr",           "g_psfMag",          "g_psfMagErr",          "u-g"),
    ("g_psfMag",          "g_psfMagErr",           "r_psfMag",          "r_psfMagErr",          "g-r"),
    ("r_psfMag",          "r_psfMagErr",           "i_psfMag",          "i_psfMagErr",          "r-i"),
    ("i_psfMag",          "i_psfMagErr",           "z_psfMag",          "z_psfMagErr",          "i-z"),
    ("z_psfMag",          "z_psfMagErr",           "y_psfMag",          "y_psfMagErr",          "z-y"),
    ("y_psfMag",          "y_psfMagErr",           "euclid_vis_psfMag", "euclid_vis_psfMagErr", "y-vis"),
    ("euclid_vis_psfMag", "euclid_vis_psfMagErr",  "euclid_y_unifMag",  "euclid_y_unifMagErr",  "vis-ey"),
    ("euclid_y_unifMag",  "euclid_y_unifMagErr",   "euclid_j_unifMag",  "euclid_j_unifMagErr",  "ey-ej"),
    ("euclid_j_unifMag",  "euclid_j_unifMagErr",   "euclid_h_unifMag",  "euclid_h_unifMagErr",  "ej-eh"),
]
REF_MAG      = "i_cModelMag"
REF_ERR      = "i_cModelMagErr"
GAAP_REF_MAG = "i_gaap1p0Mag"
GAAP_REF_ERR = "i_gaap1p0MagErr"
PSF_REF_MAG  = "i_psfMag"
PSF_REF_ERR  = "i_psfMagErr"

CATALOG_DEFAULT = os.path.join(
    os.path.dirname(__file__), "data", "dp1_ecdfs_crossmatched_catalog.parquet"
)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog loading
# ─────────────────────────────────────────────────────────────────────────────

def load_catalog(path=None, seed=42, val_frac=0.10, test_frac=0.20):
    """
    Load catalog parquet, keep extended sources with valid spec-z, and split
    into train / val / test.
    Catalog galaxies: 20,020
    Train 14,414  Val 1,601  Test 4,005

    Parameters
    ----------
    path      : path to .parquet catalog (default: CATALOG_DEFAULT)
    seed      : random seed for reproducible split
    val_frac  : fraction for validation (0 → no val, val_df is empty DataFrame)
    test_frac : fraction for test

    Returns
    -------
    tr_df, val_df, te_df : pd.DataFrame
    """
    if path is None:
        path = CATALOG_DEFAULT
    df   = pd.read_parquet(path)
    gals = df[
        (df["refExtendedness"] == 1)
        & df[TARGET_COL].notna()
        & (df[TARGET_COL] > 0)
    ].copy()
    gals[TARGET_COL] = gals[TARGET_COL].astype(float)

    rng   = np.random.RandomState(seed)
    idx   = rng.permutation(len(gals))
    n_tr  = int((1.0 - val_frac - test_frac) * len(gals))
    n_val = int(val_frac * len(gals))

    tr_df  = gals.iloc[idx[:n_tr]].reset_index(drop=True)
    val_df = gals.iloc[idx[n_tr:n_tr + n_val]].reset_index(drop=True)
    te_df  = gals.iloc[idx[n_tr + n_val:]].reset_index(drop=True)

    print(f"Catalog galaxies: {len(gals):,}")
    print(f"  Train {len(tr_df):,}  Val {len(val_df):,}  Test {len(te_df):,}")
    return tr_df, val_df, te_df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _raw_mags_errs_mask(df, mag_cols, err_cols):
    """
    Extract raw (median-imputed) mags, errors, and band-presence mask.
    Used as reconstruction targets for models with a photometric decoder.

    Returns
    -------
    mags_imp : (N, B) float32  AB mags  (NaN → column median)
    errs_imp : (N, B) float32  mag errs (NaN → column median; clipped ≥ 0.005)
    mask     : (N, B) float32  1 = band present & valid, 0 = missing/bad
    """
    # astype("float64") first: nullable/pyarrow-backed columns (e.g. from an
    # lsdb/HATS catalog) use pd.NA rather than np.nan, which breaks the raw
    # np.nanmedian calls below with "boolean value of NA is ambiguous".
    mags = df[mag_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    errs = df[err_cols].astype("float64").replace([np.inf, -np.inf], np.nan).clip(lower=0)

    m_ok = (mags.notna() & (mags > 0) & (mags < 35)).values
    e_ok = (errs.notna() & (errs > 0)).values
    mask = (m_ok & e_ok).astype(np.float32)

    med_mags = np.nanmedian(mags.values, axis=0)
    med_errs = np.nanmedian(errs.values, axis=0)
    med_mags = np.where(np.isfinite(med_mags), med_mags, 0.0)
    med_errs = np.where(np.isfinite(med_errs), med_errs, 0.1)

    mags_imp = mags.values.copy()
    errs_imp = errs.values.copy()
    for j in range(len(mag_cols)):
        mags_imp[~np.isfinite(mags_imp[:, j]), j] = med_mags[j]
        errs_imp[~np.isfinite(errs_imp[:, j]), j] = med_errs[j]

    return mags_imp.astype(np.float32), errs_imp.astype(np.float32), mask


def _build_mag_features(df, mag_cols, err_cols, scaler=None, col_medians=None, fit=False):
    """
    Magnitude-based encoder input:
      [B scaled mags | B scaled errs | B mag-miss flags | B err-miss flags]
    B=10 (with Euclid) → 40-dim; B=6 (LSST only) → 24-dim.

    Returns (X, scaler, col_medians).
    """
    mags = df[mag_cols].copy().replace([np.inf, -np.inf], np.nan)
    errs = df[err_cols].copy().replace([np.inf, -np.inf], np.nan).clip(lower=0)

    m_ok = (mags.notna()).values #& (mags > 0) & (mags < 35)
    e_ok = (errs.notna()).values # & (errs > 0)
    missing_mags = (~m_ok).astype(np.float32)
    missing_errs = (~e_ok).astype(np.float32)

    feat_df = pd.concat([mags, errs], axis=1)   # (N, 20)

    if fit:
        col_medians = feat_df.median(axis=0).values
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        scaler      = StandardScaler()
        feat_imp    = feat_df.fillna(pd.Series(col_medians, index=feat_df.columns)).fillna(0.0)
        scaler.fit(feat_imp)
    else:
        feat_imp = feat_df.fillna(pd.Series(col_medians, index=feat_df.columns)).fillna(0.0)

    feat_sc = scaler.transform(feat_imp).astype(np.float32)
    # Overwrite imputed-missing slots with an out-of-range sentinel so the
    # encoder sees a clean OOD signal instead of median≈0 for missing bands.
    # StandardScaler maps real observations to ~[-3, 3]; -50.0 is unambiguous.
    missing_mask = np.concatenate([~m_ok, ~e_ok], axis=1)  # (N, 20)
    feat_sc[missing_mask] = -50.0
    X = np.concatenate([feat_sc, missing_mags, missing_errs], axis=1)  # (N, 40)
    return X, scaler, col_medians


def _build_color_features(df, color_pairs, scaler=None, col_medians=None, fit=False,
                          ref_mag=None, ref_err=None):
    """
    Colour-based encoder input:
      [C colours | C colour-errs | i_ref | i_ref_err | 2*(C+1) missingness flags]
    C=9 (with Euclid) → 40-dim; C=5 (LSST only) → 24-dim.

    Returns (X, scaler, col_medians).
    """
    if ref_mag is None:
        ref_mag = REF_MAG
    if ref_err is None:
        ref_err = REF_ERR

    colour_cols, cerr_cols = [], []
    for m1, e1, m2, e2, cname in color_pairs:
        v1 = df[m1].replace([np.inf, -np.inf], np.nan)
        v2 = df[m2].replace([np.inf, -np.inf], np.nan)
        s1 = df[e1].replace([np.inf, -np.inf], np.nan).clip(lower=0)
        s2 = df[e2].replace([np.inf, -np.inf], np.nan).clip(lower=0)
        colour_cols.append((v1 - v2).rename(cname))
        cerr_cols.append(np.sqrt(s1 ** 2 + s2 ** 2).rename(cname + "_err"))

    ref   = df[ref_mag].replace([np.inf, -np.inf], np.nan).rename("i_ref")
    r_err = df[ref_err].replace([np.inf, -np.inf], np.nan).rename("i_ref_err")

    feat_df = pd.concat(colour_cols + cerr_cols + [ref, r_err], axis=1)  # (N, 20)
    missing = feat_df.isna().values.astype(np.float32)                   # (N, 20)

    if fit:
        col_medians = feat_df.median(axis=0).values
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        scaler      = StandardScaler()
        feat_imp    = feat_df.fillna(pd.Series(col_medians, index=feat_df.columns)).fillna(0.0)
        scaler.fit(feat_imp)
    else:
        feat_imp = feat_df.fillna(pd.Series(col_medians, index=feat_df.columns)).fillna(0.0)

    feat_sc = scaler.transform(feat_imp).astype(np.float32)
    # Overwrite imputed-missing slots with an out-of-range sentinel so the
    # encoder sees a clean OOD signal instead of median≈0 for missing bands.
    # StandardScaler maps real observations to ~[-3, 3]; -50.0 is unambiguous.
    feat_sc[feat_df.isna().values] = -50.0
    X = np.concatenate([feat_sc, missing], axis=1)  # (N, 40)
    return X, scaler, col_medians


def build_features(df, use_colors=True, use_euclid=True, use_gaap=False, use_psf=False,
                   scaler=None, col_medians=None, fit=False, target_col=TARGET_COL):
    """
    Build all tensors needed for one catalog split.

    Parameters
    ----------
    df          : pd.DataFrame from load_catalog
    use_colors  : True → colour features; False → magnitude features
    use_euclid  : True → include Euclid bands (10 bands, D=40);
                  False → LSST only (6 bands, D=24)
    use_gaap    : True → use GAAP 1.0-arcsec aperture mags for LSST bands
                  instead of cModel mags (Euclid columns are unchanged)
    use_psf     : True → use PSF mags for LSST bands instead of cModel mags
                  (Euclid columns are unchanged; ignored if use_gaap=True)
    scaler      : pre-fitted StandardScaler (required when fit=False)
    col_medians : pre-computed imputation medians (required when fit=False)
    fit         : if True, fit scaler + col_medians from df (train split only)

    Returns
    -------
    X           : (N, D) float32  encoder input; D=40 with Euclid, 24 without
    mags_obs    : (N, B) float32  raw AB mags  (imputed; for reconstruction loss)
    mag_errs    : (N, B) float32  raw mag errs (imputed; clipped ≥ 0.005)
    mask        : (N, B) float32  band-presence mask; B=10 with Euclid, 6 without
    z_spec      : (N,)   float32  spectroscopic redshift
    scaler      : fitted/passed StandardScaler
    col_medians : imputation medians
    """
    if use_gaap:
        mag_cols    = GAAP_MAG_COLS    if use_euclid else LSST_GAAP_MAG_COLS
        err_cols    = GAAP_ERR_COLS    if use_euclid else LSST_GAAP_ERR_COLS
        color_pairs = GAAP_COLOR_PAIRS if use_euclid else GAAP_COLOR_PAIRS[:5]
        ref_mag, ref_err = GAAP_REF_MAG, GAAP_REF_ERR
    elif use_psf:
        mag_cols    = PSF_MAG_COLS    if use_euclid else LSST_PSF_MAG_COLS
        err_cols    = PSF_ERR_COLS    if use_euclid else LSST_PSF_ERR_COLS
        color_pairs = PSF_COLOR_PAIRS if use_euclid else PSF_COLOR_PAIRS[:5]
        ref_mag, ref_err = PSF_REF_MAG, PSF_REF_ERR
    else:
        mag_cols    = MAG_COLS    if use_euclid else LSST_MAG_COLS
        err_cols    = ERR_COLS    if use_euclid else LSST_ERR_COLS
        color_pairs = COLOR_PAIRS if use_euclid else COLOR_PAIRS[:5]
        ref_mag, ref_err = REF_MAG, REF_ERR

    if use_colors:
        X, scaler, col_medians = _build_color_features(
            df, color_pairs, scaler=scaler, col_medians=col_medians, fit=fit,
            ref_mag=ref_mag, ref_err=ref_err)
    else:
        X, scaler, col_medians = _build_mag_features(
            df, mag_cols, err_cols, scaler=scaler, col_medians=col_medians, fit=fit)

    mags_obs, mag_errs, mask = _raw_mags_errs_mask(df, mag_cols, err_cols)
    z_spec = df[target_col].values.astype(np.float32)

    return X, mags_obs, mag_errs, mask, z_spec, scaler, col_medians


# ─────────────────────────────────────────────────────────────────────────────
# Inverse-frequency redshift weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_z_weights(z_spec, n_bins=25, z_max=5.5):
    """
    Per-galaxy inverse-frequency weights so that high-z galaxies contribute
    equally to the loss (in expectation). Weights are normalised to mean = 1.
    """
    counts, edges = np.histogram(z_spec, bins=n_bins, range=(0.0, z_max))
    counts  = np.maximum(counts, 1)
    bin_idx = np.clip(np.digitize(z_spec, edges) - 1, 0, n_bins - 1)
    w = len(z_spec) / (n_bins * counts[bin_idx])
    return (w / w.mean()).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloader(X, mags, errs, mask, z, weights,
                    batch_size, shuffle, num_workers=2):
    """
    Build a PyTorch DataLoader with a 6-tensor TensorDataset.

    Each batch yields: (x_b, mags_b, errs_b, mask_b, z_b, zw_b).

    Parameters
    ----------
    X, mags, errs, mask, z : numpy float32 arrays from build_features
    weights    : (N,) float32  per-sample loss weights (e.g. compute_z_weights)
    batch_size : int
    shuffle    : bool
    num_workers: int (default 2)
    """
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(mags),
        torch.from_numpy(errs),
        torch.from_numpy(mask),
        torch.from_numpy(z),
        torch.from_numpy(weights),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
