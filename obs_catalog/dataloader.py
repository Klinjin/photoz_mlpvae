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
                 col_medians=None, fit=False, standardize=True)
      Returns (X, mags_obs, mag_errs, mask, z_spec, scaler, col_medians).
        X           (N, D) float32   encoder input; D=40 with Euclid, 24 without
        mags_obs    (N, B) float32   raw AB mags  (imputed, for reconstruction)
        mag_errs    (N, B) float32   raw mag errs (imputed, clipped ≥ 0.005)
        mask        (N, B) float32   band-presence mask; B=10 with Euclid, 6 without
        z_spec      (N,)   float32   spectroscopic redshift
        scaler      fitted StandardScaler, or None if standardize=False
        col_medians imputation medians
      standardize=False skips StandardScaler entirely and feeds raw
      colours/mags(+errs) straight into the encoder input (still with the
      usual -50.0 missingness sentinel), EXCEPT errs are log10-transformed
      (see _log_err) -- mag errs span ~4 decades, so log-space keeps them
      roughly unit-scale even with nothing else bounding the input scale.
      standardize=True leaves errs raw (unchanged, backward-compatible with
      every existing standardized checkpoint) since StandardScaler already
      normalizes them. Regardless of standardize, any mag/err that is inf,
      NaN, or outside a physical sanity range (0 < mag < 35, 0 < err <= 5;
      see _sanitize_mag_err) is routed into that same -50.0 missingness
      path rather than passed through as a raw out-of-distribution number
      -- and a band's mag/err are sanitized jointly, so an out-of-range
      err also blanks that band's mag (and vice versa) rather than leaving
      a "valid" mag paired with a known-bad err.

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

Example – photoz_vae / photoz_mlpvae (72/8/20 split with val)
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

# Physical sanity bounds for raw AB mags/errs. A cell failing these (e.g. the
# GAAP DP2 catalog's occasional mag=-24 or mag_err=1e4 glitches) is routed
# into the same NaN -> -50.0 missingness path as a genuine non-detection,
# rather than being fed to the encoder as a raw, wildly out-of-distribution
# number -- this matters most when standardize=False, where nothing else
# bounds the input scale. Adjust _ERR_MAX if this proves too
# aggressive/lenient for a given catalog; ~5 mag error is already close to
# non-detection (S/N<1), so it costs little real signal.
_MAG_MIN, _MAG_MAX = 0.0, 35.0
_ERR_MAX = 5.0

# mag errs span ~4 decades (see obs_catalog analysis: raw i-band err p1-p99
# is ~5e-4 to 0.4 mag, worse in u/y) -- log10 compresses that into a roughly
# unit-scale feature, which matters when standardize=False since nothing
# else bounds the input scale. Only applied in the nostd path: standardize
# =True already z-scores the raw err via StandardScaler, and changing that
# would break every existing standardized checkpoint's expected input.
_ERR_LOG_FLOOR = 1e-5  # clip before log10 so a near-zero glitch can't blow up


def _log_err(err, standardize):
    """log10(err) (floored at _ERR_LOG_FLOOR) when standardize=False, else
    err unchanged. NaN (missing/sanitized-out) passes through as NaN either
    way -- computed on colour-errs AFTER the quadrature sum, magnitude-errs
    directly, always on the physical (linear) error, never before combining
    two bands' errors."""
    if standardize:
        return err
    return np.log10(err.clip(lower=_ERR_LOG_FLOOR))


def _sanitize_mag_err(mag, err, apply_range_filter=True):
    """
    inf -> NaN on both. When apply_range_filter=True (default), a further
    joint validity check: mag must be in (_MAG_MIN, _MAG_MAX) AND err must
    be in (0, _ERR_MAX], or BOTH get NaN'd together. An error that far out
    means the mag measurement for that band can't be trusted either (and
    vice versa) -- a "valid" mag paired with a missing/garbage err (or the
    reverse) is not something the encoder should see as a confident
    detection.

    apply_range_filter=False skips that range check entirely -- a band
    only counts as missing if its raw mag/err is itself NaN/inf; physically
    implausible-but-finite values (e.g. mag<=0 or err>5) pass through
    unmodified. For callers that want that looser, "raw NaN/inf only"
    missingness definition (e.g. to compare against the default).
    """
    mag = mag.replace([np.inf, -np.inf], np.nan)
    err = err.replace([np.inf, -np.inf], np.nan)
    if not apply_range_filter:
        return mag, err
    bad = (~mag.between(_MAG_MIN, _MAG_MAX, inclusive="neither")
           | ~err.between(0, _ERR_MAX, inclusive="right"))
    return mag.mask(bad), err.mask(bad)


def band_present_mask(df, mag_col, err_col, apply_range_filter=True):
    """
    Boolean Series: True where a single band's mag+err are both usable, per
    _sanitize_mag_err's joint validity rule (see apply_range_filter there).
    For filtering/reporting on band presence outside of build_features
    (e.g. "keep only galaxies with u and g both present").
    """
    mag, _ = _sanitize_mag_err(df[mag_col], df[err_col], apply_range_filter=apply_range_filter)
    return mag.notna()


def _build_mag_features(df, mag_cols, err_cols, scaler=None, col_medians=None, fit=False,
                        standardize=True, double_precision=False, apply_range_filter=True):
    """
    Magnitude-based encoder input:
      [B scaled mags | B scaled errs | B mag-miss flags | B err-miss flags]
    B=10 (with Euclid) → 40-dim; B=6 (LSST only) → 24-dim.

    standardize : True (default) → z-score via StandardScaler (fitted when
                  fit=True, else the passed-in `scaler` is applied); False →
                  pass raw mags/errs through unscaled and `scaler` stays
                  whatever was passed in (None if never fit standardized).
    double_precision : False (default) → X is float32, matching every
                  existing checkpoint's float32 weights (a float64 X raises
                  "mat1 and mat2 must have the same dtype" against them).
                  True → X is float64; only for callers that explicitly
                  need double precision and will train/load a matching
                  float64 model.
    apply_range_filter : True (default) → _sanitize_mag_err's physical
                  sanity-range check is applied (unchanged behavior). False →
                  skip that check; a band only counts as missing when its
                  raw mag/err is NaN/inf.

    Returns (X, scaler, col_medians).
    """
    dtype = np.float64 if double_precision else np.float32

    mag_series, err_series = [], []
    for mc, ec in zip(mag_cols, err_cols):
        m, e = _sanitize_mag_err(df[mc], df[ec], apply_range_filter=apply_range_filter)
        mag_series.append(m)
        err_series.append(e)
    mags = pd.concat(mag_series, axis=1)
    errs = _log_err(pd.concat(err_series, axis=1), standardize)

    m_ok = (mags.notna()).values
    e_ok = (errs.notna()).values
    missing_mags = (~m_ok).astype(dtype)
    missing_errs = (~e_ok).astype(dtype)

    feat_df = pd.concat([mags, errs], axis=1)   # (N, 20)

    if fit:
        # col_medians kept only for save/load API compatibility with older
        # checkpoints; sklearn ignores NaNs natively when fitting/transforming.
        # Computed regardless of `standardize` so downstream checkpoint
        # save/load always has a usable value.
        col_medians = feat_df.median(axis=0).values
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(feat_df)

    # transform() passes NaNs straight through; the sentinel overwrite below
    # replaces those cells, so no imputation is needed here.
    if standardize:
        feat_sc = scaler.transform(feat_df).astype(dtype)
    else:
        feat_sc = feat_df.values.astype(dtype)
    # Overwrite missing slots with an out-of-range sentinel so the encoder
    # sees a clean OOD signal instead of a real value for missing bands.
    # StandardScaler maps real observations to ~[-3, 3]; raw mags/errs are
    # always positive; -50.0 is unambiguous either way.
    missing_mask = np.concatenate([~m_ok, ~e_ok], axis=1)  # (N, 20)
    feat_sc[missing_mask] = -50.0
    X = np.concatenate([feat_sc, missing_mags, missing_errs], axis=1).astype(dtype)  # (N, 40)
    return X, scaler, col_medians


def _build_color_features(df, color_pairs, scaler=None, col_medians=None, fit=False,
                          ref_mag=None, ref_err=None, standardize=True,
                          double_precision=False, apply_range_filter=True):
    """
    Colour-based encoder input:
      [C colours | C colour-errs | i_ref | i_ref_err | 2*(C+1) missingness flags]
    C=9 (with Euclid) → 40-dim; C=5 (LSST only) → 24-dim.

    standardize : True (default) → z-score via StandardScaler (fitted when
                  fit=True, else the passed-in `scaler` is applied); False →
                  pass raw colours/mags through unscaled and `scaler` stays
                  whatever was passed in (None if never fit standardized).
    double_precision : False (default) → X is float32, matching every
                  existing checkpoint's float32 weights (a float64 X raises
                  "mat1 and mat2 must have the same dtype" against them).
                  True → X is float64; only for callers that explicitly
                  need double precision and will train/load a matching
                  float64 model.
    apply_range_filter : True (default) → _sanitize_mag_err's physical
                  sanity-range check is applied (unchanged behavior). False →
                  skip that check; a band only counts as missing when its
                  raw mag/err is NaN/inf.

    Returns (X, scaler, col_medians).
    """
    dtype = np.float64 if double_precision else np.float32

    if ref_mag is None:
        ref_mag = REF_MAG
    if ref_err is None:
        ref_err = REF_ERR

    colour_cols, cerr_cols = [], []
    for m1, e1, m2, e2, cname in color_pairs:
        v1, s1 = _sanitize_mag_err(df[m1], df[e1], apply_range_filter=apply_range_filter)
        v2, s2 = _sanitize_mag_err(df[m2], df[e2], apply_range_filter=apply_range_filter)
        colour_cols.append((v1 - v2).rename(cname))
        cerr = _log_err(np.sqrt(s1 ** 2 + s2 ** 2), standardize)
        cerr_cols.append(cerr.rename(cname + "_err"))

    ref, r_err = _sanitize_mag_err(df[ref_mag], df[ref_err], apply_range_filter=apply_range_filter)
    ref   = ref.rename("i_ref")
    r_err = _log_err(r_err, standardize).rename("i_ref_err")

    feat_df = pd.concat(colour_cols + cerr_cols + [ref, r_err], axis=1)  # (N, 20)
    missing = feat_df.isna().values.astype(dtype)                        # (N, 20)

    if fit:
        # col_medians kept only for save/load API compatibility with older
        # checkpoints; sklearn ignores NaNs natively when fitting/transforming.
        # Computed regardless of `standardize` so downstream checkpoint
        # save/load always has a usable value.
        col_medians = feat_df.median(axis=0).values
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
        if standardize:
            scaler = StandardScaler()
            scaler.fit(feat_df)

    # transform() passes NaNs straight through; the sentinel overwrite below
    # replaces those cells, so no imputation is needed here.
    if standardize:
        feat_sc = scaler.transform(feat_df).astype(dtype)
    else:
        feat_sc = feat_df.values.astype(dtype)
    # Overwrite missing slots with an out-of-range sentinel so the encoder
    # sees a clean OOD signal instead of a real value for missing bands.
    # StandardScaler maps real observations to ~[-3, 3]; raw colours/mags
    # never reach anywhere near -50 either, so it's unambiguous either way.
    feat_sc[feat_df.isna().values] = -50.0
    X = np.concatenate([feat_sc, missing], axis=1).astype(dtype)  # (N, 40)
    return X, scaler, col_medians


def build_features(df, use_colors=True, use_euclid=True, use_gaap=False, use_psf=False,
                   scaler=None, col_medians=None, fit=False, target_col=TARGET_COL,
                   standardize=True, double_precision=False, apply_range_filter=True):
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
    scaler      : pre-fitted StandardScaler (required when fit=False and
                  standardize=True; unused/should be None when standardize=False)
    col_medians : pre-computed imputation medians (required when fit=False)
    fit         : if True, fit scaler + col_medians from df (train split only)
    standardize : True (default) → z-score colours/mags+errs via StandardScaler;
                  False → pass the raw colours/mags (or raw mags/errs in
                  magnitude mode) through unscaled. `scaler` comes back None
                  in that case since none was fit.
    double_precision : False (default) → all five array outputs (X,
                  mags_obs, mag_errs, mask, z_spec) are float32, matching
                  every existing checkpoint's float32 weights. True → all
                  five are float64; only use this if you are training/
                  loading a model itself built in double precision (e.g.
                  PhotozMLPVAE(..., double_precision=True)) -- mixing
                  dtypes anywhere in model.loss()'s inputs (X vs.
                  mags_obs/mag_errs/mask/z_spec, or either vs. the model's
                  own weights) raises "mat1 and mat2 must have the same
                  dtype" or an elementwise dtype-mismatch error.
    apply_range_filter : True (default) → _sanitize_mag_err's physical
                  sanity-range check (mag in (0,35), err in (0,5]) is
                  applied when deciding what's missing (unchanged
                  behavior). False → skip that check; a band only counts
                  as missing when its raw mag/err is NaN/inf.

    Returns
    -------
    X           : (N, D) float32 (float64 if double_precision=True)
                  encoder input; D=40 with Euclid, 24 without
    mags_obs    : (N, B) float32 (float64)  raw AB mags  (imputed; for reconstruction loss)
    mag_errs    : (N, B) float32 (float64)  raw mag errs (imputed; clipped ≥ 0.005)
    mask        : (N, B) float32 (float64)  band-presence mask; 10 with Euclid, 6 without
    z_spec      : (N,)   float32 (float64)  spectroscopic redshift
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
            ref_mag=ref_mag, ref_err=ref_err, standardize=standardize,
            double_precision=double_precision, apply_range_filter=apply_range_filter)
    else:
        X, scaler, col_medians = _build_mag_features(
            df, mag_cols, err_cols, scaler=scaler, col_medians=col_medians, fit=fit,
            standardize=standardize, double_precision=double_precision,
            apply_range_filter=apply_range_filter)
        
    z_spec = df[target_col].values.astype(np.float64 if double_precision else np.float32)

    return X, z_spec, scaler, col_medians


# ─────────────────────────────────────────────────────────────────────────────
# Raw per-band mags/errs/mask (diagnostics only -- NOT part of build_features)
# ─────────────────────────────────────────────────────────────────────────────
#
# build_features()/make_dataloader() no longer return/carry mags_obs/
# mag_errs/mask (2026-08-24) -- models' loss() now derives its reconstruction
# target straight out of X (see photoz_mlpvae's reconstruction_target_from_x).
# This standalone helper (the old _raw_mags_errs_mask, made public) exists
# only for callers that still want raw per-band photometry for a handful of
# galaxies -- e.g. SED-reconstruction diagnostic plots -- and should be
# called directly on a df subset, not wired back into the training loader.

def raw_mags_errs_mask(df, mag_cols, err_cols):
    """
    Extract raw (median-imputed) mags, errors, and band-presence mask.
    For diagnostics (e.g. plot_sed_reconstructions) -- not used by
    build_features/make_dataloader or any model's loss().

    Parameters
    ----------
    df       : pd.DataFrame (or a row subset of one)
    mag_cols : list of magnitude column names, e.g. MAG_COLS/GAAP_MAG_COLS
    err_cols : matching list of magnitude-error column names

    Returns
    -------
    mags_imp : (N, B) float32  AB mags  (NaN → column median)
    errs_imp : (N, B) float32  mag errs (NaN → column median; clipped ≥ 0.005)
    mask     : (N, B) float32  1 = band present & valid, 0 = missing/bad
    """
    # astype("float64") first: nullable/pyarrow-backed columns (e.g. from an
    # lsdb/HATS catalog) use pd.NA rather than np.nan, which breaks the raw
    # np.nanmedian calls below with "boolean value of NA is ambiguous".
    mag_series, err_series = [], []
    for mc, ec in zip(mag_cols, err_cols):
        m, e = _sanitize_mag_err(df[mc].astype("float64"), df[ec].astype("float64"))
        mag_series.append(m)
        err_series.append(e)
    mags = pd.concat(mag_series, axis=1)
    errs = pd.concat(err_series, axis=1)

    m_ok = (mags.notna()).values
    e_ok = (errs.notna()).values
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


# ─────────────────────────────────────────────────────────────────────────────
# Inverse-frequency redshift weights
# ─────────────────────────────────────────────────────────────────────────────

def compute_z_weights(z_spec, n_bins=10):
    """
    Per-galaxy inverse-frequency weights so that high-z galaxies contribute
    equally to the loss (in expectation). Weights are normalised to mean = 1.
    """
    counts, edges = np.histogram(z_spec, bins=n_bins, range=(0.0, z_spec.max()))
    counts  = np.maximum(counts, 1)
    bin_idx = np.clip(np.digitize(z_spec, edges) - 1, 0, n_bins - 1)
    w = len(z_spec) / (n_bins * counts[bin_idx])
    return (w / w.mean()).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloader(X, z, weights,
                    batch_size, shuffle, num_workers=2, is_lbg=None,
                    is_interloper=None):
    """
    Build a PyTorch DataLoader with a 3-, 4-, or 5-tensor TensorDataset.

    Each batch yields (x_b, z_b, zw_b), (x_b, z_b, zw_b, is_lbg_b), or
    (x_b, z_b, zw_b, is_lbg_b, is_interloper_b), depending on which of
    is_lbg/is_interloper are given.

    Parameters
    ----------
    X, z : numpy float32 arrays from build_features
    weights    : (N,) float32  per-sample loss weights (e.g. compute_z_weights)
    batch_size : int
    shuffle    : bool
    num_workers: int (default 2)
    is_lbg     : (N,) bool array, optional -- per-row LBG-dropout routing
                 flag from compute_lbg_mask(). Appended as a 4th tensor
                 when given; omitted (existing 3-tensor behavior, every
                 other caller unaffected) when None (default).
    is_interloper : (N,) bool array, optional (2026-09-24) -- per-row
                 color-cut-false-positive routing flag (see
                 photoz_mlpvae_mdn_perpop.py's z_mixture() is_interloper
                 docstring). Appended as a 5th tensor when given; requires
                 is_lbg to also be given (fixed tensor order: is_lbg then
                 is_interloper), since every caller that uses this always
                 passes both.
    """
    tensors = [
        torch.from_numpy(X),
        torch.from_numpy(z),
        torch.from_numpy(weights),
    ]
    if is_lbg is not None:
        tensors.append(torch.from_numpy(is_lbg.astype(np.bool_)))
    if is_interloper is not None:
        if is_lbg is None:
            raise ValueError("make_dataloader(): is_interloper requires is_lbg to also be given")
        tensors.append(torch.from_numpy(is_interloper.astype(np.bool_)))
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


# ─────────────────────────────────────────────────────────────────────────────
# RTN-124 Lyman-break dropout selection (Goldrush-style color cuts)
# ─────────────────────────────────────────────────────────────────────────────
#
# Canonical home for the g/r/i/z-dropout color-color selection used to flag
# Lyman-break galaxy (LBG) candidates from observed photometry alone (no
# redshift needed -- see select_lbg_dropouts()'s own docstring). Originally
# prototyped in obs_catalog/redshift_dp2_v3p1_test.ipynb; relocated here so
# the notebook and any training script (e.g.
# photoz_mlpvae/scripts/train_dp2_lbg_perpop.py) share ONE implementation
# rather than two copies that can drift.

def select_lbg_dropouts(df, mag_type='gaap1p0', z_true_name='redshift', apply_extended_cut=True):
    """
    Apply the RTN-124 (Goldrush-style) Lyman-break dropout color-color
    selections for g-, r-, i-, and z-dropouts, and intersect each with a
    true-redshift cut to isolate true high-z positives:
        g-dropout: z_true > 3
        r-dropout: z_true > 4
        i-dropout: z_true > 5
        z-dropout: z_true > 6

    Color cuts (RTN-124):
        g-dropout: g-r > 1.0, r-i < 1.0, g-r > 1.5(r-i) + 0.8
        r-dropout: r-i > 1.2, i-z < 0.7, r-i > 1.5(i-z) + 1.0
        i-dropout: i-z > 1.5, z-y < 0.5, i-z > 2.0(z-y) + 1.1
        z-dropout: z-y > 1.6

    All four cuts are purely a function of OBSERVED colors -- no redshift
    is used to decide the `_dropout` masks, so they're safe to compute on
    any catalog (train, test, or genuinely new data) without leakage. The
    `_true` masks additionally require z_true_name > threshold and are
    only meaningful on labeled (known-z_spec) data.

    Returns
    -------
    dict with, for each band in ('g', 'r', 'i', 'z'):
        f'{band}_dropout'  -- boolean mask, color-selected (regardless of true z)
        f'{band}_true'     -- boolean mask, color-selected AND true high-z
    Masks are aligned to df's index (not reset), like plot_g_r_dropout_contamination.
    i-dropout/z-dropout require a y-band column for the given mag_type
    (present for cModel/gaap1p0/psf/kron; absent for sersic/gaap3p0).
    """
    suffix_by_mag_type = {
        'cModel': 'cModelMag', 'psf': 'psfMag', 'sersic': 'sersicMag',
        'gaap1p0': 'gaap1p0Mag', 'gaap3p0': 'gaap3p0Mag', 'kron': 'kronMag',
    }
    if mag_type not in suffix_by_mag_type:
        raise ValueError(f"Unknown mag_type: {mag_type}")
    suffix = suffix_by_mag_type[mag_type]

    def mag_col(band):
        return f'{band}_{suffix}'

    for band in ('u', 'g', 'r', 'i', 'z'):
        if mag_col(band) not in df.columns:
            raise ValueError(f"{mag_col(band)} not in df.columns for mag_type={mag_type!r}")
    has_y = mag_col('y') in df.columns

    g_r = df[mag_col('g')] - df[mag_col('r')]
    r_i = df[mag_col('r')] - df[mag_col('i')]
    i_z = df[mag_col('i')] - df[mag_col('z')]
    z_y = (df[mag_col('z')] - df[mag_col('y')]) if has_y else None

    if apply_extended_cut and 'refExtendedness' in df.columns:
        is_extended = df['refExtendedness'] == 1
    else:
        is_extended = pd.Series(True, index=df.index)

    redshift = df[z_true_name] if z_true_name in df.columns else None

    g_dropout = (
        (g_r > 1.0) &
        (r_i < 1.0) &
        (g_r > 1.5 * r_i + 0.8) &
        is_extended &
        np.isfinite(g_r) & np.isfinite(r_i)
    )
    r_dropout = (
        (r_i > 1.2) &
        (i_z < 0.7) &
        (r_i > 1.5 * i_z + 1.0) &
        is_extended &
        np.isfinite(r_i) & np.isfinite(i_z)
    )

    result = {'g_dropout': g_dropout, 'r_dropout': r_dropout}
    if redshift is not None:
        result['g_true'] = g_dropout & (redshift > 3) & np.isfinite(redshift)
        result['r_true'] = r_dropout & (redshift > 4) & np.isfinite(redshift)

    if has_y:
        i_dropout = (
            (i_z > 1.5) &
            (z_y < 0.5) &
            (i_z > 2.0 * z_y + 1.1) &
            is_extended &
            np.isfinite(i_z) & np.isfinite(z_y)
        )
        z_dropout = (
            (z_y > 1.6) &
            is_extended &
            np.isfinite(z_y)
        )
        result['i_dropout'] = i_dropout
        result['z_dropout'] = z_dropout
        if redshift is not None:
            result['i_true'] = i_dropout & (redshift > 5) & np.isfinite(redshift)
            result['z_true'] = z_dropout & (redshift > 6) & np.isfinite(redshift)
    else:
        print(f"Warning: no y-band column found for mag_type={mag_type!r}; "
              f"skipping i-dropout and z-dropout (both require z-y color).")

    return result


def compute_lbg_mask(df, mag_type='gaap1p0', z_true_name='redshift',
                     use_true_z=True, apply_extended_cut=True):
    """
    Single combined boolean LBG-dropout flag (OR across g/r/i/z-dropout),
    for use as a per-row routing signal (e.g. PhotozMLPVAE's is_lbg arg --
    see photoz_mlpvae/model/photoz_mlpvae_mdn_perpop.py) rather than for
    the per-band diagnostics select_lbg_dropouts() itself is meant for.

    use_true_z=True  (default): OR of the `_true` masks (color cut AND
        confirmed z_spec above that band's threshold) -- the purer signal,
        appropriate for TRAINING data (known z_spec), so routing gradient
        to a dedicated LBG-specialist component isn't contaminated by
        color-cut false positives (real low-z interlopers).
    use_true_z=False: OR of the `_dropout` masks (color cuts only, no
        z_spec) -- the realistic/deployable signal, appropriate for
        EVALUATION or inference on data whose true redshift is unknown
        (which is, after all, the entire point of a photo-z estimator).

    Returns (N,) bool ndarray aligned to df's row order (df.index is NOT
    consulted -- pass a df with a clean 0..N-1 index, e.g. via
    .reset_index(drop=True), matching load_parquet_galaxies's convention).
    """
    if use_true_z and z_true_name not in df.columns:
        raise ValueError(f"use_true_z=True requires {z_true_name!r} in df.columns")
    masks = select_lbg_dropouts(df, mag_type=mag_type, z_true_name=z_true_name,
                                apply_extended_cut=apply_extended_cut)
    key = '_true' if use_true_z else '_dropout'
    bands = [b for b in ('g', 'r', 'i', 'z') if f'{b}{key}' in masks]
    combined = np.zeros(len(df), dtype=bool)
    for b in bands:
        combined |= masks[f'{b}{key}'].to_numpy()
    return combined
