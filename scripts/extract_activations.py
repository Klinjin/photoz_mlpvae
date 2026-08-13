"""
extract_activations.py
=======================
Extract per-layer trunk/z-head activations plus supervised targets from a
trained PhotozMLPVAE checkpoint, for linear-probing studies of the embedding
space (does redshift/mass/etc. become linearly decodable, and at which
depth?).

Works with ANY checkpoint of ANY --model-version ({old, new, mdn}), any
dataset (dp1_v4 HDF5, DP2 parquet, synth HDF5), and any encoder_n_in --
everything needed to reproduce the exact preprocessing (scaler, col_medians,
n_in) is read from the checkpoint itself. If a config.yaml sits next to the
checkpoint (as every train_*.py script writes), model_version and
use_gaap/use_euclid/use_colors are read from it automatically; override any
of them explicitly via flags for a mismatched run.

Encoder layers tapped -- every sub-part of MLPVAEEncoder, trunk through both
heads (input_proj/res_blocks are byte-for-byte the same Sequential
definitions across old/new/mdn; only the head positions in each version's
forward()/encoder() return tuple differ, handled by `unpack_forward_output`):
  x               (N, n_in)  scaled input features (control/baseline probe)
  h0              (N, width) after input_proj (Linear+BN+ReLU)
  h1_mid          (N, width) mid-ResBlock-1 activation (block[:3] =
                              Linear+BN+ReLU, before the second Linear+BN
                              and the residual add)
  h1              (N, width) after res_blocks[0]
  h2_mid          (N, width) mid-ResBlock-2 activation
  h2              (N, width) after res_blocks[1] -- final trunk output,
                              feeds both z_head and vae_head
  z_head_hidden   (N, 128)   z_head's hidden layer, for versions that have
                              one (new/mdn: 2-layer z_head). old's z_head is
                              a bare Linear(width,1) with no hidden layer --
                              saved as all-NaN, shape (N,128), so arrays
                              stack the same way regardless of checkpoint.
  mu_15, log_var_15 (N, 15)  vae_head's raw (pre constrain_params_15) output
                              -- the unconstrained SPS latent

Decoder (frozen Speculator + FilterConv), two taps:
  log_spec        (N, 1999)  reconstructed rest-frame log-spectrum -- the
                              Speculator's output, captured via forward
                              hook; the decoder's intermediate layer.
                              (The Speculator's per-band hidden layers and
                              PCA coefficients are buffer-based, not
                              submodules, so tapping them would need the
                              band forward pass re-implemented -- not done.)
  m_ab_recon      (N, 10)    reconstructed AB magnitudes -- element 0 of
                              every version's forward() return, confirmed
                              identical position across old/new/mdn

Targets (from the "Standard diagnostic interface" `encode_theta()` every
model version implements identically, plus the raw catalog):
  z_true, z_pred, log_sigma_z, dz = z_pred - z_true
  theta_full      (N, 16)    physical SPS params [zred, logmass, logzsol, ...]
                              -- same info as mu_15 above, post the
                              constrain_params_15 physical transform
  i_ref           (N,)       reference-band magnitude (i_gaap1p0Mag or
                              i_cModelMag, whichever build_features used)
  g_minus_r       (N,)       g-r color (same GAAP/cModel convention as i_ref)
  mask            (N, n_bands) band-detection pattern (1 = detected)

Usage
-----
    ~/miniforge3/envs/WL_ML_Challenge/bin/python \\
        photoz_mlpvae/scripts/extract_activations.py \\
        --checkpoint photoz_mlpvae/trained/mlpvae_dp2_v1_lsst_gaap1p0/best.pt \\
        --catalog obs_catalog/data/412_som_test_rtn124_parquet_.../test_som_mc_matched_v3.parquet \\
        --out photoz_mlpvae/trained/mlpvae_dp2_v1_lsst_gaap1p0/probe_activations_test.npz

    # Explicit overrides if no sibling config.yaml exists:
    ~/miniforge3/envs/WL_ML_Challenge/bin/python \\
        photoz_mlpvae/scripts/extract_activations.py \\
        --checkpoint photoz_mlpvae/trained/mlpvae_v2_lsst_gaap1p0/best.pt \\
        --model-version old --use-gaap \\
        --catalog obs_catalog/data/84_test_v4_match_ecdfs_sitcomtn154_.../dp1_matched_v4_test.hdf5 \\
        --out /tmp/probe_dp1_test.npz
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import h5py
import yaml
import torch
import torch.nn as nn

_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)
from obs_catalog.dataloader import (
    build_features, TARGET_COL, REF_MAG, GAAP_REF_MAG,
)
warnings.filterwarnings("ignore")

SPECULATOR_DIR = os.path.join(_BASE, "speculator", "trained", "Inoue_IGM")
FILTER_DIR     = os.path.join(_BASE, "obs_catalog", "filters")


# ─────────────────────────────────────────────────────────────────────────────
# Catalog loading (parquet or HDF5, whichever the checkpoint's dataset used)
# ─────────────────────────────────────────────────────────────────────────────

def load_any_catalog(path: str, apply_cuts: bool = True,
                     split: str = "all", val_frac: float = 0.1,
                     split_seed: int = 42, cuts_then_split: bool = False,
                     subsample: int = 0, subsample_seed: int = 0) -> pd.DataFrame:
    """
    Load a parquet or HDF5 catalog into a DataFrame.

    `split`/`val_frac`/`split_seed` replicate the training scripts'
    split_train_val() byte-for-byte (rng.permutation of the row order;
    first val_frac of the permutation is val, rest is train), so probes can
    be fit on exactly the rows the model took gradient steps on ("train"),
    or its early-stopping set ("val").

    Cut/split ORDER matters and differs per training script:
      - train_dp1.py: no cuts at all, splits the RAW file
        -> use --no-cuts --split train (split happens on raw order).
      - train_dp2.py: cuts first (load_parquet_galaxies: refExtendedness==1
        and finite z>0, same criteria as here), THEN splits the cut frame
        -> use --cuts-then-split --split train.

    `subsample` (applied last) keeps a uniform random subset — for
    probe-fitting on DP2's ~500k-row train split, where the full set is
    unnecessarily large for ridge probes and makes the npz multi-GB.
    """
    if path.endswith(".hdf5") or path.endswith(".h5"):
        with h5py.File(path, "r") as f:
            df = pd.DataFrame({k: f[k][:] for k in f.keys()})
    else:
        df = pd.read_parquet(path)

    def _split(df):
        if split == "all":
            return df
        rng   = np.random.RandomState(split_seed)
        idx   = rng.permutation(len(df))
        n_val = int(val_frac * len(df))
        take  = idx[:n_val] if split == "val" else idx[n_val:]
        out = df.iloc[take].reset_index(drop=True)
        print(f"Split '{split}' (val_frac={val_frac}, seed={split_seed}): "
              f"{len(df):,} -> {len(out):,} rows")
        return out

    def _cuts(df):
        if not apply_cuts:
            print(f"Loaded {path}: {len(df):,} rows (no cuts)")
            return df
        n1 = len(df)
        keep = np.ones(n1, dtype=bool)
        if "refExtendedness" in df.columns:
            ext = pd.to_numeric(df["refExtendedness"], errors="coerce").to_numpy(dtype="float64")
            keep &= (ext == 1)
        if TARGET_COL in df.columns:
            z = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(dtype="float64")
            keep &= np.isfinite(z) & (z > 0)
        out = df.loc[keep].reset_index(drop=True)
        print(f"Loaded {path}: {n1:,} rows -> {len(out):,} after cuts")
        return out

    df = _split(_cuts(df)) if cuts_then_split else _cuts(_split(df))

    if subsample and subsample < len(df):
        rng = np.random.RandomState(subsample_seed)
        df = df.iloc[rng.choice(len(df), subsample, replace=False)].reset_index(drop=True)
        print(f"Subsampled to {len(df):,} rows (seed={subsample_seed})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Generic layer taps (structure shared by old/new/mdn)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def tap_trunk(encoder: nn.Module, x: torch.Tensor):
    """
    Returns (h0, h1_mid, h1, h2_mid, h2) -- generic across all model
    versions. h1_mid/h2_mid are the mid-ResBlock activations (the
    Linear+BN+ReLU at block[:3], before the second Linear+BN and the
    residual add), doubling the trunk's depth resolution.
    """
    h0 = encoder.input_proj(x)
    rb0, rb1 = encoder.res_blocks[0], encoder.res_blocks[1]
    h1_mid = rb0.block[:3](h0)
    h1     = rb0(h0)
    h2_mid = rb1.block[:3](h1)
    h2     = rb1(h1)
    return h0, h1_mid, h1, h2_mid, h2


@torch.no_grad()
def tap_z_head_hidden(z_head: nn.Module, h2: torch.Tensor, hidden_dim: int = 128):
    """
    Returns the z_head's hidden-layer activation if it has one (new/mdn: a
    2-layer nn.Sequential), else an all-NaN placeholder of the same shape
    (old: a bare Linear(width,1) with no hidden layer) so downstream arrays
    stack cleanly regardless of --model-version.
    """
    if isinstance(z_head, nn.Sequential) and len(z_head) > 1:
        return z_head[:-1](h2)
    return torch.full((h2.shape[0], hidden_dim), float("nan"), device=h2.device)


def unpack_forward_output(model_version: str, fwd_out: tuple):
    """
    model.forward(x) returns a version-specific tuple. `m_ab_recon` (the
    frozen decoder's final -- and only tapped -- layer) is element 0 in
    every version; mu_15/log_var_15 (vae_head's raw output) sit at different
    positions per version's own forward():
      old : (m_ab_recon, z_pred, mu_15,      log_var_15, log_sigma_z)
      mdn : (m_ab_recon, z_pred, sigma_z,    mu_15,      log_var_15)
      new : (m_ab_recon, z_pred, mu_z,       log_var_z,  mu_15, log_var_15)
    """
    m_ab_recon = fwd_out[0]
    if model_version == "old":
        mu_15, log_var_15 = fwd_out[2], fwd_out[3]
    elif model_version == "mdn":
        mu_15, log_var_15 = fwd_out[3], fwd_out[4]
    else:  # "new"
        mu_15, log_var_15 = fwd_out[4], fwd_out[5]
    return m_ab_recon, mu_15, log_var_15


# ─────────────────────────────────────────────────────────────────────────────
# Model + run-config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, model_version: str, device: str):
    if model_version == "new":
        from photoz_mlpvae.model.photoz_mlpvae import PhotozMLPVAE
    elif model_version == "mdn":
        from photoz_mlpvae.model.photoz_mlpvae_mdn import PhotozMLPVAE
    else:
        from photoz_mlpvae.model.photoz_mlpvae_old import PhotozMLPVAE
    model, scaler, col_medians = PhotozMLPVAE.load(
        checkpoint, SPECULATOR_DIR, FILTER_DIR, device=device)
    model.eval()
    return model, scaler, col_medians


def infer_run_config(checkpoint: str, args) -> dict:
    """
    Fill in model_version/use_gaap/use_euclid/use_colors from the
    config.yaml next to the checkpoint (written by every train_*.py script),
    for any of those not explicitly overridden on the command line.
    """
    cfg_path = os.path.join(os.path.dirname(checkpoint), "config.yaml")
    file_cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            file_cfg = yaml.safe_load(f) or {}
        print(f"Read run config from {cfg_path}")
    else:
        print(f"WARNING: no sibling config.yaml at {cfg_path} -- "
              f"relying on CLI flags / defaults (model_version=new, "
              f"use_gaap/use_euclid=False, use_colors=True unless overridden)")

    if args.model_version is None and "model_version" not in file_cfg:
        print("WARNING: config.yaml has no 'model_version' field (older "
              "checkpoints predate it) -- defaulting to 'new'. Pass "
              "--model-version explicitly if this checkpoint is actually "
              "'old' or 'mdn', or loading will fail with a state_dict "
              "mismatch.")

    return dict(
        model_version = args.model_version or file_cfg.get("model_version", "new"),
        use_gaap      = args.use_gaap   if args.use_gaap   is not None else file_cfg.get("use_gaap", False),
        use_euclid    = args.use_euclid if args.use_euclid is not None else file_cfg.get("use_euclid", False),
        use_colors    = (not args.no_colors) if args.no_colors is not None else file_cfg.get("use_colors", True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a PhotozMLPVAE .pt checkpoint (any --model-version)")
    parser.add_argument("--catalog",    required=True,
                        help="Parquet or HDF5 catalog to run through the model")
    parser.add_argument("--out",        required=True,
                        help="Output .npz path for activations + targets")
    parser.add_argument("--model-version", choices=["old", "new", "mdn"], default=None,
                        help="Default: read from the checkpoint's sibling config.yaml")
    parser.add_argument("--use-gaap",   action="store_true", default=None,
                        help="Default: read from config.yaml")
    parser.add_argument("--use-euclid", action="store_true", default=None,
                        help="Default: read from config.yaml")
    parser.add_argument("--no-colors",  action="store_true", default=None,
                        help="Default: read from config.yaml")
    parser.add_argument("--no-cuts",    action="store_true",
                        help="Skip the refExtendedness/valid-z cuts, matching "
                             "train_dp1.py's raw pre-split HDF5 loading")
    parser.add_argument("--split",      choices=["all", "train", "val"], default="all",
                        help="Replicate train_dp1.py's train/val carve-out of this "
                             "catalog and keep only the given part (default: all rows)")
    parser.add_argument("--val-frac",   type=float, default=0.1,
                        help="val fraction for --split (train_dp1.py default: 0.1)")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="RNG seed for --split (both train scripts use SEED=42)")
    parser.add_argument("--cuts-then-split", action="store_true",
                        help="Apply cuts BEFORE the train/val split, matching "
                             "train_dp2.py (train_dp1.py splits the raw file)")
    parser.add_argument("--subsample",  type=int, default=0,
                        help="Keep a uniform random subset of this many rows "
                             "(applied after split+cuts; 0 = all)")
    parser.add_argument("--subsample-seed", type=int, default=0)
    parser.add_argument("--batch",      type=int, default=1024)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_cfg = infer_run_config(args.checkpoint, args)
    print(f"Config: {run_cfg}")

    model, scaler, col_medians = load_model(args.checkpoint, run_cfg["model_version"], args.device)
    n_in = model.encoder.input_proj[0].in_features
    print(f"Loaded {args.checkpoint}  (model_version={run_cfg['model_version']}, n_in={n_in})")

    df = load_any_catalog(args.catalog, apply_cuts=not args.no_cuts,
                          split=args.split, val_frac=args.val_frac,
                          split_seed=args.split_seed,
                          cuts_then_split=args.cuts_then_split,
                          subsample=args.subsample,
                          subsample_seed=args.subsample_seed)
    X, mags_obs, mag_errs, mask, z_true, _, _ = build_features(
        df, use_colors=run_cfg["use_colors"], use_euclid=run_cfg["use_euclid"],
        use_gaap=run_cfg["use_gaap"], scaler=scaler, col_medians=col_medians,
    )
    if X.shape[1] != n_in:
        raise ValueError(
            f"Feature dim mismatch: catalog+flags built {X.shape[1]}-dim features "
            f"but checkpoint expects {n_in}-dim. Check --use-gaap/--use-euclid/"
            f"--no-colors against how this checkpoint was actually trained "
            f"(see its config.yaml)."
        )
    ref_mag_col = GAAP_REF_MAG if run_cfg["use_gaap"] else REF_MAG
    i_ref = (df[ref_mag_col].astype(float).values if ref_mag_col in df.columns
             else np.full(len(df), np.nan, dtype=np.float32))
    g_col, r_col = (("g_gaap1p0Mag", "r_gaap1p0Mag") if run_cfg["use_gaap"]
                    else ("g_cModelMag", "r_cModelMag"))
    g_minus_r = (df[g_col].astype(float).values - df[r_col].astype(float).values
                 if (g_col in df.columns and r_col in df.columns)
                 else np.full(len(df), np.nan, dtype=np.float32))

    n = len(df)
    layers = {k: [] for k in ["h0", "h1_mid", "h1", "h2_mid", "h2",
                              "z_head_hidden", "mu_15", "log_var_15",
                              "log_spec", "m_ab_recon",
                              "z_pred", "log_sigma_z", "theta_full"]}

    # The reconstructed rest-frame spectrum (batch, 1999) is internal to
    # decode() -- capture it with a forward hook on the frozen Speculator
    # (fires during model(xb); output[0] is log_spec, [1] the wl grid).
    spec_cache = {}
    hook = model.speculator.register_forward_hook(
        lambda mod, inp, out: spec_cache.__setitem__("log_spec", out[0].detach()))

    print(f"\nExtracting activations for {n:,} galaxies (batch={args.batch})...")
    with torch.no_grad():
        for i0 in range(0, n, args.batch):
            xb = torch.from_numpy(X[i0:i0 + args.batch]).to(args.device)

            h0, h1_mid, h1, h2_mid, h2 = tap_trunk(model.encoder, xb)
            zh_hidden = tap_z_head_hidden(model.encoder.z_head, h2)
            # model(xb) / encode_theta(xb) each re-run input_proj/res_blocks
            # internally -- redundant compute (3 forward passes per batch
            # instead of 1), but it keeps this script from having to
            # hand-replicate each version's differing head-conditioning
            # logic, which model.forward()/encode_theta() already
            # standardize across old/new/mdn. Fine for an offline one-shot
            # extraction pass over a model this small.
            m_ab_recon, mu_15, log_var_15 = unpack_forward_output(
                run_cfg["model_version"], model(xb))
            z_pred, log_sigma_z, theta_full = model.encode_theta(xb)

            layers["h0"].append(h0.cpu().numpy())
            layers["h1_mid"].append(h1_mid.cpu().numpy())
            layers["h1"].append(h1.cpu().numpy())
            layers["h2_mid"].append(h2_mid.cpu().numpy())
            layers["h2"].append(h2.cpu().numpy())
            layers["z_head_hidden"].append(zh_hidden.cpu().numpy())
            layers["mu_15"].append(mu_15.cpu().numpy())
            layers["log_var_15"].append(log_var_15.cpu().numpy())
            layers["log_spec"].append(spec_cache["log_spec"].cpu().numpy())
            layers["m_ab_recon"].append(m_ab_recon.cpu().numpy())
            layers["z_pred"].append(z_pred.cpu().numpy())
            layers["log_sigma_z"].append(log_sigma_z.cpu().numpy())
            layers["theta_full"].append(theta_full.cpu().numpy())
    hook.remove()

    out = {k: np.concatenate(v, axis=0) for k, v in layers.items()}
    out["x"]         = X
    out["mask"]      = mask
    out["z_true"]    = z_true
    out["i_ref"]     = i_ref
    out["g_minus_r"] = g_minus_r
    out["dz"]        = out["z_pred"] - z_true

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, **out)

    print(f"\nSaved {n:,} galaxies x {len(out)} arrays -> {args.out}")
    for k, v in out.items():
        print(f"  {k:<14s} {v.shape}  dtype={v.dtype}")


if __name__ == "__main__":
    main()
