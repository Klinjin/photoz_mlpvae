"""
train_dp2.py
============
Training script for PhotozMLPVAE on DP2 (Rubin Data Preview 2) spec-z
catalogs, loaded directly from parquet rather than the pre-split HDF5 files
train_dp1.py consumes.

Data
----
  Train : 415_clipped_train_rtn124_parquet_.../train_clipped_v3.parquet
          (554,676 rows total; galaxies + stars + AGN/QSO mixed together)
  Test  : 412_som_test_rtn124_parquet_.../test_som_mc_matched_v3.parquet
          (18,400 rows; SOM-matched Monte Carlo test set)

Both catalogs use the same LSST-band column convention as the DP1 catalogs
(u/g/r/i/z/y_{cModel,gaap1p0,psf,kron}Mag[Err], refExtendedness, redshift)
and are handled by the same obs_catalog/dataloader.py helpers -- but they
carry no Euclid columns at all, so use_euclid is hardcoded False here
(build_features would KeyError on the Euclid columns otherwise).

Differences from train_dp1.py
------------------------------
  - Loads raw parquet (not pre-split HDF5); applies the refExtendedness==1
    (galaxy) + valid-z + z_max cuts itself (load_parquet_galaxies), then
    carves a val split out of the train parquet (VAL_FRAC, default 10%),
    matching train_dp1.py's HDF5 val-carve behaviour.
  - use_euclid is not exposed as a CLI flag -- LSST-only encoder input
    (24-dim in colour mode, since no Euclid columns exist to include).
  - use_gaap defaults to True (DP2 request specifically asked for GAAP
    1.0" aperture mags); pass --cmodel to use cModel mags instead.
  - z_max (default 5.5) drops galaxies whose spec-z exceeds the range
    compute_z_weights bins over / the 'old' model's z_head can represent
    (the 'new' z-separated head's cap is model/photoz_mlpvae.py's
    _ZRED_MAX_SPEC, raised 6.5->8.5 on 2026-08-11 to cover DP2's high-z
    tail) -- <0.1% of rows in both files, so this is a cheap way to avoid
    unrepresentable labels quietly
    biasing the loss rather than a real coverage loss. This cut is
    train/val-only: the test parquet is always loaded with z_max=None
    (full, uncut redshift range), so σ_NMAD/outlier numbers reflect
    real-world performance -- including any tail the model was never
    trained on -- and stay comparable across different --z-max choices.
  - model_version defaults to "new" (the z-separated architecture; see
    photoz_mlpvae/README.md -- it holds the current best σ_NMAD/outlier
    numbers on DP1) rather than train_dp1.py's "old" default.
  - init_from defaults to the LSST-only/GAAP synthetic-SED warm-start
    checkpoint (mlpvae_synth_zsep_v1_no_euclid_gaap1p0/best.pt), the same
    one the current best DP1 recipe warm-starts from. Pass --init-from ""
    (or "none") to train from scratch instead.
  - Shared plotting/epoch-loop helpers (_Tee, sigma_nmad, split_train_val,
    plot_curves, plot_lr_and_gradnorm, plot_z_weights, run_epoch) are
    imported from train_dp1.py rather than duplicated.
  - All other architecture / training logic identical to train_dp1.py.

Usage
-----
    ~/miniforge3/envs/WL_ML_Challenge/bin/python photoz_mlpvae/scripts/train_dp2.py
    ~/miniforge3/envs/WL_ML_Challenge/bin/python photoz_mlpvae/scripts/train_dp2.py \\
        --model-name mlpvae_dp2_v1_lsst_gaap1p0 --epochs 2000 --lr 1e-3 --lam-z-min 20
"""

import os, sys, argparse, warnings, yaml, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import torch
_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)
from obs_catalog.dataloader import (
    build_features, compute_z_weights, make_dataloader,
    REF_MAG, GAAP_REF_MAG, TARGET_COL,
)
from photoz_utils import (
    plot_scatter, plot_redshift_hist, plot_metrics_vs_zpred,
    plot_metrics_paper_style, plot_pit,
    plot_corner_posteriors, plot_sed_reconstructions,
    compute_metrics,
    plot_scatter_density, plot_metrics_binned_3sig,
)
from photoz_mlpvae.scripts.train_dp1 import (
    _Tee, sigma_nmad, split_train_val,
    plot_curves, plot_lr_and_gradnorm, plot_z_weights, run_epoch,
    report_quality_cut_metrics,
)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Adjustables
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME    = "mlpvae_dp2_v3_lsst_gaap1p0"
TRAIN_PARQUET = os.path.join(
    _BASE, "obs_catalog", "data",
    "415_clipped_train_rtn124_parquet_0103491bd9870eb3d19bd89fa6cd57a9",
    "train_clipped_v3.parquet",
)
TEST_PARQUET = os.path.join(
    _BASE, "obs_catalog", "data",
    "412_som_test_rtn124_parquet_80763a5a141d268b7c6ed332ab552fbd",
    "test_som_mc_matched_v3.parquet",
)
SPECULATOR_DIR  = os.path.join(_BASE, "speculator", "trained", "Inoue_IGM")
FILTER_DIR      = os.path.join(_BASE, "obs_catalog", "filters")
OUT_BASE        = os.path.join(_BASE, "photoz_mlpvae", "trained")
SYNTH_INIT_CKPT = os.path.join(
    # v2 (2026-08-12): warm-start source retrained against the new Inoue
    # speculator (SEDs regenerated with ZMAX 5.5->6.5; see
    # speculator/retrain_inoue_zmax6p5.log). v1 was pretrained against the
    # stale z<=5.5 decoder -- using it here would warm-start the encoder to
    # decode through a speculator it was never matched to.
    _BASE, "photoz_mlpvae", "trained",
    "mlpvae_synth_zsep_v2_no_euclid_gaap1p0", "best.pt",
)

USE_COLORS = True     # default: colour-based features
VAL_FRAC   = 0.10      # fraction of train parquet reserved for validation
Z_MAX      = 5.5       # drop z_spec above this (see module docstring)

# MAX_EPOCHS reduced 2000->600 (dp2_v1 -> dp2_v2, 2026-08-12): DP2's train
# set has ~1,770 batches/epoch vs. DP1's ~24 (this recipe's epoch/warmup
# counts were originally tuned on DP1's much smaller set), so dp2_v1's
# 200-epoch warmup alone was ~354k gradient steps -- more than DP1's entire
# 2000-epoch schedule (~48k steps). Empirically, dp2_v1's rolling-avg
# σ_NMAD reached 0.0637 by epoch 500 vs. its final 0.0555 at epoch 1905 (a
# ~13% gap) while consuming half the run's ~13.3h wall-clock for that last
# stretch -- diminishing returns past roughly this point. 600 keeps some
# margin past the epoch-500 mark actually observed to still be improving.
MAX_EPOCHS  = 600
BATCH_SIZE  = 256
LR          = 1e-3
PATIENCE    = MAX_EPOCHS // 5
WARMUP_FRAC = 0.10     # Phase 1 length as a fraction of total epochs
RECON_RAMP  = 50
BETA_EPOCHS = 50
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

LAM_Z         = 20.0
LAM_R         = 0.1
LAM_Z_MIN     = 20.0   # no Phase-2 dip -- flat λ_z beat the dip-and-restore
                        # scheme on the DP1 zsep_v4 sweep (see PLAN.md)
LAM_Z_RESTORE = 50
BETA_MAX      = 0.1
SIGMA_FLOOR   = 0.3
N_ZBINS_WEIGHT = 25
Z_WEIGHT_CAP   = 10.0

PHASE2_VAE_LR_FRAC = 1.0
GRAD_NORM_REF       = 2.0
TRAIN_PLAN          = "always sigma_NMAD for best model"


# ─────────────────────────────────────────────────────────────────────────────
# Parquet loader
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet_galaxies(path: str, z_max: float = None, label: str = "catalog") -> pd.DataFrame:
    """
    Read a DP2 parquet catalog and keep extended sources (refExtendedness==1)
    with a valid, representable spec-z (0 < z, and z <= z_max if given).
    Mirrors load_catalog()'s galaxy selection in obs_catalog/dataloader.py,
    but coerces refExtendedness/redshift through pd.to_numeric first: these
    DP2 exports store them as pandas-nullable columns, and comparing those
    directly can leave stray <NA> entries in the boolean mask (pandas raises
    on "df[mask]" if any survive), whereas a coerced float64 numpy array
    turns NA into a plain NaN that compares to False as expected.
    """
    df = pd.read_parquet(path)
    n0 = len(df)

    ext = pd.to_numeric(df["refExtendedness"], errors="coerce").to_numpy(dtype="float64")
    z   = pd.to_numeric(df[TARGET_COL],        errors="coerce").to_numpy(dtype="float64")

    ext_ok  = (ext == 1)
    z_ok    = np.isfinite(z) & (z > 0)
    zmax_ok = (z <= z_max) if z_max is not None else np.ones(n0, dtype=bool)
    keep    = ext_ok & z_ok & zmax_ok

    gals = df.loc[keep].copy()
    gals[TARGET_COL] = gals[TARGET_COL].astype(float)

    n_star = int((~ext_ok).sum())
    n_badz = int((ext_ok & ~z_ok).sum())
    n_hiz  = int((ext_ok & z_ok & ~zmax_ok).sum())
    msg = (f"  {label}: {n0:,} rows  ->  refExtendedness==1: {int(ext_ok.sum()):,} "
           f"(dropped {n_star:,} non-galaxy/unknown)  ->  valid z>0: "
           f"{int((ext_ok & z_ok).sum()):,} (dropped {n_badz:,})")
    if z_max is not None:
        msg += f"  ->  z<={z_max:g}: {len(gals):,} (dropped {n_hiz:,})"
    print(msg)
    return gals.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",    default=MODEL_NAME)
    parser.add_argument("--train-parquet", default=TRAIN_PARQUET)
    parser.add_argument("--test-parquet",  default=TEST_PARQUET)
    parser.add_argument("--val-frac",      type=float, default=VAL_FRAC)
    parser.add_argument("--z-max",         type=float, default=Z_MAX,
                        help=f"Drop train/val galaxies with z_spec above this (default "
                             f"{Z_MAX:g}); <=0 disables the cut. Test is never cut -- it "
                             f"always evaluates on the full redshift range.")
    parser.add_argument("--epochs",        type=int,   default=MAX_EPOCHS)
    parser.add_argument("--patience",      type=int,   default=None,
                        help="Early-stop patience in epochs (default: epochs//5)")
    parser.add_argument("--batch",         type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--lam-z",              type=float, default=LAM_Z)
    parser.add_argument("--lam-r",              type=float, default=LAM_R)
    parser.add_argument("--lam-z-min",          type=float, default=LAM_Z_MIN,
                        help="λ_z floor at Phase 2 start (gives VAE gradient budget)")
    parser.add_argument("--lam-z-restore",      type=int,   default=LAM_Z_RESTORE,
                        help="Epochs to ramp λ_z back to full after β stabilizes")
    parser.add_argument("--sigma-floor",        type=float, default=SIGMA_FLOOR)
    parser.add_argument("--warmup-epochs",      type=int,   default=None,
                        help="Phase 1 length in epochs (default: 10%% of --epochs)")
    parser.add_argument("--phase2-vae-lr-frac",   type=float, default=PHASE2_VAE_LR_FRAC,
                        help="LR multiplier for vae_head at Phase 2 start")
    parser.add_argument("--grad-clip-max-norm", type=float, default=GRAD_NORM_REF,
                        help="Gradient-norm clip threshold for model.encoder.parameters() "
                             "(<=0 disables clipping; norm is still measured either way)")
    parser.add_argument("--z-weight-cap",       type=float, default=Z_WEIGHT_CAP,
                        help="Cap z-weights at this multiple of mean (0 = no cap)")
    parser.add_argument("--init-from",          default=SYNTH_INIT_CKPT,
                        help="Path to a checkpoint to partial-init from (shape-matched "
                             "layers); '' or 'none' disables warm-start")
    parser.add_argument("--device",        default=DEVICE)
    parser.add_argument("--no-colors",  action="store_true",
                        help="Use raw magnitude features instead of colours")
    parser.add_argument("--cmodel",     action="store_true",
                        help="Use cModel mags instead of GAAP 1.0-arcsec aperture mags "
                             "(default: GAAP, per DP2 request)")
    parser.add_argument("--model-version", choices=["old", "new", "mdn"], default="new",
                        help="'old' = 3-head photoz_mlpvae_old (16-dim joint VAE); "
                             "'new' (default) = photoz_mlpvae z-separated design, current "
                             "best on DP1; 'mdn' = z-separated with mixture-density z head")
    parser.add_argument("--out-base",   default=OUT_BASE,
                        help="Base directory for the run's output folder")
    parser.add_argument("--trial",      action="store_true",
                        help="Comparison-trial mode: train + print test metrics only, "
                             "skip all figures/diagnostics")
    args = parser.parse_args()

    if args.warmup_epochs is None:
        args.warmup_epochs = max(1, round(WARMUP_FRAC * args.epochs))

    use_colors = not args.no_colors
    use_euclid = False   # DP2 parquet catalogs carry no Euclid columns
    use_gaap   = not args.cmodel
    z_max      = args.z_max if (args.z_max is not None and args.z_max > 0) else None
    init_from  = (args.init_from if args.init_from and
                  args.init_from.strip().lower() not in ("", "none") else None)

    out_dir = os.path.join(args.out_base, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────
    log_path    = os.path.join(out_dir, "train.log")
    sys.stdout  = _Tee(sys.__stdout__, log_path)
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Training started")
    print(f"Model: {args.model_name}  |  device={args.device}  "
          f"|  use_colors={use_colors}  |  use_euclid={use_euclid}  |  use_gaap={use_gaap}")
    print(f"Outputs → {out_dir}/")

    # ── Config ────────────────────────────────────────────────────────────
    config = dict(
        model_name     = args.model_name,
        data_source    = "dp2_parquet",
        train_parquet  = args.train_parquet,
        test_parquet   = args.test_parquet,
        val_frac       = args.val_frac,
        z_max_train    = z_max,   # train/val cut only -- test is always the full range
        use_colors     = use_colors,
        use_euclid     = use_euclid,
        use_gaap       = use_gaap,
        epochs         = args.epochs,
        batch          = args.batch,
        lr             = args.lr,
        lam_z               = args.lam_z,
        lam_z_min           = args.lam_z_min,
        lam_z_restore       = args.lam_z_restore,
        lam_r               = args.lam_r,
        sigma_floor         = args.sigma_floor,
        warmup_epochs       = args.warmup_epochs,
        phase2_vae_lr_frac  = args.phase2_vae_lr_frac,
        grad_clip_max_norm  = args.grad_clip_max_norm,
        z_weight_cap        = args.z_weight_cap,
        init_from           = init_from,
        beta_max            = BETA_MAX,
        beta_epochs         = BETA_EPOCHS,
        recon_ramp          = RECON_RAMP,
        patience            = args.patience if args.patience is not None else args.epochs // 5,
        seed                = SEED,
        device              = args.device,
        n_zbins_weight      = N_ZBINS_WEIGHT,
        speculator_dir      = SPECULATOR_DIR,
        filter_dir          = FILTER_DIR,
        training_scheme     = TRAIN_PLAN,
        model_version       = args.model_version,
    )
    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config → {cfg_path}")

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"\nLoading train parquet: {args.train_parquet}")
    tr_val_df = load_parquet_galaxies(args.train_parquet, z_max=z_max, label="train parquet")
    tr_df, val_df = split_train_val(tr_val_df, args.val_frac, SEED)

    print(f"Loading test parquet:  {args.test_parquet}")
    # z_max is a training-time cut only -- test always evaluates on the full,
    # uncut redshift distribution so σ_NMAD/outlier numbers reflect real-world
    # performance (including any tail the model was never trained to handle),
    # and so different --z-max training runs stay comparable against the same
    # ground truth.
    te_df = load_parquet_galaxies(args.test_parquet, z_max=None, label="test parquet")

    print(f"  Train {len(tr_df):,}  Val {len(val_df):,}  Test {len(te_df):,}")

    (X_tr,  mags_tr,  errs_tr,  mask_tr,  z_tr,  scaler, col_med) = \
        build_features(tr_df,  use_colors=use_colors, use_euclid=use_euclid,
                       use_gaap=use_gaap, fit=True)
    (X_val, mags_val, errs_val, mask_val, z_val, _, _) = \
        build_features(val_df, use_colors=use_colors, use_euclid=use_euclid,
                       use_gaap=use_gaap, scaler=scaler, col_medians=col_med)
    (X_te,  mags_te,  errs_te,  mask_te,  z_te,  _, _) = \
        build_features(te_df,  use_colors=use_colors, use_euclid=use_euclid,
                       use_gaap=use_gaap, scaler=scaler, col_medians=col_med)

    zw_raw = compute_z_weights(z_tr)
    zw     = zw_raw
    if args.z_weight_cap > 0:
        zw = np.clip(zw_raw, None, args.z_weight_cap)
        zw = (zw / zw.mean()).astype(np.float32)
    print(f"z-weights (train): min={zw.min():.3f}  max={zw.max():.3f}  "
          f"mean={zw.mean():.3f}  (cap={args.z_weight_cap:.0f}x)")
    if not args.trial:
        plot_z_weights(z_tr, zw_raw, zw, args.z_weight_cap,
                       os.path.join(out_dir, "z_weights.png"))

    tr_loader  = make_dataloader(X_tr,  mags_tr,  errs_tr,  mask_tr,  z_tr,
                                 zw, args.batch, shuffle=True)
    val_loader = make_dataloader(X_val, mags_val, errs_val, mask_val, z_val,
                                 np.ones(len(z_val), dtype=np.float32), args.batch, shuffle=False)
    te_loader  = make_dataloader(X_te,  mags_te,  errs_te,  mask_te,  z_te,
                                 np.ones(len(z_te),  dtype=np.float32), args.batch, shuffle=False)
    print(f"Encoder input dim: {X_tr.shape[1]}")

    # ── Model ─────────────────────────────────────────────────────────────
    if args.model_version == "new":
        from photoz_mlpvae.model.photoz_mlpvae import PhotozMLPVAE
    elif args.model_version == "mdn":
        from photoz_mlpvae.model.photoz_mlpvae_mdn import PhotozMLPVAE
    else:
        from photoz_mlpvae.model.photoz_mlpvae_old import PhotozMLPVAE

    model = PhotozMLPVAE(SPECULATOR_DIR, FILTER_DIR,
                         encoder_n_in=X_tr.shape[1]).to(args.device)
    n_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Encoder trainable parameters: {n_enc:,}")

    if init_from:
        ckpt     = torch.load(init_from, map_location=args.device, weights_only=False)
        src      = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        cur      = model.state_dict()
        matched  = {k: v for k, v in src.items()
                    if k in cur and v.shape == cur[k].shape}
        skipped  = set(src) - set(matched)
        cur.update(matched)
        model.load_state_dict(cur)
        print(f"Warm-start: loaded {len(matched)}/{len(cur)} keys from {init_from}")
        if skipped:
            print(f"  Skipped (shape mismatch / new head): {sorted(skipped)}")

    # ── Phase 1 optimizer (trunk + z-head + sigma_z-head; vae_head frozen) ─
    print(f"\n── Phase 1: z-supervised warmup "
          f"({args.warmup_epochs} epochs, vae_head frozen) ──")
    model.encoder.freeze_vae_head()
    phase1_params = [p for p in model.encoder.parameters() if p.requires_grad]
    opt   = torch.optim.Adam(phase1_params, lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-6)

    best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    ckpt_path    = os.path.join(out_dir, "best.pt")
    best_nmad_avg = float("inf")
    nmad_window   = []
    patience_ctr  = 0
    patience      = args.patience if args.patience is not None else args.epochs // 5
    phase2_started = False
    NMAD_AVG_WINDOW = 5

    history = {k: [] for k in [
        "tr_total", "tr_z_sup", "tr_recon", "tr_kl", "tr_log_sigma_z",
        "val_total", "val_z_sup", "val_recon", "val_kl", "val_log_sigma_z",
        "val_sigma_nmad", "beta",
        "lr_trunk", "lr_vae", "grad_norm_mean", "grad_norm_max",
    ]}
    hist_keys = list(history.keys())
    hist_path = os.path.join(out_dir, "history.csv")
    with open(hist_path, "w") as hf:
        hf.write("epoch," + ",".join(hist_keys) + "\n")
    print(f"History → {hist_path}")

    print(f"Training for up to {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):

        # ── Phase transition ──────────────────────────────────────────────
        if epoch == args.warmup_epochs + 1 and not phase2_started:
            # Snapshot the pure z-supervised state (vae_head still frozen/
            # zero-init) before Phase 2 touches anything -- a clean fallback/
            # warm-start anchor independent of best.pt's rolling-average
            # tracking, and predating the historically more crash-prone
            # reconstruction+KL fine-tuning (see PLAN.md Failure 13).
            phase1_ckpt_path = os.path.join(out_dir, "phase1_end.pt")
            model.save(phase1_ckpt_path, scaler=scaler, col_medians=col_med)
            print(f"  Phase 1 end checkpoint (epoch {epoch - 1}) → {phase1_ckpt_path}")
            print(f"\n── Phase 2: joint fine-tuning "
                  f"(epoch {epoch}, vae_head unfrozen, "
                  f"trunk/z_head lr unchanged (cosine continues), "
                  f"vae lr ×{args.phase2_vae_lr_frac}) ──\n")
            model.encoder.unfreeze_vae_head()
            opt.add_param_group({
                "params":       list(model.encoder.vae_head.parameters()),
                "lr":           args.lr * args.phase2_vae_lr_frac,
                "weight_decay": 1e-5,
            })
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - epoch + 1, eta_min=1e-6)
            patience_ctr   = 0
            phase2_started = True

        # ── Loss weights for this epoch ───────────────────────────────────
        kl_epoch  = max(0, epoch - args.warmup_epochs)

        if epoch <= args.warmup_epochs:
            lam_r_eff = 0.0
        elif epoch <= args.warmup_epochs + RECON_RAMP:
            lam_r_eff = args.lam_r * (epoch - args.warmup_epochs) / RECON_RAMP
        else:
            lam_r_eff = args.lam_r

        beta = min(BETA_MAX, BETA_MAX * kl_epoch / max(BETA_EPOCHS - 1, 1))

        if epoch <= args.warmup_epochs:
            lam_z_eff = args.lam_z
        elif kl_epoch <= BETA_EPOCHS:
            lam_z_eff = args.lam_z_min
        elif kl_epoch <= BETA_EPOCHS + args.lam_z_restore:
            frac      = (kl_epoch - BETA_EPOCHS) / max(args.lam_z_restore, 1)
            lam_z_eff = args.lam_z_min + (args.lam_z - args.lam_z_min) * frac
        else:
            lam_z_eff = args.lam_z

        lr_trunk = opt.param_groups[0]["lr"]
        lr_vae   = opt.param_groups[1]["lr"] if len(opt.param_groups) > 1 else None

        import time as _time
        t0 = _time.time()

        tr_loss, tr_grad_stats = run_epoch(model, tr_loader,  args.device, opt=opt,
                             lam_z=lam_z_eff, lam_r=lam_r_eff, beta=beta,
                             sigma_floor=args.sigma_floor,
                             use_nll_z=True,
                             grad_clip_max_norm=args.grad_clip_max_norm)
        val_loss, _ = run_epoch(model, val_loader, args.device, opt=None,
                             lam_z=lam_z_eff, lam_r=lam_r_eff, beta=beta,
                             sigma_floor=args.sigma_floor,
                             use_nll_z=True)

        model.eval()
        z_pred_val = []
        with torch.no_grad():
            for x_b, *_ in val_loader:
                _, z_mu, _ = model.predict_z(x_b.to(args.device), n_samples=1)
                z_pred_val.append(z_mu.cpu().numpy())
        z_pred_val = np.concatenate(z_pred_val)
        nmad = sigma_nmad(z_pred_val, z_val)

        sched.step()
        dt = _time.time() - t0

        for k in ("total", "z_sup", "recon", "kl", "log_sigma_z"):
            history[f"tr_{k}"].append(tr_loss[k])
            history[f"val_{k}"].append(val_loss[k])
        history["val_sigma_nmad"].append(nmad)
        history["beta"].append(beta)
        history["lr_trunk"].append(lr_trunk)
        history["lr_vae"].append(lr_vae)
        history["grad_norm_mean"].append(tr_grad_stats["mean"])
        history["grad_norm_max"].append(tr_grad_stats["max"])

        with open(hist_path, "a") as hf:
            row = ["" if history[k][-1] is None else f"{history[k][-1]:.6g}"
                   for k in hist_keys]
            hf.write(f"{epoch}," + ",".join(row) + "\n")

        nmad_window.append(nmad)
        if len(nmad_window) > NMAD_AVG_WINDOW:
            nmad_window.pop(0)
        nmad_avg = float(np.mean(nmad_window))

        vae_lr_str = f"{lr_vae:.1e}" if lr_vae is not None else "-"
        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"tr={tr_loss['total']:.4f}  val={val_loss['total']:.4f}  "
              f"(z={val_loss['z_sup']:.4f} r={val_loss['recon']:.4f} "
              f"kl={val_loss['kl']:.4f})  "
              f"σ_NMAD={nmad:.4f}(avg={nmad_avg:.4f})  σ_z={val_loss['log_sigma_z']:.3f}  "
              f"β={beta:.3f}  λ_z={lam_z_eff:.1f}  lam_r={lam_r_eff:.2f}  "
              f"lr={lr_trunk:.1e}/{vae_lr_str}  "
              f"gnorm={tr_grad_stats['mean']:.2f}/{tr_grad_stats['max']:.2f}  t={dt:.1f}s")

        if np.isnan(tr_loss["total"]) or np.isnan(val_loss["total"]):
            print(f"  NaN detected at epoch {epoch} — stopping.")
            break

        if not np.isnan(nmad_avg) and nmad_avg < best_nmad_avg:
            best_nmad_avg = nmad_avg
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            patience_ctr  = 0
            model.save(ckpt_path, scaler=scaler, col_medians=col_med)
            print(f"  ✓ New best avg σ_NMAD={nmad_avg:.4f} (inst={nmad:.4f})  saved → {ckpt_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping (no improvement for {patience} epochs).")
                break

    # ── Final evaluation ──────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    z_pred_te, z_std_te, z_samples_te = [], [], []
    with torch.no_grad():
        for x_b, *_ in te_loader:
            z_samps, z_mu, z_sd = model.predict_z(x_b.to(args.device), n_samples=200)
            z_pred_te.append(z_mu.cpu().numpy())
            z_std_te.append(z_sd.cpu().numpy())
            z_samples_te.append(z_samps.cpu().numpy())
    z_pred_te    = np.concatenate(z_pred_te)
    z_std_te     = np.concatenate(z_std_te)
    z_samples_te = np.concatenate(z_samples_te)

    dz_te = (z_pred_te - z_te) / (1.0 + z_te)
    bias_te, nmad_te, fout_te, rms_te, fcat_te = compute_metrics(dz_te, z_te)
    print(f"\n=== Test-set metrics ===")
    print(f"  σ_NMAD  : {nmad_te:.4f}")
    print(f"  Bias    : {bias_te:.4f}")
    print(f"  Outliers: {fout_te*100:.2f}%")
    print(f"  RMS     : {rms_te:.4f}")
    print(f"  f_cat   : {fcat_te*100:.2f}%")

    # report_quality_cut_metrics(dz_te, z_te, z_std_te)

    # ── Save predictions ──────────────────────────────────────────────────
    pred_path = os.path.join(out_dir, "test_predictions.csv")
    pd.DataFrame({
        "z_true":  z_te,
        "z_pred":  z_pred_te,
        "z_std":   z_std_te,
        "delta_z": (z_pred_te - z_te) / (1.0 + z_te),
    }).to_csv(pred_path, index=False)
    print(f"  Predictions → {pred_path}")

    # ── Diagnostics + figures (skipped entirely in --trial mode) ──────────
    if args.trial:
        print("\nTrial mode: skipping diagnostics and figures.")
    else:
        _targets = [
            (0.5, 0.5, "good_lowz"),
            (3.5, 3.5, "good_highz"),
            (0.5, 1.5, "mild_overest"),
            (0.5, 3.0, "severe_overest"),
            (3.5, 2.5, "mild_underest"),
            (3.5, 1.0, "severe_underest"),
        ]
        sel_idx = []
        for zt, zp, _ in _targets:
            score = np.abs(z_te - zt) + np.abs(z_pred_te - zp)
            sel_idx.append(int(np.argmin(score)))

        sel_labels = [lbl for _, _, lbl in _targets]
        sel_z_true = z_te[sel_idx]
        sel_z_pred = z_pred_te[sel_idx]
        sel_X      = X_te[sel_idx]
        sel_mags   = mags_te[sel_idx]
        sel_errs   = errs_te[sel_idx]
        sel_mask   = mask_te[sel_idx]

        print("\n── Failure-case diagnostics ──")
        for lbl, zt, zp in zip(sel_labels, sel_z_true, sel_z_pred):
            print(f"  {lbl:<15s}  z_true={zt:.3f}  z_pred={zp:.3f}")

        plot_corner_posteriors(
            model, sel_X, sel_z_true, sel_z_pred,
            sel_labels, args.device,
            os.path.join(out_dir, "corner"),
            n_samples=2000,
        )
        plot_sed_reconstructions(
            model, sel_X, sel_mags, sel_errs, sel_mask,
            sel_z_true, sel_z_pred, sel_labels,
            args.device,
            os.path.join(out_dir, "test_sed_reconstructions.png"),
        )

        # ── Figures ───────────────────────────────────────────────────────
        plot_scatter(z_te, z_pred_te,
                     os.path.join(out_dir, "test_scatter.png"),
                     title=f"{args.model_name}  test  σ_NMAD={nmad_te:.3f}")
        plot_redshift_hist(z_te, z_pred_te,
                           os.path.join(out_dir, "test_zdist.png"))
        plot_metrics_vs_zpred(z_te, z_pred_te,
                              os.path.join(out_dir, "test_metrics_vs_z.png"))
        if args.lam_z_min == args.lam_z:
            _lz_drop    = f"λ_z stays {args.lam_z:g} (no dip)"
            _lz_restore = f"λ_z flat at {args.lam_z:g}"
        else:
            _lz_drop    = f"λ_z {args.lam_z:g}→{args.lam_z_min:g}"
            _lz_restore = f"λ_z restored {args.lam_z_min:g}→{args.lam_z:g}"
        events = [
            (args.warmup_epochs,
             f"Phase 2: VAE unfrozen; {_lz_drop}; "
             f"λ_r 0→{args.lam_r:g} & β 0→{BETA_MAX:g} over {RECON_RAMP} ep"),
            (args.warmup_epochs + RECON_RAMP,
             f"λ_r={args.lam_r:g}, β={BETA_MAX:g} fully on"),
            (args.warmup_epochs + BETA_EPOCHS + args.lam_z_restore,
             _lz_restore),
        ]
        plot_curves(history, os.path.join(out_dir, "train_curves.png"), events=events)
        plot_lr_and_gradnorm(history, os.path.join(out_dir, "lr_and_gradnorm.png"),
                             grad_norm_ref=(args.grad_clip_max_norm if args.grad_clip_max_norm > 0
                                            else GRAD_NORM_REF),
                             clip_applied=(args.grad_clip_max_norm > 0), events=events)
        plot_metrics_paper_style(z_te, z_pred_te, args.model_name,
                                 os.path.join(out_dir, "test_metrics_paper_style.png"))
        plot_pit(z_te, z_pred_te, z_samples=z_samples_te,
                 save_path=os.path.join(out_dir, "test_pit.png"),
                 title=args.model_name)

        # ── Paper-style figures (density scatter + binned metrics) ────────
        plot_scatter_density(z_te, z_pred_te,
                             os.path.join(out_dir, "test_scatter_density.png"))
        plot_metrics_binned_3sig(z_te, dz_te,
                                  bins=np.arange(0.0, 3.2, 0.2),
                                  xlabel=r"$z_{\rm spec}$",
                                  save_path=os.path.join(out_dir, "test_metrics_vs_z_binned.png"))
        _ref_mag_col = GAAP_REF_MAG if use_gaap else REF_MAG
        plot_metrics_binned_3sig(te_df[_ref_mag_col].values, dz_te,
                                  bins=np.arange(18.0, 25.5, 0.5),
                                  xlabel="Magnitude",
                                  save_path=os.path.join(out_dir, "test_metrics_vs_mag.png"))

    print(f"\nDone. Best avg σ_NMAD={best_nmad_avg:.4f}")
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Training finished")
    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
