"""
train_hdf5.py
=============
Training script for PhotozMLPVAE using pre-split HDF5 catalogs.

Differences from train.py
--------------------------
  - Data loaded from HDF5 train/test files (pre-split by upstream pipeline)
  - Val split carved out of the HDF5 train file (VAL_FRAC, default 10%)
  - use_euclid defaults to False (LSST-only, 24-dim encoder input)
  - All other architecture / training logic identical to train.py

Usage
-----
    ~/miniforge3/envs/WL_ML_Challenge/bin/python photoz_mlpvae/scripts/train_hdf5.py
    ~/miniforge3/envs/WL_ML_Challenge/bin/python photoz_mlpvae/scripts/train_hdf5.py \\
        --model-name mlpvae_v2_lsst_colors
    ~/miniforge3/envs/WL_ML_Challenge/bin/python photoz_mlpvae/scripts/train_hdf5.py \\
        --no-colors
"""

import os, sys, time, argparse, warnings, yaml, datetime
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)
from obs_catalog.dataloader import (
    build_features, compute_z_weights, make_dataloader,
    REF_MAG, GAAP_REF_MAG,
)
from photoz_utils import (
    plot_scatter, plot_redshift_hist, plot_metrics_vs_zpred,
    plot_metrics_paper_style, plot_pit,
    plot_corner_posteriors, plot_sed_reconstructions,
    compute_metrics,
    plot_scatter_density, plot_metrics_binned_3sig,
)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, stream, fpath):
        self._stream = stream
        self._file   = open(fpath, "a", buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


# ─────────────────────────────────────────────────────────────────────────────
# Adjustables
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME     = "mlpvae_v1_lsst"
TRAIN_HDF5     = os.path.join(
    _BASE, "obs_catalog", "data",
    "83_training_v4_match_ecdfs_sitcomtn154_27184412354f233d1b9f512bdf350d58",
    "dp1_matched_v4_train.hdf5",
)
TEST_HDF5      = os.path.join(
    _BASE, "obs_catalog", "data",
    "84_test_v4_match_ecdfs_sitcomtn154_1ec71937ff3e4a0a8283535193764ec7",
    "dp1_matched_v4_test.hdf5",
)
SPECULATOR_DIR = os.path.join(_BASE, "speculator", "trained", "Inoue_IGM")
FILTER_DIR     = os.path.join(_BASE, "obs_catalog", "filters")
OUT_BASE       = os.path.join(_BASE, "photoz_mlpvae", "trained")

USE_COLORS     = True    # default: colour-based features
USE_EUCLID     = False   # LSST-only (24-dim encoder input)
VAL_FRAC       = 0.10    # fraction of HDF5 train file reserved for validation

MAX_EPOCHS    = 500
BATCH_SIZE    = 256
LR            = 1e-4
PATIENCE      = MAX_EPOCHS//5
WARMUP_EPOCHS = 50
RECON_RAMP    = 50
BETA_EPOCHS   = 50
SEED          = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

LAM_Z        = 20.0
LAM_R        = 0.1
LAM_Z_MIN    = 2.0         #  λ_z reduction at Phase 2 (v1 scheme)
LAM_Z_RESTORE = 50
BETA_MAX     = 0.1
SIGMA_FLOOR  = 0.3
N_ZBINS_WEIGHT = 25
Z_WEIGHT_CAP  = 10.0        # cap z-weights at this multiple of mean (prevent 277× extremes)

PHASE2_TRUNK_LR_FRAC = 0.1  # no LR scaling at Phase 2 (v1 scheme)
PHASE2_VAE_LR_FRAC   = 1.0  # vae_head at full LR (v1 scheme)
SIGMA_Z_WARMUP       = 0    # 0 = always NLL on; sq_err.detach() already protects z_pred
NMAD_AVG_WINDOW      = 5    # rolling-average window for checkpoint criterion (smooths 677-gal noise)
TRAIN_PLAN          = "always sigma_NMAD for best model"  # "phase1_warmup_then_phase2" or "joint_training"
# ─────────────────────────────────────────────────────────────────────────────
# HDF5 loader
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5(path: str) -> pd.DataFrame:
    """Read an HDF5 catalog into a pandas DataFrame."""
    with h5py.File(path, "r") as f:
        return pd.DataFrame({k: f[k][:] for k in f.keys()})


def split_train_val(df: pd.DataFrame, val_frac: float, seed: int):
    """Random train/val split of a DataFrame."""
    rng    = np.random.RandomState(seed)
    idx    = rng.permutation(len(df))
    n_val  = int(val_frac * len(df))
    tr_df  = df.iloc[idx[n_val:]].reset_index(drop=True)
    val_df = df.iloc[idx[:n_val]].reset_index(drop=True)
    return tr_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def sigma_nmad(z_pred, z_true):
    dz  = (z_pred - z_true) / (1.0 + z_true)
    med = np.nanmedian(dz)
    return 1.4826 * np.nanmedian(np.abs(dz - med))


# ─────────────────────────────────────────────────────────────────────────────
# Loss curves plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(history: dict, path: str, events: list = None):
    """
    Two-row layout: one wide total-loss panel on top, four breakdown panels below.
    events: list of (epoch_index, label) for phase/lambda change annotations.
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 7))
    gs  = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1.4, 1],
                            hspace=0.35, wspace=0.3)
    ax_top  = fig.add_subplot(gs[0, :])
    ax_subs = [fig.add_subplot(gs[1, i]) for i in range(4)]

    keys   = ["total", "z_sup", "recon", "kl", "log_sigma_z"]
    labels = ["Total loss", "z NLL", "Reconstruction", "KL", "mean log σ_z"]

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    event_colors = [prop_cycle[(i + 2) % len(prop_cycle)]
                    for i in range(len(events) if events else 0)]

    def _draw_events(ax, label_in_legend):
        if not events:
            return
        for (x, lbl), color in zip(events, event_colors):
            ax.axvline(x, ls=":", color=color, lw=1.4,
                       label=f"ep {x}: {lbl}" if label_in_legend else None)

    # ── Top: total loss ──────────────────────────────────────────────────────
    k, lab = "total", labels[0]
    if f"tr_{k}" in history and history[f"tr_{k}"]:
        ax_top.plot(history[f"tr_{k}"],  label="train")
        ax_top.plot(history[f"val_{k}"], label="val", ls="--")
    _draw_events(ax_top, label_in_legend=True)
    ax_top.set_title(lab, fontsize=12)
    ax_top.set_xlabel("epoch")
    ax_top.legend(fontsize=9, loc="upper right")

    # ── Bottom: breakdown losses ─────────────────────────────────────────────
    for ax, k, lab in zip(ax_subs, keys[1:], labels[1:]):
        if f"tr_{k}" in history and history[f"tr_{k}"]:
            ax.plot(history[f"tr_{k}"],  label="train")
            ax.plot(history[f"val_{k}"], label="val", ls="--")
        _draw_events(ax, label_in_legend=False)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)

    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curves → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, device, opt=None,
              lam_z=LAM_Z, lam_r=LAM_R, beta=0.0,
              sigma_floor=SIGMA_FLOOR, use_nll_z=True):
    is_train = (opt is not None)
    model.train() if is_train else model.eval()

    totals = {k: 0.0 for k in ("total", "z_sup", "recon", "kl", "log_sigma_z")}
    n = 0

    ctx = torch.enable_grad if is_train else torch.no_grad

    with ctx():
        for x_b, mags_b, errs_b, mask_b, z_b, zw_b in loader:
            x_b    = x_b.to(device)
            mags_b = mags_b.to(device)
            errs_b = errs_b.to(device)
            mask_b = mask_b.to(device)
            z_b    = z_b.to(device)
            zw_b   = zw_b.to(device)

            if is_train:
                opt.zero_grad()

            losses = model.loss(
                x_b, mags_b, errs_b, mask_b, z_b,
                lam_z=lam_z, lam_r=lam_r, beta=beta,
                sigma_floor=sigma_floor,
                z_weights=zw_b,
                use_nll_z=use_nll_z,
            )

            if is_train:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 2.0)
                opt.step()

            bs = len(x_b)
            for k in totals:
                totals[k] += losses[k].item() * bs
            n += bs

    return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name",    default=MODEL_NAME)
    parser.add_argument("--train-hdf5",    default=TRAIN_HDF5)
    parser.add_argument("--test-hdf5",     default=TEST_HDF5)
    parser.add_argument("--val-frac",      type=float, default=VAL_FRAC)
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
    parser.add_argument("--warmup-epochs",      type=int,   default=WARMUP_EPOCHS)
    parser.add_argument("--sigma-z-warmup",     type=int,   default=SIGMA_Z_WARMUP,
                        help="Phase-2 epochs before enabling z NLL and σ_z calibration")
    parser.add_argument("--phase2-trunk-lr-frac", type=float, default=PHASE2_TRUNK_LR_FRAC,
                        help="LR multiplier for trunk/z_head at Phase 2 start")
    parser.add_argument("--phase2-vae-lr-frac",   type=float, default=PHASE2_VAE_LR_FRAC,
                        help="LR multiplier for vae_head at Phase 2 start")
    parser.add_argument("--z-weight-cap",       type=float, default=Z_WEIGHT_CAP,
                        help="Cap z-weights at this multiple of mean (0 = no cap)")
    parser.add_argument("--init-from",          default=None,
                        help="Path to a checkpoint to partial-init from (shape-matched layers)")
    parser.add_argument("--device",        default=DEVICE)
    parser.add_argument("--no-colors",  action="store_true",
                        help="Use raw magnitude features instead of colours")
    parser.add_argument("--euclid",     action="store_true",
                        help="Include Euclid bands (default: LSST-only)")
    parser.add_argument("--use-gaap",   action="store_true",
                        help="Use GAAP 1.0-arcsec aperture mags for LSST bands instead of cModel")
    args = parser.parse_args()

    use_colors = not args.no_colors
    use_euclid = args.euclid
    use_gaap   = args.use_gaap

    out_dir = os.path.join(OUT_BASE, args.model_name)
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
        data_source    = "hdf5_presplit",
        train_hdf5     = args.train_hdf5,
        test_hdf5      = args.test_hdf5,
        val_frac       = args.val_frac,
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
        sigma_z_warmup      = args.sigma_z_warmup,
        phase2_trunk_lr_frac = args.phase2_trunk_lr_frac,
        phase2_vae_lr_frac  = args.phase2_vae_lr_frac,
        z_weight_cap        = args.z_weight_cap,
        init_from           = args.init_from,
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

    )
    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config → {cfg_path}")

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"\nLoading train HDF5: {args.train_hdf5}")
    tr_val_df = load_hdf5(args.train_hdf5)
    tr_df, val_df = split_train_val(tr_val_df, args.val_frac, SEED)

    print(f"Loading test HDF5:  {args.test_hdf5}")
    te_df = load_hdf5(args.test_hdf5)

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

    zw = compute_z_weights(z_tr)
    if args.z_weight_cap > 0:
        zw = np.clip(zw, None, args.z_weight_cap)
        zw = (zw / zw.mean()).astype(np.float32)
    print(f"z-weights (train): min={zw.min():.3f}  max={zw.max():.3f}  "
          f"mean={zw.mean():.3f}  (cap={args.z_weight_cap:.0f}x)")

    tr_loader  = make_dataloader(X_tr,  mags_tr,  errs_tr,  mask_tr,  z_tr,
                                 zw, args.batch, shuffle=True)
    val_loader = make_dataloader(X_val, mags_val, errs_val, mask_val, z_val,
                                 np.ones(len(z_val), dtype=np.float32), args.batch, shuffle=False)
    te_loader  = make_dataloader(X_te,  mags_te,  errs_te,  mask_te,  z_te,
                                 np.ones(len(z_te),  dtype=np.float32), args.batch, shuffle=False)
    print(f"Encoder input dim: {X_tr.shape[1]}")

    # ── Model ─────────────────────────────────────────────────────────────
    from photoz_mlpvae.model.photoz_mlpvae import PhotozMLPVAE

    model = PhotozMLPVAE(SPECULATOR_DIR, FILTER_DIR,
                         encoder_n_in=X_tr.shape[1]).to(args.device)
    n_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Encoder trainable parameters: {n_enc:,}")

    if args.init_from:
        ckpt     = torch.load(args.init_from, map_location=args.device, weights_only=False)
        src      = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        cur      = model.state_dict()
        matched  = {k: v for k, v in src.items()
                    if k in cur and v.shape == cur[k].shape}
        skipped  = set(src) - set(matched)
        cur.update(matched)
        model.load_state_dict(cur)
        print(f"Warm-start: loaded {len(matched)}/{len(cur)} keys from {args.init_from}")
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
    best_nmad_avg = float("inf")   # rolling-average σ_NMAD criterion
    nmad_window   = []             # recent σ_NMAD values for rolling average
    patience_ctr  = 0
    patience      = args.patience if args.patience is not None else args.epochs // 5
    phase2_started = False

    history = {k: [] for k in [
        "tr_total", "tr_z_sup", "tr_recon", "tr_kl", "tr_log_sigma_z",
        "val_total", "val_z_sup", "val_recon", "val_kl", "val_log_sigma_z",
        "val_sigma_nmad", "beta",
    ]}

    print(f"Training for up to {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):

        # ── Phase transition ──────────────────────────────────────────────
        if epoch == args.warmup_epochs + 1 and not phase2_started:
            print(f"\n── Phase 2: joint fine-tuning "
                  f"(epoch {epoch}, vae_head unfrozen, "
                  f"trunk lr ×{args.phase2_trunk_lr_frac}, "
                  f"vae lr ×{args.phase2_vae_lr_frac}) ──\n")
            model.encoder.unfreeze_vae_head()
            # Scale down existing trunk/z_head/sigma_z_head LR (preserve Adam momentum).
            for pg in opt.param_groups:
                pg["lr"] *= args.phase2_trunk_lr_frac
            # Add vae_head as new param group with higher LR (learns from scratch).
            opt.add_param_group({
                "params":       list(model.encoder.vae_head.parameters()),
                "lr":           args.lr * args.phase2_vae_lr_frac,
                "weight_decay": 1e-5,
            })
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - epoch + 1, eta_min=1e-6)
            patience_ctr   = 0   # fresh patience budget for Phase 2
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

        # λ_z drops at Phase 2 start to give VAE gradient budget,
        # then restores linearly after β has stabilized.
        if epoch <= args.warmup_epochs:
            lam_z_eff = args.lam_z
        elif kl_epoch <= BETA_EPOCHS:
            lam_z_eff = args.lam_z_min
        elif kl_epoch <= BETA_EPOCHS + args.lam_z_restore:
            frac      = (kl_epoch - BETA_EPOCHS) / max(args.lam_z_restore, 1)
            lam_z_eff = args.lam_z_min + (args.lam_z - args.lam_z_min) * frac
        else:
            lam_z_eff = args.lam_z

        # Delay σ_z NLL until VAE has warmed up (mu_15 is informative).
        use_nll_z = (epoch > args.warmup_epochs + args.sigma_z_warmup)

        t0 = time.time()

        tr_loss  = run_epoch(model, tr_loader,  args.device, opt=opt,
                             lam_z=lam_z_eff, lam_r=lam_r_eff, beta=beta,
                             sigma_floor=args.sigma_floor,
                             use_nll_z=use_nll_z)
        val_loss = run_epoch(model, val_loader, args.device, opt=None,
                             lam_z=lam_z_eff, lam_r=lam_r_eff, beta=beta,
                             sigma_floor=args.sigma_floor,
                             use_nll_z=use_nll_z)

        model.eval()
        z_pred_val = []
        with torch.no_grad():
            for x_b, *_ in val_loader:
                _, z_mu, _ = model.predict_z(x_b.to(args.device), n_samples=1)
                z_pred_val.append(z_mu.cpu().numpy())
        z_pred_val = np.concatenate(z_pred_val)
        nmad = sigma_nmad(z_pred_val, z_val)

        sched.step()
        dt = time.time() - t0

        for k in ("total", "z_sup", "recon", "kl", "log_sigma_z"):
            history[f"tr_{k}"].append(tr_loss[k])
            history[f"val_{k}"].append(val_loss[k])
        history["val_sigma_nmad"].append(nmad)
        history["beta"].append(beta)

        # Rolling-average σ_NMAD: smooths per-epoch noise from small val set (677 gal).
        nmad_window.append(nmad)
        if len(nmad_window) > NMAD_AVG_WINDOW:
            nmad_window.pop(0)
        nmad_avg = float(np.mean(nmad_window))

        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"tr={tr_loss['total']:.4f}  val={val_loss['total']:.4f}  "
              f"(z={val_loss['z_sup']:.4f} r={val_loss['recon']:.4f} "
              f"kl={val_loss['kl']:.4f})  "
              f"σ_NMAD={nmad:.4f}(avg={nmad_avg:.4f})  σ_z={val_loss['log_sigma_z']:.3f}  "
              f"β={beta:.3f}  λ_z={lam_z_eff:.1f}  lam_r={lam_r_eff:.2f}  t={dt:.1f}s")

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

    # ── Save predictions ──────────────────────────────────────────────────
    pred_path = os.path.join(out_dir, "test_predictions.csv")
    pd.DataFrame({
        "z_true":  z_te,
        "z_pred":  z_pred_te,
        "z_std":   z_std_te,
        "delta_z": (z_pred_te - z_te) / (1.0 + z_te),
    }).to_csv(pred_path, index=False)
    print(f"  Predictions → {pred_path}")

    # ── Failure-case diagnostics ──────────────────────────────────────────
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

    # ── Figures ───────────────────────────────────────────────────────────
    plot_scatter(z_te, z_pred_te,
                 os.path.join(out_dir, "test_scatter.png"),
                 title=f"{args.model_name}  test  σ_NMAD={nmad_te:.3f}")
    plot_redshift_hist(z_te, z_pred_te,
                       os.path.join(out_dir, "test_zdist.png"))
    plot_metrics_vs_zpred(z_te, z_pred_te,
                          os.path.join(out_dir, "test_metrics_vs_z.png"))
    events = [
        (args.warmup_epochs,
         "Phase 2: VAE unfrozen, λ_z drops, λ_r & β begin"),
        (args.warmup_epochs + args.sigma_z_warmup,
         "σ_z NLL begins"),
        (args.warmup_epochs + RECON_RAMP,
         "λ_r & β fully on, λ_z restoring"),
        (args.warmup_epochs + BETA_EPOCHS + args.lam_z_restore,
         "λ_z restored"),
    ]
    plot_curves(history, os.path.join(out_dir, "train_curves.png"), events=events)
    plot_metrics_paper_style(z_te, z_pred_te, args.model_name,
                             os.path.join(out_dir, "test_metrics_paper_style.png"))
    plot_pit(z_te, z_pred_te, z_samples=z_samples_te,
             save_path=os.path.join(out_dir, "test_pit.png"),
             title=args.model_name)

    # ── Paper-style figures (density scatter + binned metrics) ────────────
    plot_scatter_density(z_te, z_pred_te,
                         os.path.join(out_dir, "dp1_v4_test_scatter.png"))
    plot_metrics_binned_3sig(z_te, dz_te,
                              bins=np.arange(0.0, 3.2, 0.2),
                              xlabel=r"$z_{\rm spec}$",
                              save_path=os.path.join(out_dir, "dp1_v4_test_metrics_vs_z.png"))
    _ref_mag_col = GAAP_REF_MAG if use_gaap else REF_MAG
    plot_metrics_binned_3sig(te_df[_ref_mag_col].values, dz_te,
                              bins=np.arange(18.0, 25.5, 0.5),
                              xlabel="Magnitude",
                              save_path=os.path.join(out_dir, "dp1_v4_test_metrics_vs_mag.png"))

    print(f"\nDone. Best avg σ_NMAD={best_nmad_avg:.4f}")
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Training finished")
    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
