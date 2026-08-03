"""
train.py
========
Training script for PhotozMLPVAE: 2-stage MLP z-head + 15-param VAE.

Architecture recap
------------------
  Shared ResBlock trunk → split heads:
    z_head       : MLP redshift prediction  (sigmoid × 6.0)
    sigma_z_head : SED-informed log aleatoric z uncertainty
                   Input: [trunk_features | mu_15.detach()]
                   mu_15 weights zero-initialized → phase 1 = pure trunk-based σ_z;
                   phase 2 activates SED pathway as vae_head becomes informative.
    vae_head     : 15-param VAE posterior (μ_15, log σ²_15)

  Frozen decoder: theta_full → SpeculatorInoueIGM → FilterConv → 10 AB mags
  z_pred is detached before the decoder → reconstruction trains theta_15 only;
  z_pred receives gradients exclusively from the supervised z loss.

Training phases
---------------
  Phase 1 (warmup_epochs): lam_r=0, beta=0, vae_head frozen, use_nll_z=False
      Trunk + z_head + sigma_z_head learn from z-supervised MSE.
  Phase 2 (joint): vae_head added to optimizer via add_param_group() (preserves
      trunk/z_head Adam momentum); LR scaled by phase2_lr_frac for all groups;
      lam_z dropped IMMEDIATELY to lam_z_min (not ramped) so the VAE competes;
      lam_r ramped, beta annealed.
  Phase 2a (first sigma_z_warmup epochs): use_nll_z=False; sigma_z_head frozen
      while mu_15 is still uninformative. z_head trains with plain MSE at lam_z_min.
  Phase 2b (after sigma_z_warmup): use_nll_z=True; sigma_z_head calibrates using
      stabilized mu_15 (SED posterior) for better z uncertainty.

Warm-start
----------
  Use --init-from path/to/best.pt to load compatible weights from a prior
  checkpoint (e.g. mlpvae_synth_v6/best.pt).  Only keys whose shapes match
  the new architecture are loaded; mismatches (e.g. sigma_z_head) are skipped.

Feature modes (USE_COLORS)
---------------------------
  True  : colour-based 40-dim input [9 colours | 9 colour-errs | i_ref | i_ref_err
                                      | 20 missingness flags]
  False : magnitude-based 40-dim input [10 mags | 10 errs | 20 missingness flags]

  Both modes produce n_in=40; the encoder architecture is identical.

Outputs  (photoz_mlpvae/trained/<MODEL_NAME>/)
---------------------------------------------
  best.pt                    model checkpoint (best val σ_NMAD)
  config.yaml                all hyperparameters
  train.log                  stdout mirror
  train_curves.png           loss curves
  test_scatter.png
  test_predictions.csv       z_true, z_pred, z_std, delta_z
  test_metrics_paper_style.png
  test_pit.png

Usage
-----
    conda run -n WL_ML_Challenge python photoz_mlpvae/scripts/train.py
    conda run -n WL_ML_Challenge python photoz_mlpvae/scripts/train.py \\
        --model-name mlpvae_v1_colors
    conda run -n WL_ML_Challenge python photoz_mlpvae/scripts/train.py \\
        --no-colors  # use raw magnitude features instead
"""

import os, sys, time, argparse, warnings, yaml, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)
from obs_catalog.dataloader import (
    load_catalog, build_features, compute_z_weights, make_dataloader,
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

MODEL_NAME     = "mlpvae_v1"

CATALOG        = os.path.join(_BASE, "obs_catalog", "data",
                              "dp1_ecdfs_crossmatched_catalog.parquet")
SPECULATOR_DIR = os.path.join(_BASE, "speculator", "trained", "Inoue_IGM")
FILTER_DIR     = os.path.join(_BASE, "obs_catalog", "filters")
OUT_BASE       = os.path.join(_BASE, "photoz_mlpvae", "trained")

USE_COLORS     = True    # default: colour-based features (40-dim)

MAX_EPOCHS     = 500
BATCH_SIZE     = 256
LR             = 1e-4
PATIENCE       = 100        # early stopping on val total loss per normalization window
WARMUP_EPOCHS  = 50         # phase 1: z-supervised only, vae_head frozen
RECON_RAMP     = 50         # epochs to ramp lam_r from 0 → full after warmup
BETA_EPOCHS    = 50         # KL annealing: β rises linearly over this many epochs
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

LAM_Z          = 20.0
LAM_R          = 0.1
BETA_MAX       = 0.1
SIGMA_FLOOR    = 0.3
N_ZBINS_WEIGHT = 25
PHASE2_TRUNK_LR_FRAC = 0.02  # LR multiplier for trunk/z_head/sigma_z_head at phase 2 start
PHASE2_VAE_LR_FRAC   = 0.3   # LR multiplier for vae_head at phase 2 start

# λ_z phase-2 schedule
# At phase 2 start, λ_z drops IMMEDIATELY to lam_z_min so the VAE can compete with
# the z-supervised loss. After KL stabilizes (beta_epochs), λ_z restores linearly.
# Optimizer momentum is never reset — phases are purely loss-weight changes.
LAM_Z_MIN      = 2.0       # floor for λ_z during VAE KL ramp (phase 2a)
LAM_Z_RESTORE  = 50        # epochs to ramp λ_z back to full after KL stabilizes (phase 2b)

# sigma_z suppression at phase 2 start
# NLL calibration (use_nll_z=True) is delayed for the first SIGMA_Z_WARMUP epochs of
# phase 2 so sigma_z_head doesn't train on noisy z predictions / uninformative mu_15.
# During this window z_head trains with plain MSE at lam_z_min.
SIGMA_Z_WARMUP = 30        # phase-2 epochs before enabling z NLL and sigma_z_head

TRAIN_PLAN      = "total loss, whenever lambdas change, reset best loss and patience."  # "always NAMD" or "total loss, whenever lambdas change, reset best loss."

# ─────────────────────────────────────────────────────────────────────────────
# Catalog columns (shared by both feature modes)
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
N_BANDS    = len(MAG_COLS)   # 10

# Adjacent-band colour pairs (same as photoz_mlp)
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
# Reference magnitude (breaks absolute-flux degeneracy in colour mode)
REF_MAG = "i_cModelMag"
REF_ERR = "i_cModelMagErr"
N_COLORS = len(COLOR_PAIRS)   # 9


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def sigma_nmad(z_pred, z_true):
    dz  = (z_pred - z_true) / (1.0 + z_true)
    med = np.nanmedian(dz)
    return 1.4826 * np.nanmedian(np.abs(dz - med))


def outlier_frac(z_pred, z_true, threshold=0.15):
    dz = np.abs((z_pred - z_true) / (1.0 + z_true))
    return float(np.nanmean(dz > threshold))


def rms_dz(z_pred, z_true):
    dz = (z_pred - z_true) / (1.0 + z_true)
    return float(np.sqrt(np.nanmean(dz[np.isfinite(dz)] ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# Plots
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
    """One train or validation epoch. Returns dict of mean losses."""
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
    parser.add_argument("--model-name",      default=MODEL_NAME)
    parser.add_argument("--epochs",          type=int,   default=MAX_EPOCHS)
    parser.add_argument("--batch",           type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=LR)
    parser.add_argument("--lam-z",           type=float, default=LAM_Z)
    parser.add_argument("--lam-r",           type=float, default=LAM_R)
    parser.add_argument("--sigma-floor",     type=float, default=SIGMA_FLOOR)
    parser.add_argument("--warmup-epochs",   type=int,   default=WARMUP_EPOCHS)
    parser.add_argument("--recon-ramp",      type=int,   default=RECON_RAMP,
                        help="Epochs to ramp lam_r from 0 to full after warmup")
    parser.add_argument("--beta-max",        type=float, default=BETA_MAX,
                        help="Maximum KL weight (beta)")
    parser.add_argument("--beta-epochs",     type=int,   default=BETA_EPOCHS,
                        help="Epochs over which beta is annealed 0 → beta_max")
    parser.add_argument("--phase2-trunk-lr-frac", type=float, default=PHASE2_TRUNK_LR_FRAC,
                        help="LR multiplier for trunk/z_head/sigma_z_head at phase 2 start")
    parser.add_argument("--phase2-vae-lr-frac",   type=float, default=PHASE2_VAE_LR_FRAC,
                        help="LR multiplier for vae_head at phase 2 start")
    parser.add_argument("--lam-z-min",       type=float, default=LAM_Z_MIN,
                        help="Floor for λ_z during phase 2a KL ramp "
                             "(dropped from lam_z → lam_z_min over beta_epochs, "
                             "then restored over lam_z_restore epochs)")
    parser.add_argument("--lam-z-restore",   type=int,   default=LAM_Z_RESTORE,
                        help="Epochs to ramp λ_z from lam_z_min back to lam_z "
                             "after beta_epochs have elapsed")
    parser.add_argument("--sigma-z-warmup", type=int,   default=SIGMA_Z_WARMUP,
                        help="Phase-2 epochs before enabling z NLL / sigma_z_head "
                             "calibration (use_nll_z stays False during this window)")
    parser.add_argument("--init-from",       type=str,   default=None,
                        help="Path to best.pt for warm-start (e.g. mlpvae_synth_v6/best.pt)")
    parser.add_argument("--device",          default=DEVICE)
    parser.add_argument("--no-colors",  action="store_true",
                        help="Use raw magnitude features instead of colours")
    parser.add_argument("--no-euclid", action="store_true",
                        help="Use LSST-only photometry (6 bands)")
    args = parser.parse_args()

    use_colors = not args.no_colors
    use_euclid = not args.no_euclid

    out_dir = os.path.join(OUT_BASE, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────
    log_path    = os.path.join(out_dir, "train.log")
    sys.stdout  = _Tee(sys.__stdout__, log_path)
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Training started")
    print(f"Model: {args.model_name}  |  device={args.device}  "
          f"|  use_colors={use_colors}  |  use_euclid={use_euclid}")
    print(f"Outputs → {out_dir}/")

    # ── Config ────────────────────────────────────────────────────────────
    config = dict(
        model_name      = args.model_name,
        data_source     = "real_obs_catalog",
        use_colors      = use_colors,
        use_euclid      = use_euclid,
        epochs          = args.epochs,
        batch           = args.batch,
        lr              = args.lr,
        lam_z           = args.lam_z,
        lam_r           = args.lam_r,
        sigma_floor     = args.sigma_floor,
        warmup_epochs   = args.warmup_epochs,
        recon_ramp      = args.recon_ramp,
        beta_max        = args.beta_max,
        beta_epochs     = args.beta_epochs,
        phase2_trunk_lr_frac = args.phase2_trunk_lr_frac,
        phase2_vae_lr_frac   = args.phase2_vae_lr_frac,
        lam_z_min       = args.lam_z_min,
        lam_z_restore   = args.lam_z_restore,
        sigma_z_warmup  = args.sigma_z_warmup,
        init_from       = args.init_from,
        patience        = PATIENCE,
        seed            = SEED,
        device          = args.device,
        n_zbins_weight  = N_ZBINS_WEIGHT,
        catalog         = CATALOG,
        speculator_dir  = SPECULATOR_DIR,
        filter_dir      = FILTER_DIR,
        training_scheme  = TRAIN_PLAN,
    )
    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Config → {cfg_path}")

    # ── Data ─────────────────────────────────────────────────────────────
    tr_df, val_df, te_df = load_catalog(CATALOG)

    (X_tr,  mags_tr,  errs_tr,  mask_tr,  z_tr,  scaler, col_med) = \
        build_features(tr_df,  use_colors=use_colors, use_euclid=use_euclid, fit=True)
    (X_val, mags_val, errs_val, mask_val, z_val, _, _) = \
        build_features(val_df, use_colors=use_colors, use_euclid=use_euclid,
                       scaler=scaler, col_medians=col_med)
    (X_te,  mags_te,  errs_te,  mask_te,  z_te,  _, _) = \
        build_features(te_df,  use_colors=use_colors, use_euclid=use_euclid,
                       scaler=scaler, col_medians=col_med)

    tr_loader  = make_dataloader(X_tr,  mags_tr,  errs_tr,  mask_tr,  z_tr,
                                 compute_z_weights(z_tr), args.batch, shuffle=True)
    val_loader = make_dataloader(X_val, mags_val, errs_val, mask_val, z_val,
                                 np.ones(len(z_val), dtype=np.float32), args.batch, shuffle=False)
    te_loader  = make_dataloader(X_te,  mags_te,  errs_te,  mask_te,  z_te,
                                 np.ones(len(z_te),  dtype=np.float32), args.batch, shuffle=False)

    zw = compute_z_weights(z_tr)
    print(f"z-weights (train): min={zw.min():.3f}  max={zw.max():.3f}  "
          f"mean={zw.mean():.3f}  (high-z upweight={zw.max():.1f}x)")

    # ── Model ─────────────────────────────────────────────────────────────
    from photoz_mlpvae.model.photoz_mlpvae_old import PhotozMLPVAE

    torch.manual_seed(SEED)
    model = PhotozMLPVAE(SPECULATOR_DIR, FILTER_DIR,
                         encoder_n_in=X_tr.shape[1]).to(args.device)
    n_enc = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"Encoder trainable parameters: {n_enc:,}")

    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=args.device, weights_only=False)
        old_state = ckpt["model_state"]
        new_state = model.state_dict()
        compatible = {k: v for k, v in old_state.items()
                      if k in new_state and new_state[k].shape == v.shape}
        new_state.update(compatible)
        model.load_state_dict(new_state)
        skipped = set(old_state) - set(compatible)
        print(f"Warm-start: loaded {len(compatible)}/{len(new_state)} keys "
              f"from {args.init_from}")
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
    best_total   = float("inf")
    patience_ctr = 0

    # Epochs where loss normalization changes — best_total resets to inf so
    # patience is judged within a window of comparable loss magnitudes.
    #   epoch W+1      : phase 2 start (lam_z drops, recon/KL terms appear)
    #   epoch W+S+1    : NLL activates (z_sup switches MSE → NLL)
    #   epoch W+B+1    : lam_z restore begins (lam_z ramps 2 → 20)
    _w = args.warmup_epochs
    norm_reset_epochs = {
        _w + 1,
        _w + args.sigma_z_warmup + 1,
        _w + args.beta_epochs + 1,
    }

    phase2_started = False

    history = {k: [] for k in [
        "tr_total", "tr_z_sup", "tr_recon", "tr_kl", "tr_log_sigma_z",
        "val_total", "val_z_sup", "val_recon", "val_kl", "val_log_sigma_z",
        "val_sigma_nmad", "beta",
    ]}

    print(f"Training for up to {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):

        # ── Phase transition: unfreeze vae_head at end of warmup ──────────
        if epoch == args.warmup_epochs + 1 and not phase2_started:
            print(f"\n── Phase 2: joint fine-tuning "
                  f"(epoch {epoch}, vae_head unfrozen, "
                  f"trunk lr ×{args.phase2_trunk_lr_frac}, "
                  f"vae lr ×{args.phase2_vae_lr_frac}) ──\n")
            model.encoder.unfreeze_vae_head()
            # Trunk/z_head/sigma_z_head: very low LR to protect z-head representations.
            # vae_head: higher LR so it can learn SED reconstruction from scratch.
            # Splicing into the existing optimizer preserves Adam momentum for trunk groups.
            for pg in opt.param_groups:
                pg["lr"] = args.lr * args.phase2_trunk_lr_frac
            opt.add_param_group({
                "params":       list(model.encoder.vae_head.parameters()),
                "lr":           args.lr * args.phase2_vae_lr_frac,
                "weight_decay": 1e-5,
            })
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - epoch + 1, eta_min=1e-6)
            phase2_started = True

        # ── Loss weights for this epoch ───────────────────────────────────
        kl_epoch = max(0, epoch - args.warmup_epochs)

        if epoch <= args.warmup_epochs:
            lam_r_eff = 0.0
        elif epoch <= args.warmup_epochs + args.recon_ramp:
            lam_r_eff = args.lam_r * (epoch - args.warmup_epochs) / args.recon_ramp
        else:
            lam_r_eff = args.lam_r

        beta = min(args.beta_max,
                   args.beta_max * kl_epoch / max(args.beta_epochs - 1, 1))

        # λ_z phase-2 schedule:
        #   Phase 1  (epoch ≤ warmup)                           : lam_z (full)
        #   Phase 2a (kl_epoch ≤ beta_epochs)                  : lam_z_min (immediate drop)
        #     z NLL dominates total loss when β is small and VAE is untrained.
        #     Dropping λ_z immediately gives the VAE gradient budget from epoch 1 of phase 2.
        #   Phase 2b (kl_epoch in [beta_epochs, beta_epochs+lam_z_restore]):
        #     lam_z_min → lam_z (linear restore, KL has now stabilized)
        #   Phase 2c (kl_epoch > beta_epochs + lam_z_restore): lam_z (full)
        if epoch <= args.warmup_epochs:
            lam_z_eff = args.lam_z
        elif kl_epoch <= args.beta_epochs:
            lam_z_eff = args.lam_z_min   # immediate drop; restored after KL stabilizes
        elif kl_epoch <= args.beta_epochs + args.lam_z_restore:
            frac      = (kl_epoch - args.beta_epochs) / max(args.lam_z_restore, 1)
            lam_z_eff = args.lam_z_min + (args.lam_z - args.lam_z_min) * frac
        else:
            lam_z_eff = args.lam_z

        # use_nll_z delayed by sigma_z_warmup: sigma_z_head waits until VAE has
        # established informative mu_15 before calibrating on z predictions.
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

        # σ_NMAD on validation set
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

        print(f"Ep {epoch:3d}/{args.epochs}  "
              f"tr={tr_loss['total']:.4f}  val={val_loss['total']:.4f}  "
              f"(z={val_loss['z_sup']:.4f} r={val_loss['recon']:.4f} "
              f"kl={val_loss['kl']:.4f})  "
              f"σ_NMAD={nmad:.4f}  σ_z={val_loss['log_sigma_z']:.3f}  "
              f"β={beta:.3f}  λ_z={lam_z_eff:.1f}  lam_r={lam_r_eff:.2f}  t={dt:.1f}s")

        if np.isnan(tr_loss["total"]) or np.isnan(val_loss["total"]):
            print(f"  NaN detected at epoch {epoch} — stopping.")
            break

        # ── Reset patience at loss-normalization boundaries ───────────────
        if epoch in norm_reset_epochs:
            best_total   = float("inf")
            patience_ctr = 0
            print(f"  ↺ Loss normalization reset at epoch {epoch} "
                  f"(λ_z={lam_z_eff:.1f}  use_nll_z={use_nll_z})")

        # ── Val total loss — checkpoint + patience ────────────────────────
        val_total = val_loss["total"]
        if val_total < best_total and not np.isnan(val_total):
            best_total   = val_total
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            patience_ctr = 0
            model.save(ckpt_path, scaler=scaler, col_medians=col_med)
            print(f"  ✓ New best val_total={val_total:.4f}  σ_NMAD={nmad:.4f}  "
                  f"saved → {ckpt_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping (val total stalled for {PATIENCE} epochs).")
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

    # ── Corner + SED plots for 4 failure-case galaxies ───────────────────
    # Select: mild over, severe over, mild under, severe under
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

    sel_labels   = [lbl for _, _, lbl in _targets]
    sel_z_true   = z_te[sel_idx]
    sel_z_pred   = z_pred_te[sel_idx]
    sel_X        = X_te[sel_idx]
    sel_mags     = mags_te[sel_idx]
    sel_errs     = errs_te[sel_idx]
    sel_mask     = mask_te[sel_idx]

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
    _w = args.warmup_epochs
    events = [
        (_w,
         "Phase 2: VAE unfrozen, λ_z drops, λ_r & β begin"),
        (_w + args.sigma_z_warmup,
         "σ_z NLL calibration begins"),
        (_w + max(args.recon_ramp, args.beta_epochs),
         "λ_r & β fully on, λ_z restoring"),
        (_w + args.beta_epochs + args.lam_z_restore,
         "λ_z fully restored"),
    ]
    plot_curves(history, os.path.join(out_dir, "train_curves.png"), events=events)
    plot_metrics_paper_style(z_te, z_pred_te, args.model_name,
                             os.path.join(out_dir, "test_metrics_paper_style.png"))
    plot_pit(z_te, z_pred_te, z_samples=z_samples_te,
             save_path=os.path.join(out_dir, "test_pit.png"),
             title=args.model_name)

    # ── DP1 paper-style figures (density scatter + binned metrics) ────────
    plot_scatter_density(z_te, z_pred_te,
                         os.path.join(out_dir, "dp1_v4_test_scatter.png"))
    plot_metrics_binned_3sig(z_te, dz_te,
                              bins=np.arange(0.0, 3.2, 0.2),
                              xlabel=r"$z_{\rm spec}$",
                              save_path=os.path.join(out_dir, "dp1_v4_test_metrics_vs_z.png"))
    plot_metrics_binned_3sig(te_df[REF_MAG].values, dz_te,
                              bins=np.arange(18.0, 25.5, 0.5),
                              xlabel="Magnitude",
                              save_path=os.path.join(out_dir, "dp1_v4_test_metrics_vs_mag.png"))

    print(f"\nDone. Best val total={best_total:.4f}")
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] Training finished")
    if isinstance(sys.stdout, _Tee):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
