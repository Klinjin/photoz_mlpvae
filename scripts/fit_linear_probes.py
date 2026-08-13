"""
fit_linear_probes.py
=====================
Fit linear (ridge) regression probes per encoder layer, for a battery of
target quantities, to characterize how R^2 changes with model depth -- "at
which layer does redshift / SPS parameters / color become linearly
decodable, and how much is already there?"

Input: activations + targets produced by extract_activations.py (one or two
.npz files). Probes are ALWAYS fit on one split and scored on a separate,
held-out split -- an R^2 measured on the same rows a probe was fit on is
optimistic (more so for wide, correlated activation vectors like h0-h2's
512 dims), so this script never reports train-set R^2 as the headline
number.

Layers probed (see extract_activations.py's docstring for what each one
is): trunk x, h0, h1_mid, h1, h2_mid, h2; heads z_head_hidden, z_out
(= [z_pred, log_sigma_z], assembled here), mu_15; decoder log_spec,
m_ab_recon. z_head_hidden/z_out and mu_15 are PARALLEL branches off h2
(z-branch and SPS-branch), and the frozen decoder consumes BOTH branches'
outputs (theta_full = [z_pred, theta_15]).

Plots use a normalized model-depth x-axis with a per-question terminal:
  - redshift + SPS figures (HEAD ladder): depth 1 = the head outputs,
    where the model reads out (z_pred, sigma_z) / mu_15 -- the decoder is
    off the z inference path (it consumes the prediction) and is not shown.
  - input-property figure, i mag / g-r (DECODER ladder): depth 1 = the
    decoder's last layer m_ab_recon, with log_spec (reconstructed
    rest-frame spectrum) as the decoder's intermediate.
Each target is drawn as two lines that share the trunk and fork at h2:
solid = VAE branch, dashed = z branch. A layer that's all-NaN or absent in
a given extraction (e.g. z_head_hidden for a --model-version old
checkpoint, or the newer taps in a pre-2026-08-12 npz) is skipped
automatically.

Targets: z_true, log_sigma_z (redshift "mean" and "std"), i_ref magnitude,
g_minus_r color, and each of the 15 SPS parameters (read from theta_full's
columns 1-15, ordered per model/photoz_mlpvae.py's PARAM_NAMES_15 -- column
0, zred, is skipped here since it's just z_true's own model-side estimate,
already covered by z_true/log_sigma_z).

Usage
-----
    # Two separate extractions (recommended -- e.g. train catalog vs. test):
    ~/miniforge3/envs/WL_ML_Challenge/bin/python \\
        photoz_mlpvae/scripts/fit_linear_probes.py \\
        --train-npz photoz_mlpvae/trained/mlpvae_dp2_v1_lsst_gaap1p0/probe_activations_train.npz \\
        --test-npz  photoz_mlpvae/trained/mlpvae_dp2_v1_lsst_gaap1p0/probe_activations_test.npz \\
        --out-dir   photoz_mlpvae/trained/mlpvae_dp2_v1_lsst_gaap1p0/probes

    # Single extraction, split internally:
    ~/miniforge3/envs/WL_ML_Challenge/bin/python \\
        photoz_mlpvae/scripts/fit_linear_probes.py \\
        --npz photoz_mlpvae/trained/mlpvae_dp2_v1_lsst_gaap1p0/probe_activations_test.npz \\
        --test-frac 0.2 --out-dir /tmp/probes

Outputs (in --out-dir)
-----------------------
    probe_r2.csv             every (target, layer) R^2 + train/test row
                             counts + depth_heads/depth_decoder ladder
                             positions (+ sigma_nmad, z_true target only --
                             R^2 on z is catastrophic-outlier-dominated,
                             see fit_and_score)
    probe_r2_redshift.png    z + log sigma_z on the HEAD ladder
    probe_r2_photometry.png  i mag + g-r on the DECODER ladder
    probe_r2_sps_params.png  all 15 SPS params on one axes, HEAD ladder
                             (hue = physical group, lightness+marker = param)
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)
from photoz_mlpvae.model.photoz_mlpvae import PARAM_NAMES_15

# Layers to probe (union over both ladders below). Trunk now includes the
# mid-ResBlock activations h1_mid/h2_mid (block[:3] = Linear+BN+ReLU) for
# double depth resolution. z_out = [z_pred, log_sigma_z], the z head's
# final readout, is assembled in main() from arrays every extraction saves,
# so older npz files still work (missing layers are skipped everywhere).
TRUNK_LAYERS = ["x", "h0", "h1_mid", "h1", "h2_mid", "h2"]
LAYERS = TRUNK_LAYERS + ["z_head_hidden", "z_out", "mu_15",
                         "log_spec", "m_ab_recon"]

# Two depth ladders, one per question:
#  - HEAD ladder (redshift + SPS targets): the representation question ends
#    where the model reads out (z_pred, sigma_z) / mu_15, so depth 1 = the
#    head outputs. The decoder is off the z inference path (it CONSUMES the
#    prediction) and is deliberately not shown on these figures.
#  - DECODER ladder (input-property targets, i mag / g-r): what survives the
#    full round trip, so depth 1 = the decoder's last layer m_ab_recon, with
#    the reconstructed rest-frame spectrum log_spec as its intermediate.
#    Both branches converge into the decoder (theta_full = [z_pred, theta_15]).
HEAD_BRANCHES = [  # (path past h2, linestyle, legend label)
    (["mu_15"],                  "-",  "VAE branch (→ mu_15)"),
    (["z_head_hidden", "z_out"], "--", "z branch (→ hidden → z readout)"),
]
DECODER_BRANCHES = [
    (["mu_15", "log_spec", "m_ab_recon"],         "-",  "VAE branch (→ mu_15 → spectrum → m_AB)"),
    (["z_head_hidden", "log_spec", "m_ab_recon"], "--", "z branch (→ hidden → spectrum → m_AB)"),
]
XLABEL = "model depth"
_TICK_NAME = {"z_head_hidden": "z_hid", "m_ab_recon": "m_AB",
              "log_spec": "spec", "h1_mid": "h1'", "h2_mid": "h2'",
              "z_out": "z out"}


def build_ladder(branches, layers_present: list):
    """
    Depth map for one ladder, restricted to the layers actually present:
    trunk evenly spaced from 0, each branch's LAST layer at exactly 1,
    intermediate branch layers evenly spaced between trunk end and 1.
    Returns (trunk, branches, depths, ticks, ticklabels).
    """
    trunk = [l for l in TRUNK_LAYERS if l in layers_present]
    branches = [([l for l in path if l in layers_present], ls, lab)
                for path, ls, lab in branches]
    branches = [(p, ls, lab) for p, ls, lab in branches if p]
    n_trunk = len(trunk) - 1
    n_steps = n_trunk + max((len(p) for p, _, _ in branches), default=1)
    depths = {l: i / n_steps for i, l in enumerate(trunk)}
    for path, _, _ in branches:
        for j, l in enumerate(path):
            depths[l] = (n_trunk + (j + 1) * (n_steps - n_trunk) / len(path)) / n_steps
    ticks, labels = [], []
    for d in sorted({round(v, 6) for v in depths.values()}):
        at = sorted((l for l, v in depths.items() if round(v, 6) == d),
                    key=LAYERS.index)
        ticks.append(d)
        labels.append(f"{d:.2g}\n" + "/".join(_TICK_NAME.get(l, l) for l in at))
    return trunk, branches, depths, ticks, labels

PRIMARY_TARGETS = [
    ("z_true",      "z (redshift)"),
    ("log_sigma_z", "log σ_z (model uncertainty)"),
    ("i_ref",       "i magnitude"),
    ("g_minus_r",   "g−r color"),
]

# All 15 SPS parameters share one axes, so identity can't ride on 15 raw hues
# (indistinguishable, CVD-unsafe). Composite encoding instead: one hue per
# PHYSICAL group, lightness-stepped within the group (light -> dark in list
# order -- for the SFH ratios that's bin order, so the ramp is meaningful),
# plus a distinct marker per line within the group.
SPS_GROUPS = [  # (params, base hue)
    (["logmass", "logzsol"],                     "#57534e"),  # stellar (gray)
    (["dust2", "dust1_fraction", "dust_index"],  "#eb6834"),  # dust (orange)
    (["gas_logz", "igm_scale"],                  "#1baf7a"),  # gas/IGM (aqua)
    (["fagn", "agn_tau"],                        "#6a5acd"),  # AGN (violet)
    ([f"logsfr_ratios_{i}" for i in range(6)],   "#2a78d6"),  # SFH (blue)
]
_MARKERS = ["o", "s", "^", "D", "v", "P"]


def _shade(hex_color: str, f: float) -> str:
    """Blend a hex color toward white (f>0) or black (f<0), f in [-1, 1]."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    rgb = rgb + (255.0 - rgb) * f if f >= 0 else rgb * (1.0 + f)
    return "#" + "".join(f"{int(round(c)):02x}" for c in rgb)


# param -> (color, marker); lightness ramp spans the group, markers cycle.
SPS_STYLE = {
    p: (_shade(base, f), _MARKERS[i % len(_MARKERS)])
    for params, base in SPS_GROUPS
    for i, (p, f) in enumerate(zip(params, np.linspace(0.35, -0.25, len(params))
                                   if len(params) > 1 else [0.0]))
}

ALPHAS  = np.logspace(-3, 3, 13)
Y_LIMIT = (-0.1, 1.05)   # shared R^2 axis range across all plots


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def get_target_vector(data: dict, key: str) -> np.ndarray:
    """
    z_true/log_sigma_z/i_ref/g_minus_r are top-level 1-D arrays; SPS
    parameters are addressed as "sps:<name>" and read from theta_full's
    columns 1-15 (column 0 is zred).
    """
    if key.startswith("sps:"):
        idx = PARAM_NAMES_15.index(key[len("sps:"):])
        return data["theta_full"][:, 1 + idx]
    return data[key]


# ─────────────────────────────────────────────────────────────────────────────
# Probe fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_score(X_tr, y_tr, X_te, y_te, min_rel_std=1e-3,
                  dead_dim_rel_std=1e-6, clip_sigma=10.0, z_for_nmad=None):
    """
    Standardize activations on train (ridge penalizes raw coefficient
    magnitude, so unscaled dims would be regularized unevenly), fit
    RidgeCV (internal CV picks alpha), score R^2 on the held-out split.
    Rows with a NaN in X or y are dropped independently per split.

    Guards against a near-constant target: R^2's denominator is the target's
    own variance, so a target that's collapsed to near-constant (e.g. an SPS
    parameter the vae_head barely varies for out-of-distribution inputs --
    see PLAN.md's Failure 4b) makes R^2 numerically explode into meaningless
    values like -8 rather than reporting "no signal". Below `min_rel_std`
    (relative to |mean|, or absolute if mean~0), R^2 is reported as NaN
    instead.

    Guards against activation dims that are near-dead on TRAIN but active on
    TEST (post-ReLU units the training split never lights up): StandardScaler
    divides by the tiny train std, so on test those dims reach thousands of
    "train sigmas" and one small ridge coefficient explodes the prediction
    (h2 showed test/train std ratios up to 3e9, driving held-out R^2 to -26
    while a test-internal fit of the same cell scored +0.8). Two layers of
    defense: dims whose train std is below `dead_dim_rel_std` x the layer's
    median dim-std are dropped outright, and every scaled feature is clipped
    to +-`clip_sigma` train-sigmas on both splits.

    If `z_for_nmad` (test-split z_true) is given, also returns the photo-z
    robust scatter of the probe's predictions, sigma_NMAD = 1.4826 x
    median(|d - median(d)|), d = (pred - z)/(1 + z) -- R^2 on redshift is
    dominated by the rare catastrophic outliers (the model's own z_pred only
    reaches R^2 = 0.26 on dp1_v4 test despite sigma_NMAD = 0.03), so for the
    z target R^2 alone misranks layers.
    """
    tr_ok = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    te_ok = np.isfinite(X_te).all(axis=1) & np.isfinite(y_te)
    if tr_ok.sum() < 20 or te_ok.sum() < 20:
        return np.nan, np.nan, int(tr_ok.sum()), int(te_ok.sum())

    y_te_ok = y_te[te_ok]
    rel_std = np.std(y_te_ok) / max(np.abs(np.mean(y_te_ok)), 1e-8)
    if rel_std < min_rel_std:
        return np.nan, np.nan, int(tr_ok.sum()), int(te_ok.sum())

    dim_std = X_tr[tr_ok].std(axis=0)
    keep = dim_std > dead_dim_rel_std * max(np.median(dim_std), 1e-12)
    X_tr, X_te = X_tr[:, keep], X_te[:, keep]

    scaler = StandardScaler().fit(X_tr[tr_ok])
    Xtr_s  = np.clip(scaler.transform(X_tr[tr_ok]), -clip_sigma, clip_sigma)
    Xte_s  = np.clip(scaler.transform(X_te[te_ok]), -clip_sigma, clip_sigma)

    reg = RidgeCV(alphas=ALPHAS)
    reg.fit(Xtr_s, y_tr[tr_ok])
    pred = reg.predict(Xte_s)
    r2 = r2_score(y_te_ok, pred)

    sigma_nmad = np.nan
    if z_for_nmad is not None:
        z = z_for_nmad[te_ok]
        d = (pred - z) / (1.0 + z)
        sigma_nmad = float(1.4826 * np.median(np.abs(d - np.median(d))))
    return float(r2), sigma_nmad, int(tr_ok.sum()), int(te_ok.sum())


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _r2_series(df: pd.DataFrame, target: str, layers: list) -> list:
    sub = df[df["target"] == target].set_index("layer")
    return [sub.loc[l, "r2"] for l in layers]


def _plot_target(ax, df: pd.DataFrame, target: str, ladder, color: str,
                 lw: float = 2.0, marker: str = "o"):
    """
    One target = up to two lines sharing the trunk and forking at h2, one
    per branch of the given ladder (solid = VAE branch, dashed = z branch).
    The exact-trunk overlap makes the shared part read as a single line
    that splits at the fork.
    """
    trunk, branches, depths, _, _ = ladder
    for branch_path, ls, _ in branches:
        path = trunk + branch_path
        ax.plot([depths[l] for l in path], _r2_series(df, target, path),
                marker=marker, lw=lw, ms=4 * lw / 1.5, color=color, ls=ls)


def _branch_handles(ladder):
    _, branches, _, _, _ = ladder
    return [Line2D([0], [0], color="black", ls=ls, lw=1.5, label=lab)
            for _, ls, lab in branches]


def _set_depth_axis(ax, ladder, fontsize=None):
    _, _, _, ticks, labels = ladder
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(*Y_LIMIT)
    ax.grid(alpha=0.25)


def _plot_target_pair(df, layers, branches, targets_colors, title, out_path):
    """Shared body of the redshift / photometry two-target figures."""
    ladder = build_ladder(branches, layers)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    handles = []
    for key, label, color in targets_colors:
        if key not in df["target"].values:
            continue
        _plot_target(ax, df, key, ladder, color)
        handles.append(Line2D([0], [0], color=color, lw=2, marker="o",
                              ms=4, label=label))
    _set_depth_axis(ax, ladder, fontsize=9)
    ax.set_ylabel(r"Ridge probe $R^2$ (held-out)")
    ax.set_xlabel(XLABEL)
    ax.set_title(title)
    leg1 = ax.legend(handles=handles, fontsize=9, frameon=False,
                     loc="lower right")
    ax.add_artist(leg1)
    ax.legend(handles=_branch_handles(ladder), fontsize=8, frameon=False,
              loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_redshift(df: pd.DataFrame, layers: list, out_path: str):
    """
    The target-goal figure: z (mean) + log σ_z (std) on the HEAD ladder —
    depth 1 = the model's own readout, the decoder is off the z path. The
    z_out point is the readout itself, so its log σ_z probe is trivially
    ~1.0 and its z probe ≈ the model's own performance — it benchmarks the
    upstream layers rather than adding information.
    """
    _plot_target_pair(
        df, layers, HEAD_BRANCHES,
        [("z_true",      "z (redshift)",                "#2a78d6"),
         ("log_sigma_z", "log σ_z (model uncertainty)", "#eb6834")],
        "Linear-probe R² vs. model depth — redshift (mean + std)", out_path)


def plot_photometry(df: pd.DataFrame, layers: list, out_path: str):
    """
    The input-data-properties figure: i mag + g−r on the DECODER ladder —
    depth 1 = the decoder's last layer, log_spec (reconstructed rest-frame
    spectrum) as its intermediate: what survives the full round trip.
    """
    _plot_target_pair(
        df, layers, DECODER_BRANCHES,
        [("i_ref",      "i magnitude", "#1baf7a"),
         ("g_minus_r",  "g−r color",   "#898781")],
        "Linear-probe R² vs. model depth — input photometry properties", out_path)


def plot_sps_params(df: pd.DataFrame, layers: list, out_path: str):
    """
    All 15 SPS parameters on one axes. Identity = group hue + lightness step
    + marker (see SPS_STYLE); branch = line style, as everywhere else. The
    legend lists every parameter with its color+marker, grouped by hue.
    """
    ladder = build_ladder(HEAD_BRANCHES, layers)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    param_handles = []
    for pname in PARAM_NAMES_15:
        color, marker = SPS_STYLE[pname]
        _plot_target(ax, df, f"sps:{pname}", ladder, color, lw=1.6, marker=marker)
        param_handles.append(Line2D([0], [0], color=color, marker=marker,
                                    ms=5, lw=1.6, label=pname))
    _set_depth_axis(ax, ladder, fontsize=9)
    ax.set_ylabel(r"Ridge probe $R^2$ (held-out)")
    ax.set_xlabel(XLABEL)
    ax.set_title("Linear-probe R² vs. model depth — SPS parameters")
    leg1 = ax.legend(handles=param_handles, fontsize=8, frameon=False,
                     loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     title="SPS parameter (hue = group)", title_fontsize=8,
                     alignment="left")
    ax.add_artist(leg1)
    ax.legend(handles=_branch_handles(ladder), fontsize=8, frameon=False,
              loc="lower left")
    fig.tight_layout()
    # bbox_extra_artists: bbox_inches="tight" ignores legends attached via
    # add_artist, silently cropping the outside-right legend without it.
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                bbox_extra_artists=(leg1,))
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", default=None,
                        help="Activations npz to fit probes on")
    parser.add_argument("--test-npz",  default=None,
                        help="Activations npz to score probes on (held out)")
    parser.add_argument("--npz",       default=None,
                        help="Single npz to split internally, instead of --train-npz/--test-npz")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out-dir",   required=True)
    args = parser.parse_args()

    if args.npz:
        data = load_split(args.npz)
        n = len(data["z_true"])
        rng = np.random.RandomState(args.seed)
        idx = rng.permutation(n)
        n_te = int(args.test_frac * n)
        te_idx, tr_idx = idx[:n_te], idx[n_te:]
        train = {k: v[tr_idx] for k, v in data.items()}
        test  = {k: v[te_idx] for k, v in data.items()}
        print(f"Split {args.npz}: {len(tr_idx):,} train / {len(te_idx):,} test "
              f"(internal, seed={args.seed})")
    else:
        assert args.train_npz and args.test_npz, "Need --train-npz + --test-npz, or --npz"
        train = load_split(args.train_npz)
        test  = load_split(args.test_npz)
        print(f"Train: {args.train_npz} ({len(train['z_true']):,})")
        print(f"Test:  {args.test_npz} ({len(test['z_true']):,})")

    os.makedirs(args.out_dir, exist_ok=True)

    # z_out = the z head's final readout, derivable from every extraction.
    for split in (train, test):
        if "z_out" not in split:
            split["z_out"] = np.stack(
                [split["z_pred"], split["log_sigma_z"]], axis=1)

    layers = [l for l in LAYERS if l in train and not np.isnan(train[l]).all()]
    dropped = [l for l in LAYERS if l not in layers]
    if dropped:
        print(f"Skipping missing/all-NaN layers (not in this extraction): {dropped}")

    # Per-ladder depths for the CSV (NaN where a layer is off that ladder).
    d_heads   = build_ladder(HEAD_BRANCHES,    layers)[2]
    d_decoder = build_ladder(DECODER_BRANCHES, layers)[2]

    targets = list(PRIMARY_TARGETS) + [(f"sps:{p}", p) for p in PARAM_NAMES_15]

    print(f"\nFitting {len(targets)} targets x {len(layers)} layers "
          f"({len(targets) * len(layers)} probes)...")
    rows = []
    for key, label in targets:
        y_tr = get_target_vector(train, key)
        y_te = get_target_vector(test, key)
        for layer in layers:
            r2, nmad, n_tr, n_te = fit_and_score(
                train[layer], y_tr, test[layer], y_te,
                z_for_nmad=test["z_true"] if key == "z_true" else None)
            rows.append(dict(target=key, label=label, layer=layer,
                             depth_heads=d_heads.get(layer, np.nan),
                             depth_decoder=d_decoder.get(layer, np.nan),
                             r2=r2, sigma_nmad=nmad,
                             n_train=n_tr, n_test=n_te))
        r2_by_layer = ", ".join(f"{l}={r['r2']:.3f}" for l, r in
                                zip(layers, rows[-len(layers):]))
        print(f"  {label:<22s} {r2_by_layer}")
        if key == "z_true":
            nmad_by_layer = ", ".join(f"{l}={r['sigma_nmad']:.4f}" for l, r in
                                      zip(layers, rows[-len(layers):]))
            print(f"  {'  (z sigma_NMAD)':<22s} {nmad_by_layer}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "probe_r2.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    stale = os.path.join(args.out_dir, "probe_r2_primary.png")
    if os.path.exists(stale):
        os.remove(stale)   # replaced by the redshift/photometry pair
    plot_redshift(df, layers, os.path.join(args.out_dir, "probe_r2_redshift.png"))
    plot_photometry(df, layers, os.path.join(args.out_dir, "probe_r2_photometry.png"))
    plot_sps_params(df, layers, os.path.join(args.out_dir, "probe_r2_sps_params.png"))


if __name__ == "__main__":
    main()
