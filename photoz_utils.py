"""
photoz_utils.py
===============
Shared plotting and metrics utilities for photoz_mlp, photoz_bnn, photoz_vae.
All functions are importable; the matplotlib backend is switched to "Agg" on
import so that plots work in headless environments.

Usage
-----
    import sys
    sys.path.insert(0, "/astro/users/lindajin")
    from photoz_utils import (
        delta_z, compute_metrics,
        plot_scatter, plot_redshift_hist,
        plot_metrics_vs_zpred, plot_metrics_vs_zspec,
        plot_metrics_paper_style, plot_calibration,
    )
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ndtr  # Normal CDF, used for Gaussian PIT


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def delta_z(z_pred, z_true):
    """Normalised residual: (z_pred - z_true) / (1 + z_true)"""
    return (z_pred - z_true) / (1.0 + z_true)


def compute_metrics(dz, z_true=None):
    """Return (bias, nmad, fout, rms, f_cat) for an array of Δz values (NaN-safe).

    f_cat: 3σ catastrophic outlier fraction — |dz| > 3·σ_NMAD.
    Requires z_true to compute f_cat; returns np.nan for f_cat if omitted.
    """
    dz = np.asarray(dz, dtype=float)
    finite = np.isfinite(dz)
    if z_true is not None:
        z_true = np.asarray(z_true, dtype=float)
        finite &= np.isfinite(z_true)
        z_true = z_true[finite]
    dz = dz[finite]
    if len(dz) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    med  = np.median(dz)
    bias = np.mean(dz)
    nmad = 1.4826 * np.median(np.abs(dz - med))
    rms  = np.sqrt(np.mean(dz ** 2))
    fout = np.mean(np.abs(dz) > 0.15)
    if z_true is not None and nmad > 0:
        f_3sig = float(np.mean((np.abs(dz) > 3.0 * nmad)))
    else:
        f_3sig = np.nan
    return bias, nmad, fout, rms, f_3sig


def compute_metrics_dp1(dz, z_true=None):
    """Return (bias, nmad, fout, rms, f_3sig) for an array of Δz values (NaN-safe).

    f_3sig: 3σ catastrophic outlier fraction — |dz| > 3·σ_NMAD.
    Requires z_true to compute f_3sig; returns np.nan for f_3sig if omitted.
    """
    dz = np.asarray(dz, dtype=float)
    finite = np.isfinite(dz)
    if z_true is not None:
        z_true = np.asarray(z_true, dtype=float)
        finite &= np.isfinite(z_true)
        z_true = z_true[finite]
    dz = dz[finite]
    if len(dz) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    med  = np.median(dz)
    bias = np.mean(dz)
    nmad = 1.4826 * np.median(np.abs(dz - med))
    rms  = np.sqrt(np.mean(dz ** 2))
    fout = np.mean(np.abs(dz) > 0.2)
    if z_true is not None and nmad > 0:
        f_3sig = float(np.mean((np.abs(dz) > 3.0 * nmad)))
    else:
        f_3sig = np.nan
    return bias, nmad, fout, rms, f_3sig


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – scatter coloured by log10(counts)
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(z_true, z_pred, save_path="photoz_scatter.png", title=""):
    """2-D density scatter of z_true vs z_pred coloured by log10(counts)."""
    mask   = np.isfinite(z_true) & np.isfinite(z_pred)
    z_true = z_true[mask]; z_pred = z_pred[mask]
    dz     = delta_z(z_pred, z_true)
    bias, nmad, fout, rms, _ = compute_metrics(dz)

    zmin = np.floor(float(np.nanmin(z_true)) / 0.5) * 0.5
    zmax = np.ceil(float(np.nanmax(z_true))  / 0.5) * 0.5

    fig, ax = plt.subplots(figsize=(6, 5))
    n_bins  = 100
    edges   = np.linspace(zmin, zmax, n_bins + 1)
    H, xe, ye = np.histogram2d(z_true, z_pred, bins=[edges, edges])
    Hlog = np.log10(np.where(H > 0, H, np.nan))
    pcm  = ax.pcolormesh(xe, ye, Hlog.T, cmap="viridis", shading="auto")
    plt.colorbar(pcm, ax=ax, label=r"$\log_{10}(\mathrm{counts})$")
    zline = np.array([zmin, zmax])
    ax.plot(zline, zline, color="red", lw=1.5, zorder=5)
    ax.set_xlim(zmin, zmax); ax.set_ylim(zmin, zmax)
    ax.set_xlabel(r"$z_{\rm true}$",  fontsize=13)
    ax.set_ylabel(r"$z_{\rm pred}$", fontsize=13)
    if title:
        ax.set_title(title)
    info = (f"Bias: {bias:.3f}\nNMAD: {nmad:.3f}\n"
            f"Outliers: {fout*100:.1f}%\nRMS: {rms:.3f}")
    ax.text(0.05, 0.95, info, transform=ax.transAxes,
            va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot → {save_path}")
    return bias, nmad, fout, rms


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – redshift distribution histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_redshift_hist(z_true, z_pred, save_path="photoz_zdist.png",
                       highz_weight=0.0, highz_thresh=None):
    """Overlaid step histograms of z_true and z_pred.

    Parameters
    ----------
    highz_weight : float
        When > 1.0, draw a vertical line at *highz_thresh* marking the
        high-z upweighting boundary.
    highz_thresh : float or None
        Redshift value of the high-z threshold line (only drawn when
        highz_weight > 1.0 and highz_thresh is not None).
    """
    mask   = np.isfinite(z_true) & np.isfinite(z_pred)
    z_true = z_true[mask]; z_pred = z_pred[mask]
    zmax   = np.ceil(max(float(np.nanmax(z_true)), float(np.nanmax(z_pred))) / 0.5) * 0.5
    bins   = np.arange(0, zmax + 0.2, 0.2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(z_true, bins=bins, histtype="step", lw=2, color="steelblue", label=r"$z_{\rm true}$")
    ax.hist(z_pred, bins=bins, histtype="step", lw=2, color="tomato",    label=r"$z_{\rm pred}$", linestyle="--")
    if highz_weight > 1.0 and highz_thresh is not None:
        ax.axvline(highz_thresh, color="gray", ls=":", lw=1.5,
                   label=f"high-z threshold ({highz_thresh})")
    ax.set_xlabel("Redshift", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved redshift histogram → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – metrics vs redshift (shared helper + public wrappers)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_metrics_binned(z_bin_by, dz, xlabel, save_path):
    """4-panel metrics plot binned by z_bin_by (z_pred or z_true)."""
    z_lo  = np.floor(float(np.nanmin(z_bin_by)) / 0.2) * 0.2
    z_hi  = np.ceil(float(np.nanmax(z_bin_by))  / 0.2) * 0.2 + 0.2
    edges   = np.round(np.arange(z_lo, z_hi, 0.2), 6)
    centers = (edges[:-1] + edges[1:]) / 2.0

    biases, nmads, fouts, rms_arr = [], [], [], []
    bias_errs, nmad_errs, fout_errs, rms_errs = [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z_bin_by >= lo) & (z_bin_by < hi)
        n = m.sum()
        dz_bin = dz[m]
        b, nm, fo, rm, _ = compute_metrics(dz_bin)
        biases.append(b);   bias_errs.append(np.std(dz_bin) / np.sqrt(max(n, 1)))
        nmads.append(nm);   nmad_errs.append(nm / np.sqrt(max(2 * n, 1)))
        fouts.append(fo);   fout_errs.append(np.sqrt(fo * (1 - fo) / max(n, 1)))
        rms_arr.append(rm); rms_errs.append(rm  / np.sqrt(max(2 * n, 1)))

    biases  = np.array(biases,  dtype=float)
    nmads   = np.array(nmads,   dtype=float)
    fouts   = np.array(fouts,   dtype=float)
    rms_arr = np.array(rms_arr, dtype=float)
    valid   = np.isfinite(biases)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)
    eb_kw = dict(fmt="o-", color="steelblue", capsize=3, ms=4, lw=1.5)

    ax = axes[0, 0]
    ax.errorbar(centers[valid], fouts[valid], yerr=np.array(fout_errs)[valid], **eb_kw)
    ax.axhline(0.10, color="gray", ls="--", lw=1)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r"Fraction of outliers ($f_{\rm out}$)", fontsize=10)
    ax.set_xlim(edges[0], edges[-1])

    ax = axes[0, 1]
    ax.errorbar(centers[valid], biases[valid], yerr=np.array(bias_errs)[valid], **eb_kw)
    ax.axhline( 0.003, color="gray", ls="--", lw=1)
    ax.axhline(-0.003, color="gray", ls="--", lw=1)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Bias", fontsize=11)
    ax.set_xlim(edges[0], edges[-1])

    ax = axes[1, 0]
    ax.errorbar(centers[valid], nmads[valid], yerr=np.array(nmad_errs)[valid], **eb_kw)
    ax.axhline(0.02, color="gray", ls="--", lw=1)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(r"Scatter ($\sigma_{\rm NMAD}$)", fontsize=10)
    ax.set_xlim(edges[0], edges[-1])

    ax = axes[1, 1]
    ax.errorbar(centers[valid], rms_arr[valid], yerr=np.array(rms_errs)[valid], **eb_kw)
    ax.axhline(0.2, color="gray", ls="--", lw=1)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("RMS", fontsize=11)
    ax.set_xlim(edges[0], edges[-1])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved metrics plot → {save_path}")


def plot_metrics_vs_zpred(z_true, z_pred, save_path="photoz_metrics_vs_zpred.png"):
    """Metrics binned by predicted photo-z."""
    mask   = np.isfinite(z_true) & np.isfinite(z_pred)
    z_true = z_true[mask]; z_pred = z_pred[mask]
    dz     = delta_z(z_pred, z_true)
    _plot_metrics_binned(z_pred, dz, xlabel=r"$z_{\rm photo}$", save_path=save_path)


def plot_metrics_vs_zspec(z_true, z_pred, save_path="photoz_metrics_vs_zspec.png"):
    """Metrics binned by spectroscopic redshift (z_spec), matching paper plots."""
    mask   = np.isfinite(z_true) & np.isfinite(z_pred)
    z_true = z_true[mask]; z_pred = z_pred[mask]
    dz     = delta_z(z_pred, z_true)
    _plot_metrics_binned(z_true, dz, xlabel=r"$z_{\rm spec}$", save_path=save_path)


def plot_metrics_paper_style(z_true, z_pred, title, save_path):
    """Single-panel metrics vs z_spec matching Jones+24 figure style.

    Three series on one axes:
      - bias (black dots)
      - sigma_NMAD (blue triangles)
      - outlier fraction (red circles)
    x-axis: z_spec 0 to data max, y-axis: -0.8 to 0.8
    """
    mask   = np.isfinite(z_true) & np.isfinite(z_pred)
    z_true = z_true[mask]; z_pred = z_pred[mask]
    dz     = delta_z(z_pred, z_true)

    z_max   = np.ceil(float(np.nanmax(z_true)) / 0.2) * 0.2
    edges   = np.round(np.arange(0.0, z_max + 0.2, 0.2), 6)
    centers = (edges[:-1] + edges[1:]) / 2.0

    biases, nmads, fouts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z_true >= lo) & (z_true < hi)
        dz_bin = dz[m]
        b, nm, fo, *_ = compute_metrics(dz_bin)
        biases.append(b); nmads.append(nm); fouts.append(fo)

    biases = np.array(biases, dtype=float)
    nmads  = np.array(nmads,  dtype=float)
    fouts  = np.array(fouts,  dtype=float)
    valid  = np.isfinite(biases)

    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(centers[valid], biases[valid], "k.",  ms=5, label="bias")
    ax.plot(centers[valid], nmads[valid],  "b^",  ms=5, label=r"$\sigma$")
    ax.plot(centers[valid], fouts[valid],  "ro",  ms=5, label="outliers")
    ax.set_xlim(0, z_max)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel(r"$z_{\rm spec}$", fontsize=11)
    ax.set_ylabel("metrics", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved paper-style plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3b – grayscale density scatter with ±3σ / ±0.2 dashed lines
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter_density(z_true, z_pred, save_path,
                         zmin=0.0, zmax=3.0, n_bins=100):
    """Grayscale 2D density scatter matching the reference paper style.

    Draws four red dashed lines at ±3·σ_NMAD·(1+z) and ±0.2·(1+z),
    matching the two outlier thresholds shown in the statistics box.

    Parameters
    ----------
    z_true, z_pred : array-like  spectroscopic and photometric redshifts
    save_path      : str         output file path
    zmin, zmax     : float       axis / histogram range
    n_bins         : int         histogram resolution per axis
    """
    from matplotlib.colors import LogNorm

    mask   = np.isfinite(z_true) & np.isfinite(z_pred)
    z_true = np.asarray(z_true)[mask]
    z_pred = np.asarray(z_pred)[mask]
    dz     = delta_z(z_pred, z_true)

    med_dz  = np.median(dz)
    bias    = float(np.mean(dz))
    nmad    = float(1.4826 * np.median(np.abs(dz - med_dz)))
    fout_3s = float(np.mean(np.abs(dz) > 3.0 * nmad))
    fout_02 = float(np.mean(np.abs(dz) > 0.2))

    edges    = np.linspace(zmin, zmax, n_bins + 1)
    H, xe, ye = np.histogram2d(z_true, z_pred, bins=[edges, edges])
    H_masked  = np.where(H > 0, H, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    pcm  = ax.pcolormesh(xe, ye, H_masked.T,
                         cmap="Greys_r", norm=LogNorm(vmin=1), shading="auto")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Density", fontsize=11)

    zline = np.linspace(zmin, zmax, 300)
    ax.plot(zline, zline, color="red", lw=1.5, zorder=5)
    for sign in (-1, 1):
        ax.plot(zline, zline + sign * 3.0 * nmad * (1.0 + zline),
                "r--", lw=1.0, zorder=4)
        ax.plot(zline, zline + sign * 0.2 * (1.0 + zline),
                "r--", lw=1.0, zorder=4)

    ax.set_xlim(zmin, zmax)
    ax.set_ylim(zmin, zmax)
    ax.set_xlabel("True Redshift",      fontsize=12)
    ax.set_ylabel("Estimated Redshift", fontsize=12)

    info = (
        f"Δz = {bias:.4f}\n"
        f"σz = {nmad:.4f}\n"
        f"outlier rate (>3σ) = {fout_3s:.4f}\n"
        f"outlier rate (>0.2) = {fout_02:.4f}"
    )
    ax.text(0.04, 0.96, info,
            transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.85))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter density → {save_path}")
    return bias, nmad, fout_3s, fout_02


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3c – two-panel binned metrics with global 3-sigma clipping
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_binned_3sig(x_vals, dz_all, bins, xlabel, save_path,
                              ylim_top=None):
    """Two-panel binned metrics figure with global 3-sigma clipping.

    Top panel  : per-bin σz (orange), outlier rate (green), bias (blue+)
                 computed after removing |dz| > 3·global_σ_NMAD.
    Bottom panel: scatter of dz vs x with running 5th/16th/84th/95th
                  percentile envelopes (blue dashed).

    Parameters
    ----------
    x_vals    : (N,)  binning variable (e.g. z_spec or i-band magnitude)
    dz_all    : (N,)  (z_phot − z_spec) / (1 + z_spec)
    bins      : bin edges array
    xlabel    : x-axis label
    save_path : output file path
    ylim_top  : (ymin, ymax) for top panel, or None for auto-scale
    """
    fin    = np.isfinite(x_vals) & np.isfinite(dz_all)
    x_vals = np.asarray(x_vals)[fin]
    dz_all = np.asarray(dz_all)[fin]

    med_dz      = np.median(dz_all)
    nmad_global = 1.4826 * np.median(np.abs(dz_all - med_dz))
    outlier     = np.abs(dz_all) > 3.0 * nmad_global
    centers     = (bins[:-1] + bins[1:]) / 2.0

    biases, sigmas, fouts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x_vals >= lo) & (x_vals < hi)
        if m.sum() < 3:
            biases.append(np.nan); sigmas.append(np.nan); fouts.append(np.nan)
            continue
        fouts.append(float(outlier[m].mean()))
        dz_clean = dz_all[m & ~outlier]
        if len(dz_clean) < 2:
            biases.append(np.nan); sigmas.append(np.nan)
        else:
            biases.append(float(np.mean(dz_clean)))
            sigmas.append(float(
                1.4826 * np.median(np.abs(dz_clean - np.median(dz_clean)))))

    biases  = np.array(biases, dtype=float)
    sigmas  = np.array(sigmas, dtype=float)
    fouts   = np.array(fouts,  dtype=float)
    valid   = np.isfinite(biases)

    pct_vals = {p: [] for p in (5, 16, 84, 95)}
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x_vals >= lo) & (x_vals < hi)
        for p in pct_vals:
            pct_vals[p].append(
                float(np.percentile(dz_all[m], p)) if m.sum() >= 2 else np.nan)
    pct_vals = {p: np.array(v) for p, v in pct_vals.items()}

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.subplots_adjust(hspace=0.05)

    ax_top.axhline(0, color="gray", lw=0.8)
    ax_top.plot(centers[valid], sigmas[valid],
                color="orange", lw=1.5, label=r"$\sigma_z$")
    ax_top.plot(centers[valid], fouts[valid],
                color="green",  lw=1.5, label="Outlier rate")
    ax_top.errorbar(centers[valid], biases[valid],
                    fmt="+-", color="blue", lw=1.5, ms=6, capsize=3,
                    label="Bias")
    ax_top.set_ylabel("Statistics", fontsize=11)
    ax_top.set_title("Bias, Sigma, and Outlier rates w/ 3 sigma clipping",
                     fontsize=11)
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.set_xlim(bins[0], bins[-1])
    if ylim_top is not None:
        ax_top.set_ylim(*ylim_top)

    ax_bot.scatter(x_vals, dz_all,
                   s=1, c="black", alpha=0.25, rasterized=True, zorder=1)
    for p in (5, 16, 84, 95):
        v = np.isfinite(pct_vals[p])
        ax_bot.plot(centers[v], pct_vals[p][v],
                    "b--", lw=1.0, alpha=0.75, zorder=2)
    ax_bot.axhline(0, color="gray", lw=0.5)
    ax_bot.set_xlabel(xlabel, fontsize=11)
    ax_bot.set_ylabel(
        r"$(z_{\rm phot} - z_{\rm spec})/(1+z_{\rm spec})$", fontsize=10)
    ax_bot.set_xlim(bins[0], bins[-1])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved binned metrics → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 – calibration: predicted sigma vs |delta_z| in z-bins (BNN-specific)
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration(z_true, z_pred, z_std, save_path="photoz_calibration.png"):
    """Compare predicted uncertainty sigma against actual |delta_z| in z-bins.

    A well-calibrated model has sigma ~ |delta_z| on average.
    """
    mask   = np.isfinite(z_true) & np.isfinite(z_pred) & np.isfinite(z_std)
    z_true = z_true[mask]; z_pred = z_pred[mask]; z_std = z_std[mask]
    abs_dz = np.abs(delta_z(z_pred, z_true))

    z_lo   = np.floor(float(np.nanmin(z_pred)) / 0.5) * 0.5
    z_hi   = np.ceil(float(np.nanmax(z_pred))  / 0.5) * 0.5 + 0.5
    edges  = np.round(np.arange(z_lo, z_hi, 0.5), 6)
    centers = (edges[:-1] + edges[1:]) / 2.0

    mean_sigma, mean_absdz = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z_pred >= lo) & (z_pred < hi)
        if m.sum() < 5:
            mean_sigma.append(np.nan); mean_absdz.append(np.nan)
        else:
            mean_sigma.append(np.nanmean(z_std[m]))
            mean_absdz.append(np.nanmean(abs_dz[m]))

    mean_sigma = np.array(mean_sigma)
    mean_absdz = np.array(mean_absdz)
    valid = np.isfinite(mean_sigma) & np.isfinite(mean_absdz)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(centers[valid], mean_sigma[valid], "o-",  color="steelblue", lw=1.5, ms=5,
            label=r"mean $\hat{\sigma}$ (predicted)")
    ax.plot(centers[valid], mean_absdz[valid], "s--", color="tomato",    lw=1.5, ms=5,
            label=r"mean $|\Delta z|$ (actual)")
    ax.set_xlabel("pred-z", fontsize=12)
    ax.set_ylabel("Uncertainty", fontsize=12)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved calibration plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – QQ + PIT calibration plot (Schmidt et al. 2020, DC1 style)
# ─────────────────────────────────────────────────────────────────────────────

def plot_pit(z_true, z_pred, z_std=None, z_samples=None,
             save_path="photoz_pit.png", title="", n_bins=100,
             n_quants=101):
    """QQ + PIT calibration plot for probabilistic photo-z predictions.

    PIT_i = ∫₀^{z_true,i} p_i(z) dz — the quantile of the true redshift
    within each galaxy's photo-z posterior.  A perfectly calibrated model
    produces PIT values uniform on [0, 1].

    Top panel: PIT histogram (blue bars, right axis "Number") with the
    ideal uniform level (solid black line, N / n_bins), overlaid with the
    QQ curve (red, left axis): Q_data — the quantiles of the photo-z PIT
    distribution — against Q_theory — the corresponding quantiles of the
    calibrated (uniform) reference built from the true redshifts.  A
    calibrated model traces the dashed diagonal.

    Bottom panel: ΔQ = Q_data − Q_theory.  Excursions below zero at small
    Q_theory (or above zero near 1) reflect the over-confidence spikes at
    PIT ≈ 0 / 1.

    Parameters
    ----------
    z_true    : (N,) spectroscopic redshifts.
    z_pred    : (N,) predicted mean redshifts.
    z_std     : (N,) predicted standard deviations.  Used when z_samples is
                None; PIT is then Φ((z_true − z_pred) / z_std) where Φ is
                the standard-Normal CDF.
    z_samples : (N, S) posterior samples (e.g. from VAE MC draws or BNN
                stochastic forward passes).  When provided the PIT is
                computed empirically: PIT_i = fraction of samples < z_true_i.
    save_path : output file path.
    title     : label shown in the boxed legend (e.g. the model name).
    n_bins    : number of histogram bins (default 100).
    n_quants  : number of quantile points for the QQ curve (default 101).
    """
    if z_samples is not None:
        mask = np.isfinite(z_true) & np.isfinite(z_pred)
        z_true    = z_true[mask]
        z_samples = z_samples[mask]
        pit = np.mean(z_samples < z_true[:, np.newaxis], axis=1)
    else:
        if z_std is None:
            raise ValueError("Provide either z_std (Gaussian) or z_samples (empirical).")
        mask  = np.isfinite(z_true) & np.isfinite(z_pred) & np.isfinite(z_std) & (z_std > 0)
        z_true = z_true[mask]
        z_pred = z_pred[mask]
        z_std  = z_std[mask]
        pit = ndtr((z_true - z_pred) / z_std)

    q_theory = np.linspace(0.0, 1.0, n_quants)
    q_data   = np.quantile(pit, q_theory)
    delta_q  = q_data - q_theory

    fig = plt.figure(figsize=(5.0, 6.2))
    gs  = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.30)

    # ── Top panel: QQ curve (left axis) + PIT histogram (right axis) ──────
    ax_q = fig.add_subplot(gs[0])
    ax_h = ax_q.twinx()

    ax_h.hist(pit, bins=n_bins, range=(0.0, 1.0),
              color="#5c9dc7", edgecolor="white", lw=0.2)
    ideal = len(pit) / n_bins
    ax_h.axhline(ideal, color="black", lw=1.2)
    ax_h.set_ylabel("Number", fontsize=12)
    ax_h.set_ylim(bottom=0)

    ax_q.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax_q.plot(q_theory, q_data, color="red", lw=2.5)
    ax_q.set_xlim(0.0, 1.0)
    ax_q.set_ylim(0.0, 1.0)
    ax_q.set_ylabel(r"$Q_{data}$", fontsize=12)
    # Draw the QQ axes above the histogram axes.
    ax_q.set_zorder(ax_h.get_zorder() + 1)
    ax_q.patch.set_visible(False)
    if title:
        ax_q.text(0.05, 0.96, title, transform=ax_q.transAxes,
                  ha="left", va="top", fontsize=11,
                  bbox=dict(boxstyle="square,pad=0.4", facecolor="white",
                            edgecolor="black", lw=0.8))

    # ── Bottom panel: ΔQ = Q_data − Q_theory ──────────────────────────────
    ax_d = fig.add_subplot(gs[1])
    ax_d.axhline(0.0, color="black", ls="--", lw=1.2)
    ax_d.plot(q_theory, delta_q, color="red", lw=2.5)
    ax_d.set_xlim(0.0, 1.0)
    # ±0.12 as in Schmidt et al. (2020); widen symmetrically if ΔQ exceeds it.
    dq_lim = max(0.12, 1.08 * float(np.max(np.abs(delta_q))))
    ax_d.set_ylim(-dq_lim, dq_lim)
    if dq_lim <= 0.12:
        ax_d.set_yticks([-0.1, 0.0, 0.1])
    ax_d.set_xlabel(r"$Q_{theory}$ / PIT Value", fontsize=12)
    ax_d.set_ylabel(r"$\Delta Q$", fontsize=12)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved QQ/PIT plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SPS parameter labels (shared by PhotozVAE and PhotozMLPVAE)
# ─────────────────────────────────────────────────────────────────────────────

_SPS_LATEX_LABELS = [
    r"$z$",
    r"$\log M_*$",
    r"$\log Z_*$",
    r"$\hat{\tau}_2$",
    r"$f_{\rm dust1}$",
    r"$n_{\rm dust}$",
    r"$\log Z_{\rm gas}$",
    r"$f_{\rm AGN}$",
    r"$\tau_{\rm AGN}$",
    r"$s_{\rm IGM}$",
    r"$\log r_0$",
    r"$\log r_1$",
    r"$\log r_2$",
    r"$\log r_3$",
    r"$\log r_4$",
    r"$\log r_5$",
]

_ZRED_IDX    = 0
_LOGMASS_IDX = 1
_BAND_NAMES  = ["u", "g", "r", "i", "z", "y", "VIS", "Y", "J", "H"]


# ─────────────────────────────────────────────────────────────────────────────
# Prior sampler (parameters only – no FSPS required)
# ─────────────────────────────────────────────────────────────────────────────

def sample_prior_physical(n_samples, gallazzi_path, rng=None):
    """
    Draw n_samples from the joint prior over physical SPS parameters,
    matching the distributions in generate_sed_dataset.py::sample_prior().
    FSPS/Prospector are NOT required.

    Returns
    -------
    samples : (n_samples, 16) float32 ndarray
              [zred, logmass, logzsol, dust2, dust1_fraction, dust_index,
               gas_logz, fagn, agn_tau, igm_scale, logsfr_ratios_0..5]
    """
    from scipy.stats import truncnorm, t as student_t

    if rng is None:
        rng = np.random.default_rng(42)
    n = n_samples
    out = np.empty((n, 16), dtype=np.float32)

    # 0: zred – Uniform [0, 5.5]
    out[:, 0] = rng.uniform(0.0, 5.5, n)

    # 1: logmass – Uniform [7, 12.5]
    out[:, 1] = rng.uniform(7.0, 12.5, n)

    # 2: logzsol – Truncated-normal from Gallazzi+05 mass–metallicity relation
    gallazzi = np.loadtxt(gallazzi_path)  # cols: logmass, median, p16, p84
    loc   = np.interp(out[:, 1], gallazzi[:, 0], gallazzi[:, 1])
    upper = np.interp(out[:, 1], gallazzi[:, 0], gallazzi[:, 3])
    lower = np.interp(out[:, 1], gallazzi[:, 0], gallazzi[:, 2])
    scale = np.maximum(upper - lower, 1e-6)
    z_mini, z_maxi = -1.98, 0.19
    a = (z_mini - loc) / scale
    b = (z_maxi - loc) / scale
    out[:, 2] = truncnorm.rvs(a, b, loc=loc, scale=scale,
                               random_state=int(rng.integers(0, 2**31))).astype(np.float32)

    # 3: dust2 – ClippedNormal N(0.3, 1.0) ∈ [0, 4]
    _buf = rng.normal(0.3, 1.0, n * 10)
    _buf = _buf[(_buf >= 0.0) & (_buf <= 4.0)]
    while len(_buf) < n:
        _buf = np.concatenate([_buf, rng.normal(0.3, 1.0, n * 5)])
        _buf = _buf[(_buf >= 0.0) & (_buf <= 4.0)]
    out[:, 3] = _buf[:n].astype(np.float32)

    # 4: dust1_fraction – ClippedNormal N(1.0, 0.3) ∈ [0, 2]
    _buf = rng.normal(1.0, 0.3, n * 5)
    _buf = _buf[(_buf >= 0.0) & (_buf <= 2.0)]
    while len(_buf) < n:
        _buf = np.concatenate([_buf, rng.normal(1.0, 0.3, n * 5)])
        _buf = _buf[(_buf >= 0.0) & (_buf <= 2.0)]
    out[:, 4] = _buf[:n].astype(np.float32)

    # 5: dust_index – Uniform [-1, 0.4]
    out[:, 5] = rng.uniform(-1.0, 0.4, n)

    # 6: gas_logz – Uniform [-2, 0.5]
    out[:, 6] = rng.uniform(-2.0, 0.5, n)

    # 7: fagn – log-uniform [1e-5, 3], returned in linear space
    out[:, 7] = (10.0 ** rng.uniform(np.log10(1e-5), np.log10(3.0), n)).astype(np.float32)

    # 8: agn_tau – log-uniform [5, 150], returned in linear space
    out[:, 8] = (10.0 ** rng.uniform(np.log10(5.0), np.log10(150.0), n)).astype(np.float32)

    # 9: igm_scale – Uniform [0, 2]
    out[:, 9] = rng.uniform(0.0, 2.0, n)

    # 10–15: logsfr_ratios_0..5 – Clipped Student-t σ=0.3, ν=2, |r|≤5
    sigma_sfh, df_sfh, clip = 0.3, 2.0, 5.0
    for col in range(10, 16):
        _buf = student_t.rvs(df=df_sfh, loc=0.0, scale=sigma_sfh,
                              size=n * 20,
                              random_state=int(rng.integers(0, 2**31)))
        _buf = _buf[np.abs(_buf) <= clip]
        while len(_buf) < n:
            _buf = np.concatenate([_buf, student_t.rvs(
                df=df_sfh, loc=0.0, scale=sigma_sfh, size=n * 5,
                random_state=int(rng.integers(0, 2**31)))])
            _buf = _buf[np.abs(_buf) <= clip]
        out[:, col] = _buf[:n].astype(np.float32)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6 – corner posteriors for individual galaxies
# ─────────────────────────────────────────────────────────────────────────────

def plot_corner_posteriors(model, X_sel, z_true_vals, z_pred_vals,
                           labels, device, save_prefix, n_samples=2000,
                           gallazzi_path=None, theta_true=None):
    """
    Corner plot of the joint 16-parameter posterior for selected galaxies,
    with the prior overplotted as red contours/histograms.

    The model must implement:
      model.sample_theta_posterior(x, n_samples, device, rng) → (N_gal, n_samples, 16)

    z_true is marked with a gold vertical line.  When theta_true is supplied
    (e.g. training on synthetic SEDs with known SPS parameters), all 16 true
    values are marked; otherwise only z_true is marked and the 15 SPS params
    receive nan truth markers.
    Prior (red) is sampled via sample_prior_physical() from the Gallazzi+05
    mass–metallicity file; auto-detected at sed_generation/model/ relative to
    photoz_utils.py if gallazzi_path is None.

    Parameters
    ----------
    model        : PhotozVAE or PhotozMLPVAE (eval mode expected)
    X_sel        : (N_gal, D)   np.ndarray  scaled encoder features (D=42 or 26)
    z_true_vals  : (N_gal,)     np.ndarray  true redshifts
    z_pred_vals  : (N_gal,)     np.ndarray  predicted redshifts (for title)
    labels       : list[str]    short identifier for each galaxy
    device       : str
    save_prefix  : str          output path prefix (e.g. "/path/to/corner")
    n_samples    : int          posterior samples per galaxy
    gallazzi_path: str or None  path to gallazzi_05_massmet.txt
    theta_true   : (N_gal, 16) np.ndarray or None
                   When provided (synthetic SED training), all 16 physical SPS
                   parameter truth values are shown as gold lines.  When None,
                   only z_true is shown and the 15 SPS params get nan markers.
    """
    import os as _os
    try:
        import corner as corner_lib
    except ImportError:
        print("  [warn] `corner` not installed – skipping corner plots.")
        return

    # ── Auto-detect Gallazzi path and sample prior ────────────────────────
    if gallazzi_path is None:
        gallazzi_path = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)),
            "sed_generation", "model", "gallazzi_05_massmet.txt",
        )
    prior_samples = None
    if _os.path.isfile(gallazzi_path):
        try:
            prior_samples = sample_prior_physical(
                10_000, gallazzi_path, rng=np.random.default_rng(1)
            )
            print(f"  Prior: {prior_samples.shape[0]:,} samples drawn for overplot")
        except Exception as _e:
            print(f"  [warn] Prior sampling failed: {_e}")
    else:
        print(f"  [warn] Gallazzi file not found at {gallazzi_path} – prior not overplotted")

    model.eval()
    rng = np.random.default_rng(0)
    with __import__("torch").no_grad():
        all_samples = model.sample_theta_posterior(X_sel, n_samples, device, rng)

    n_gal = X_sel.shape[0]
    for i in range(n_gal):
        samples = all_samples[i]                              # (n_samples, 16)
        if theta_true is not None:
            truths = [float(theta_true[i, j]) for j in range(16)]
        else:
            truths = [float(z_true_vals[i])] + [np.nan] * 15

        base_fig = plt.figure(figsize=(32, 32))
        fig = corner_lib.corner(
            samples,
            labels=_SPS_LATEX_LABELS,
            truths=truths,
            truth_color="#FF0044",         
            truth_kwargs={"lw": 3.5},
            show_titles=True,
            title_kwargs={"fontsize": 30},
            label_kwargs={"fontsize": 32},
            labelpad=0.5,
            bins=40,
            color="#4C72B0",
            hist_kwargs={"density": True,"linewidth": 2.5, "alpha": 0.8},
            quantiles=[0.16, 0.5, 0.84],
            fig=base_fig,
        )

        # ── Overplot prior as red step histograms + unfilled contours ────
        if prior_samples is not None:
            corner_lib.corner(
                prior_samples,
                fig=fig,
                color="red",
                plot_datapoints=False,
                fill_contours=False,
                plot_density=False,
                show_titles=False,
                quantiles=[],
                bins=40,
                smooth=1.0,
                hist_kwargs={"linewidth": 1.5, "alpha": 0.6},
                contour_kwargs={"linewidths": 1.5, "alpha": 0.6},
            )

        for ax in fig.axes:
            ax.tick_params(labelsize=22)
        fig.suptitle(
            f"{labels[i]}   $z_{{\\rm true}}={z_true_vals[i]:.3f}$  "
            f"$z_{{\\rm pred}}={z_pred_vals[i]:.3f}$",
            fontsize=42, y=1.002,
        )
        outpath = f"{save_prefix}_{labels[i]}.png"
        fig.savefig(outpath, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Corner → {outpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 7 – SED reconstruction figure for individual galaxies
# ─────────────────────────────────────────────────────────────────────────────

def plot_sed_reconstructions(model, X_sel, mags_obs_sel, mag_errs_sel, mask_sel,
                              z_true_vals, z_pred_vals, labels, device, save_path,
                              theta_true=None):
    """
    4-panel SED reconstruction figure (one panel per galaxy).

    Each panel shows:
      - Grey curve   : full SED redshifted to z_pred (observer-frame λ)
      - Bisque curve : full SED redshifted to z_true
      - Black dots   : observed photometry with error bars
      - Red crosses  : model-reconstructed photometry at z_pred
      - Orange diamonds: reconstruction with z substituted by z_true

    The model must implement:
      model.encode_theta(x_t) → (z_pred, log_sigma_z, theta_full)
    and expose:
      model.speculator(theta) → (log_spec, wl_rest)
      model.filter_conv(log_spec, z, logmass) → m_ab

    Parameters
    ----------
    model        : PhotozVAE or PhotozMLPVAE (eval mode expected)
    X_sel        : (N_gal, D)   np.ndarray  scaled encoder input (D=42 or 26)
    mags_obs_sel : (N_gal, B)   np.ndarray  observed AB mags (B=10 or 6)
    mag_errs_sel : (N_gal, B)   np.ndarray  observed mag errors
    mask_sel     : (N_gal, B)   np.ndarray  band-presence mask
    z_true_vals  : (N_gal,)     np.ndarray
    z_pred_vals  : (N_gal,)     np.ndarray  (title only; model recomputes)
    labels       : list[str]
    device       : str
    save_path    : str
    """
    import torch

    # Infer band count from mask; decoder always returns 10 bands so we slice.
    n_obs_bands = mask_sel.shape[1]   # 6 (LSST only) or 10 (LSST+Euclid)
    band_names  = _BAND_NAMES[:n_obs_bands]

    model.eval()
    x_t = torch.from_numpy(X_sel.astype(np.float32)).to(device)

    with torch.no_grad():
        z_pred_t, _, theta_full_t = model.encode_theta(x_t)
        z_safe = z_pred_t.clamp(max=5.5)
        theta_full_t = torch.cat([
            z_safe.unsqueeze(1),
            theta_full_t[:, 1:],
        ], dim=1)

        # SED at z_pred
        log_spec_t, wl_rest_t = model.speculator(theta_full_t)
        m_ab_pred_t = model.filter_conv(
            log_spec_t,
            theta_full_t[:, _ZRED_IDX],
            theta_full_t[:, _LOGMASS_IDX],
        )

        # SED at z_true: use actual true SPS params if available, else substitute z only
        if theta_true is not None:
            theta_ztrue_t = torch.from_numpy(theta_true.astype(np.float32)).to(device)
        else:
            z_true_t      = torch.from_numpy(z_true_vals.astype(np.float32)).to(device)
            theta_ztrue_t = theta_full_t.clone()
            theta_ztrue_t[:, _ZRED_IDX] = z_true_t
        log_spec_zt_t, _ = model.speculator(theta_ztrue_t)
        m_ab_ztrue_t = model.filter_conv(
            log_spec_zt_t,
            theta_ztrue_t[:, _ZRED_IDX],
            theta_ztrue_t[:, _LOGMASS_IDX],
        )

    wl_rest_np     = wl_rest_t.cpu().numpy()
    log_spec_np    = log_spec_t.cpu().numpy()
    log_spec_zt_np = log_spec_zt_t.cpu().numpy()
    theta_full_np  = theta_full_t.cpu().numpy()
    theta_ztrue_np = theta_ztrue_t.cpu().numpy()
    # Decoder outputs 10 bands; slice to the observed band set.
    m_pred_np  = m_ab_pred_t.cpu().numpy()[:,  :n_obs_bands]
    m_ztrue_np = m_ab_ztrue_t.cpu().numpy()[:, :n_obs_bands]
    z_pred_np  = z_pred_t.cpu().numpy()

    _log_unit  = float(model.filter_conv._log_unit)
    z_grid_np  = model.filter_conv.z_grid.cpu().numpy()
    dl_grid_np = model.filter_conv.dl_grid.cpu().numpy()

    wl_eff_obs = np.zeros(n_obs_bands)
    for j in range(n_obs_bands):
        wl_f = getattr(model.filter_conv, f"filt_wl_{j}").cpu().numpy()
        T_f  = getattr(model.filter_conv, f"filt_T_{j}").cpu().numpy()
        wl_eff_obs[j] = np.trapz(T_f * wl_f, wl_f) / (np.trapz(T_f, wl_f) + 1e-30)

    n_gal = X_sel.shape[0]
    ncols = min(n_gal, 2)
    nrows = (n_gal + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes[:n_gal]):
        z_p = float(z_pred_np[i])
        z_t = float(z_true_vals[i])

        wl_obs_pred = wl_rest_np * (1.0 + z_p)
        wl_obs_true = wl_rest_np * (1.0 + z_t)

        band_ok        = mask_sel[i].astype(bool)
        logmass_pred_i = float(theta_full_np[i,  _LOGMASS_IDX])
        logmass_true_i = float(theta_ztrue_np[i, _LOGMASS_IDX])
        dl_pred_i      = float(np.interp(z_p, z_grid_np, dl_grid_np))
        dl_true_i      = float(np.interp(z_t, z_grid_np, dl_grid_np))

        lfs_pred = (logmass_pred_i + _log_unit
                    + np.log10(max(1.0 + z_p, 1e-6))
                    - 2.0 * np.log10(max(dl_pred_i, 1.0)))
        lfs_true = (logmass_true_i + _log_unit
                    + np.log10(max(1.0 + z_t, 1e-6))
                    - 2.0 * np.log10(max(dl_true_i, 1.0)))

        m_ab_curve_pred = np.clip(-2.5 * (log_spec_np[i]    + lfs_pred), 10.0, 40.0)
        m_ab_curve_true = np.clip(-2.5 * (log_spec_zt_np[i] + lfs_true), 10.0, 40.0)

        ax.plot(np.log10(wl_obs_pred), m_ab_curve_pred,
                color="0.75", lw=0.8, zorder=1, label="SED (z_pred)")
        ax.plot(np.log10(wl_obs_true), m_ab_curve_true,
                color="bisque", lw=0.8, zorder=1, label="SED (z_true)")

        ax.scatter(np.log10(wl_eff_obs), m_pred_np[i],
                   marker="x", s=80, linewidths=2.0, color="red",
                   zorder=4, label="Recon (z_pred)")
        ax.scatter(np.log10(wl_eff_obs), m_ztrue_np[i],
                   marker="D", s=40, facecolors="none", edgecolors="darkorange",
                   linewidths=1.5, zorder=3, label="Recon (z_true)")

        obs_x     = np.log10(wl_eff_obs)[band_ok]
        obs_y     = mags_obs_sel[i][band_ok]
        obs_e     = mag_errs_sel[i][band_ok]
        obs_names = [band_names[b] for b in range(n_obs_bands) if band_ok[b]]
        ax.errorbar(obs_x, obs_y, yerr=obs_e,
                    fmt="o", color="black", ms=5, lw=1.2,
                    zorder=5, label="Observed")
        for bx, by, bn in zip(obs_x, obs_y, obs_names):
            ax.annotate(bn, (bx, by), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=12, color="black")

        ax.invert_yaxis()
        ax.set_xlabel(r"$\log_{10}(\lambda_{\rm obs} / \AA)$", fontsize=10)
        ax.set_ylabel("AB magnitude", fontsize=10)
        ax.set_title(
            f"{labels[i]}\n$z_{{\\rm true}}={z_t:.3f}$  $z_{{\\rm pred}}={z_p:.3f}$",
            fontsize=10,
        )
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Hide any extra axes in the grid
    for ax in axes[n_gal:]:
        ax.set_visible(False)

    fig.suptitle("SED reconstructions — failure cases", fontsize=13)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  SED reconstructions → {save_path}")
