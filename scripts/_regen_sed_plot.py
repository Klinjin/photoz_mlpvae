"""
_regen_sed_plot.py
------------------
Regenerate test_sed_reconstructions.png for an already-trained model
WITHOUT re-running the full training loop.

Reproduces the exact test-set noise realization by advancing the RNG
past the train (200K×10) and val (80K×10) draws, then regenerates the
test set and calls plot_sed_reconstructions with theta_true.
"""
import os, sys
import numpy as np
import torch
import h5py
import pandas as pd

_BASE = "/astro/users/lindajin"
sys.path.insert(0, _BASE)

from obs_catalog.dataloader import build_features, MAG_COLS, ERR_COLS, TARGET_COL
from photoz_utils import plot_sed_reconstructions
from photoz_mlpvae.model.photoz_mlpvae_old import PhotozMLPVAE

# ── Constants (must match train_synth.py) ─────────────────────────────────────
MODEL_NAME       = "mlpvae_synth_v1"
SYNTH_VAL_H5     = os.path.join(_BASE, "sed_generation/data/sed_val.h5")
SPECULATOR_DIR   = os.path.join(_BASE, "speculator/trained/Inoue_IGM")
FILTER_DIR       = os.path.join(_BASE, "obs_catalog/filters")
OUT_DIR          = os.path.join(_BASE, f"photoz_mlpvae/trained/{MODEL_NAME}")

M_5SIG           = np.array([25.0,26.5,26.3,25.7,25.0,23.4,25.5,24.0,24.0,24.0],
                             dtype=np.float32)
CALIB_FLOOR      = 0.01
DETECT_SIGMA_MAX = 5.0
N_TRAIN_MAX      = 200_000
N_VAL_MAX        =  80_000
N_TEST_MAX       =  20_000
SEED             = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")

# ── Load model ────────────────────────────────────────────────────────────────
ckpt = os.path.join(OUT_DIR, "best.pt")
print(f"Loading model from {ckpt} …")
model, scaler, col_med = PhotozMLPVAE.load(ckpt, SPECULATOR_DIR, FILTER_DIR, device)
model.eval()

# ── Load theta_te (last N_TEST_MAX rows of sed_val.h5) ───────────────────────
with h5py.File(SYNTH_VAL_H5, "r") as f:
    n_total  = f["parameters"].shape[0]
    theta_te = f["parameters"][n_total - N_TEST_MAX : n_total].astype(np.float32)
print(f"theta_te shape: {theta_te.shape}")

# ── Generate noiseless test photometry ───────────────────────────────────────
def gen_phot(theta):
    out = []
    for i in range(0, len(theta), 1024):
        t = torch.from_numpy(theta[i:i+1024]).to(device)
        with torch.no_grad():
            ls, _ = model.speculator(t)
            m     = model.filter_conv(ls, t[:, 0], t[:, 1])
        out.append(m.cpu().numpy())
    return np.concatenate(out).astype(np.float32)

print("Generating noiseless photometry …")
mags_te_clean = gen_phot(theta_te)

# ── Reproduce RNG state ───────────────────────────────────────────────────────
# Original script: rng draws noise for train (N_TRAIN_MAX×10) then val (N_VAL_MAX×10)
# before drawing test noise.  Advance with dummy draws to match state exactly.
rng = np.random.default_rng(SEED)
rng.standard_normal((N_TRAIN_MAX, 10))   # advance past train noise
rng.standard_normal((N_VAL_MAX,   10))   # advance past val noise

# ── Add test noise (same function as train_synth.py) ─────────────────────────
m5   = M_5SIG[None, :]
snr  = np.clip(5.0 * np.power(10.0, -0.4 * (mags_te_clean - m5)), 1e-3, None)
errs = np.sqrt((2.5 / np.log(10.0) / snr) ** 2 + CALIB_FLOOR ** 2).astype(np.float32)

noise    = rng.normal(0.0, errs).astype(np.float32)
mags_te  = (mags_te_clean + noise).astype(np.float32)
detected = (errs < DETECT_SIGMA_MAX) & np.isfinite(mags_te)
mask_te  = detected.astype(np.float32)
mags_te[~detected] = np.nan
errs[~detected]    = np.nan

frac = mask_te.mean()
print(f"test band-detection rate: {frac*100:.1f}%")

# ── Build encoder features ────────────────────────────────────────────────────
data = {mc: mags_te[:, j] for j, mc in enumerate(MAG_COLS)}
data.update({ec: errs[:, j] for j, ec in enumerate(ERR_COLS)})
data[TARGET_COL] = theta_te[:, 0]

X_te, mags_te, errs_te, mask_te, z_te, _, _ = build_features(
    pd.DataFrame(data), use_colors=True, scaler=scaler, col_medians=col_med
)

# ── Load saved z_pred ─────────────────────────────────────────────────────────
z_pred_te = pd.read_csv(os.path.join(OUT_DIR, "test_predictions.csv"))["z_pred"].values

# ── Select failure cases ──────────────────────────────────────────────────────
_targets = [
    (0.5, 0.5,  "good_lowz"),
    (3.5, 3.5,  "good_highz"),
    (0.5, 1.5,  "mild_overest"),
    (0.5, 3.0,  "severe_overest"),
    (3.5, 2.5,  "mild_underest"),
    (3.5, 1.0,  "severe_underest"),
]
sel_idx = [
    int(np.argmin(np.abs(z_te - zt) + np.abs(z_pred_te - zp)))
    for zt, zp, _ in _targets
]
sel_labels = [lbl for _, _, lbl in _targets]

for lbl, i in zip(sel_labels, sel_idx):
    print(f"  {lbl:<15s}  z_true={z_te[i]:.3f}  z_pred={z_pred_te[i]:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
save_path = os.path.join(OUT_DIR, "test_sed_reconstructions.png")
plot_sed_reconstructions(
    model,
    X_te[sel_idx], mags_te[sel_idx], errs_te[sel_idx], mask_te[sel_idx],
    z_te[sel_idx], z_pred_te[sel_idx],
    sel_labels, device, save_path,
    theta_true=theta_te[sel_idx],
)
print(f"Done → {save_path}")
