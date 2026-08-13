# photoz_mlpvae

2-Stage MLP-VAE for photometric redshift estimation. Redshift and the
remaining 15 SPS parameters are inferred by two heads (their split differs
by model version, see below) sharing a common ResBlock trunk; the 15 SPS
parameters are always a VAE latent.

---

## Architecture

### Model versions (`--model-version {old,new}`)

Both training scripts select between two architectures sharing the same
trunk, decoder, and 3-term loss:

| | `old` (default) — `model/photoz_mlpvae_old.py` | `new` (z-separated) — `model/photoz_mlpvae.py` |
|---|---|---|
| Diagram | below | below |
| z estimate | deterministic z_head + separate `sigma_z_head` reading `[trunk \| mu_15.detach()]` | dedicated 1-dim z latent: 2-layer MLP → (mu_z, log_var_z); σ_z via delta-method push-forward, no separate head |
| vae_head input | `[trunk, z_pred.detach()]` | `[h.detach(), z_pred.detach()]` — recon/KL gradient **cannot** reach trunk/z_head (isolation baked in from the start) |
| Best dp1_v4 result | σ_NMAD 0.0587 / bias **0.0106** (`mlpvae_v2_lsst_gaap1p0`) | σ_NMAD **0.0298** / bias 0.0244 / RMS 0.2514 (`mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20`) |

The z-separated model holds the current σ_NMAD/outlier records; the old
model's bias number is a core/tail cancellation, not better calibration —
see PLAN.md's Open Problems for the bias decomposition and the remaining
data-limited low-z tail.

#### `old` — 3-head joint-gradient model (`model/photoz_mlpvae_old.py`)

```
x_phot (B, D)   D = 26 (LSST-only+det), 24 (LSST-only), 42 (LSST+Euclid+det), 40 (LSST+Euclid)
    │
    ▼
┌────────────────────────────────────────┐
│           Shared trunk                 │
│  Linear(D → 512) + BatchNorm + ReLU   │
│  ResBlock(512, dropout=0.1)            │
│  ResBlock(512, dropout=0.1)            │
└───┬─────────────┬──────────────┬───────┘
    │             │              │
┌───▼────┐  ┌────▼─────────┐  ┌─────▼──────────────────────────────────┐
│z_head  │  │sigma_z_head  │  │           vae_head                     │
│Linear  │  │Linear        │  │  input = cat([trunk, z_pred.detach()])  │
│(512→1) │  │([trunk|      │  │  Linear(513→30) → mu_15, log_var_15    │
│sigmoid │  │ mu_15.detach]│  │  + reparameterize → z_raw_15           │
│ ×5.5   │  │ →512→1)      │  └─────────────┬──────────────────────────┘
└───┬────┘  │→ log σ_z     │               │
    │       └──────────────┘      constrain_params_15(z_raw_15)
    │                                      │         (B, 15)
    └──────────────┬────────────────────────┘
                   │
          theta_full = cat([z_pred, theta_15], dim=1)   (B, 16)
                   │
                   ▼
        ┌──────────────────────┐
        │   Speculator (frozen)│
        │  theta → log_spec    │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │     FilterConv       │  (frozen)
        │  log_spec + z + mass │
        │  → 10 AB mags        │
        └──────────────────────┘
```

`z_head` is a bare deterministic `Linear(512→1)` → `sigmoid×5.5` point
estimate — no distribution, no reparameterization for z. **z_head injection
into vae_head**: the VAE posterior head receives
`[trunk_features (512) | z_pred (1)]` (513-dim input, trunk NOT detached) so
SPS inference is conditioned on redshift; `z_pred` itself is detached before
concatenation, but reconstruction/KL gradient can still reach the shared
trunk through the undetached `trunk_features` term (patched afterward per
Failure 4c/obs_v6 in PLAN.md, not structurally prevented).

**sigma_z_head** reads `[trunk | mu_15.detach()]` so that the predicted SED
posterior informs per-galaxy uncertainty estimates. The mu_15 portion is
zero-initialized to prevent early noise from corrupting the trunk gradient;
it activates gradually as mu_15 becomes informative during Phase 2.

**z_mu vs z_sample**: `z_pred` (z_mu) is the point estimate for NMAD/bias/OLF
metrics. `z_samples = N(z_pred, σ_z²)` adds aleatoric noise and should only
be used for calibration / PIT tests.

#### `new` — z-separated model (`model/photoz_mlpvae.py`)

```
x_phot (B, D)   D = 26 (LSST-only+det), 24 (LSST-only), 42 (LSST+Euclid+det), 40 (LSST+Euclid)
    │
    ▼
┌────────────────────────────────────────┐
│           Shared trunk                 │
│  Linear(D → 512) + BatchNorm + ReLU   │
│  ResBlock(512, dropout=0.1)            │
│  ResBlock(512, dropout=0.1)            │
└───┬────────────────────────┬───────────┘
    │ h                      │ h.detach()
┌───▼──────────────┐   ┌─────▼──────────────────────────────┐
│     z_head        │   │             vae_head                │
│ Linear(512→128)   │   │  input = cat([h.detach(),           │
│ + ReLU +          │   │               z_pred.detach()])     │
│ Linear(128→2)     │   │  Linear(513→30) → mu_15, log_var_15 │
│ → mu_z, log_var_z │   │  + reparameterize → z_raw_15        │
└───┬───────────────┘   └─────────────┬────────────────────────┘
    │ reparameterize                  │
    │ z_raw ~ N(mu_z, var_z)          constrain_params_15(z_raw_15)
    │ z_pred = sigmoid(z_raw)×_ZRED_MAX_SPEC (8.5) │  (B, 15)
    └──────────────┬───────────────────┘
                   │
          theta_full = cat([z_pred, theta_15], dim=1)   (B, 16)
                   │
                   ▼
        ┌──────────────────────┐
        │   Speculator (frozen)│
        │  theta → log_spec    │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │     FilterConv       │  (frozen)
        │  log_spec + z + mass │
        │  → 10 AB mags        │
        └──────────────────────┘
```

`z_head` is a dedicated 2-layer MLP — not a shared output row — producing
`(mu_z, log_var_z)` for a private 1-dim redshift latent. `z_pred` is the
**reparameterized sample** `sigmoid(z_raw)×5.5` (not the mean, and not a
bare point-regression output as in the old model): this is what gives
`log_var_z` a real calibration gradient from the label instead of drifting
under KL pressure alone (redshift itself carries no KL term — it's purely
supervised via NLL).

**No separate sigma_z_head**: σ_z is derived analytically from `log_var_z`
via a delta-method push-forward through the sigmoid transform
(`_zred_sigma_phys`) — the same mechanism `photoz_vae.py` uses for its
jointly-encoded zred. This removes the old model's SED→z-uncertainty
information channel (`sigma_z_head` reading `mu_15.detach()`) entirely.

**vae_head isolation baked in from the start**: reads
`[h.detach(), z_pred.detach()]`, so reconstruction/KL gradient can never
reach the trunk or z branch, by construction — not a later patch like the
old model's Failure 4c/obs_v6 fix.

**z_mu vs z_sample**: same convention as the old model — `z_pred` is the
point estimate for NMAD/bias/OLF metrics; `z_samples` (drawn from the
z-latent's own posterior) are for calibration / PIT only.

### SPS parameter ordering in `theta_full` (16 dims)

| Index | Parameter        | Source      | Transform                        | Range       |
|-------|------------------|-------------|----------------------------------|-------------|
| 0     | zred             | MLP z-head  | sigmoid(·) × 6.0                 | [0, 6.0]    |
| 1     | logmass          | VAE latent  | Φ(·) × 6.0 + 7.0                | [7, 13.0]   |
| 2     | logzsol          | VAE latent  | Φ(·) × 2.17 − 1.98              | [−1.98, 0.19] |
| 3     | dust2            | VAE latent  | Φ(·) × 4.0                      | [0, 4.0]    |
| 4     | dust1_fraction   | VAE latent  | Φ(·) × 2.0                      | [0, 2.0]    |
| 5     | dust_index       | VAE latent  | Φ(·) × 1.4 − 1.0                | [−1.0, 0.4] |
| 6     | gas_logz         | VAE latent  | Φ(·) × 2.5 − 2.0                | [−2.0, 0.5] |
| 7     | fagn             | VAE latent  | 10^(Φ(·) × 5.477 − 5)           | [1e-5, ~3]  |
| 8     | agn_tau          | VAE latent  | 10^(Φ(·) × 1.477 + 0.699)       | [5, 150]    |
| 9     | igm_scale        | VAE latent  | Φ(·) × 2.0                      | [0, 2.0]    |
| 10–15 | logsfr_ratios_0..5 | VAE latent | Φ(·) × 10.0 − 5.0             | [−5, 5]     |

Φ = standard-normal CDF (maps N(0,1) latent → Uniform over each range).

---

## Datasets

### dp1_v4 ECDFS (real observations — `train_dp1.py`)

Pre-split HDF5 catalogs from the LSST DP1 ECDFS field matched to
CDFS spectroscopic redshifts (SITCOMTN-154):

| Split | File | N |
|-------|------|---|
| Train | `83_training_v4_match_ecdfs_sitcomtn154_.../dp1_matched_v4_train.hdf5` | 6,101 (10% held out as val → 5,491 train / 610 val) |
| Val   | carved from train at runtime (val_frac=0.10)                            | 677 |
| Test  | `84_test_v4_match_ecdfs_sitcomtn154_.../dp1_matched_v4_test.hdf5`       | 2,905 |

Total labeled galaxies: ~9,006. Small dataset — warm-starting from synthetic
pre-training is important (see `--init-from`).

### DP2 SOM-matched (real observations — `train_dp2.py`)

Loads parquet catalogs directly (not pre-split HDF5) and applies the
galaxy/valid-z cuts itself:

| Split | File | N (raw → after `refExtendedness==1`) |
|-------|------|---|
| Train | `415_clipped_train_rtn124_parquet_.../train_clipped_v3.parquet` | 554,676 → 503,689 |
| Val   | carved from train at runtime (`--val-frac`, default 10%)         | — |
| Test  | `412_som_test_rtn124_parquet_.../test_som_mc_matched_v3.parquet` | 18,400 → 17,278 |

No Euclid columns in either file (`use_euclid` hardcoded False). `--z-max`
(default 5.5) drops train/val galaxies above that spec-z — test always
evaluates on the full, uncut range. `--use-gaap`-equivalent GAAP mags are
the default here (`--cmodel` to switch to cModel). Also saves
`phase1_end.pt` — the pure z-supervised checkpoint at the instant Phase 2
would otherwise unfreeze `vae_head`, independent of `best.pt`'s
rolling-average tracking. See PLAN.md for the z_max A/B test and the
NaN-crash investigation (Failure 13).

### Synthetic SEDs (`train_synth.py`)

Generated by the frozen Speculator+FilterConv pipeline from the SPS prior:

| Split | Source HDF5 | N (after i-SNR≥5 cut) |
|-------|-------------|----------------------|
| Train | `sed_train.h5` (≤200,000 drawn) | ~78,091 |
| Val   | `sed_val.h5` (80,000 drawn) | ~31,411 |
| Test  | `sed_val.h5` last 20,000 | ~7,774 |

Noiseless AB magnitudes are assigned photometric uncertainties using LSST
DP1 ECDFS 5σ depths. The i-band SNR cut removes galaxies too faint to be
detectable in the real survey.

---

## Photometry modes

### cModel vs GAAP magnitudes

| Mode | Column pattern | Description |
|------|---------------|-------------|
| cModel (default) | `{u,g,r,i,z,y}_cModelMag` | PSF-model fit; adapts aperture per object |
| GAAP 1.0 (--use-gaap) | `{u,g,r,i,z,y}_gaap1p0Mag` | Fixed 1.0-arcsec Gaussian aperture |

GAAP aperture magnitudes are better matched between the LSST bands and the
synthetic training apertures assumed by the noise model, reducing the
domain gap when fine-tuning on real data. Use `--use-gaap` for both
`train_synth.py` and `train_dp1.py` to keep the feature distributions aligned.

### Encoder input dimensions

| use_euclid | use_gaap | +det features | Input dim | Bands |
|------------|----------|---------------|-----------|-------|
| False       | either   | Yes (default) | **26**    | 6 LSST |
| False       | either   | No            | 24        | 6 LSST |
| True        | either   | Yes (default) | **42**    | 6 LSST + 4 Euclid |
| True        | either   | No            | 40        | 6 LSST + 4 Euclid |

`train_dp1.py` defaults to LSST-only (`use_euclid=False`). When warm-starting
from a synth checkpoint, ensure both scripts use the same `use_euclid` setting.

### Detection features (+2 dims)

Two scalar features are appended to every encoder input to break the high-z
feature degeneracy (see Failure 5):

| Feature | Value | Interpretation |
|---------|-------|----------------|
| `frac_detected` | n_detected / n_bands | Overall photometric coverage |
| `frac_blue_detected` | (u+g+r detected) / 3 | Lyman-break proxy: 0 → all blue bands below limit (z > 4.5) |

These are in [0, 1] and require no additional scaling.

### Feature encoding (USE_COLORS=True, default)

| Input bands | Features (D/2 scaled) | Missingness (D/2 flags) |
|-------------|-----------------------|-------------------------|
| 6 LSST      | 5 colours + 5 colour errs + i-ref + i-ref err (12) | 12 flags → 24-dim |
| 10 LSST+Eu  | 9 colours + 9 colour errs + i-ref + i-ref err (20) | 20 flags → 40-dim |

Toggle magnitudes-only with `--no-colors`.

---

## Loss function

```
L = λ_z × NLL_z(z_pred, z_spec, σ_z)                    MLP z head
  + λ_r × Σ_i mask_i (m̂_i − m_i)² / (σ_i² + σ_floor²)  reconstruction
  + β   × D_KL[ N(μ_15, σ_15²) ‖ N(0, I) ]               KL on 15-param latent
```

- **NLL_z**: `0.5 × (z_pred − z_spec)².detach() / σ_z² + log σ_z`
  (`sq_err.detach()` prevents σ_z from diluting the z gradient; z_pred trains
  on plain MSE and σ_z calibrates independently)
- **Reconstruction**: noise-weighted chi-squared over observed bands
- **KL**: standard VAE divergence in unconstrained space, β-annealed

Default weights: `λ_z = 20`, `λ_r = 0.1` (real data) / `0.3` (synthetic), `β_max = 0.1`

---

## Training phases

### Phase 1 — z-supervised warmup
- Duration: `--warmup-epochs`; `train_dp1.py` defaults to **10% of total
  epochs**, `train_synth.py` to a fixed 50
- `vae_head` weights **frozen**
- `λ_r = 0`, `β = 0`
- `use_nll_z = True` from epoch 1 (both scripts) — z trains on NLL so
  `log_var_z` / σ_z is calibrated throughout, never on bare MSE
- Only trunk + z branch (+ `sigma_z_head` in the old model) are updated

### Phase 2 — joint fine-tuning (remaining epochs)
- `vae_head` **unfrozen** via `opt.add_param_group()` (preserves trunk Adam
  momentum; prevents z_pred spike from optimizer reset)
- Trunk / z-branch LR: `train_dp1.py` keeps it on its single continuous
  cosine schedule (no Phase-2 rescale — the z-separated model's `h.detach()`
  makes the old ×0.1 cut unnecessary); `train_synth.py` still scales it by
  `phase2_trunk_lr_frac` (default 0.1) against its stronger noiseless
  reconstruction gradients
- `λ_r` ramped 0 → `lam_r` over `recon_ramp` epochs (default 50)
- `β` annealed 0 → `beta_max` over `beta_epochs` epochs (default 50)
- `λ_z` drops to `lam_z_min` (default **2.0**) at Phase 2 start, then
  restored to `lam_z` over `lam_z_restore` epochs (default 50) — gives the
  VAE head gradient budget during the β ramp without z_pred degrading

### Gradient clipping (both phases)
`clip_grad_norm_(encoder.parameters(), --grad-clip-max-norm)` (default
**2.0**) is applied every step and binds on essentially every batch — it
acts as always-on gradient normalization, not an emergency brake, and
removing it measurably hurts every test metric (see PLAN.md, Failure 11).
Per-epoch LR and pre-clip gradient norms are logged and plotted to
`lr_and_gradnorm.png`.

### λ_z schedule (Phase 2)

```
epoch < warmup:              λ_z = lam_z        (full, Phase 1)
warmup ≤ epoch < warmup+beta_epochs:
                             λ_z = lam_z_min    (reduced; VAE has budget)
warmup+beta_epochs ≤ epoch < warmup+beta_epochs+lam_z_restore:
                             λ_z linearly restored lam_z_min → lam_z
epoch ≥ warmup+beta_epochs+lam_z_restore:
                             λ_z = lam_z        (full, restored)
```

---

## Usage

### Synthetic pre-training

```bash
CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/WL_ML_Challenge/bin/python \
    photoz_mlpvae/scripts/train_synth.py \
    --model-name mlpvae_synth_v3_no_euclid_gaap1p0 \
    --use-gaap --no-euclid
```

### Real-data fine-tuning (warm-start from synth)

```bash
CUDA_VISIBLE_DEVICES=0 ~/miniforge3/envs/WL_ML_Challenge/bin/python \
    photoz_mlpvae/scripts/train_dp1.py \
    --model-name mlpvae_v4_lsst_gaap1p0_init_synth_v3 \
    --use-gaap \
    --init-from photoz_mlpvae/trained/mlpvae_synth_v3_no_euclid_gaap1p0/best.pt
```

### Key arguments (both scripts)

| Flag | Default (synth / dp1) | Description |
|------|-----------------------|-------------|
| `--model-name` | `mlpvae_synth_v1` / `mlpvae_v1_lsst` | Output directory |
| `--model-version` | `old` | `old` = 3-head joint-VAE model; `new` = z-separated model |
| `--use-gaap` | off | Use GAAP 1.0-arcsec aperture mags for LSST bands |
| `--no-euclid` | off (synth) / on (dp1) | LSST-only 6-band encoder |
| `--no-colors` | off | Raw magnitudes instead of colours |
| `--warmup-epochs` | 50 / **10% of `--epochs`** | Phase 1 duration (z-supervised, vae_head frozen) |
| `--lam-z` | 20.0 | Supervised z loss weight |
| `--lam-z-min` | 2.0 | λ_z floor during Phase 2 β ramp |
| `--lam-z-restore` | 50 | Epochs to ramp λ_z back from min to full |
| `--lam-r` | 0.3 / 0.1 | Reconstruction loss weight |
| `--sigma-floor` | 0.3 | Magnitude error floor for Speculator mismatch |
| `--phase2-trunk-lr-frac` | 0.1 (synth only) | LR multiplier for trunk/z_head at Phase 2 start (removed from `train_dp1.py`) |
| `--phase2-vae-lr-frac` | 1.0 | LR multiplier for vae_head at Phase 2 start |
| `--grad-clip-max-norm` | 2.0 (dp1 only) | Encoder gradient-norm clip; ≤0 disables (norm still logged) |
| `--beta-max` | 0.1 | Peak β for KL annealing |
| `--lr` | 1e-4 | Initial learning rate |
| `--i-band-snr-min` | 5.0 (synth only) | Min i-band SNR for galaxy inclusion |
| `--init-from` | None (dp1 only) | Checkpoint path for shape-matched warm-start |
| `--z-weight-cap` | — / 10.0 | Cap z-weights at this multiple of mean |
| `--trial` | off (dp1 only) | Comparison-trial mode: train + test metrics only, no figures |
| `--out-base` | `trained/` (dp1 only) | Base directory for the run's output folder |

### Outputs (`photoz_mlpvae/trained/<model-name>/`)

- `best.pt` — checkpoint at best rolling-average σ_NMAD (window=5 epochs)
- `config.yaml` — all hyperparameters
- `train.log` — full stdout mirror
- `train_curves.png` — loss / σ_NMAD curves with phase annotations
- `lr_and_gradnorm.png` — per-epoch LR (both param groups) + pre-clip
  gradient norm vs the clip threshold
- `test_scatter.png`, `test_zdist.png`, `test_metrics_vs_z.png`
- `test_metrics_paper_style.png`, `test_pit.png`
- `test_predictions.csv` — `z_true`, `z_pred`, `z_std`, `delta_z`
- `dp1_v4_test_scatter.png`, `dp1_v4_test_metrics_vs_z.png`,
  `dp1_v4_test_metrics_vs_mag.png` — density scatter + binned metrics on dp1_v4 test set
- `corner_*.png` — 16-parameter posterior corner plots for failure-case galaxies
- `test_sed_reconstructions.png` — SED diagnostic for same 4 galaxies

### Load a checkpoint

```python
from photoz_mlpvae.model import PhotozMLPVAE

model, scaler, col_medians = PhotozMLPVAE.load(
    "photoz_mlpvae/trained/mlpvae_v4_lsst_gaap1p0_init_synth_v3/best.pt",
    speculator_dir="speculator/trained/Inoue_IGM",
    filter_dir="obs_catalog/filters",
    device="cpu",
)
# x_phot: (N, 26) scaled features built with build_features(use_gaap=True)
z_samples, z_mean, z_std = model.predict_z(x_phot, n_samples=500)
theta_full = model.predict_params(x_phot)   # (N, 16) physical SPS params
```

---

## Current best

**Current best (real data, saved checkpoint)**:
`mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20` (z-separated model, dp1_v4,
LSST-only, GAAP 1.0", 2000 epochs / lr 1e-3 / `lam_z_min=20`, warm-started
from `mlpvae_synth_zsep_v1`) — **σ_NMAD = 0.0298, bias = 0.0244,
RMS = 0.2514** (test). See PLAN.md's changelog and Open Problems for the
full tuning history and the bias decomposition (the old model's
bias = 0.0106 is a core/tail cancellation, not better calibration).

---

## File structure

```
photoz_mlpvae/
├── README.md
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── photoz_mlpvae.py          # z-separated PhotozMLPVAE (--model-version new)
│   └── photoz_mlpvae_old.py      # 3-head joint-VAE PhotozMLPVAE (--model-version old)
├── scripts/
│   ├── __init__.py
│   ├── train_synth.py            # Synthetic SED pre-training
│   ├── train_dp1.py              # Real dp1_v4 ECDFS fine-tuning (HDF5 presplit)
│   ├── train_dp2.py              # Real DP2 SOM-matched fine-tuning (parquet, loads+cuts itself)
│   ├── extract_activations.py    # Per-layer activations + targets → .npz (representation study)
│   └── fit_linear_probes.py      # Ridge probes per (layer, target): R² vs depth
├── representation/               # Representation study: docs, probe npz's, results
│   └── README.md                 #   (does the SED latent add info for z / σ_z?)
└── trained/
    └── <model-name>/
        ├── best.pt
        ├── config.yaml
        ├── train.log
        └── *.png / *.csv
```

## Dependencies

Imports from sibling packages (must be on `sys.path`):
- `photoz_ae.model.vae_encoder._ResBlock` — residual block
- `photoz_ae.model.speculator_torch.SpeculatorInoueIGM` — frozen SED emulator
- `photoz_ae.model.filter_conv.FilterConv` — frozen photometric integration
- `photoz_utils` — plotting and metrics utilities
- `obs_catalog.dataloader` — shared feature builder, GAAP/cModel column lists

---

See [PLAN.md](PLAN.md) for the full experiment log, diagnosed failure modes,
and open problems.
