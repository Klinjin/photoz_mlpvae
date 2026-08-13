# Representation study — does the SED latent carry *new* information for photo-z?

Goal: probe whether the MLP+VAE model actually gains information from its
physically-bounded SED latent space (the 15-dim `mu_15` produced by
`vae_head`, constrained to SPS parameter ranges) for the purpose of redshift
estimation **with uncertainty** — or whether everything the z-branch uses is
already present in the shared trunk, with the SED latent merely re-expressing
it.

The operational question, in probe terms: at which depth does each quantity
(z, log σ_z, magnitude, color, each SPS parameter) become linearly decodable,
and does the SPS branch (`mu_15`) hold information the z-branch inputs (`h2`)
do not — or vice versa?

## Pipeline (2 existing scripts, both in `../scripts/`)

```
checkpoint (best.pt + sibling config.yaml)          catalog (.hdf5 / .parquet)
        └───────────────┬───────────────────────────────────┘
                        ▼
        1. extract_activations.py   →  probe_activations_{train,test}.npz
                        ▼
        2. fit_linear_probes.py     →  probe_r2.csv + R²-vs-depth plots
```

### 1. `scripts/extract_activations.py` — activation + target extraction

Runs a catalog through a trained `PhotozMLPVAE` checkpoint and saves every
tapped layer plus supervised targets to a single `.npz`.

- **Checkpoint-agnostic**: works with any `--model-version` (`old`, `new`,
  `mdn`), any dataset (dp1_v4 HDF5, DP2 parquet, synth HDF5). Preprocessing
  (scaler, `col_medians`, `n_in`) is read from the checkpoint;
  `model_version` / `use_gaap` / `use_euclid` / `use_colors` are read from the
  sibling `config.yaml` (CLI flags override).
- **Layers tapped** (encoder trunk → both heads; decoder is frozen so only
  its final output is kept):

  | key | shape | what it is |
  |---|---|---|
  | `x` | (N, n_in) | scaled input features — control/baseline probe |
  | `h0` | (N, 512) | after `input_proj` (Linear+BN+ReLU) |
  | `h1_mid`, `h2_mid` | (N, 512) | mid-ResBlock activations (`block[:3]` = Linear+BN+ReLU, before the second Linear+BN and the residual add) |
  | `h1`, `h2` | (N, 512) | ResBlock outputs — `h2` is the trunk output, feeds both heads |
  | `z_head_hidden` | (N, 128) | z_head hidden layer (`new`/`mdn` only; all-NaN for `old`) |
  | `z_out` | (N, 2) | `[z_pred, log σ_z]` — the z head's final readout (assembled by the probe script from saved arrays) |
  | `mu_15`, `log_var_15` | (N, 15) | vae_head raw output — the unconstrained SPS latent |
  | `log_spec` | (N, 1999) | reconstructed rest-frame log-spectrum — the Speculator's output, the decoder's intermediate layer (captured via forward hook) |
  | `m_ab_recon` | (N, 10) | reconstructed AB magnitudes (FilterConv output, the decoder's last layer) |

  Not tapped (would need the band forward pass re-implemented — they're
  frozen buffers, not submodules): the Speculator's per-band hidden layers
  (3×256 per band) and PCA coefficients; FilterConv internals.

- **Targets saved alongside**: `z_true`, `z_pred`, `log_sigma_z`,
  `dz = z_pred − z_true`, `theta_full` (N, 16 — post-`constrain_params_15`
  physical SPS params, col 0 = zred), `i_ref` (reference-band magnitude,
  GAAP or cModel per config), `g_minus_r`, `mask` (band-detection pattern).

### 2. `scripts/fit_linear_probes.py` — ridge probes, R² vs depth

Fits `RidgeCV` linear probes per (layer, target) pair on one split, scores
held-out R² on the other. Never reports train-set R² as the headline number.

- **Layers probed**: trunk `x, h0, h1_mid, h1, h2_mid, h2`; heads
  `z_head_hidden, z_out, mu_15` (parallel branches off `h2`); decoder
  `log_spec, m_ab_recon` (consumes **both** branches:
  `theta_full = [z_pred, theta_15]`).
- **Two depth ladders**, one per question, both drawn as two lines sharing
  the trunk and forking at `h2` (**solid = VAE branch, dashed = z
  branch**):
  - **HEAD ladder** (redshift + SPS figures): depth 1 = the head outputs,
    where the model reads out `(z_pred, σ_z)` / `mu_15`. The decoder is
    off the z inference path — it *consumes* the prediction — so it is
    deliberately not shown on these figures.
  - **DECODER ladder** (input-property figure, i mag / g−r): depth 1 =
    the decoder's last layer `m_ab_recon`, with `log_spec` as the
    decoder's intermediate — what survives the full round trip.
  The SPS figure puts all 15 parameters on one axes: hue = physical group
  (stellar gray, dust orange, gas/IGM aqua, AGN violet, SFH blue),
  lightness step + marker = parameter. Missing/all-NaN layers (e.g.
  `z_head_hidden` for `old` checkpoints, new taps in older npz files) are
  skipped.
- **Targets**: `z_true`, `log_sigma_z`, `i_ref`, `g_minus_r`, plus each of
  the 15 SPS parameters (from `theta_full[:, 1:16]`, ordered per
  `PARAM_NAMES_15`; zred column skipped as redundant with `z_true`).
- **Guards**: activations standardized on train before ridge; near-constant
  targets (collapsed SPS params) reported as NaN instead of exploding R²;
  rows with NaNs dropped per split; alpha chosen by internal CV over
  `logspace(-3, 3, 13)`; activation dims near-dead on train are dropped and
  scaled features clipped to ±10 train-σ, so post-ReLU units the train split
  never activates can't explode the held-out predictions (diagnosis in
  [../PLAN.md](../PLAN.md), Failure 14).
- **Robust z metric**: for the `z_true` target the CSV also carries
  `sigma_nmad` of the probe predictions — R² on redshift is dominated by
  rare catastrophic outliers (the model's own z_pred only reaches R² = 0.26
  on dp1_v4 test despite σ_NMAD = 0.030), so R² alone misranks layers on z.
- **Outputs** (in `--out-dir`): `probe_r2.csv` (incl. `depth_heads` /
  `depth_decoder` ladder positions), `probe_r2_redshift.png` (z + log σ_z,
  HEAD ladder — the target-goal figure), `probe_r2_photometry.png`
  (i mag + g−r, DECODER ladder — the input-data-properties figure),
  `probe_r2_sps_params.png` (all 15 SPS params on one axes, HEAD ladder,
  grouped-hue encoding as above).

## Current experiment: `mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20`

The promoted z-separated model (`model_version: new`, GAAP LSST-only,
σ_NMAD ≈ 0.030 on dp1_v4 test). Its architecture makes the probe question
sharp: `vae_head` reads `[h.detach(), z_pred.detach()]`, so **no
reconstruction gradient ever reaches the trunk or z_head** — any information
in `mu_15` that helps decode z or σ_z beyond what `h2` holds was put there by
the SED reconstruction objective alone.

Catalogs (from the checkpoint's `config.yaml`, same pre-split files used in
training). The probe splits match the model's **exactly**: probes are fit on
the 6,101 galaxies the model took gradient steps on (`--no-cuts --split
train` replicates `train_dp1.py`'s seed-42 val carve-out on the raw file —
`train_dp1.py` applies no refExtendedness/valid-z cuts) and scored on the
full 2,905-row test file the model was tested on.

- train: `obs_catalog/data/83_training_v4_match_ecdfs_sitcomtn154_*/dp1_matched_v4_train.hdf5`
- test: `obs_catalog/data/84_test_v4_match_ecdfs_sitcomtn154_*/dp1_matched_v4_test.hdf5`

**Frozen-decoder pinning**: the 2026-08-12 Inoue zmax-6.5 speculator
retrain replaced `speculator/trained/Inoue_IGM` on disk with different PCA
dimensionalities, which used to crash `PhotozMLPVAE.load()` for every
pre-retrain checkpoint. `load()` now restores the checkpoint's *own*
frozen-decoder buffers when disk shapes mismatch (prints a NOTE), so these
extractions decode with the exact decoder the model was trained against.

Commands (env: `~/miniforge3/envs/WL_ML_Challenge/bin/python`, run from
`/astro/users/lindajin`; cap BLAS threads — unlimited threads on a loaded
head node stretched this ~1-min fit past 10 min):

```bash
PY=~/miniforge3/envs/WL_ML_Challenge/bin/python
CKPT=photoz_mlpvae/trained/mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20/best.pt
OUT=photoz_mlpvae/representation/mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20

$PY photoz_mlpvae/scripts/extract_activations.py --checkpoint $CKPT \
    --catalog obs_catalog/data/83_training_v4_match_ecdfs_sitcomtn154_27184412354f233d1b9f512bdf350d58/dp1_matched_v4_train.hdf5 \
    --no-cuts --split train \
    --out $OUT/probe_activations_train.npz

$PY photoz_mlpvae/scripts/extract_activations.py --checkpoint $CKPT \
    --catalog obs_catalog/data/84_test_v4_match_ecdfs_sitcomtn154_1ec71937ff3e4a0a8283535193764ec7/dp1_matched_v4_test.hdf5 \
    --no-cuts \
    --out $OUT/probe_activations_test.npz

OMP_NUM_THREADS=4 $PY photoz_mlpvae/scripts/fit_linear_probes.py \
    --train-npz $OUT/probe_activations_train.npz \
    --test-npz  $OUT/probe_activations_test.npz \
    --out-dir   $OUT/probes
```

Outputs live under `representation/<run_name>/` (npz's + `probes/`), keeping
this study separate from the training artifacts in `trained/<run_name>/`.

## Reading the results

- `z_true` R² rising through h0→h2 and peaking at `z_head_hidden` = the trunk
  builds z information; `mu_15` matching or exceeding `h2` on `z_true` /
  `log_sigma_z` would mean the SED latent retains (or sharpens) the
  z-relevant information despite being trained only on reconstruction.
- SPS params decodable from `mu_15` but *not* from `h2`/`x` = information the
  VAE branch genuinely adds beyond photometry — the "new information"
  candidate. Decodable already from `x` = re-expressed photometry.
- `log_sigma_z` is the uncertainty half of the question: if it probes better
  from `mu_15` than from `z_head_hidden`, the physically-bounded latent is
  informing the error budget, not just the point estimate.
- **Circularity caveat**: the SPS "targets" are `theta_full` — the model's
  own output, a deterministic transform of `mu_15` (+ `z_pred`). `mu_15`
  decoding them near-perfectly is trivial; the informative comparisons are
  `h2` vs `x` (how much SPS structure the trunk builds from photometry) and
  external targets (`z_true`, `i_ref`, `g_minus_r`).

## First results — zsep_v4, exact model train→test splits (2026-08-12)

Probe battery: 19 targets × 11 layers, guards on, fit on the model's 6,101
gradient-step galaxies, scored on its full 2,905-galaxy test set. Headline
numbers (`probes/probe_r2.csv`, depth-ladder plots in `probes/`; h1/h1_mid
and h2_mid omitted here for width — see the CSV):

| target | x | h0 | h2 | z_head_hidden | z_out | mu_15 | log_spec | m_ab_recon |
|---|---|---|---|---|---|---|---|---|
| z σ_NMAD (probe) | 0.174 | 0.083 | **0.069** | **0.054** | 0.040 | 0.090 | 0.051 | 0.290 |
| log σ_z R² | 0.539 | 0.694 | 0.897 | **0.966** | 1.000 | 0.672 | 0.791 | 0.474 |
| i mag R² | 1.000 | 0.998 | 0.985 | 0.723 | 0.551 | 0.733 | 0.641 | 0.817 |
| g−r R² | 1.000 | 0.999 | 0.986 | 0.581 | 0.132 | 0.844 | 0.133 | 0.674 |

(`z_out` = the readout itself, so its log σ_z probe is trivially 1.000 and
its z probe ≈ the model's own performance — it benchmarks the layers
upstream of it rather than adding information. Its z σ_NMAD of 0.040
vs. the model's 0.030 reflects the ridge probe optimizing MSE, not robust
scatter.)

Takeaways so far:

1. **The trunk builds z information monotonically** (σ_NMAD 0.174 → 0.069
   through the trunk), and `z_head_hidden` probes best (0.054, approaching
   the model's own 0.030). Sanity checks pass: log σ_z probes at 0.966 from
   `z_head_hidden` (it *is* a linear readout of that layer); i and g−r probe
   at 1.000 from `x` (they are inputs).
2. **The SED latent does not add linearly-decodable z or σ_z information
   beyond the trunk**: `mu_15` is worse than `h2` on both z (σ_NMAD 0.090 vs
   0.069) and log σ_z (0.672 vs 0.897). Architecturally it can't add any —
   `vae_head` reads `[h.detach(), z_pred.detach()]` — so this is the
   expected ceiling; the observation is that 15 physically-bounded dims
   *retain most* of the 512-dim trunk's z signal while reorganizing it into
   SPS coordinates.
3. **The trunk already encodes most SPS structure the vae_head expresses**:
   `h2` decodes theta at R² 0.8–0.95 for most params vs 0.1–0.6 from raw
   photometry `x` (mu_15's near-1.0 values are circular, see caveat above).
4. **The decoder is a lossy re-projection, and the spectrum tap localizes
   where**: the reconstructed rest-frame spectrum `log_spec` still carries
   z almost perfectly (σ_NMAD 0.051 — unsurprising, `z_pred` is literally
   one of the Speculator's 16 inputs and IGM absorption is z-dependent)
   and retains SED-shape params at R² ≈ 1 (dust_index 1.000, logzsol
   0.997). The collapse happens at the **FilterConv integration step**:
   spectrum → 10 broadband mags drops z to σ_NMAD 0.290 and most SPS
   params to R² 0.1–0.6. Conversely the *observed-frame* color g−r is
   nearly absent from the rest-frame spectrum (R² 0.133) and only emerges
   after FilterConv redshifts it (0.674). This is expected forward physics,
   **not** evidence that reconstruction confused z training — full
   analysis in [../PLAN.md](../PLAN.md), "Decoder-step probe degradation".
5. Curiosity: `z_head_hidden` has 83/128 dims dead on the train split — the
   z branch effectively uses ~45 dims. The mid-ResBlock taps (`h1_mid`,
   `h2_mid`) probe slightly *worse* than their block outputs on photometric
   targets — the pre-residual-add state is less linearly readable, and the
   skip connection restores it.

**Open next steps** (the non-circular version of the "new information"
question): probe against *external* ground truth (e.g. matched COSMOS
stellar masses), and repeat the battery across checkpoints that differ in
the SED objective — no-zsup ablation (photoz_vae v12), synth-pretrain vs
from-scratch, old (entangled) vs new (z-separated) — to see whether the SED
objective changes what the *trunk* learns, which is the only path by which
it could help z here.

## dp2_v3: MLP-only vs MLP+VAE on identical rows (2026-08-12)

The comparison PLAN.md anticipated (Failure 13 round 3, item 5):
`mlpvae_dp2_v3_lsst_gaap1p0` saves both `phase1_end.pt` (epoch 60 — pure
z-supervised MLP, vae_head still synth-initialized, recon never on) and
`best.pt` (epoch ~597, full MLP+VAE joint training). Both probed on the
SAME rows: a 50k-row subsample (seed 0) of the model's 453,321
gradient-step galaxies (`--cuts-then-split --split train` replicates
train_dp2.py's cuts-first-then-split order) and the full 17,278-row DP2
SOM test set. Outputs in `mlpvae_dp2_v3_lsst_gaap1p0/{best,phase1_end}/`.

Key numbers (test-split probes; z rows are σ_NMAD, others R²):

| target @ layer | phase1_end (MLP-only) | best (MLP+VAE) |
|---|---|---|
| z @ h2 (trunk) | **0.074** | **0.077** |
| z @ z_head_hidden | 0.079 | 0.066 |
| z @ mu_15 | 0.077 | 0.085 |
| log σ_z @ mu_15 | 0.713 | 0.825 |
| i mag @ mu_15 | 0.766 | 0.895 |
| i mag @ m_ab_recon | 0.456 | **0.933** |
| g−r @ m_ab_recon | 0.287 | 0.615 |

Takeaways:

1. **The VAE phase adds no z information to the shared trunk**: z
   decodability at `h2` is statistically identical before and after joint
   training (σ_NMAD 0.074 vs 0.077) — the cleanest evidence yet for the
   study's core question: same run, same architecture, identical rows,
   540 epochs of joint training apart. The trunk's z content had already
   saturated by the end of the 60-epoch warmup.
2. **The z readout keeps improving through Phase 2** (`z_head_hidden`
   0.079 → 0.066), but that is the continued z-NLL training of the head —
   a path the VAE objective cannot touch (`h.detach()`).
3. **What joint training actually does is align the SED latent and decoder
   with real photometry**: the reconstruction goes from synth-initialized
   garbage to photometry-matching (i mag @ `m_ab_recon` 0.456 → 0.933,
   g−r 0.287 → 0.615), `mu_15` picks up photometric fidelity (i mag
   0.766 → 0.895) and aligns better with the model's own σ_z
   (0.713 → 0.825) — while its z decodability *slightly drops*
   (0.077 → 0.085).
4. DP2 caveat: probe σ_NMAD from `z_out` (0.11–0.12) sits well above the
   layer probes on this dataset — DP2's heavier catastrophic-outlier tail
   makes the MSE-optimal linear recalibration shrink harder, so the
   `z_out` point under-benchmarks here.
