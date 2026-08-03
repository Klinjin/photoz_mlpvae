# photoz_mlpvae — Problem Log & Open Investigations

This file holds the history: diagnosed failures, reverted experiments, open
questions, and the full run-by-run experiment log. For the current
architecture and usage, see [README.md](README.md).

---

## Open problems

### Euclid bands degrade real dp1_v4 fine-tuning despite better synth pretrain (Failure 7)

Like-for-like comparison of `mlpvae_v2_lsst_gaap1p0` (LSST-only, 24-dim
encoder) vs `mlpvae_v2_euclid_gaap1p0` (LSST+Euclid, 40-dim encoder), holding
every training hyperparameter fixed (epochs, lr, warmup, `lam_z`/`lam_z_min`=20
(no drop), `phase2_trunk_lr_frac`=1.0, `phase2_vae_lr_frac`=1.0,
`z_weight_cap`=10, patience=100, seed=42): adding Euclid bands *worsened* test
σ_NMAD (0.0587 → 0.0842), bias (0.0106 → 0.0266), outlier rate (14.32% →
17.69%), and RMS (0.2532 → 0.2615). Only `f_cat` improved (12.05% → 8.33%).

This held even after training a proper Euclid-native synthetic-pretraining
checkpoint (`mlpvae_synth_v1_euclid_gaap1p0`, same recipe as
`mlpvae_synth_v3_no_euclid_gaap1p0` but with `--euclid`) to warm-start from,
instead of reusing the LSST-only synth checkpoint. That Euclid-native synth
checkpoint was itself *more* accurate on synthetic test data than the
LSST-only one (σ_NMAD 0.0184 vs 0.0301) — so the degradation is specific to
fine-tuning on the small real dp1_v4 catalog, not a defect in the Euclid
pretraining itself.

**Suspected root cause (unconfirmed)**: the real dp1_v4 training set is small
(6,101 galaxies). The extra 16 encoder input dims from Euclid mags/errs/colors
likely let the model overfit during real-data fine-tuning; real Euclid
photometry in the DP1-matched catalog may also be noisier/lower-SNR relative
to the noiseless synthetic assumption than LSST photometry is.

**Status**: open — not yet mitigated. Candidate follow-ups: stronger
regularization (dropout/weight decay) specifically on the Euclid input-facing
weights, more real Euclid-matched training galaxies, or a reduced
`phase2_trunk_lr_frac` (as in the Phase-2 NLL oscillation fix below) to slow
overfitting during Phase 2.

### σ_z calibration not re-checked against the current best checkpoint

Badly overconfident on the mistuned-baseline runs (PIT heavily U-shaped,
~27%/50% coverage within 1σ/2σ vs. the ideal 68.3%/95.4% — see
`mlpvae_obs_v6_gaap1p0`). Not yet re-checked against
`mlpvae_v4b_lsst_gaap1p0_init_synth_v1` (the actual best real-data checkpoint
from the mistuned-baseline family, see "Reconstruction-ambiguity → σ_z
coupling" below and the experiment log). If it's still overconfident there,
post-hoc isotonic regression on a held-out val fraction can recalibrate
coverage.

### Reconstruction-ambiguity/σ_z coupling — re-ablate against the correct baseline

An attempted extra training signal for `sigma_z_head` (see "Reconstruction-
ambiguity → σ_z coupling" under Resolved failures) measurably improved σ_z
informativeness and calibration (Spearman(σ_z,|error|) 0.32→0.56, coverage
within 1σ 27%→74%) relative to a *mistuned* baseline (`obs_v6`). It was
reverted because a properly-tuned baseline (`v4b`, using `train_dp1.py`'s own
defaults) beats both the mistuned baseline and the coupling-augmented model
built on it. The coupling's effect was never tested against the properly-tuned
baseline, so it's unconfirmed whether it helps, hurts, or is neutral once the
hyperparameter gap is closed. Re-run the ablation against `v4b`'s
hyperparameters as the baseline, not `obs_v6`'s. Mechanism is preserved in git
history (commit `558d2b2` diff) if revisited.

### Other remaining limitations (future work)

- **Unimodal P(z|photo)**: N(z_pred, σ_z²) cannot represent color-redshift
  degeneracy multimodality. A mixture density network or normalizing flow
  would better represent the full posterior.
- **No n_detected_bands feature**: partially addressed by `frac_detected` and
  `frac_blue_detected`, but explicit integer band counts may help further.
- **Continuous SED never directly trained**: only 10 filter-integrated
  magnitudes are constrained; the SED curve in `test_sed_reconstructions.png`
  is diagnostic and can diverge even when filter points are close. On
  synthetic data, supervised theta loss (lam_theta, proposed) would fix this
  directly.

---

## Resolved failures

### Failure 1 — Phase 2 optimizer reset spikes z_pred (observed in obs_v5)

**Symptom**: Best checkpoint saved during Phase 1. Phase 2 degrades σ_NMAD
from 0.054 → 0.070 immediately and early stopping fires before recovery.

**Root cause**: Phase 2 created a new `Adam` optimizer, discarding trunk/z_head
momentum. Freshly-initialized vae_head KL (~10.5 nats) + fresh reconstruction
gradient drives a large update that pushes z_pred off its Phase 1 solution.

**Fix (obs_v6+)**: `opt.add_param_group()` splices the vae_head parameters into
the *existing* optimizer. Trunk/z_head groups keep Adam momentum. All groups
have LR scaled by `phase2_trunk_lr_frac` before Phase 2 starts.

---

### Failure 2 — High-z trunk collapse (obs_v5, 40.6% of test set)

**Symptom**: 8,126/20,000 test galaxies predicted at exactly z_pred=3.458,
z_std=0.343, spanning wide z_true range with 67% outlier rate.

**Root cause**: Flat z prior → all training weights ≈ 1. Lyman-break galaxies
(z > 3) have many blue bands below detection limit → sparse, near-identical
feature vectors → trunk maps them to the same representation → z_head outputs
its bias-dominated value sigmoid(0.31) × 6.0 ≈ 3.46.

**Fix (obs_v6+)**: After `compute_z_weights`, multiply by `z_boost_factor`
(5.0) for z > `z_boost_threshold` (3.0), re-normalised to mean=1.

---

### Failure 3 — Persistent z_pred mode collapse (obs_v6/v7)

**Symptom**: Multiple test galaxies (z_true = 1.6–5.0) receive bit-for-bit
identical predictions: z_pred=4.1048045, z_std=0.38891384.

**Root cause**: z-boost raises weights for z > 2 but photometry for z > 2.5 is
genuinely degenerate after the Lyman break. z_head minimises average MSE across
the entire high-z confusion cloud. Increasing boost factor alone cannot break
this because the *information* in the photometry is insufficient.

**Partial fix**: Detection features (`frac_detected`, `frac_blue_detected`)
appended to the encoder input (see Failure 5). These provide explicit band-count
signals that break the high-z degeneracy without requiring the trunk to
reverse-engineer it from missingness flags.

---

### Failure 4 — SED reconstruction fails even with correct z (obs_v6/v7)

**Symptom**: `test_sed_reconstructions.png` shows reconstructed photometry
5–8 magnitudes below the observed points even for well-predicted redshifts.

Three contributing causes:

**4a — logmass underestimate**: KL regularisation pushes mu_15 toward zero
→ logmass → Φ(0)×6+7 = 10.0 regardless of true mass. Reconstruction gradient
too weak (lam_r=1 vs lam_z=10) to break the KL degeneracy.

**4b — Multi-task conflict**: Shared trunk trained 10× harder for z than for
SPS parameters. Trunk learns z-optimised features; VAE head receives
representations not built to encode dust, metallicity, or stellar mass.

**4c — sigma_floor too permissive**: sigma_floor=0.3 dominates photometric
errors of 0.1–0.2 mag for faint galaxies, killing the reconstruction gradient.

**Fixes**: Reduce lam_r dominance gap (lam_z=20, lam_r=0.1 with noiseless
synth using lam_r=0.3); lower sigma_floor to 0.3 (kept; further reduction to
0.1 is proposed); detach h from vae_head path (obs_v6 fix).

---

### Failure 5 — Synthetic stripe galaxies / feature degeneracy

**Symptom**: High-z galaxies cluster in horizontal stripes in the z_pred vs
z_true scatter plot. Multiple galaxies with different true redshifts receive
the same z_pred because the trunk maps their near-identical feature vectors to
the same representation.

**Root cause**: Noiseless synthetic photometry means ALL detected bands carry
exactly the same magnitude regardless of noise. Galaxies near the Lyman break
(z > 2) lose their blue bands, producing sparse feature vectors that differ
only in a handful of detected red bands. The 20 missingness flags alone are
insufficient to distinguish z=2.5 from z=4.0 if both have the same detected
bands above the i-band limit.

Additionally, the high i-band SNR cut (i-SNR≥20 in v1/v2) removed 70% of
synthetic galaxies (59,644/200,000 pass vs 78,091 at i-SNR≥5), discarding
faint high-z objects that are actually well-represented in the real DP1 survey.

**Fix (synth_v3+)**:
1. **Detection features**: append `frac_detected` and `frac_blue_detected` to
   the encoder input (+2 dims). These explicit band-count scalars break the
   degeneracy without requiring the trunk to reverse-engineer it from flags.
2. **i-SNR cut lowered from 20 → 5**: includes faint galaxies whose photometry
   can still be modelled (the noise model assigns proper large σ_m to faint bands).
3. **Phase 2 trunk LR = 0.1**: noiseless reconstruction gradients are ×10
   stronger than on real data (no noise floor variance); reducing the trunk LR
   prevents the reconstruction loss from overwriting the z-prediction features.

---

### Failure 6 — Val NLL oscillation at Phase 2 start (lsst_v3)

**Symptom**: At epoch 50, val z NLL spikes to +2 and oscillates wildly (train
NLL smoothly decreases to −1). Val total loss diverges from train by ~20 nats.

**Root cause**: Two compounding factors:
1. **sigma_z_head is cold at Phase 2 start** — Phase 1 uses `use_nll_z=False`
   so sigma_z_head receives zero gradient and exits Phase 1 with synth-calibrated
   σ_z ≈ exp(−2.2) ≈ 0.11. Real DP1 residuals are larger, so the NLL gradient
   immediately pushes σ_z up; with `0.5 × sq_err / σ_z²` still large during
   recalibration, val galaxies with large residuals and small σ_z generate huge
   NLL spikes: `0.5 × 0.09 / 0.012 + (−2.2) ≈ +1.5` per galaxy.
2. **Phase 2 trunk LR = 1.0 (lsst_v3)** — mu_15 features (read by sigma_z_head)
   change dramatically each epoch, so sigma_z_head chases a moving target and
   cannot stabilize calibration.

**Fix (lsst_v4+)**: Set `phase2_trunk_lr_frac = 0.1` and `lam_z_min = 2.0`.
Reduced trunk LR slows mu_15 drift; the λ_z drop gives the VAE gradient budget
without z_pred destabilisation, allowing sigma_z_head to recalibrate gradually.

---

### Failure 7 — see "Open problems" above (still unresolved)

---

### Reconstruction-ambiguity → σ_z coupling: tried, debugged, then reverted because the comparison baseline itself was mistuned (Failure 8)

**Attempted**: gave `sigma_z_head` a second training signal — whether
`theta_15`, re-fit for a small sampled perturbation of z, still reconstructs
the photometry — combined with the calibrated σ_z in quadrature. Full
mechanism (two iterations: v1 shared weights with `sigma_z_head` and had a
runaway train-only reconstruction-loss bug from that sharing; v2 fixed it
with a dedicated, capped `z_ambiguity_head`) is preserved in git history
(commit `558d2b2` diff) if revisited.

**Why reverted**: the comparison used to evaluate it (`mlpvae_obs_v6_gaap1p0`
as baseline, σ_NMAD=0.1133) turned out to itself be trained with a mistuned
recipe — `phase2_trunk_lr_frac=0.02`, `phase2_vae_lr_frac=0.3`,
`sigma_z_warmup=30`, `epochs=500`, chosen to match that one checkpoint rather
than validated as good. `mlpvae_v4b_lsst_gaap1p0_init_synth_v1`
(σ_NMAD=0.074–0.093 depending on run), using `train_dp1.py`'s own defaults
(`phase2_trunk_lr_frac=0.1`, `phase2_vae_lr_frac=1.0`, `sigma_z_warmup=0`,
`epochs=1000`, `patience=200`) substantially outperforms both the baseline
*and* the coupling-augmented model built on top of it. The coupling's own
effect (it did measurably improve σ_z's informativeness and calibration
relative to obs_v6 — Spearman(σ_z,|error|) 0.32→0.56, coverage within 1σ
27%→74%) was never tested against a properly-tuned baseline, so it's
unconfirmed whether it helps, hurts, or is neutral once the hyperparameter
gap is closed.

**Status**: `model/photoz_mlpvae.py` reverted to the pre-coupling
architecture (no `z_ambiguity_head`). See "Open problems" above for the
re-ablation follow-up.

---

### obs_v6/v8 recipe vs v4b

`mlpvae_v4b_lsst_gaap1p0_init_synth_v1` uses `train_dp1.py`'s own defaults
(`phase2_trunk_lr_frac=0.1`, `phase2_vae_lr_frac=1.0`, `sigma_z_warmup=0`,
`epochs=1000`, `patience=200`) and beats `obs_v6`/`obs_v8` by a wide margin
(σ_NMAD 0.093 vs 0.113/0.115) despite otherwise-similar setup — `obs_v6`'s
recipe (`phase2_trunk_lr_frac=0.02`, `phase2_vae_lr_frac=0.3`,
`sigma_z_warmup=30`, `epochs=500`) was mistuned. The overall best real-data
checkpoint in the experiment log is still the LSST-only `mlpvae_v2_lsst_gaap1p0`
at 0.0587 — see Failure 7 above.

v3→v4 key changes: trunk LR 1.0→0.1, λ_z_min 20→2, i-SNR cut 20→5 (more
training galaxies).

---

## Experiment log

| Model | Data | use_gaap | Synth init | Phase2 trunk LR | λ_z_min | σ_NMAD (test) |
|-------|------|----------|------------|-----------------|---------|---------------|
| mlpvae_obs_v3 | old catalog | No | — | 0.02 | 20.0 | — |
| mlpvae_synth_v1_no_euclid_gaap1p0 | synth | Yes | — | 0.1 | — | — |
| mlpvae_synth_v2_no_euclid_gaap1p0_dp1 | synth | Yes | — | 0.1 | 20.0 (no drop) | — |
| mlpvae_v2_lsst_gaap1p0 | dp1_v4 | Yes | synth_v7 | 1.0 | 20.0 (no drop) | **0.0587** |
| mlpvae_v3_lsst_gaap1p0_init_synth_v2 | dp1_v4 | Yes | synth_v2 | **1.0** | **20.0 (no drop)** | — |
| mlpvae_synth_v3_no_euclid_gaap1p0 | synth | Yes | — | **0.1** | **2.0** | in progress |
| mlpvae_v4_lsst_gaap1p0_init_synth_v3 | dp1_v4 | Yes | synth_v3 | **0.1** | **2.0** | pending |
| mlpvae_synth_v1_euclid_gaap1p0 | synth (+Euclid) | Yes | — | 0.1 | 2.0 | 0.0184 (synth test) |
| mlpvae_v2_euclid_gaap1p0 | dp1_v4 (+Euclid) | Yes | synth_v1_euclid | 1.0 | 20.0 (no drop) | 0.0842 |
| mlpvae_v4b_lsst_gaap1p0_init_synth_v1 | dp1_v4 (+Euclid) | Yes | synth_v1_euclid | **0.1 (script default)** | 2.0 | **0.0929** (0.0739 on an earlier run in the same log) |
| mlpvae_obs_v6_gaap1p0 | dp1_v4 (+Euclid) | Yes | synth_v1_euclid | 0.02 | 2.0 | 0.1133 |
| mlpvae_obs_v8_zambig_fixed_gaap1p0 | dp1_v4 (+Euclid) | Yes | synth_v1_euclid | 0.02 | 2.0 | 0.1146 (reverted, see Failure 8) |
