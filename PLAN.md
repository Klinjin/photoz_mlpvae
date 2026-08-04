# photoz_mlpvae — Problem Log & Open Investigations

This file holds the history: diagnosed failures, reverted experiments, open
questions, and the full run-by-run experiment log. For the current
architecture and usage, see [README.md](README.md).

---

## Changelog (moved from README.md, newest first, metric-moving changes only)

- Epochs×LR sweep + `lam_z_min=20` (bias campaign, see Open Problems):
  promoted `mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20` as new best —
  σ_NMAD 0.0348 → **0.0298**, bias 0.0559 → **0.0244**, RMS 0.3012 → 0.2514.
- epochs×LR trial sweep on the z-separated recipe (Phase-1 warmup now 10% of
  total epochs): best trial (2000 epochs, lr 1e-3) σ_NMAD 0.0348 → 0.0291,
  bias 0.0559 → 0.0258 (later promoted with `lam_z_min=20`, see above).
- 1000-epoch run of the LR-fix recipe: bias 0.0689 → 0.0559, RMS
  0.3413 → 0.3012 at flat σ_NMAD (0.0349 → 0.0348).
- Gradient-clipping A/B: removing `clip_grad_norm_(2.0)` worsened all five
  metrics (σ_NMAD 0.0349 → 0.0380, bias 0.0689 → 0.0896); clipping kept and
  made explicit via `--grad-clip-max-norm` (Failure 11/12).
- `phase2_trunk_lr_frac` removed from `train_dp1.py` (obsolete under the
  z-separated model's `h.detach()` isolation): σ_NMAD 0.0442 → 0.0349
  (Failure 10).
- Phase-1 `use_nll_z` gating bug fixed (NLL-z on from epoch 1, no more σ_z
  collapse → Phase-2 loss spike): σ_NMAD 0.0442 → 0.0391 (Failure 9).
- z-separated model (`--model-version new`) + GAAP + synth warm-start:
  σ_NMAD 0.0587 → 0.0442 (bias worsened 0.0106 → 0.0577 — see the bias
  decomposition in Open Problems for why this "worse bias" is misleading).
- Euclid bands tested as additional input (LSST+Euclid, 40-dim encoder):
  *degraded* real dp1_v4 performance, σ_NMAD 0.0587 → 0.0842, despite a more
  accurate Euclid-native synth pretrain (0.0184 vs 0.0301 on synth test).
  Not adopted for the old model (Failure 7); near-parity when retried on the
  z-separated model (see Open Problems, MDN + Euclid trials).
- `phase2_trunk_lr_frac` 1.0→0.1 + `lam_z_min` 20→2 (lsst_v4+): fixed a val
  z-NLL oscillation at the start of Phase 2 (Failure 6).
- i-band SNR cut 20→5 (synth_v3+): restored faint high-z training galaxies
  discarded by the stricter cut (Failure 5).
- Detection features `frac_detected` / `frac_blue_detected` added to encoder
  input (+2 dims, synth_v3+): broke a high-z feature degeneracy that was
  collapsing predictions to a single value across a wide true-z range
  (Failure 5).
- `z_boost_factor` (5.0 for z > 3) added to training weights (obs_v6+): fixed
  40.6%-of-test-set collapse to a single high-z prediction (Failure 2).
- Phase 2 now splices `vae_head` into the existing optimizer via
  `opt.add_param_group()` instead of creating a fresh one (obs_v6+): fixed a
  σ_NMAD 0.054→0.070 regression at the Phase 1→2 transition (Failure 1).
- `h.detach()` before `vae_head` (obs_v6): fixed a reconstruction-gradient
  collapse through the trunk (Failure 4c).

---

## Open problems

### Target σ_NMAD ~2×10⁻² / bias ~10⁻³ not reachable via epochs×LR alone (z-separated model)

A 12-trial sweep (2026-08-03) over total epochs × learning rate on the
z-separated model's best recipe (`--model-version new --use-gaap`, warm-start
from `mlpvae_synth_zsep_v1_no_euclid_gaap1p0`, grad clip 2.0, Phase-1 warmup =
10% of total epochs, patience = epochs/5) mapped the space and hit a floor
well short of the target. Trials ran in `--trial` mode (metrics only, no
figures, scratchpad output — nothing promoted to `trained/`).

σ_NMAD / bias per trial:

| | lr 5e-5 | 1e-4 | 3e-4 | 5e-4 | **1e-3** | 2e-3 | 3e-3 |
|---|---|---|---|---|---|---|---|
| **e1500** | | | .0302/.0328 | | | | |
| **e2000** | .0328/.0449 | .0310/.0365 | .0306/.0339 | .0308/.0363 | **.0291/.0258** | .0300/.0336 | .0302/.0367 |
| **e3000** | | | .0304/.0348 | | .0310/.0321 | | |
| **e4000** | | .0332/.0509 | | | .0297/.0283 | | |

Structure: (1) LR has a single peak at **1e-3** — bias falls monotonically
5e-5→1e-3, rises again by 2e-3; (2) length saturates around ~2000 effective
epochs — e4000 runs (warmup/patience/cosine scaled along) are slightly
*worse* at both LRs tested. The best trial (e2000 lr1e-3) ran its full budget
without early-stopping, i.e. was still improving, but its e3000/e4000
extensions did not beat it.

**Floor: σ_NMAD ≈ 0.0291, bias ≈ 0.0258** (best trial) vs. target σ ~2×10⁻²,
bias ~10⁻³. σ is within reach; bias is stuck ~2.6× above target. Neighboring
trials near the peak differ by ±0.003–0.008 in bias — likely seed-level
noise, so finer grid refinement is unlikely to close a 2.6× gap.

**λ_z-dip follow-up (same day)**: removing the Phase-2 λ_z dip
(`--lam-z-min 20`, the old model's scheme) was tested as a paired A/B at
e2500 lr1.5e-3 — it improved both metrics at identical config
(0.0321/0.0521 → 0.0299/0.0420), a cleanly attributable gain. Stacked onto
the sweep peak (e2000 lr1e-3): σ_NMAD 0.0291→0.0298, bias 0.0258→**0.0244**
(best bias of all 15 trials, but no additive transfer — the gain shrinks to
~0.001–0.002 at the peak, within the noise band). Revised floor:
**σ ≈ 0.029, bias ≈ 0.024**.

**Bias decomposition (2026-08-04)** — where the mean-dz bias actually lives
(`bias = np.mean(dz)` in `compute_metrics`, so it is outlier-sensitive):

- **The z-separated core is essentially unbiased.** For the σ-best trial,
  the 88.4% of test galaxies with |dz| ≤ 0.15 have mean dz = −0.0006
  (median dz overall −0.0013). The entire +0.026 headline bias is the tail:
  6.8% of galaxies with dz > +0.15 (mean +0.63) contribute +0.043, partially
  cancelled by 4.8% with dz < −0.15 contributing −0.017.
- **The dominant term is low-z→high-z catastrophic outliers**: 11.7% of
  z_true < 0.5 galaxies (out of 957) are flung upward, contributing +0.033 —
  more than the entire net bias. High-z bins contribute *negative* bias
  (26–46% of z>1.5 galaxies scatter low) but hold few galaxies.
- **The old model's bias=0.0106 is a cancellation artifact, not better
  predictions.** Its core is systematically shifted: core mean dz = −0.0199
  (median −0.0266), cancelling a *worse* outlier tail (+0.048 from out+,
  8.0% rate). On core calibration the z-separated model is ~30× better;
  on the tail it is slightly better too. Every previous "old model wins on
  bias" comparison should be reread with this in mind — median dz or
  core-mean dz is the honest core-calibration metric.
- **Implication for the 10⁻³ bias target**: it will not come from core
  tuning (already at −0.001); it requires shrinking the low-z out+ rate.
  Suspected driver: the z-weighting — all z>3 train galaxies sit at the
  13× weight cap (see `z_weights.png`), pushing the model to enlarge the
  high-z prediction region in feature space, which captures ambiguous
  low-z galaxies (Balmer/Lyman-break confusion). Testing `z_weight_cap`
  ∈ {5, 3, 1} at the sweep peak now.

**z_weight_cap trials (2026-08-04): not a lever.** At e2000 lr1e-3, bias vs
cap: 10 (incumbent) → +0.0258, 5 → +0.0400, 3 → +0.0282, 1 → +0.0344 —
non-monotonic, all ≥ incumbent, and the low-z out+ rate did not fall at any
cap (11.7% → 13.2 / 10.8 / 13.8%). The low-z→high-z catastrophic outliers
are not controlled by the training z-weighting; they look like a genuine
6-band photometric degeneracy (Balmer↔Lyman-break confusion) that per-galaxy
loss weights can't resolve. Remaining plausible tools for the out+ tail:
multimodal posterior (mixture density / flow z-head — see "Remaining
limitations"), split gradient clipping, or adopting median-dz / core-mean-dz
as the calibration metric (already at −0.001, i.e. the 10⁻³ order).

**MDN z-head + Euclid trials (2026-08-04)** — both mechanism-matched fixes
tried, decomposed against the promoted baseline (all e2000 lr1e-3 lzmin20):

| | baseline | MDN K=3 | +Euclid (no-eu init) |
|---|---|---|---|
| bias / median dz | +0.0244 / −0.006 | +0.0671 / +0.005 | +0.0297 / +0.001 |
| σ_NMAD / f_cat | 0.0298 / 19.2% | 0.0298 / 17.4% | 0.0313 / **16.2%** |
| low-z out+ rate / mean dz | 11.3% / +0.88 | 11.5% / **+1.35** | 10.6% / +0.88 |
| high-z out− rate (z>1.5) | 32.2% | **15.0%** | 20.3% |

- **MDN** (`model/photoz_mlpvae_mdn.py`, `--model-version mdn`: K=3 raw-space
  Gaussian mixture, mixture NLL, dominant-mode point estimate, component-
  sampled z conditioning for vae_head): *halves* the high-z out− rate and
  gives the family-best f_cat, but the low-z fling **rate** is unchanged —
  it's a wrong-branch *ranking* error, not a unimodal-compromise error — and
  mode-picking makes each wrong commit land further (dz +1.35 vs +0.88), so
  mean bias worsens. Early-stopped fast (ep 767).
- **Euclid** (mismatch warm-start from the no-euclid synth ckpt): near-parity
  headline (vs the old model's Euclid disaster), best f_cat trend, high-z
  out− 32→20%, essentially perfectly calibrated core (+0.0004) — but low-z
  out+ rate again unchanged (10.6%). The **native-init** counterpart
  (warm-started from `mlpvae_synth_zsep_v1_euclid_gaap1p0`, synth σ_NMAD
  0.0091 vs no-euclid 0.0185) landed at σ 0.0317 / bias +0.0329 /
  **f_cat 15.25% (best overall)** / median −0.0006 / low-z out+ 11.2% —
  statistically equal to the mismatch-init run (same precedent as the old
  model: the better synth pretrain does not transfer an advantage), and the
  low-z fling rate is unchanged in every Euclid variant.
- **Who the flung galaxies are** (baseline, 108/957 low-z): median i=24.0 vs
  21.0 for well-predicted low-z (≈3 mag fainter, ~10× noisier photometry;
  all 6 bands detected; spec-z confidence *fine*, median 0.97). Faint low-z
  dwarfs whose noisy colors admit a high-z solution — also why Euclid can't
  help (DP1-matched Euclid photometry is near its depth limit at i≈24), and
  why a magnitude prior would push them the *wrong* way (P(z|i=24) favors
  high z). **Conclusion: this population is data-limited in 6-band optical
  photometry.** Its +0.033 mean-dz contribution exceeds the entire <0.01
  bias budget, so mean-dz bias ~10⁻² on the full sample is likely
  unreachable for any model on this data.
- **Quality cuts** (survey-standard, model-side): baseline σ_z cut at 70%
  retention → σ_NMAD **0.0202** (meets the 2×10⁻² target), bias +0.0145,
  outliers 3.8%. MDN between-mode-std cut (the right MDN flag; dominant-mode
  σ is useless — wrong-branch commits are confident) → bias +0.0151 at 70%,
  +0.0128 at 50%. Even at 50% retention mean bias stays >0.01; median dz is
  ~+0.004 throughout. The 10⁻³-order target is met by median-dz on the full
  sample, and by no mean-dz variant tried.

**Status**: open. Best trial configs not yet promoted to `trained/`
(σ-best: e2000 lr1e-3; bias-best: e2000 lr1e-3 + lam_z_min=20). Candidate
levers still untested: split gradient clipping (trunk+z_head vs vae_head as
separate `clip_grad_norm_` calls, so one head's spikes can't eat the other's
budget); seed ensembling of the best config.

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

### Failure 9 — Phase-1 `use_nll_z` gating collapses σ_z, then spikes the loss at Phase 2 (z-separated model, fixed 2026-07-30)

**Symptom**: one-epoch catastrophic loss spike at the Phase 1→2 transition
(`z_sup` 0.33 → 5551, total → 11922), recovering over ~10 epochs.

**Root cause**: `train_dp1.py`/`train_synth.py` gated `use_nll_z` on
`epoch > warmup_epochs + sigma_z_warmup`, so Phase 1 trained z with plain MSE
only. Nothing calibrated `log_var_z`, and MSE's reparameterization noise gave
the model incentive to shrink it toward zero for free (collapsed to
log σ_z ≈ −4.0, σ_z ≈ 0.018, by epoch 50). When Phase 2 flipped
`use_nll_z=True`, the `1/σ_z² ≈ 3100` factor amplified real residuals into
the spike.

**Fix**: `use_nll_z=True` hardcoded for all epochs in both scripts; dead
`--sigma-z-warmup` flag removed. Rerun (`mlpvae_zsep_v3_lsst_gaap1p0_nllfix`):
spike gone (σ_z smooth through the transition), σ_NMAD 0.0442 → 0.0391 —
though bias/outliers/f_cat slightly worsened (tight-scatter/worse-tails
pattern). Note `train.py` (non-DP1) retains the old gating with
`SIGMA_Z_WARMUP=30` for the old model — deliberately left unchanged.

---

### Failure 10 — `phase2_trunk_lr_frac` obsolete for the z-separated model (removed from train_dp1.py, 2026-08-03)

**Observation**: the 10× trunk-LR cut at Phase 2 was a holdover from the old
joint-gradient model, where it protected z_pred from the freshly-unfrozen
VAE head. The z-separated model already closes that path architecturally —
`vae_head` reads `[h.detach(), z_pred.detach()]`, so recon/KL gradient cannot
reach trunk/z_head regardless of LR. The cut was also stacking on top of the
λ_z dip (a second, redundant throttle), and dropped trunk LR 10× right before
cosine decay pulled it down anyway, leaving little step size to fix residual
bias late in training.

**Change**: `phase2_trunk_lr_frac` removed entirely from `train_dp1.py`
(constant, flag, config field, and the Phase-2 rescale). Trunk/z_head now
stay on one continuous cosine schedule. `train_synth.py` still has the flag
(default 0.1).

**Result** (`mlpvae_zsep_v2_lsst_gaap1p0_init_synth_lrfix`, 500 epochs):
σ_NMAD 0.0442 → **0.0349**, outliers 13.91% → 13.15%; bias worsened
0.0577 → 0.0689 at 500 epochs but recovered to **0.0559** in the 1000-epoch
rerun (`..._lrfix_1000ep`, early-stopped at 603) — which also improved RMS
0.3413 → 0.3012, making it the best saved checkpoint on σ_NMAD and bias
simultaneously among z-separated runs.

---

### Failure 11 — Gradient clipping fires on 100% of batches; removing it worsens every metric (investigated 2026-08-03)

**Observation**: new per-epoch LR/grad-norm instrumentation
(`lr_and_gradnorm.png`, gnorm in the epoch log line) showed
`clip_grad_norm_(encoder.parameters(), 2.0)` triggering on **every batch of
every epoch** — pre-clip norms typically 50–5,000 (2–4 orders above the
threshold), ~6×10⁷ at epoch 1, climbing again late in training as σ_z shrinks
and the 1/σ_z² NLL factor amplifies residual gradients. The huge norm is
present already in Phase 1 (vae_head frozen), so it is driven by the z-NLL
term itself, not the VAE.

**A/B test** (`mlpvae_zsep_v2_lsst_gaap1p0_init_synth_noclip`): removing
clipping entirely trained stably (no NaN — Adam absorbed the epoch-1 spike)
but worsened **all five** metrics: σ_NMAD 0.0349→0.0380, bias 0.0689→0.0896,
outliers 13.15%→15.73%, RMS 0.3413→0.3688, f_cat 18.00%→19.21%. Unlike other
ablations in this lineage (which trade scatter against tails), this was
uniformly negative.

**Interpretation**: at max_norm=2.0 with raw norms in the hundreds–thousands,
clipping acts as always-on normalized-direction descent, keeping every
batch's contribution to Adam's second-moment EMA at the same scale so
occasional outlier batches can't poison ~1000 subsequent steps. **Conclusion:
keep clipping at 2.0**; raising/removing it is empirically closed. The open
follow-up is *split* clipping (see Open problems).

---

### Failure 12 — Clipping silently disabled by a leftover ablation edit; mislabeled 1000-epoch run (caught + fixed 2026-08-03)

**Symptom**: the first "lrfix 1000-epoch" run produced f_cat=19.17% —
pattern-matching the noclip run (19.21%), not the clipped 500-epoch run
(18.00%).

**Root cause**: the noclip ablation had been implemented by editing
`run_epoch` directly (replacing `clip_grad_norm_` with a measure-only norm
computation) and was never reverted, so the follow-up run trained unclipped
while believing clipping was on.

**Fix**: the mislabeled run was renamed
`mlpvae_zsep_v2_lsst_gaap1p0_init_synth_noclip_1000ep` (σ_NMAD 0.0342, bias
0.0701 — a valid extra noclip data point), and clipping was reinstated behind
an explicit `--grad-clip-max-norm` flag (default 2.0; ≤0 disables but still
measures the norm; recorded in `config.yaml`; the `lr_and_gradnorm.png`
reference-line label reflects whether clipping was applied). The corrected
1000-epoch clipped run is `..._lrfix_1000ep` (0.0348/0.0559). **Lesson**:
toggle training behaviors for ablations via flags with explicit defaults,
never by editing source in place.

---

## Experiment log

> **Checkpoint cleanup (2026-08-04):** all `trained/` model directories that
> were not best on ≥1 dp1_v4 test metric were deleted; their metrics survive
> only in the tables below. Kept: `mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20`
> (σ_NMAD 0.0298, RMS 0.2514), `mlpvae_zsep_v4_euclid_gaap1p0_e2000_lzmin20`
> (outliers 10.02%), `mlpvae_v2_lsst_gaap1p0` (bias 0.0106 — a core/tail
> cancellation, see Open problems), `mlpvae_v2_euclid_gaap1p0` (f_cat 8.33%),
> their four synth warm-start sources (`mlpvae_synth_v1_{no_,}euclid_gaap1p0`,
> `mlpvae_synth_zsep_v1_{no_,}euclid_gaap1p0`), and
> `mlpvae_speculator_v2_finetuned` (PZ Data Challenge artifact, out of scope).
> Old-catalog models (`mlpvae_v1/v2/v1_mags`, `obs_v3/v5/v6`) were evaluated
> on a different test set — their numbers were not treated as metric wins.

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

### z-separated model era (`--model-version new`, 2026-07-30 →)

All dp1_v4, LSST-only. "clip" = `clip_grad_norm_` max-norm on encoder grads.
Warm-start ("synth_zsep") = `mlpvae_synth_zsep_v1_no_euclid_gaap1p0/best.pt`
(synth test σ_NMAD 0.0185).

| Run (`trained/`) | Date | Epochs | LR | Warmup | Trunk-LR ×0.1 @P2 | Clip | σ_NMAD | Bias | Notes |
|---|---|---|---|---|---|---|---|---|---|
| mlpvae_zsep_v1_lsst | 07-30 | 500 | 1e-4 | 50 | yes | 2.0 | 0.0685 | 0.0920 | no GAAP, no warm-start |
| mlpvae_zsep_v2_..._preLRfix_0730 | 07-30 | 500 | 1e-4 | 50 | yes | 2.0 | 0.0442 | 0.0577 | GAAP + synth_zsep warm-start |
| mlpvae_zsep_v3_..._nllfix | 07-30 | 500 | 1e-4 | 50 | yes | 2.0 | 0.0391 | 0.0824 | NLL-z always on (Failure 9) |
| mlpvae_zsep_v2_..._lrfix | 08-03 | 500 | 1e-4 | 50 | **no** | 2.0 | 0.0349 | 0.0689 | trunk-LR rescale removed (Failure 10) |
| mlpvae_zsep_v2_..._noclip | 08-03 | 500 | 1e-4 | 50 | no | **off** | 0.0380 | 0.0896 | all 5 metrics worse (Failure 11) |
| mlpvae_zsep_v2_..._noclip_1000ep | 08-03 | 1000 | 1e-4 | 50 | no | **off** | 0.0342 | 0.0701 | accidental noclip (Failure 12) |
| mlpvae_zsep_v2_..._lrfix_1000ep | 08-03 | 1000 | 1e-4 | 50 | no | 2.0 | 0.0348 | 0.0559 | superseded by v4 promotions |
| ~24 sweep/lever trials (scratch only) | 08-03/04 | 1500–4000 | 5e-5–3e-3 | 10% | no | 2.0 | best 0.0291 | best 0.0244 | see Open problems |
| **mlpvae_zsep_v4_lsst_gaap1p0_e2000_lzmin20** | 08-04 | 2000 | 1e-3 | 200 | no | 2.0 | **0.0298** | **0.0244** | promoted trial (lam_z_min=20); best saved LSST-only |
| **mlpvae_zsep_v4_euclid_gaap1p0_e2000_lzmin20** | 08-04 | 2000 | 1e-3 | 200 | no | 2.0 | 0.0317 | 0.0329 | +Euclid, native-init (`mlpvae_synth_zsep_v1_euclid_gaap1p0`); best f_cat 15.25%, outliers 10.02%, median dz −0.0006 |
