"""
photoz_mlpvae_mdn_perpop.py
============================
Per-population variant of the MDN-z PhotozMLPVAE (model/photoz_mlpvae_mdn.py):
each of the K redshift-mixture components gets its OWN SPS/VAE decoder head
and its OWN prior in the 15-dim SPS latent, instead of all K components
sharing one `vae_head` conditioned on the collapsed marginal `z_pred`. See
`.claude/plans/the-k-3-mdn-archetecture-polymorphic-toast.md` for the full
design rationale: the shared-decoder design silently drops the losing
mixture mode's reconstruction signal entirely (only the single highest-
density mode ever conditions the SPS decoder), and any per-component
reconstruction scheme must reuse the same closed-form posterior
responsibility already validated for the z-NLL rather than a hard/
Gumbel-Softmax component pick (this exact codebase already found a hard
per-step component pick for z itself starves `log_pi`'s gradient to a
stuck uniform 1/K -- see `photoz_mlpvae_mdn.py`'s `loss()` docstring).

Architecture delta vs. `photoz_mlpvae_mdn.py` (identical otherwise -- same
shared trunk / z_head / z_mixture / z_point / z-NLL / outlier_head /
i-mag BPZ z-prior):

    - `vae_head` (one shared `Linear(width+1, 30)`) -> `vae_heads`: an
      `nn.ModuleList` of K independent `Linear(width+1, 30)` heads. Head k
      is conditioned on `[h.detach(), z_k_phys.detach()]`, where
      `z_k_phys = z_m_i * softplus(reparameterize(mu_mix, log_var_mix)[:, k])`
      is component k's OWN physical redshift anchor (same physical scale/
      units as `z_pred`, which conditions the shared design's single head)
      -- NOT the raw pre-softplus `mu_mix[:, k]` itself, which lives on an
      unbounded logit-like scale the frozen speculator was never calibrated
      against. The reparameterized draw (2026-09-17) means each head trains
      against a range of z consistent with that component's own predicted
      uncertainty rather than one point estimate; `reparameterize()` is a
      no-op returning the bare mean in eval mode. `z_k_phys` is fully
      detached by default (`allow_zk_grad_frac=0.0`, the default and the
      only value used until 2026-09-17), so ordinarily this is pure input
      noising with no gradient path from `decode()` back into the
      z-mixture -- `allow_zk_grad_frac` (0..1) is an A/B knob that reopens
      that path via a straight-through partial-gradient trick (1.0 = fully
      open, equivalent to no detach at all); see PLAN.md for why this is
      expected to hurt (4+ prior "open a detached gradient path" ablations
      across this codebase all came back flat-to-negative, plus a
      mechanism-specific negative result: SED-reconstruction-likelihood
      reranking over MDN mode candidates made z ~4x worse on the
      disagreement subset, diagnosed cause being the frozen decoder's 15
      free params letting it "cheat" a reasonable-looking fit at nearly
      any candidate z).
    - `decode()` runs the frozen speculator/filter_conv on all K
      components at once (reshaped to a single `(batch*K, ...)` call, not
      a Python loop over K) to get per-component reconstructed photometry.
    - `loss()`'s reconstruction and KL terms are responsibility-weighted
      across the K components using
      `gamma_k = softmax(log_pi + log_comp_k(z_spec))`, DETACHED -- the
      exact posterior responsibility the z-NLL term already computes
      inside its own `logsumexp`, reused here rather than recomputed, and
      detached so recon/KL can't reshape `log_pi`/`mu_mix` through a
      second, uncontrolled gradient path (same isolation convention as
      `h.detach()`/`z_pred.detach()` elsewhere in this project, which has
      repeatedly been re-confirmed to matter).
    - Each component's SPS latent is regularized (when `beta>0`; off by
      default, matching the base model) toward its OWN prior
      `N(sps_prior_mu[k], I)` instead of a shared `N(0, I)` -- a FIXED
      (non-learned) buffer. Fixed rather than learned: a learnable
      per-component anchor needs an explicit repulsion term to keep it
      from drifting back toward the other anchors under gradient descent
      (a per-component KL alone doesn't prevent that), which isn't
      implemented here. Components 1..K-1 keep `sps_prior_mu[k] =
      spread[k] * ones(15)` with `spread = linspace(-1.5, 1.5, n_mix)` (the
      same numeric convention already used for `z_head`'s own mu-bias
      spread-init) -- deliberately a uniform shift across all 15 raw dims,
      not a hand-picked archetype direction, since assigning a specific
      sign to each of the 15 physical SPS parameters without real domain
      input would be actively misleading. Component 0 is the exception
      (2026-09-21): it's reserved as a dedicated LBG (Lyman-break dropout)
      specialist -- see the `is_lbg` routing in `z_mixture()`/`forward()`/
      `loss()` below -- so it DOES get a hand-picked physical archetype
      (massive, low-dust, actively-star-forming; see its construction in
      `MLPVAEEncoderMDNPerPop.__init__` for the full derivation and the
      physical targets it was inverted from).
    - Each `vae_heads[k]`'s bias is initialized so it starts by predicting
      `mu_15 ~= sps_prior_mu[k]` (near-zero-gain weights + bias set to the
      component's own prior mean) -- ties initialization and prior
      together so all K components are differentiated from step 0, giving
      the responsibility-weighted reconstruction loss something asymmetric
      to specialize from instead of K identical heads pulled toward
      identical solutions by symmetric gradients.

Because per-component decoding changes `mu_15`/`log_var_15`/`z_raw_15`'s
shape from `(batch, 15)` to `(batch, K, 15)`, this is a genuinely new model
class (not a drop-in state_dict match for `photoz_mlpvae_mdn.py`
checkpoints) -- train it from scratch or warm-start only the shared-trunk/
z-head parameters that still match by name.

Usage
-----
    model = PhotozMLPVAE(speculator_dir, filter_dir, n_mix=3)
    losses = model.loss(x_phot, z_spec)
    z_samples, z_mean, z_std = model.predict_z(x_phot, n_samples=500)
    z_grid, weights, component_pdf, mixture_pdf = model.predict_z_pdf(x_phot)
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from photoz_vae.model.vae_encoder import _ResBlock
from photoz_vae.model.speculator_torch import SpeculatorInoueIGM
from photoz_vae.model.filter_conv import FilterConv

from photoz_mlpvae.model.photoz_mlpvae import (
    PARAM_NAMES, PARAM_NAMES_15, N_PARAMS_15, N_PARAMS_FULL,
    ZRED_IDX_FULL, LOGMASS_IDX_FULL,
    _ZRED_SIGMA_FLOOR, _LATENT_CLAMP,
    constrain_params_15, _I_BAND_IDX,
    _i_mag_from_x, _unstandardize_scaled, _resolve_feat_scaler,
    _ZPRIOR_INIT, fit_zprior, log_p_bpz, predict_z_map)
from photoz_mlpvae.model.photoz_mlpvae_mdn import (
    _mixture_component_log_pdf, _LOG_VAR_CLAMP, _LOG_VAR_15_CLAMP,
    _ZRED_SIGMA_FLOOR_FRAC)

# Neutral i_mag substituted for missing photometry in z_m() -- see that
# method's docstring. 24.0 is a population-typical magnitude for this
# survey depth (roughly the median observed i_mag), chosen so "we don't
# know this galaxy's brightness" maps to an unremarkable/uninformative
# z_m(i) rather than the extreme-bright collapse the raw sentinel used to
# cause.
MISSING_I_MAG_DEFAULT = 24.0

# Soft routing prior strength added to component 0's raw logit for
# is_lbg=True rows -- see z_mixture()'s is_lbg docstring. z_head's last
# layer starts near-zero (xavier_uniform_ gain=0.1, zero bias on the
# logit half), so softmax([6, 0, 0]) ~= [0.9975, 0.0012, 0.0012] at
# initialization -- a strong, but not absolute, default toward the
# specialist for confirmed candidates. Deliberately far below
# _LATENT_CLAMP_Z-scale scores a well-trained gate could in principle
# reach, so genuine learned disagreement (the trunk's own features
# strongly favoring a different component) can still override it as
# training progresses -- unlike the old -1e9 hard mask, this is a prior,
# not a constraint.
_LBG_LOGIT_BIAS = 6.0

# z-mixture's own `mu` clamp (z_mixture()'s log_pi/mu/log_var split, NOT the
# imported `_LATENT_CLAMP` -- that one is shared across every model version
# in photoz_mlpvae.py/_mdn.py/_mdn_perpop.py AND used for BOTH the z-latent
# AND the unrelated 15-dim SPS-parameter latent, so widening it globally
# would also change every OTHER model class's SPS-latent range for no
# reason). z_pred = z_m(i) * softplus(mu), and the reachable ceiling is
# z_m(i) * softplus(_LATENT_CLAMP) = z_m(i) * ~15 under the original value.
# 2026-09-22 diagnosis of mlpvae_mdn_perpop_dp2_lbg_v1's persistent val
# z-NLL plateau: z_m()'s missing-i_mag fix (above) removes the sentinel-
# driven cases, but genuine very-bright REAL detections (e.g. i_mag~10-19,
# rare but present) are deliberately left untouched by that fix (they're
# real astrophysical outliers, not missing data) and still drive z_m(i)
# toward its 1e-3 floor for their own (small but nonzero) i_mag. Direct
# computation over the full DP2 train+val+test population shows this
# population is bimodal, not a smooth tail: raising the clamp from 15 to
# 30 already drops the unreachable count from 15 to 3, and pushing all
# the way to 100 doesn't help at all -- the SAME 3 rows (i_mag=9.83/12.60/
# 13.95, all suspiciously bright for a refExtendedness==1 "galaxy" at this
# survey depth, likely blended/saturated-star artifacts rather than
# confirmed real detections) stay stuck until ~632. 30 covers every
# plausibly-genuine outlier (ratio up to ~27) with margin, and leaves
# those 3 likely-bad-data rows as ordinary high-loss outliers rather than
# stretching the latent's effective dynamic range 20x to chase them.
# Ordinary well-behaved galaxies need |mu|~O(1-3), nowhere near this
# ceiling, and softplus's gradient (sigmoid) doesn't vanish for large
# positive inputs, so widening this ceiling doesn't introduce a dead-zone
# the way widening a clamp on an unbounded-both-directions latent might.
_LATENT_CLAMP_Z = 30.0


class MLPVAEEncoderMDNPerPop(nn.Module):
    """Shared ResBlock trunk feeding an MDN z head and K per-component SPS heads."""

    def __init__(self, n_in: int = 40, width: int = 512,
                 n_latent: int = N_PARAMS_15, n_mix: int = 3,
                 dropout: float = 0.1,
                 zred_sigma_floor: float = _ZRED_SIGMA_FLOOR,
                 zred_sigma_floor_frac: float = _ZRED_SIGMA_FLOOR_FRAC,
                 detach_trunk_for_vae: bool = True,
                 allow_zk_grad_frac: float = 0.0,
                 use_colors: bool = True,
                 use_z_prior_sampling: bool = False,
                 z_sample_method: str = "normal",
                 router_noise_std: float = 0.0,
                 interloper_prior_archetype: bool = False,
                 zprior_params: dict = None,
                 feat_mean=None, feat_scale=None):
        super().__init__()
        self.interloper_prior_archetype = interloper_prior_archetype
        self.n_latent = n_latent
        self.n_mix    = n_mix
        self.zred_sigma_floor = zred_sigma_floor
        self.zred_sigma_floor_frac = zred_sigma_floor_frac
        self.detach_trunk_for_vae = detach_trunk_for_vae
        if not (0.0 <= allow_zk_grad_frac <= 1.0):
            raise ValueError(f"allow_zk_grad_frac must be in [0, 1], got {allow_zk_grad_frac}")
        self.allow_zk_grad_frac = allow_zk_grad_frac
        if router_noise_std < 0.0:
            raise ValueError(f"router_noise_std must be >= 0, got {router_noise_std}")
        # Noisy-gating exploration (Shazeer+17 "Outrageously Large Neural
        # Networks", arXiv:1701.06538 -- fixed-scale variant, not their
        # learned per-example noise head): Gaussian noise added to
        # z_mixture()'s raw router logits, TRAINING ONLY (self.training
        # gate lives in z_mixture() itself). Mutable post-construction
        # (the training script anneals it epoch-by-epoch by assigning
        # this attribute directly, same convention as beta/lam_r/lam_z
        # being recomputed per-epoch rather than baked into the model) --
        # not a registered buffer, so it does NOT round-trip through
        # save()/load()'s state_dict; save() persists the run's INITIAL
        # value in encoder_config for provenance only. Rationale: without
        # this, a component that loses the responsibility race early
        # (pure init luck, before the trunk has learned anything
        # informative) can go permanently under-trained -- the same
        # gradient-starvation failure mode this file's module docstring
        # already documents for the *hard*-pick z_point() case, just
        # unaddressed here for the soft router.
        self.router_noise_std = router_noise_std
        self.use_colors = use_colors
        self.use_z_prior_sampling = use_z_prior_sampling
        if z_sample_method not in ("normal", "sample", "risk"):
            raise ValueError(f"z_sample_method must be 'normal', 'sample', or 'risk', got {z_sample_method!r}")
        self.z_sample_method = z_sample_method
        n_scaled = 2 * (n_in // 4)
        feat_mean_t  = torch.zeros(n_scaled) if feat_mean  is None else torch.as_tensor(feat_mean,  dtype=torch.float32)
        feat_scale_t = torch.ones(n_scaled)  if feat_scale is None else torch.as_tensor(feat_scale, dtype=torch.float32)
        if feat_mean_t.shape != (n_scaled,) or feat_scale_t.shape != (n_scaled,):
            raise ValueError(
                f"feat_mean/feat_scale must have shape ({n_scaled},) to match "
                f"x_phot's scaled portion (n_in={n_in} -> n_bands={n_in // 4}); "
                f"got {tuple(feat_mean_t.shape)}/{tuple(feat_scale_t.shape)}")
        self.register_buffer("feat_mean", feat_mean_t)
        self.register_buffer("feat_scale", feat_scale_t)

        zprior_params = zprior_params or _ZPRIOR_INIT
        for name, val in zprior_params.items():
            self.register_buffer(f"zprior_{name}", torch.tensor(float(val), dtype=torch.float32))

        self.input_proj = nn.Sequential(
            nn.Linear(n_in, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            _ResBlock(width, dropout),
            _ResBlock(width, dropout),
        )
        self.z_head = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * n_mix),
        )
        # Dedicated catastrophic-outlier classifier head -- see loss()'s
        # lam_outlier docstring. Always reads h.detach(), same rationale as
        # photoz_mlpvae_mdn.py's outlier_head.
        self.outlier_head = nn.Sequential(
            nn.Linear(width, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        nn.init.xavier_uniform_(self.z_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.z_head[-1].bias)
        with torch.no_grad():
            spread = torch.linspace(-1.5, 1.5, n_mix)
            self.z_head[-1].bias[n_mix:2 * n_mix] = spread
        nn.init.xavier_uniform_(self.outlier_head[-1].weight, gain=0.1)
        nn.init.zeros_(self.outlier_head[-1].bias)

        # Per-component SPS prior anchor -- FIXED (not learned), one row per
        # mixture component. Components 1..K-1 keep the deliberately-
        # uninformative uniform-shift placeholder (see module docstring for
        # why a hand-picked archetype was avoided there). Component 0 is
        # reserved as the dedicated LBG (Lyman-break dropout) specialist
        # (see z_mixture()'s is_lbg routing above), so IT gets a hand-picked
        # physical archetype: massive (logmass=10.5 dex, safely above the
        # >10 dex "massive" LBG threshold), modest/sub-average dust
        # (dust2=0.2, dust1_fraction=0.8 -- below this repo's own SED-
        # generation prior means of 0.3/1.0, see
        # sed_generation/scripts/generate_sed_dataset.py's sample_prior),
        # Calzetti-like dust_index=0, sub-solar logzsol/gas_logz=-0.5,
        # negligible AGN (fagn=1e-3), fiducial igm_scale=1.0 (a line-of-
        # sight effect, not intrinsic to the galaxy), and logsfr_ratios
        # biased +0.3 in every one of the 6 non-parametric SFH bins -- a
        # modest, uniform recent-SFR-dominated (rising-toward-the-present)
        # star-formation history. This qualitatively encodes "hot,
        # O/B-star-dominated rest-frame UV, main-sequence actively star-
        # forming" without literally targeting the sSFR(z)=0.2(1+z)^1.5
        # Gyr^-1 relation: that relation is redshift-dependent (the SFH
        # age-bin edges themselves are built from tuniv(z), see
        # _build_agebins() in generate_sed_dataset.py) but this fixed
        # per-component anchor cannot vary per-galaxy-z, so an exact match
        # is only possible at one chosen representative z, not the whole
        # population -- a rising-SFH bias is the honest qualitative analog.
        # Values are RAW (pre-constrain_params_15) latent coordinates,
        # each dim's own sigmoid/tanh/log10 transform inverted to hit the
        # physical targets above exactly (round-trip verified through
        # constrain_params_15's own formulas):
        #   logmass=10.5, logzsol=-0.5, dust2=0.2, dust1_fraction=0.8,
        #   dust_index=0.0, gas_logz=-0.5, fagn=1e-3, agn_tau=20.0,
        #   igm_scale=1.0, logsfr_ratios_{0..5}=+0.3 each.
        _lbg_prior_mu_raw = torch.tensor([
            0.4700,   # logmass
            0.3769,   # logzsol
           -3.1781,   # dust2
           -0.7538,   # dust1_fraction
            0.4377,   # dust_index
            0.2177,   # gas_logz
           -0.6931,   # fagn
           -0.3590,   # agn_tau
           -0.4055,   # igm_scale
        ] + [0.0500] * 6, dtype=torch.float32)  # logsfr_ratios_0..5

        # Component 1's OPTIONAL hand-picked interloper archetype
        # (2026-09-24, A/B alternative to the default flexible/uninformative
        # placeholder -- see is_interloper's docstring in z_mixture() for
        # why flexible is the DEFAULT: interlopers are photometrically
        # heterogeneous, so a narrow archetype risks being actively
        # misleading the same way it would for components 2..K-1). Gated
        # behind interloper_prior_archetype (default False) purely to A/B
        # against that default -- only meaningful when n_mix >= 2 AND the
        # caller also routes is_interloper to component 1 (see
        # z_mixture()/PhotozMLPVAE, --interloper-component in
        # scripts/train_dp2_lbg_perpop.py).
        #
        # Physical targets (old/low-mass/DIM/dusty low-z classic Balmer-break
        # LBG-color-cut contaminant -- a REMARKABLE/strong 4000-Angstrom/
        # Balmer break mimics the Lyman break's color signature): logmass=
        # 8.0 (2026-09-26, strengthened from 9.0 -- a clearly sub-L* dwarf,
        # matching the "dim" end of the strong-Balmer-break low-z population
        # v7 vs. v4 barely moved, vs. component 0's 10.5), logzsol=-0.5
        # (sub-solar, nominal -- same as component 0, not a differentiating
        # axis here),
        # dust2=1.0 and dust1_fraction=1.8 (dusty -- both comfortably above
        # this repo's own SED-generation prior means of 0.3/1.0, and above
        # component 0's modest/sub-average 0.2/0.8), dust_index=0.0
        # (nominal/Calzetti-like attenuation-curve SHAPE -- this axis is
        # NOT what makes the break remarkable, see logsfr_ratios below for
        # that), sub-solar gas_logz=-0.5 (nominal, same as component 0 --
        # an old/quiescent-leaning population has little ongoing nebular
        # emission to begin with, so this axis isn't a meaningful
        # differentiator either way), negligible AGN (fagn=1e-3,
        # agn_tau=20.0, same as component 0), igm_scale=2.0 ("high" as
        # requested -- NOTE this is a fixed, z-INDEPENDENT raw-latent
        # anchor, same caveat as component 0's own igm_scale=1.0 choice,
        # but doubly so here: igm_scale scales the Inoue+14 IGM optical
        # depth computed at the galaxy's OWN redshift, which for a
        # genuinely LOW-z interloper is already near-zero regardless of
        # igm_scale's value -- this differentiates the raw latent
        # coordinate from component 0's, but is not expected to be
        # physically load-bearing for this component's actual SED shape;
        # the dust axes are what should be doing the real reddening work),
        # and logsfr_ratios biased -2.5 (2026-09-26, strengthened from -1.5 --
        # v7's archetype vs. v4's baseline barely moved the strong-Balmer-
        # break/dim/low-z subpopulation specifically) in every one of the 6
        # SFH bins -- the Balmer/4000-break's actual STRENGTH is set by the
        # star-formation history, not by dust_index: a REMARKABLE break needs
        # recent star formation to be essentially ABSENT (so no hot OB-star
        # UV/blue continuum dilutes the break), not just modestly reduced.
        # -2.5 implies a ~316x SFR drop between adjacent bins (10**-2.5), vs.
        # ~30x (10**-1.5) for the original archetype and only ~2x (10**0.3)
        # for component 0's mild +0.3 "rising" bias -- a near-total
        # quenching, not just decisively quenched, star-formation history.
        # Same "cannot vary per-galaxy-z" caveat as component 0's own
        # archetype comment applies throughout. Raw values inverted from
        # these targets through constrain_params_15's own sigmoid/tanh/log10
        # formulas, round-trip verified to the targets above to <1e-3 (in
        # fact <1e-6 in float64):
        _interloper_prior_mu_raw = torch.tensor([
            -1.2040,   # logmass (target 8.0)
             0.3769,   # logzsol
            -1.3863,   # dust2
             0.9445,   # dust1_fraction
             0.4377,   # dust_index
             0.2177,   # gas_logz
            -0.6931,   # fagn
            -0.3590,   # agn_tau
             1.3863,   # igm_scale
        ] + [-0.4437] * 6, dtype=torch.float32)  # logsfr_ratios_0..5 (target -2.5 each)

        sps_prior_mu = spread.unsqueeze(1).expand(n_mix, n_latent).contiguous()
        if n_mix >= 1:
            assert _lbg_prior_mu_raw.shape == (n_latent,), (
                f"LBG prior anchor has {_lbg_prior_mu_raw.shape[0]} dims, "
                f"expected n_latent={n_latent}")
            sps_prior_mu[0] = _lbg_prior_mu_raw
        if interloper_prior_archetype and n_mix >= 2:
            assert _interloper_prior_mu_raw.shape == (n_latent,), (
                f"Interloper prior anchor has {_interloper_prior_mu_raw.shape[0]} dims, "
                f"expected n_latent={n_latent}")
            sps_prior_mu[1] = _interloper_prior_mu_raw
        self.register_buffer("sps_prior_mu", sps_prior_mu)

        self.vae_heads = nn.ModuleList([nn.Linear(width + 1, n_latent * 2) for _ in range(n_mix)])
        for k, head in enumerate(self.vae_heads):
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                # Start each component predicting mu_15 ~= its own prior
                # anchor (log_var half of the bias stays 0) -- ties init and
                # prior together so all K heads are differentiated from step 0.
                head.bias[:n_latent] = self.sps_prior_mu[k]

    def z_mixture(self, h: torch.Tensor, is_lbg: torch.Tensor = None,
                 is_interloper: torch.Tensor = None):
        out = self.z_head(h)
        K = self.n_mix
        logits   = out[:, :K]
        mu       = out[:, K:2 * K].clamp(-_LATENT_CLAMP_Z, _LATENT_CLAMP_Z)
        log_var  = out[:, 2 * K:].clamp(*_LOG_VAR_CLAMP)
        if is_lbg is not None or is_interloper is not None:
            bias = torch.zeros_like(logits)
            if is_lbg is not None:
                bias[:, 0] = is_lbg.to(logits.dtype) * _LBG_LOGIT_BIAS
            if is_interloper is not None and K > 1:
                bias[:, 1] = is_interloper.to(logits.dtype) * _LBG_LOGIT_BIAS
            logits = logits + bias
        if self.training and self.router_noise_std > 0.0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        log_pi   = torch.log_softmax(logits, dim=1)
        return log_pi, mu, log_var

    def prior_params(self):
        """(alpha, beta, z0, km, pm) for the i-mag redshift prior, floor-clamped."""
        alpha = self.zprior_alpha.clamp(min=1e-3)
        beta  = self.zprior_beta.clamp(min=1e-3)
        z0    = self.zprior_z0.clamp(min=1e-4)
        km    = self.zprior_km.clamp(min=0.0)
        pm    = self.zprior_pm.clamp(min=0.0)
        return alpha, beta, z0, km, pm

    def z_m(self, i_mag: torch.Tensor) -> torch.Tensor:
        """Characteristic (turnover) redshift z_m(i).

        Missing i_mag is substituted with 24.0 (a neutral, population-
        typical magnitude for this survey depth -- see MISSING_I_MAG_DEFAULT)
        before evaluating the fit. BUGFIX (2026-09-22): the -50.0 raw
        missingness sentinel, round-tripped through build_features's
        standardize/unstandardize, resolves to an off-scale value (NOT
        literally -50; empirically ~-85 for one run's scaler, and scaler-
        dependent in general) that this formula's linear z0+km*d term
        (d=i_mag-16, only linear below the d>0 power-law turnover) then
        reads as "impossibly bright," collapsing z_m toward its 1e-3 floor
        regardless of the galaxy's actual redshift. Combined with z_pred's
        architectural ceiling z_m*softplus(_LATENT_CLAMP_Z), this made
        ~0.29% of the full DP2 train+val+test population (1,504/520,967,
        99% of them exactly the missingness-sentinel value) PERMANENTLY
        unable to reach their true z_spec, no matter how much training ran
        -- diagnosed via a persistent, non-decreasing val z-NLL plateau
        (~23,300, vs. ~0.27 for a comparable run without this pathology)
        that 280+ epochs never resolved. Detected via the same physical
        sanity range obs_catalog/dataloader.py's own _sanitize_mag_err
        uses (0 < mag < 35) rather than a hardcoded numeric sentinel value,
        since the exact round-tripped sentinel is scaler-dependent.
        Genuine (even if unusually bright, e.g. i_mag~10-14) REAL
        detections are NOT touched here -- those are real astrophysical
        outliers, not missing data, and are instead covered by
        _LATENT_CLAMP_Z's widened ceiling.
        """
        _, beta, z0, km, pm = self.prior_params()
        is_missing = (i_mag <= 0.0) | (i_mag >= 35.0)
        i_mag_safe = torch.where(is_missing, torch.full_like(i_mag, MISSING_I_MAG_DEFAULT), i_mag)
        d  = i_mag_safe - 16.0
        zm = z0 + km * d + pm * d.clamp(min=0.0).pow(beta)
        return zm.clamp(min=1e-3)

    def z_point(self, log_pi, mu, log_var, z_m_i, i_mag=None,
                n_grid: int = 500, z_max: float = 10.0):
        z_grid = torch.linspace(1e-3, z_max, n_grid, device=mu.device, dtype=mu.dtype)

        log_comp = _mixture_component_log_pdf(
            z_grid.view(1, 1, -1), mu.unsqueeze(-1), log_var.unsqueeze(-1),
            z_m_i.view(-1, 1, 1), self.zred_sigma_floor, self.zred_sigma_floor_frac)  # (batch, K, G)
        weights     = log_pi.exp()
        mixture_pdf = (weights.unsqueeze(-1) * log_comp.exp()).sum(dim=1)             # (batch, G)

        # Gated on self.training (not just use_z_prior_sampling): the prior
        # is also applied explicitly in loss()'s `-lam_prior * log_prior`
        # term against this SAME z_pred, so folding it in here again at eval
        # (self.training=False, i.e. every predict_z()/predict_z_pdf()/
        # forward() call after model.eval()) would double-apply it -- once
        # via the training-time loss/point-estimate coupling, once more via
        # this multiplicative posterior reweight on top of an already
        # prior-shaped estimate. Inference always uses the raw mixture
        # likelihood only.
        prior_pdf = None
        if self.use_z_prior_sampling and self.training:
            if i_mag is None:
                raise ValueError("z_point(): use_z_prior_sampling=True needs i_mag")
            alpha, beta, z0, km, pm = self.prior_params()
            prior_pdf = log_p_bpz(z_grid.view(1, -1), i_mag.view(-1, 1),
                                  alpha, beta, z0, km, pm).exp()

        method = self.z_sample_method if self.training else "risk"
        z_pred, sigma_z = predict_z_map(z_grid, mixture_pdf, prior_pdf, sample_method=method)
        z_pred = z_pred.clamp(min=1e-3)

        mu_k    = (weights * mu).sum(dim=1)
        log_v_k = (weights * log_var).sum(dim=1)
        return z_pred, sigma_z, mu_k, log_v_k

    def forward(self, x: torch.Tensor, is_lbg: torch.Tensor = None,
               is_interloper: torch.Tensor = None):
        """Returns z_pred, sigma_z, log_pi, mu_mix, log_var_mix, mu_15 (batch,K,15),
        log_var_15 (batch,K,15), z_m, mu_k, log_v_k, z_k_phys (batch,K), outlier_logit.
        is_lbg/is_interloper: (batch,) bool, optional -- see z_mixture()'s docstring."""
        h = self.input_proj(x)
        h = self.res_blocks(h)

        log_pi, mu_mix, log_var_mix = self.z_mixture(h, is_lbg=is_lbg, is_interloper=is_interloper)
        i_mag = _i_mag_from_x(x, self.use_colors, self.feat_mean, self.feat_scale)
        z_m_i = self.z_m(i_mag)
        z_pred, sigma_z, mu_k, log_v_k = self.z_point(log_pi, mu_mix, log_var_mix, z_m_i, i_mag)

        z_raw_k      = self.reparameterize(mu_mix, log_var_mix)                  # (batch, K)
        z_k_phys_raw = (z_m_i.unsqueeze(1) * F.softplus(z_raw_k)).clamp(min=1e-3)  # (batch, K)
        if self.allow_zk_grad_frac > 0.0:
            z_k_phys = (z_k_phys_raw.detach()
                       + self.allow_zk_grad_frac * (z_k_phys_raw - z_k_phys_raw.detach()))
        else:
            z_k_phys = z_k_phys_raw.detach()

        h_for_vae = h.detach() if self.detach_trunk_for_vae else h
        mu_15_list, log_var_15_list = [], []
        for k, head in enumerate(self.vae_heads):
            cond_k = torch.cat([h_for_vae, z_k_phys[:, k:k + 1]], dim=1)
            out_k  = head(cond_k)
            mu_15_list.append(out_k[:, :self.n_latent].clamp(-_LATENT_CLAMP, _LATENT_CLAMP))
            log_var_15_list.append(out_k[:, self.n_latent:].clamp(*_LOG_VAR_15_CLAMP))
        mu_15      = torch.stack(mu_15_list, dim=1)      # (batch, K, 15)
        log_var_15 = torch.stack(log_var_15_list, dim=1) # (batch, K, 15)

        outlier_logit = self.outlier_head(h.detach()).squeeze(-1)

        return (z_pred, sigma_z, log_pi, mu_mix, log_var_mix, mu_15, log_var_15,
                z_m_i, mu_k, log_v_k, z_k_phys, outlier_logit)

    def reparameterize(self, mu, log_var):
        """Reparameterisation trick: z = mu + eps * exp(0.5 * log_var). Shape-agnostic."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + torch.randn_like(std) * std
        return mu

    def freeze_vae_head(self):
        """Freeze all K SPS heads (stage 1 / z-supervised warmup)."""
        for head in self.vae_heads:
            for p in head.parameters():
                p.requires_grad_(False)

    def unfreeze_vae_head(self):
        """Unfreeze all K SPS heads (stage 2 / joint fine-tuning)."""
        for head in self.vae_heads:
            for p in head.parameters():
                p.requires_grad_(True)


class PhotozMLPVAE(nn.Module):
    """Per-population MDN-z variant of PhotozMLPVAE. Same external API as
    photoz_mlpvae_mdn.py's PhotozMLPVAE, except mu_15/log_var_15/z_raw_15
    carry an extra per-component axis (batch, K, 15) throughout."""

    def __init__(self,
                 speculator_dir: str,
                 filter_dir: str,
                 encoder_width: int = 512,
                 encoder_dropout: float = 0.1,
                 encoder_n_in: int = 40,
                 n_mix: int = 3,
                 zred_sigma_floor: float = _ZRED_SIGMA_FLOOR,
                 zred_sigma_floor_frac: float = _ZRED_SIGMA_FLOOR_FRAC,
                 detach_trunk_for_vae: bool = True,
                 allow_zk_grad_frac: float = 0.0,
                 use_colors: bool = True,
                 use_z_prior_sampling: bool = False,
                 z_sample_method: str = "normal",
                 router_noise_std: float = 0.0,
                 interloper_prior_archetype: bool = False,
                 double_precision: bool = False,
                 zprior_params: dict = None,
                 feat_mean=None,
                 feat_scale=None,
                 scaler=None):
        super().__init__()
        self.use_colors = use_colors
        if scaler is not None:
            feat_mean, feat_scale = scaler.mean_, scaler.scale_
        self._double_precision = double_precision

        self.encoder = MLPVAEEncoderMDNPerPop(
            n_in=encoder_n_in, width=encoder_width,
            n_latent=N_PARAMS_15, n_mix=n_mix, dropout=encoder_dropout,
            zred_sigma_floor=zred_sigma_floor,
            zred_sigma_floor_frac=zred_sigma_floor_frac,
            detach_trunk_for_vae=detach_trunk_for_vae,
            allow_zk_grad_frac=allow_zk_grad_frac,
            use_colors=use_colors,
            use_z_prior_sampling=use_z_prior_sampling,
            z_sample_method=z_sample_method,
            router_noise_std=router_noise_std,
            interloper_prior_archetype=interloper_prior_archetype,
            zprior_params=zprior_params,
            feat_mean=feat_mean,
            feat_scale=feat_scale,
        )

        self.speculator  = SpeculatorInoueIGM(speculator_dir)
        self.filter_conv = FilterConv(filter_dir, self.speculator.wl_rest)

        for p in self.speculator.parameters():
            p.requires_grad_(False)
        for p in self.filter_conv.parameters():
            p.requires_grad_(False)

        if double_precision:
            self.double()

    def encode(self, x_phot: torch.Tensor, is_lbg: torch.Tensor = None,
              is_interloper: torch.Tensor = None):
        """Returns z_pred, sigma_z, log_pi, mu_mix, log_var_mix, mu_15 (batch,K,15),
        log_var_15 (batch,K,15), z_raw_15 (batch,K,15), z_m, mu_k, log_v_k,
        z_k_phys (batch,K), outlier_logit.
        is_lbg/is_interloper: (batch,) bool, optional -- see z_mixture()'s docstring."""
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_m, mu_k, log_v_k, z_k_phys, outlier_logit) = self.encoder(
            x_phot, is_lbg=is_lbg, is_interloper=is_interloper)
        z_raw_15 = self.encoder.reparameterize(mu_15, log_var_15) \
            .clamp(-_LATENT_CLAMP, _LATENT_CLAMP)
        return (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
                mu_15, log_var_15, z_raw_15, z_m, mu_k, log_v_k, z_k_phys, outlier_logit)

    def decode(self, z_k_phys: torch.Tensor, z_raw_15: torch.Tensor) -> torch.Tensor:
        """Returns m_ab, (batch, K, 10) reconstructed AB magnitudes per mixture component.
        Runs the frozen speculator/filter_conv once on a (batch*K, ...) reshape,
        not a Python loop over K."""
        batch, K, n_latent = z_raw_15.shape
        theta_15   = constrain_params_15(z_raw_15.reshape(batch * K, n_latent))
        zred_dec   = z_k_phys.detach().reshape(batch * K, 1)
        theta_full = torch.cat([zred_dec, theta_15], dim=1)
        log_spec, _ = self.speculator(theta_full)
        m_ab = self.filter_conv(log_spec,
                                theta_full[:, ZRED_IDX_FULL],
                                theta_full[:, LOGMASS_IDX_FULL])
        return m_ab.reshape(batch, K, -1)

    def forward(self, x_phot: torch.Tensor):
        """Returns m_ab_recon (batch,K,n_bands), z_pred, sigma_z, mu_15 (batch,K,15),
        log_var_15 (batch,K,15)."""
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_raw_15, _z_m, _mu_k, _log_v_k,
         z_k_phys, _outlier_logit) = self.encode(x_phot)
        m_ab_recon = self.decode(z_k_phys, z_raw_15)
        return m_ab_recon, z_pred, sigma_z, mu_15, log_var_15

    def encode_theta(self, x: torch.Tensor):
        """Standard diagnostic interface: returns z_pred, log_sigma_z,
        theta_full_per_k (batch, K, 16) -- per-component physical SPS params
        from each component's own MAP posterior mean and own z_k_phys (unlike
        the base MDN model's single-component encode_theta())."""
        (z_pred, sigma_z, _, _, _, mu_15, _, _, _, _, z_k_phys, _) = self.encoder(x)
        log_sigma_z = torch.log(sigma_z)
        batch, K, n_latent = mu_15.shape
        theta_15   = constrain_params_15(mu_15.reshape(batch * K, n_latent)).reshape(batch, K, -1)
        theta_full = torch.cat([z_k_phys.unsqueeze(-1), theta_15], dim=-1)
        return z_pred, log_sigma_z, theta_full

    def encode_theta_per_k(self, x: torch.Tensor):
        """Same per-component theta_full_per_k as encode_theta(), plus the
        mixture weights needed to plot/weight each component separately.
        Returns z_pred, log_sigma_z, theta_full_per_k (batch, K, 16),
        weights (batch, K) (softmax(log_pi), i.e. normalized to sum to 1
        across K)."""
        (z_pred, sigma_z, log_pi, _, _, mu_15, _, _, _, _, z_k_phys, _) = self.encoder(x)
        log_sigma_z = torch.log(sigma_z)
        batch, K, n_latent = mu_15.shape
        theta_15   = constrain_params_15(mu_15.reshape(batch * K, n_latent)).reshape(batch, K, -1)
        theta_full = torch.cat([z_k_phys.unsqueeze(-1), theta_15], dim=-1)
        weights    = log_pi.exp()
        return z_pred, log_sigma_z, theta_full, weights

    @torch.no_grad()
    def sample_theta_posterior(self, x, n_samples: int, device: str, rng):
        """Draw posterior samples per galaxy (z AND SPS params both drawn from
        the SAME sampled component k); returns (N_gal, n_samples, 16) numpy float32."""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(device=device, dtype=next(self.parameters()).dtype)
        (_, _, log_pi, mu_mix, log_var_mix, mu_15, log_var_15, z_m, _, _, _, _) = self.encoder(x)
        pi_np     = log_pi.exp().cpu().numpy()
        mu_np     = mu_mix.cpu().numpy()
        std_np    = np.exp(0.5 * log_var_mix.cpu().numpy())
        mu_15_np  = mu_15.cpu().numpy()          # (N_gal, K, 15)
        std_15_np = np.exp(0.5 * log_var_15.cpu().numpy())
        z_m_np    = z_m.cpu().numpy()

        n_gal = x.shape[0]
        all_samples = np.empty((n_gal, n_samples, N_PARAMS_FULL), dtype=np.float32)
        for i in range(n_gal):
            k       = rng.choice(self.encoder.n_mix, size=n_samples, p=pi_np[i] / pi_np[i].sum())
            eps_z   = rng.standard_normal(n_samples).astype(np.float32)
            z_raw_s = mu_np[i, k] + eps_z * std_np[i, k]
            z_s     = z_m_np[i] * np.logaddexp(0.0, z_raw_s)

            # SPS params drawn from the SAME per-sample picked component k,
            # not a single shared posterior -- mu_15_np[i][k]/std_15_np[i][k]
            # index (n_samples, 15) out of (K, 15) via k's own advanced index.
            eps_15   = rng.standard_normal((n_samples, N_PARAMS_15)).astype(np.float32)
            z_raw_15 = mu_15_np[i][k] + eps_15 * std_15_np[i][k]
            with torch.no_grad():
                theta_15 = constrain_params_15(
                    torch.from_numpy(z_raw_15).to(device)
                ).cpu().numpy()
            all_samples[i] = np.concatenate([z_s[:, None], theta_15], axis=1)
        return all_samples

    def loss(self,
             x_phot:    torch.Tensor,
             z_spec:    torch.Tensor,
             lam_z:     float = 20.0,
             lam_r:     float = 1.0,
             beta:      float = 0.0,
             lam_prior: float = 1.0,
             sigma_floor: float = 0.3,
             z_weights: torch.Tensor = None,
             use_nll_z: bool = True,
             lam_outlier: float = 0.0,
             outlier_threshold: float = 0.15,
             outlier_pos_weight: float = None,
             is_lbg: torch.Tensor = None,
             is_interloper: torch.Tensor = None,
             lam_aux: float = 0.0,
             return_grad_probe: bool = False) -> dict:
        """Compute the 3-term MDN loss (closed-form mixture z-NLL + i-mag
        prior, responsibility-weighted reconstruction, responsibility-weighted
        KL against each component's own SPS prior), plus an optional 4th BCE
        term for the dedicated outlier_head.

        Unlike photoz_mlpvae_mdn.py, `recon`/`kl` are no longer single scalar
        terms from one shared decoder -- each of the K components decodes its
        own SPS posterior against the SAME observed photometry, and the two
        terms are combined via `gamma_k = softmax(log_pi + log_comp_k(z_spec))`
        (the exact posterior responsibility the z-NLL term below already
        computes inside its own logsumexp), DETACHED before being used as a
        loss weight so recon/KL can't reshape log_pi/mu_mix through a second
        gradient path -- z-NLL is the only term allowed to move the mixture
        itself. See module docstring for why this reuses z-NLL's exact
        responsibility rather than a hard per-example component pick.

        Also returns `gamma_entropy` (mean posterior-responsibility entropy,
        log(K) is the ceiling at exactly-uniform) as a diagnostic -- watch
        this during training to confirm components are actually specializing
        rather than staying stuck near uniform responsibility (the same
        failure this design's docstring warns a hard component pick would
        risk causing to log_pi itself).

        lam_aux (2026-09-22, reworked 2026-09-23): weight on a K-way
        auxiliary load-balancing loss, added now that is_lbg routing is a
        soft prior rather than a hard mask (see z_mixture()'s docstring) --
        nothing else discourages the K-way gate from collapsing
        responsibility onto too few components.

        2026-09-23: the original formulation here was GShard's (Lepikhin+20,
        arXiv:2006.16668, Algorithm 1 line 13) l_aux = (1/E) sum_e (c_e/S) *
        m_e, adapted to this fully-differentiable soft mixture as f_k
        (mean posterior responsibility, gamma.mean(0)) times m_k (mean
        PRIOR gate weight, log_pi.exp().mean(0)). That formula's implicit
        target is uniform (1/E) traffic for every expert -- right for
        GShard's own use case (don't waste compute capacity) but WRONG
        here: component 0 is deliberately supposed to carry low usage
        (LBGs are a genuinely rare subpopulation), so pushing it toward
        1/K usage fought the entire point of having a specialist (flagged
        as a CAUTION in the previous version of this docstring, never
        fixed). Replaced with a TARGET-AWARE balancing loss instead of a
        uniform one: KL(f_k || p*_k), where p*_k is not a hand-tuned
        constant but read directly off THIS batch's own `is_lbg` labels --
        p*_0 = is_lbg.float().mean() (the batch's own empirical
        LBG-candidate rate) and the remaining 1 - p*_0 split uniformly
        across the other K-1 components (same "no hand-picked archetype
        without real domain input" stance the module docstring already
        takes for components 1..K-1's SPS prior anchors). This keeps
        GShard's actual goal -- prevent outright responsibility collapse
        -- while making the target match this specific rare-subpopulation
        setting instead of fighting it. Falls back to a uniform 1/K target
        when is_lbg is None (e.g. the plain, non-LBG-routed use of this
        model class, where there's no rarity signal to read a target
        from). Keep lam_aux small (default 0.01, unchanged) so this only
        discourages outright collapse rather than pinning f_k to the
        target exactly; default 0.0 (off, matching lam_outlier/lam_prior's
        existing None/0-means-omit convention elsewhere in this file).

        return_grad_probe (2026-10-18, Sever robust-training hook -- see
        scripts/train_dp2_lbg_perpop.py's --sever flag): when True, calls
        `.retain_grad()` on `log_pi`/`mu_mix`/`log_var_mix` (the z_head's
        raw per-mixture-component output, split into its 3 natural slices)
        and adds the UN-reduced, un-detached per-galaxy z-NLL (`per_gal`,
        shape (batch,), pre-`z_weights`/pre-`.mean()`) to the returned
        dict so a caller can backward a per-sample loss of its own choosing
        (e.g. `(z_weights * per_gal).sum()`) and then read
        `log_pi.grad`/`mu_mix.grad`/`log_var_mix.grad` as a cheap, exact
        per-sample gradient matrix (d(per_gal_i)/d(z_head output_i), no
        per-example backward pass needed -- each row of these tensors only
        ever feeds its OWN row's contribution to any sum built from them,
        since z_head/z_mixture/z_point are all row-wise with no
        BatchNorm/attention downstream of z_head). Costs nothing when
        False (default): no retain_grad calls, no extra dict entries.
        """
        (z_pred, sigma_z, log_pi, mu_mix, log_var_mix,
         mu_15, log_var_15, z_raw_15, z_m, mu_k, log_v_k,
         z_k_phys, outlier_logit) = self.encode(x_phot, is_lbg=is_lbg, is_interloper=is_interloper)
        if return_grad_probe and log_pi.requires_grad:
            log_pi.retain_grad()
            mu_mix.retain_grad()
            log_var_mix.retain_grad()
        m_ab_recon = self.decode(z_k_phys, z_raw_15)  # (batch, K, n_bands)

        log_comp = _mixture_component_log_pdf(
            z_spec.unsqueeze(1), mu_mix, log_var_mix, z_m.unsqueeze(1),
            self.encoder.zred_sigma_floor, self.encoder.zred_sigma_floor_frac)  # (batch, K)

        if use_nll_z:
            log_mix  = torch.logsumexp(log_pi + log_comp, dim=1)  # (batch,)
            per_gal  = -log_mix
        else:
            per_gal = (z_pred - z_spec) ** 2

        alpha, beta_prior, z0, km, pm = self.encoder.prior_params()
        i_mag = _i_mag_from_x(x_phot, self.use_colors, self.encoder.feat_mean, self.encoder.feat_scale)
        log_prior = log_p_bpz(z_pred, i_mag, alpha, beta_prior, z0, km, pm)
        per_gal   = per_gal - lam_prior * log_prior

        if z_weights is not None:
            loss_z = (z_weights * per_gal).mean()
        else:
            loss_z = per_gal.mean()

        # Posterior responsibility, reused from log_comp above (NOT recomputed
        # differently) and detached -- see loss()'s own docstring above.
        # 2026-09-22: no longer hard-masked by is_lbg -- routing is now a
        # soft prior baked into log_pi itself (see z_mixture()'s is_lbg
        # docstring), so gamma here is the model's genuine, fully-learned
        # posterior responsibility, free to route recon/KL gradient to
        # whichever component the data actually supports.
        gamma = torch.softmax(log_pi + log_comp, dim=1) #.detach()  # (batch, K)

        n_bands, n_c = x_phot.shape[1] // 4, x_phot.shape[1] // 4 - 1
        x_real = _unstandardize_scaled(x_phot, self.encoder.feat_mean, self.encoder.feat_scale)
        if self.use_colors:
            target     = torch.cat([x_real[:, :n_c], x_real[:, 2 * n_c:2 * n_c + 1]], dim=1)
            target_err = torch.cat([x_real[:, n_c:2 * n_c], x_real[:, 2 * n_c + 1:2 * n_c + 2]], dim=1)
            valid      = 1.0 - torch.cat([x_real[:, 2 * n_c + 2:3 * n_c + 2],
                                          x_real[:, 4 * n_c + 2:4 * n_c + 3]], dim=1)
            recon      = torch.cat([m_ab_recon[:, :, :n_c] - m_ab_recon[:, :, 1:n_bands],
                                    m_ab_recon[:, :, _I_BAND_IDX:_I_BAND_IDX + 1]], dim=2)
        else:
            target, target_err = x_real[:, :n_bands], x_real[:, n_bands:2 * n_bands]
            valid  = 1.0 - x_real[:, 2 * n_bands:3 * n_bands]
            recon  = m_ab_recon[:, :, :n_bands]

        _zero = torch.zeros(1, device=x_phot.device).squeeze()
        if lam_r > 0.0:
            sig2       = (target_err ** 2 + sigma_floor ** 2).unsqueeze(1)         # (batch,1,C)
            chi2       = valid.unsqueeze(1) * (recon - target.unsqueeze(1)) ** 2 / sig2  # (batch,K,C)
            n_valid    = valid.sum(dim=1).clamp(min=1.0)                           # (batch,)
            chi2_per_k = chi2.sum(dim=2) / n_valid.unsqueeze(1)                     # (batch,K)
            loss_r     = (gamma * chi2_per_k).sum(dim=1).mean()
            loss_r     = torch.nan_to_num(loss_r, nan=0.0, posinf=0.0)
        else:
            loss_r = _zero

        if beta > 0.0:
            prior_mu  = self.encoder.sps_prior_mu.unsqueeze(0)  # (1, K, 15)
            kl_per_k  = 0.5 * (
                log_var_15.exp() + (mu_15 - prior_mu) ** 2 - 1.0 - log_var_15
            ).sum(dim=2)                                        # (batch, K)
            # BUGFIX (2026-09-21): this summed kl_per_k unweighted across all
            # K components for every row, contradicting this method's own
            # docstring ("responsibility-weighted ... KL") -- every galaxy's
            # KL gradient reached EVERY component's vae_head/mu_15/log_var_15
            # regardless of which component it actually belongs to, unlike
            # loss_r (correctly gamma-weighted two lines below). Found while
            # adding is_lbg routing: an unweighted KL term would leak
            # gradient across the is_lbg=True/False boundary regardless of
            # gamma's routing mask, defeating "component 0 only trains on
            # LBG rows" for any run with beta>0. Affects every prior
            # mdn_perpop checkpoint trained with beta>0 (e.g. the
            # "...beta1..." runs) -- their per-component SPS-prior
            # regularization was not actually per-component; z_pred itself
            # is unaffected (it depends on log_pi/mu_mix/log_var_mix, not
            # mu_15/log_var_15), so σ_NMAD/bias/outlier numbers should be
            # unaffected, but SED-reconstruction quality/calibration per
            # component may have been subtly cross-contaminated.
            loss_kl = (gamma * kl_per_k).sum(dim=1).mean()
        else:
            loss_kl = _zero

        if lam_outlier > 0.0:
            dz_now        = (z_pred.detach() - z_spec) / (1.0 + z_spec)
            outlier_label = (dz_now.abs() > outlier_threshold).to(outlier_logit.dtype)
            pos_weight_t  = (torch.as_tensor(outlier_pos_weight, device=x_phot.device,
                                              dtype=outlier_logit.dtype)
                              if outlier_pos_weight is not None else None)
            loss_outlier = F.binary_cross_entropy_with_logits(
                outlier_logit, outlier_label, pos_weight=pos_weight_t)
        else:
            loss_outlier = _zero

        if lam_aux > 0.0:
            # Target-aware load-balancing loss -- see loss()'s own
            # docstring (2026-09-23 rework) for why this replaced the
            # earlier GShard-style uniform-target formula. f_k is the
            # empirical usage (mean posterior responsibility); target_p is
            # read off this batch's is_lbg/is_interloper labels rather than
            # hand-tuned, falling back to uniform when neither is available.
            # 2026-09-24: extended to also target component 1's usage off
            # is_interloper when given (see z_mixture()'s is_interloper
            # docstring), splitting remaining mass over components 2..K-1.
            f_k = gamma.mean(dim=0)  # (K,) mean posterior responsibility
            K   = f_k.shape[0]
            if is_lbg is not None and K > 1:
                p0 = is_lbg.to(f_k.dtype).mean().clamp(1e-4, 1.0 - 1e-4)
                if is_interloper is not None and K > 2:
                    p1       = is_interloper.to(f_k.dtype).mean().clamp(1e-4, 1.0 - 1e-4)
                    rest     = (1.0 - p0 - p1).clamp(min=1e-4) / (K - 2)
                    target_p = torch.full_like(f_k, rest)
                    target_p[0] = p0
                    target_p[1] = p1
                else:
                    target_p = torch.full_like(f_k, (1.0 - p0) / (K - 1))
                    target_p[0] = p0
            else:
                target_p = torch.full_like(f_k, 1.0 / K)
            loss_aux = (f_k.clamp_min(1e-8) * (f_k.clamp_min(1e-8) / target_p).log()).sum()
        else:
            loss_aux = _zero

        total = (lam_z * loss_z + lam_r * loss_r + beta * loss_kl
                 + lam_outlier * loss_outlier + lam_aux * loss_aux)

        gamma_entropy = -(gamma.clamp_min(1e-12) * gamma.clamp_min(1e-12).log()).sum(dim=1).mean()

        out = {
            "total":         total,
            "z_sup":         loss_z,
            "recon":         loss_r,
            "kl":            loss_kl,
            "outlier":       loss_outlier,
            "aux":           loss_aux,
            "log_sigma_z":   torch.log(sigma_z).mean().detach(),
            "z_m_mean":      z_m.mean().detach(),
            "gamma_entropy": gamma_entropy.detach(),
        }
        if return_grad_probe:
            out["per_gal"] = per_gal  # (batch,), NOT detached/reduced -- see return_grad_probe docstring above
            out["grad_probe_tensors"] = (log_pi, mu_mix, log_var_mix)
        return out

    @torch.no_grad()
    def predict_z(self, x_phot: torch.Tensor, n_samples: int = 500, is_lbg: torch.Tensor = None,
                  is_interloper: torch.Tensor = None):
        """Returns z_samples (full mixture), z_mean/z_std (the model's own
        point estimate) -- identical to photoz_mlpvae_mdn.py's predict_z(),
        since z-space sampling is unchanged by the per-component SPS design.
        is_lbg/is_interloper: (batch,) bool, optional -- see z_mixture()'s
        docstring. At inference (unknown z_spec), pass the SAME color-cut-
        only flag (compute_lbg_mask(..., use_true_z=False)) for BOTH -- see
        is_interloper's docstring for why there's no separate realistic
        variant for it specifically."""
        (z_mean, z_std, log_pi, mu_mix, log_var_mix,
         _, _, z_m, _, _, _, _) = self.encoder(x_phot, is_lbg=is_lbg, is_interloper=is_interloper)

        ks      = torch.distributions.Categorical(logits=log_pi).sample((n_samples,)).T
        mu_s    = mu_mix.gather(1, ks)
        std_s   = torch.exp(0.5 * log_var_mix.gather(1, ks))
        z_raw_s = mu_s + std_s * torch.randn_like(std_s)
        z_samples = z_m.unsqueeze(1) * F.softplus(z_raw_s)

        return z_samples, z_mean, z_std

    @torch.no_grad()
    def predict_z_pdf(self, x_phot: torch.Tensor, z_grid: torch.Tensor = None,
                       n_grid: int = 500, z_max: float = 10.0, is_lbg: torch.Tensor = None,
                       is_interloper: torch.Tensor = None):
        """Identical to photoz_mlpvae_mdn.py's predict_z_pdf() -- the z-mixture
        posterior only depends on log_pi/mu_mix/log_var_mix/z_m, none of which
        the per-component SPS redesign touches.
        is_lbg/is_interloper: (batch,) bool, optional -- see predict_z()'s docstring."""
        (_, _, log_pi, mu_mix, log_var_mix, _, _, z_m, _, _, _, _) = self.encoder(
            x_phot, is_lbg=is_lbg, is_interloper=is_interloper)

        if z_grid is None:
            z_grid = torch.linspace(1e-3, z_max, n_grid, device=x_phot.device)
        z_grid = z_grid.to(device=x_phot.device, dtype=mu_mix.dtype)

        weights   = log_pi.exp()                     # (batch, K)
        sigma_raw = torch.exp(0.5 * log_var_mix)      # (batch, K)

        z   = z_grid.view(1, 1, -1)                   # (1,     1, G)
        zm  = z_m.view(-1, 1, 1)                       # (batch, 1, 1)
        mu  = mu_mix.unsqueeze(-1)                     # (batch, K, 1)
        sig = sigma_raw.unsqueeze(-1)                  # (batch, K, 1)

        ratio = (z / zm).clamp(min=1e-8)
        raw   = ratio + torch.log1p(-torch.exp(-ratio))
        log_pdf_raw = (-0.5 * ((raw - mu) / sig) ** 2
                       - torch.log(sig) - 0.5 * math.log(2.0 * math.pi))
        log_jac       = torch.log(zm) + F.logsigmoid(raw)
        component_pdf = torch.exp(log_pdf_raw - log_jac)    # (batch, K, G)

        mixture_pdf = (weights.unsqueeze(-1) * component_pdf).sum(dim=1)  # (batch, G)

        return z_grid, weights, component_pdf, mixture_pdf

    @torch.no_grad()
    def predict_outlier_risk(self, x_phot: torch.Tensor, threshold: float = 0.15,
                              z_grid: torch.Tensor = None, n_grid: int = 500,
                              z_max: float = 10.0) -> torch.Tensor:
        """Identical convention to photoz_mlpvae_mdn.py's predict_outlier_risk()."""
        _, z_pred, _ = self.predict_z(x_phot, n_samples=1)
        zg, _, _, mixture_pdf = self.predict_z_pdf(
            x_phot, z_grid=z_grid, n_grid=n_grid, z_max=z_max)

        band_halfwidth = threshold * (1.0 + z_pred).unsqueeze(1)          # (batch, 1)
        in_band = (zg.view(1, -1) - z_pred.unsqueeze(1)).abs() <= band_halfwidth
        dz_grid = zg[1] - zg[0]
        total_mass   = mixture_pdf.sum(dim=1) * dz_grid
        in_band_mass = (mixture_pdf * in_band.float()).sum(dim=1) * dz_grid
        return 1.0 - (in_band_mass / total_mass.clamp_min(1e-12))

    @torch.no_grad()
    def predict_z_prior_pdf(self, x_phot: torch.Tensor, z_grid: torch.Tensor = None,
                             n_grid: int = 500, z_max: float = 6.5):
        """Learned i-mag-conditioned BPZ-style prior p(z|i) -- identical to
        photoz_mlpvae_mdn.py's predict_z_prior_pdf(); this is the z-space
        prior (zprior_* buffers), unrelated to the new per-component SPS
        prior (`sps_prior_mu`). See that method's docstring for the
        distinction from `predict_z_pdf()`'s posterior mixture."""
        i_mag = _i_mag_from_x(x_phot, self.use_colors,
                              self.encoder.feat_mean, self.encoder.feat_scale)
        alpha, beta_prior, z0, km, pm = self.encoder.prior_params()

        if z_grid is None:
            z_grid = torch.linspace(1e-3, z_max, n_grid, device=x_phot.device)
        z_grid = z_grid.to(device=x_phot.device, dtype=i_mag.dtype)

        log_prior = log_p_bpz(z_grid.view(1, -1), i_mag.view(-1, 1),
                              alpha, beta_prior, z0, km, pm)
        return z_grid, log_prior.exp()

    @torch.no_grad()
    def predict_params(self, x_phot: torch.Tensor) -> torch.Tensor:
        """Returns theta_full_per_k, (batch, K, 16) posterior-mean physical SPS
        params PER COMPONENT (unlike the base MDN model's single-component
        predict_params()) -- there is no longer one unambiguous "the" params
        estimate once each mixture component owns its own SPS posterior."""
        (z_pred, _, _, _, _, mu_15, _, _, _, _, z_k_phys, _) = self.encoder(x_phot)
        batch, K, n_latent = mu_15.shape
        theta_15 = constrain_params_15(mu_15.reshape(batch * K, n_latent)).reshape(batch, K, -1)
        return torch.cat([z_k_phys.unsqueeze(-1), theta_15], dim=-1)

    @torch.no_grad()
    def predict_outlier_prob(self, x_phot: torch.Tensor) -> torch.Tensor:
        """Identical to photoz_mlpvae_mdn.py's predict_outlier_prob()."""
        *_, outlier_logit = self.encoder(x_phot)
        return torch.sigmoid(outlier_logit)

    def save(self, path: str, scaler=None, col_medians=None):
        """Save model weights + optional preprocessor state."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        payload = {
            "model_state": self.state_dict(),
            "encoder_config": {
                "n_in":     self.encoder.input_proj[0].in_features,
                "width":    self.encoder.input_proj[0].out_features,
                "n_latent": self.encoder.n_latent,
                "n_mix":    self.encoder.n_mix,
                "detach_trunk_for_vae": self.encoder.detach_trunk_for_vae,
                "allow_zk_grad_frac": self.encoder.allow_zk_grad_frac,
                "use_colors": self.use_colors,
                "use_z_prior_sampling": self.encoder.use_z_prior_sampling,
                "z_sample_method": self.encoder.z_sample_method,
                # Persisted for provenance only -- router_noise_std is
                # mutated live during training (see __init__'s docstring)
                # and is typically 0.0 by the time a checkpoint is saved
                # (post-anneal) or at eval (the attribute doesn't gate
                # anything outside self.training anyway).
                "router_noise_std": self.encoder.router_noise_std,
                "interloper_prior_archetype": self.encoder.interloper_prior_archetype,
                "double_precision": self._double_precision,
                "zprior_params": {
                    "z0": self.encoder.zprior_z0.item(),
                    "km": self.encoder.zprior_km.item(),
                    "pm": self.encoder.zprior_pm.item(),
                    "alpha": self.encoder.zprior_alpha.item(),
                    "beta": self.encoder.zprior_beta.item(),
                },
                "feat_mean":  self.encoder.feat_mean.tolist(),
                "feat_scale": self.encoder.feat_scale.tolist(),
            },
            "scaler":      scaler,
            "col_medians": col_medians,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, speculator_dir: str, filter_dir: str,
             device: str = "cpu"):
        """Load model from a saved checkpoint, migrating older config formats."""
        payload = torch.load(path, map_location=device, weights_only=False)
        cfg     = payload["encoder_config"]
        n_in       = cfg.get("n_in", 40)
        use_colors = cfg.get("use_colors", True)
        feat_mean, feat_scale = _resolve_feat_scaler(cfg, n_in, use_colors)
        model   = cls(speculator_dir, filter_dir,
                      encoder_width=cfg["width"],
                      encoder_n_in=n_in,
                      n_mix=cfg.get("n_mix", 3),
                      detach_trunk_for_vae=cfg.get("detach_trunk_for_vae", True),
                      allow_zk_grad_frac=cfg.get("allow_zk_grad_frac", 0.0),
                      use_colors=use_colors,
                      use_z_prior_sampling=cfg.get("use_z_prior_sampling", False),
                      z_sample_method=cfg.get("z_sample_method", "normal"),
                      router_noise_std=cfg.get("router_noise_std", 0.0),
                      interloper_prior_archetype=cfg.get("interloper_prior_archetype", False),
                      double_precision=cfg.get("double_precision", False),
                      zprior_params=cfg.get("zprior_params"),
                      feat_mean=feat_mean, feat_scale=feat_scale)
        model.load_state_dict(payload["model_state"], strict=False)
        model.to(device)
        model.eval()
        return model, payload.get("scaler"), payload.get("col_medians")
