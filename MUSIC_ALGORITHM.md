# MUSIC Algorithm: Accuracy Improvements and Current Workflow

This document describes the implementation in this repository as of September 8,
2026. It covers the accuracy improvements, the active notebook configuration,
the blind estimation workflow, the use of ground truth, and the remaining
limitations. It describes implemented code, not a proposed method.

## 1. Executive Summary

The active method is **Blind RANK-MUSIC Detection + covariance fit**:

1. Construct one covariance matrix per BS sector.
2. Estimate the effective signal-subspace dimension using a rank criterion.
3. Evaluate MUSIC with the actual Sionna array geometry and sector orientation.
4. Extract distinct local spectrum maxima and refine their angles.
5. Jointly refine directions and nonnegative gains by fitting the covariance.
6. Use the resulting anonymous MUSIC directions and estimated gains for nulling.

This is **not currently MDL-MUSIC**. MDL estimates are still computed and logged,
but RANK determines the signal-subspace dimension in the notebook.

The estimator does not receive true user-pair keys: `pair_keys=None`.
True directions and true gains are not used to optimize the anonymous MUSIC
peaks. However, the current analytic covariance is synthesized from the
simulation's true channels. The experiment is therefore an idealized
covariance-based estimation experiment, not a demonstration on measured data.

The additional **MUSIC u + true g** beamforming experiment has been removed.
Gain-estimation accuracy against true g remains available. The separate
**All true u,g (oracle)** reference remains available.

## 2. Active Experiment Configuration

These are the explicit notebook settings. They can differ from the default
arguments of the underlying Python functions.

| Setting | Current value | Meaning |
| --- | --- | --- |
| Scene | `Denver_scene/10kmwithfigure/10km.xml` | Sionna scene |
| Deployment | 4 BS sites, 3 sectors per site | 12 independently processed sector arrays |
| Receivers | 20 NTN, 200 TN | Current Monte Carlo deployment |
| TX array | 8 x 8 | 64 array elements per sector |
| Carrier | 7 GHz | Scene/channel carrier |
| Sector downtilt | 6 degrees | Positive Sionna pitch, downward boresight |
| Path depth | `max_depth=0` | Current direct-path/LOS configuration |
| Macro simulations | 10 | Current notebook run length |
| `music_covariance_mode` | `"analytic"` | Population covariance |
| `music_source_estimation` | `"rank"` | Effective-rank source count |
| `music_num_sources` | `None` | No fixed-K override |
| `music_rank_relative_threshold` | `1e-5` | Signal-eigenvalue dynamic-range cutoff |
| `music_rank_noise_margin` | `1e-3` | Noise-related rank floor |
| `music_user_powers` | `None` | Unit powers in covariance generation |
| `music_noise_var` | `N0_bs / Tx_power` | Noise in the normalized observation model |
| `music_num_snapshots` | 800 | Used in sample mode, not to construct analytic Rxx |
| `music_std_channel_mode` | `"conj"` | Work with conjugated channel vectors |
| `music_use_sector_orientation` | `True` | Include each sector's yaw, pitch, and roll |
| `music_rotation_order` | `"zyx"` | Local-to-global rotation convention |
| `music_sector_forward_only` | `True` | Front-side representative search |
| `music_sector_forward_cos_min` | `0.0` | Front hemisphere, not the former 60-degree cone |
| Initial azimuth grid | 0 to 359.5 degrees, step 0.5 | Periodic azimuth |
| Initial zenith grid | 0 to 180 degrees, step 0.5 | Zenith, not elevation |
| Minimum angular peak separation | 0 degrees on both axes | Correlation handles duplicate directions |
| `music_peak_max_correlation` | 0.98 | Steering-vector correlation separation |
| `music_peak_local_maxima` | `True` | Select local maxima instead of arbitrary high bins |
| `music_peak_max_noise_projection` | 0.2 | Candidate noise-subspace projection cutoff |
| `music_peak_refine` | `True` | Local continuous angular refinement |
| Angular refinement | Half-width 1 degree; max 40 iterations | Refinement around initial grid peaks |
| `music_covariance_refine` | `True` | Joint covariance-based direction/gain refinement |
| Joint refinement | Max 60 iterations; coordinate step bound 0.1 | Bound is in direction cosines, not degrees |
| `max_detected_b_terms` | `"all"` | Use all retained positive-gain estimated terms |
| Nulling lambda values | `1e10, 1e11, 1e12` | Interference-penalty weights |

The scene's physical sector pattern and service coverage are separate from the
MUSIC search domain. A broader estimation domain does not reconfigure the
physical sectors or remove their downtilt.

The notebook still contains some historical names such as
`blind_mdl_music_detection_name`. The displayed method label is constructed
from the selected method; the variable name does not make the estimator MDL.

## 3. Implemented Accuracy Improvements

### 3.1 Use the Actual Array Geometry

The dictionary uses wavelength-normalized element positions from the Sionna
array and the actual orientation of each sector. This ties the phase response
to the same element ordering, spacing, coordinate axes, and rotations as the
channel simulation.

The active exact-geometry path does not independently reconstruct an ideal UPA
using legacy flattening or horizontal-sign assumptions. Those compatibility
parameters still exist for other paths, but the exact position list controls
the current array response.

The raw channel manifold uses a positive phase sign. With the active
`channel_mode="conj"`, the scan uses the opposite phase sign. Estimated vectors
are conjugated back before they enter the raw-channel nulling calculation.

### 3.2 Expand the Search Support

The previous search required:

```text
direction dot sector_boresight >= cos(60 degrees) = 0.5
```

The current search requires:

```text
direction dot sector_boresight >= 0
```

This expands a 60-degree half-angle cone to the front hemisphere. It allows
phase directions that the narrower cone could exclude, including tilted
directions and front-side representatives of rear arrivals.

It does **not** delete rear signals from Rxx. It also does not physically
resolve the front/back ambiguity of a planar phase-only array.

### 3.3 Preserve Weaker Signal-Subspace Components

The relative rank threshold was reduced from `1e-4` to `1e-5`. This can retain
weaker covariance eigenmodes that were previously assigned to the noise
subspace.

K is an effective spatial rank, not the number of NTN terminals. Nearly
collinear channels, weak links, and the limited aperture can make K much
smaller than the terminal count. Lowering the cutoff is not guaranteed to help
in noisier or model-mismatched experiments.

### 3.4 Improve Peak Selection and Refinement

The peak extractor now:

- Evaluates the stable noise projection directly as `||En^H a||^2`.
- Selects local spectrum maxima, with periodic azimuth handling on a full grid.
- Uses steering correlation to reject redundant candidates.
- Refines grid candidates continuously in angle.
- Rejects candidates whose refined noise projection exceeds 0.2.
- Does not force the number of accepted peaks to equal K.
- Excludes invalid zenith values outside [0, 180] degrees.

A fixed separation in azimuth/elevation is a poor proxy for array-space
separation. Correlation compares the vectors that actually enter nulling.

The 0.2 cutoff is a heuristic. It is not a calibrated false-alarm probability.
The filtering occurs during MUSIC peak extraction; the later covariance
optimizer has a different acceptance objective.

Exact-geometry dictionary evaluation is also batched to reduce Python overhead
and bound temporary memory use.

### 3.5 Replace Clipped Least Squares with Scaled NNLS

Gain estimation now solves a genuinely nonnegative least-squares problem,
rather than solving unconstrained least squares and clipping negative gains.

The complex covariance equation is stacked into real and imaginary parts.
The target covariance is normalized for numerical conditioning, then gains
are rescaled afterward. This matters because channel powers can be extremely
small in this scene.

Analytic mode subtracts the configured noise variance. Sample mode estimates
noise from the noise-subspace eigenvalue tail. The old absolute `1e-12` noise
floor was removed because it could exceed the actual noise level.

Exactly zero fitted gains are discarded before constructing the nulling terms.

### 3.6 Add Joint Covariance Fitting

After MUSIC initializes the directions, `refine_music_covariance` jointly
adjusts direction cosines and nonnegative gains.

For the current local yz-plane array, the optimizer uses two coordinates
per direction, qy and qz. For a forward cosine bound c, it enforces:

```text
qy^2 + qz^2 <= 1 - c^2
qx = sqrt(1 - qy^2 - qz^2)
```

For each trial direction set, it solves NNLS for the gains. SLSQP then updates
the directions using an analytic gradient. Each direction-cosine coordinate
is restricted to within 0.1 of its MUSIC initialization.

A candidate update must be finite, feasible, and improve the covariance-fit
objective. Otherwise, the initial solution is retained. Numerical boundary
correction is checked so that it does not leave a worse fit.

This refinement uses covariance, array geometry, orientation, and noise.
It does not use true directions, true gains, or true pair identities.

It is a local optimizer. It does not create missing sources, guarantee the
global optimum, or guarantee better INR or better per-direction correlation
for every sample.

### 3.7 Correct Numerical Diagnostics and Accuracy Reporting

Additional changes include:

- Scale-normalized eigenvalues and a relative floor in MDL.
- Global correlation-based assignment for anonymous u/g accuracy evaluation,
  replacing greedy matching.
- Clipping computed correlations to [0, 1] against floating-point overshoot.
- Removing the absolute `1e-12` denominator floor from gain relative error.
- Reporting lower-tail and power-weighted direction quality, covariance error,
  covered true power, and covariance-fit residuals.
- Separate placement, satellite, and snapshot seeds.
- Timestamped outputs and cached channels for same-channel comparisons.

The gain-error definition changed. Old and new printed gain errors should not
be directly compared without recomputing both using the same metric.

## 4. Current End-to-End Algorithm

### Step 1: Generate the Scene Channels

`Nulling_CDF.ipynb` calls `run_nulling_cdf_experiment`. For each macro,
the scene creates positions and paths, then collapses the CIR into narrowband
channel tensors:

```text
h_ntn_all: [NTN receiver, RX antenna, sector TX, TX antenna]
h_tn_all:  [TN receiver,  RX antenna, sector TX, TX antenna]
```

The experiment injects the scene's exact array positions and per-sector
orientations into the MUSIC call. Sector processing is independent; there is
no joint three-sector DOA solver in the current implementation.

### Step 2: Form Rxx

For one sector, let h_j be the observation-model channel vector of source j:

```text
x[t] = sum_j sqrt(p_j) h_j s_j[t] + w[t]
E[s s^H] = I
E[w w^H] = sigma^2 I
```

The current analytic branch constructs:

```text
Rxx = sum_j p_j h_j h_j^H + sigma^2 I
```

The optional sample branch generates independent complex Gaussian source
symbols and noise, then computes:

```text
Rxx_hat = X X^H / T
```

Analytic mode has no finite-snapshot estimation error. Increasing
`music_num_snapshots` does not improve its covariance. In sample mode,
increasing T can reduce sampling error under the static independent-source
model, but it cannot remove array ambiguity or dictionary mismatch.

The covariance is average power, not accumulated energy. Noise is therefore
not multiplied by T when comparing these two branches under this model.

### Step 3: Estimate K and Split the Eigenspaces

The covariance is eigendecomposed. For RANK, define signal eigenvalues as
`s_i = max(eigenvalue_i - noise_power, 0)`. The cutoff is:

```text
floor = max(
    rank_relative_threshold * s_max,
    rank_noise_margin * noise_power,
    machine_epsilon * s_max
)
```

K counts signal eigenvalues above this floor and is capped at M - 1 so that a
noise subspace remains. Zero signal can produce K = 0.

The leading K eigenvectors form the signal subspace; the remaining eigenvectors
form En. MDL and eigengap estimates are also recorded for comparison.

A fixed integer `music_num_sources` overrides automatic selection. With
`None`, the selected `music_source_estimation` determines K.

In analytic mode, MDL currently uses the surrogate count
`max(num_ntn * num_rx_ant, M + 1)`, not the configured 800 snapshots.
Consequently, simply switching this run to MDL is not the same as testing
standard finite-snapshot MDL on observations.

### Step 4: Scan the Exact-Geometry MUSIC Dictionary

For normalized global element positions p_m and a unit direction d:

```text
a_m(d) = exp(j * phase_sign * 2*pi * p_m dot d) / sqrt(M)
D(d) = ||En^H a(d)||^2
P_MUSIC(d) = 1 / max(D(d), numerical_floor)
```

The grid is specified in global azimuth and zenith. Each sector's orientation
defines its allowed forward domain and its rotated element positions.

Peak extraction and angular refinement produce anonymous candidate vectors.
No user identity is needed for this step.

### Step 5: Fit Directions and Gains

For unit-norm vectors u_k, the noncoherent model is:

```text
R_signal ~= sum_k g_k u_k u_k^H
g_k >= 0
```

For fixed directions, NNLS minimizes:

```text
||Rxx - sigma^2 I - sum_k g_k u_k u_k^H||_F^2
```

Joint fitting updates directions in the same model and resolves gains at each
trial. A final NNLS solve is performed at the retained directions.

The primary outputs are `peak_u_hat_raw` and `peak_g_hat`, with sector indices
and angle diagnostics. The estimated g is a covariance/power coefficient,
not a signed or complex path amplitude. Its square root provides an equivalent
amplitude magnitude; absolute path phase is not recovered.

If several physical links have effectively the same steering vector, a fitted
gain can represent their combined power rather than one link's individual g.

### Step 6: Construct the Nulling Beam

`build_music_tx_lookup` uses the anonymous peak outputs. The retained terms form:

```text
B_hat = sum_k g_hat_k u_hat_k u_hat_k^H
A = h_tn w_r w_r^H h_tn^H
Q = A - lambda * B_hat
```

`nulling_bf_music_noncoh` chooses the principal eigenvector of the Hermitian Q
as the transmit beam.

The current `max_detected_b_terms="all"` retains all accepted positive-gain
terms. It does not mean that all physical users or all true paths were resolved.

### Step 7: Evaluate Link Performance on the True Simulation Channels

The beam is evaluated on the actual simulated TN and NTN channels to compute
SNR, SINR, and INR. This use of true channels measures the performance of the
estimated beam; it does not substitute true g into that beam.

The active comparisons are:

| Branch | Beam-design inputs |
| --- | --- |
| Raw baseline | Desired TN channel, no MUSIC nulling penalty |
| MUSIC estimate | MUSIC u_hat and covariance-fitted g_hat |
| All true u,g (oracle) | All retained true-channel direction/power terms |

The `MUSIC u + true g` branch, including its SNR/SINR/INR computation,
aggregation, plots, legends, and metric export, has been removed.

The generic direct-true-covariance branch still exists in the utility API but
is disabled by `lambda_ranges=None` in the notebook. INR and SINR plotting are
active; the separate SNR plotting cell remains commented out.

## 5. Where Ground Truth Is and Is Not Used

| Operation | Ground-truth involvement |
| --- | --- |
| Simulation and analytic Rxx generation | Uses the simulated true channels |
| Dictionary construction | Uses known array geometry and sector orientation |
| Anonymous MUSIC peak extraction | No true pair keys or true source directions |
| NNLS and joint covariance refinement | No true gains or true source directions |
| Estimated nulling beam | Uses anonymous estimated u and g |
| Per-user scores, pair mappings, angle errors | Truth-assisted evaluation after peak estimation |
| u/g accuracy metrics | Matches estimated vectors to true channel vectors |
| Oracle beam | Deliberately uses true u and true g |
| INR/SNR/SINR measurement | Uses actual simulated propagation channels |

The estimator entry uses `pair_keys=None` and
`compute_user_scores=False` during subspace estimation. Per-user evaluation
is performed later. The `pair_*` outputs are evaluation mappings, not the input
to the active anonymous nulling branch.

`music_threshold=3` controls the truth-assisted per-user score evaluation.
It is not the blind peak threshold. Blind peak acceptance uses the MUSIC
projection, local-maxima, and separation settings.

Thus, the improvement is not produced by giving correct pair keys to the
estimator. Nevertheless, the current simulator supplies an exact population
covariance and known geometry, which are substantially easier conditions than
finite, noisy, calibrated-imperfect observations.

## 6. Accuracy Metrics and Their Interpretation

For a true channel h, the reference decomposition is:

```text
g_true = ||h||^2
u_true = h / ||h||
rho = |u_hat^H u_true|       # after unit normalization
u_err = 1 - rho
g_relative_error = |g_hat - g_true| / max(g_true, float_tiny)
```

Anonymous estimates are assigned to true vectors by a global maximum-total-
correlation assignment. If there are more estimates than true vectors,
unassigned estimates fall back to their best-correlated reference, allowing
reference reuse.

- **u_rho_mean / median / p10:** Direction-vector correlation summaries.
  Rho is not a percentage of correctly identified users.
- **u_rho_power_weighted:** Correlation weighted by matched true powers.
  This is conditional on the matched terms, not a missed-source metric.
- **g_rel_err_mean / median:** Accuracy of g_hat against matched true g.
  Very weak references and merged sources can produce a large mean.
- **Power coverage at rho >= 0.95:** Fraction of total true channel power whose
  direction has a sufficiently correlated active estimate. It includes missed
  sectors/sources but allows one estimate to cover several similar directions.
  It is not user-identity recall.
- **Covariance NRMSE:** Square root of the sum of sector covariance-error
  squared Frobenius norms divided by the sum of true-covariance squared norms.
- **Covariance fit before/after:** Relative fit residual for the optimizer's
  noise-subtracted covariance target.
- **Angle MAE:** Physical-angle evaluation, not the same quantity as vector
  correlation. A planar front/back ambiguity can leave a large azimuth MAE
  even when the phase-equivalent u is extremely accurate.

The true-channel covariance diagnostic currently assumes the unit source
powers used by the notebook. If `music_user_powers` is changed, it should not
be interpreted as a correctly source-power-weighted observation-fit metric
without adapting the reference calculation.

## 7. Observed Results and Limits of the Evidence

### 7.1 Controlled Same-Channel Comparison

The earlier implementation and the improved implementation were compared on
two identical cached Denver channel realizations. Both were evaluated using
the corrected matching and accuracy metrics.

| Metric | Macro 0: previous | Macro 0: improved | Macro 1: previous | Macro 1: improved |
| --- | ---: | ---: | ---: | ---: |
| Mean u correlation | 0.881165 | 0.998136 | 0.916495 | 0.999545 |
| Power-weighted correlation | 0.969085 | 0.999940 | 0.979432 | 0.999926 |
| Covariance NRMSE | 0.867848 | 0.000272 | 0.049859 | 0.000228 |
| Power coverage at rho >= 0.95 | 0.842401 | 0.999957 | 0.985711 | 0.999988 |

With the corrected engine and hemisphere search but without joint fitting,
the two covariance NRMSE values were 0.018498 and 0.011047. Joint fitting
reduced them to approximately 0.000272 and 0.000228. It did not improve every
individual u statistic.

The gains reflect multiple changes together. The full old/new table does not
isolate the causal contribution of each individual setting.

Only 16 INR samples were available in this small comparison. The INR median
at lambda = 1e12 became worse, although the median paired change was slightly
better. Improved covariance/u metrics do not prove uniformly improved INR.

Artifacts:
`result/music_improved_20260908_163344_202810/`.

### 7.2 Subsequent Ten-Macro Run

Saved metrics:
`result/music_improved_20260908_170020_651618/nulling_cdf_metrics.npz`.

| Summary | Value |
| --- | ---: |
| Arithmetic mean of the 10 per-macro mean u correlations | 0.994767 |
| Range of per-macro mean u correlations | 0.979088 to 0.999821 |
| Arithmetic mean of per-macro covariance NRMSE | 0.002370 |
| Largest per-macro covariance NRMSE | 0.021026 |
| Arithmetic mean of per-macro mean g relative errors | 0.729269 |
| Arithmetic mean of per-macro median g relative errors | 0.014659 |
| Macros contributing to the link CDFs | 7 of 10 |
| INR samples | 131 |
| SNR/SINR samples | 96 each |

These averages weight macros equally, not individual matched terms. The mean
of macro medians is not a pooled median. The large gap between gain mean and
median summaries indicates that some matched gains remain much less accurate.

CDF scheduling uses the minimum paired-TN count across sectors. A macro with
an empty TN sector contributes no small-round CDF samples, although its MUSIC
accuracy diagnostics are retained.

This run predates removal of the true-g beamforming branch. Its historical
files remain unchanged and may contain the old branch's fields. New runs no
longer produce those fields.

The ten-macro run is not an independent, large held-out benchmark: it shares
the initial seeded realizations with the small validation. Results should
also be tested under sample covariance, different seeds, larger source counts,
and model mismatch before making broader accuracy claims.

## 8. How to Run and Verify

Run the notebook from its import cell so the updated modules are reloaded.
The notebook writes to a fresh timestamped directory:

```text
result/music_improved_<timestamp>/
    run_config.json
    nulling_cdf_metrics.npz
    nulling_inr_cdf.png / .pdf
    nulling_tn_sinr_cdf.png / .pdf
    channels/
        channels_0000.npz
        music_0000.npz
        ...
```

The independent notebook seeds are:

```python
position_rng_seed = 20260908
satellite_rng_seed = 20260909
music_rng_seed = 20260910
```

Channel caches are created exclusively to prevent accidental overwrite.
Changing snapshot seeds alone has no effect on analytic covariance.

Run the regression suite from the project root in the Sionna environment:

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

At the time of this document, 21 tests passed. They cover geometry/conjugation,
sector orientations, numerical scales, empty cases, peak boundaries,
refinement, seeded sampling, accuracy matching, cache behavior, notebook
syntax, and removal of true-g beamforming while retaining gain accuracy.

INR and SINR plotting were also executed using the existing ten-macro data
without the removed fields, writing only to a temporary directory.

Replay cached MUSIC estimation without ray tracing:

```sh
OPENBLAS_NUM_THREADS=1 python replay_music_accuracy.py \
    result/music_improved_20260908_170020_651618
```

Replay variants compare the current corrected engine with:

- `cone60`: 60-degree half-angle cone, no joint covariance refinement.
- `hemisphere`: front hemisphere, no joint covariance refinement.
- `improved`: front hemisphere with joint covariance refinement.

These variants share the corrected engine and the run's rank cutoff.
`cone60` does not recreate all historical code behavior. The replay evaluates
MUSIC accuracy; it is not a full rerun of the link-performance CDF experiment.

## 9. Code Map

| File / function | Responsibility |
| --- | --- |
| [Nulling_CDF.ipynb](Nulling_CDF.ipynb) | Experiment settings, execution, metrics and CDF plotting |
| [SceneConfigSionna.py](SceneConfigSionna.py) | Scene geometry, deployment, arrays and channels |
| [ntn_music_detection.py](ntn_music_detection.py): `run_music_standard_pipeline` | Public pipeline entry and blind/paired dispatch |
| `_run_music_standard_blind_pipeline` | Anonymous MUSIC estimation and subsequent evaluation mappings |
| `detect_ntn_music_from_hi` | Covariance construction, eigendecomposition and K estimates |
| `array_position_steering_global` | Exact-position steering vector |
| `music_top_peaks` | Spectrum scan, candidate selection and angular refinement |
| `estimate_noncoh_gains_from_covariance` | Scaled NNLS gain fit |
| `refine_music_covariance` | Joint direction-cosine and gain refinement |
| [nulling_cdf_utils.py](nulling_cdf_utils.py): `build_music_tx_lookup` | Anonymous peak outputs to per-sector nulling inputs |
| `summarize_music_noncoh_quality` | Matched u/g accuracy |
| `summarize_music_covariance_quality` | Covariance reconstruction and true-power coverage |
| `run_small_round` | Estimated/oracle beam generation and link-performance evaluation |
| `run_nulling_cdf_experiment` | Monte Carlo orchestration, geometry injection and caches |
| `save_experiment_metrics` | Metric export, including retained g accuracy |
| [BeamformingCalc.py](BeamformingCalc.py): `nulling_bf_music_noncoh` | Noncoherent covariance penalty and principal-eigenvector beam |
| [replay_music_accuracy.py](replay_music_accuracy.py) | Same-channel MUSIC ablation replay |
| [tests/test_music_accuracy.py](tests/test_music_accuracy.py) | Regression tests |

## 10. Remaining Limitations

The implementation still assumes a static narrowband observation model,
independent source symbols, known array geometry, and white noise. The active
experiment uses analytic covariance and direct paths.

The planar phase-only manifold remains front/back ambiguous. Searching only
the front hemisphere selects a representative; it does not determine the
physical side of arrival. A high u correlation must not be presented as proof
of correct physical azimuth.

The current optimizer cannot recover a direction omitted by the initial
subspace/peak stage. Rank thresholds, peak correlation, and projection cutoffs
are heuristics, not universally optimal settings for every NTN count or SNR.

Not implemented in this update: a full initial direction-cosine scan, a joint
multi-sector physical DOA solver, new spatial smoothing, direct estimated-
covariance nulling as an additional branch, or a broad held-out parameter
sweep. These should not be credited for the reported improvement.
