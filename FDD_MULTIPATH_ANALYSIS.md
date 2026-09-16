# FDD Nulling: Multipath Processing and Frequency-Transfer Analysis

This document accompanies `Nulling_CDF_SectorDrop.ipynb`. Sector placement and
association rules are documented in [TN_SECTOR_DROP.md](TN_SECTOR_DROP.md).

## Fixed DL array and UL frequency sweep

The parameter cell of `Nulling_CDF_SectorDrop.ipynb` now includes:

```python
f_dl = 7e9
ul_frequency_percentages = [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
num_macro_sims = 20
ul_to_dl_mode = "angle"
lambda_ranges_music_est = [1e10, 1e11, 1e12]
plot_oracle = False
```

Percentages are percentage points: `+5` gives 7.35 GHz and `-5` gives 6.65 GHz.
The 8x8 BS has a fixed physical spacing of `c/(2*f_dl)`, about 21.414 mm.
At UL, Sionna's wavelength-normalized spacing is `0.5*f_ul/f_dl`; its element
positions in meters remain unchanged. Positive offsets can therefore introduce
spatial aliasing; the code does not shrink the physical array to avoid it.

Each macro draws positions and traces DL TN/NTN channels once. Every percentage
uses those same positions, satellite direction, sector orientations, DL TN
association and scheduling. Nonzero offsets retrace the NTN channel at UL using
the retained endpoints and fixed array. UL tracing uses reciprocal BS-to-NTN
paths in the existing channel-vector convention, and restores DL scene state
on success or failure. Zero offset reuses the original DL sensing channel;
`ul_frequency_percentages=[0]` is the regression control.

In `angle` mode, blind MUSIC uses the actual UL electrical array geometry.
Its anonymous estimated angles rebuild DL steering vectors; the estimated UL
power weights are retained. No true DL channels, paired-user angles or DL gains
are supplied to the transfer. This tests angular transfer and UL-derived weights,
not instantaneous FDD channel reciprocity. Errors in angular estimation and UL
aliasing propagate into the DL nulls. In `raw` mode, the UL vectors themselves
are reused, providing a direct cross-frequency mismatch comparison. Both modes
calculate INR/SNR/SINR against the same DL channels and DL evaluation mask.

The current settings produce 3 x 7 = 21 estimated curves plus one no-nulling
baseline per CDF. The color legend lists the three lambdas and the black
no-nulling baseline. A separate line-style legend lists each UL percentage and
frequency; readers combine the two keys. Set `plot_oracle=True` for one additional true-DL
oracle per lambda, or `plot_snr=True` for the optional SNR plot. The oracle does
not depend on UL frequency and is only drawn once per lambda.

Runs are saved under `result/<YYYYMMDD_HHMMSS_microseconds>/`. Access one case as:

```python
case = nulling_cdf_results["by_percentage"][5.0]
values = case["est_inr_db"][1e11]
```

`run_config.json` records DL/UL frequencies, fixed spacing, bandwidths and the
transfer mode. Common DL caches remain in `channels/channels_<macro>.npz`;
UL sensing and MUSIC diagnostics live under `channels/ul_<index>/`. Diagnostics
save both UL peak vectors and `peak_u_used_for_dl`. The combined metrics archive
uses `ul_<index>/...` keys; separate per-percentage archives retain the existing
metric names. Use `*_db_index_<lambda_index>` plus `lambda_ranges_*` for arbitrary
lambdas; these indexed arrays avoid collisions from legacy rounded lambda names.

Nonzero offsets are checked for band separation using
`abs(f_ul-f_dl) > (B+B_ul)/2`. With both bandwidths at 200 MHz, +/-2% at 7 GHz
has overlapping bands; reduce both bandwidths below 140 MHz to treat it as a
disjoint FDD case. Zero is exempt as a mathematical control. The optional
`enforce_disjoint_bands=False` permits mathematical overlapping-band studies.
This remains a narrowband array/channel experiment; bandwidth sets noise and
checks separation, without modeling within-band beam squint.

The original element-pattern models and normalized sensing SNR are retained:
`music_noise_var = N0_bs / Tx_power`. For a 23 dBm NTN-UE sensing-power experiment,
explicitly use `Tx_power_handheld` in that denominator. No RF retuning transient,
training duty-cycle overhead, cross-band calibration or DL gain prediction is
introduced here. `music_covariance_mode="analytic"` still uses ideal covariance;
choose `"sample"` to introduce the existing finite-snapshot estimator.

Validation: zero-offset regression and paired DL baseline/oracle invariance;
positive/negative UL angle transfer, vector conventions, weights and state
restoration; Cartesian-product plotting and archive round trips. A real Sionna
LoS check verified fixed physical positions and frequency-dependent phase/gain.
A small end-to-end Denver run used one macro, three frequencies, two lambdas,
12 TNs, eight NTN UEs and the 8x8 array, including saving and both CDF plots.

## Coherent multipath and NLOS controls

`Nulling_CDF_SectorDrop.ipynb` now imports the shared
`multipath_support.collapse_cir_to_narrowband` helper. The existing
`ntn_music_detection.collapse_cir_to_narrowband` import remains compatible.
The notebook currently uses **`max_depth = 3`**. Setting `max_depth = 0`
selects the legacy direct-path estimator. Example controls:

```python
max_depth = 3  # Sionna maximum path interaction depth
ntn_los_mode = "natural"  # or "nlos_only"
music_spatial_smoothing = True
music_smoothing_rows = 6
music_smoothing_cols = 6
music_forward_backward = True
multipath_top_k = None
multipath_energy_fraction = 1.0
```

Rerun the parameter cell and all subsequent cells. After loading updated Python
modules, restart the kernel and run all cells so an old scene instance is not
retained. `max_depth` is the only propagation-depth control and is passed to
Sionna unchanged for TN DL, NTN DL and NTN UL. It specifies the maximum number
of interactions along a path, not how many paths must exist. There is no separate
`multipath` enable switch. Zero depth uses the original estimator; positive depth
automatically selects coherent-path estimation and replaces diagonal covariance
refinement with the correlated fit. Logs, plot titles and result archives record
`max_depth` so runs with depths 1, 2, etc. remain distinguishable. If omitted from
`run_nulling_cdf_experiment`'s path arguments, the experiment explicitly uses 0.
Other Sionna propagation settings remain available. At zero depth, disabling NTN
LOS leaves no NTN path; that case is reported without fabricating indirect paths.

`natural` includes all valid direct and indirect paths that the environment and
selected propagation mechanisms support. A link is classified as NLOS if it has
at least one valid nonzero indirect path and no valid nonzero direct path.
`nlos_only` suppresses direct paths **only on NTN links**, at both UL and DL;
this is an artificial direct-component removal test. TN links retain natural
LOS/NLOS and gain multipath as well, keeping their serving-sector admission
rule. A link with no valid path is reported separately; it is not called NLOS.

Depth, reflection/refraction/scattering/diffraction switches, tracing budgets,
and solver seed are shared between NTN UL and DL. Direct-path mode preserves
the original solver defaults. For positive depths, the notebook defaults to specular reflection and
refraction, 100,000 launched samples and a 100,000 path cap per source.
Choose the depth directly with `max_depth`; the notebook currently uses 3.
Diffuse scattering and diffraction are configurable and disabled by default;
scattering also requires suitable nonzero material scattering coefficients.
Path finding at finite sampling budgets is approximate. Raising the depth or
sampling budget can increase runtime and the number of discoverable paths.

### CIR synthesis and time axes

Sionna's `paths.cir()` returns baseband path coefficients `a` and delays `tau`.
Use those coefficients, not the passband `paths.a` attribute. Their carrier
propagation phase is already included.

```python
# a: [RX, RX_ANT, TX, TX_ANT, PATH, TIME]
h = collapse_cir_to_narrowband(a)                 # sum PATH, choose TIME=0
h_t = collapse_cir_to_narrowband(a, time_index=3)  # another time sample
h_all_times = collapse_cir_to_narrowband(a, time_index=None)
h_offset = collapse_cir_to_narrowband(
    a, tau=tau, frequency_offset_hz=1e6, valid_mask=paths.valid,
)
```

Summation is complex and coherent: two equal opposite-phase paths cancel.
The helper never sums time samples or discards weak paths. Five-dimensional
single-time CIRs and four-dimensional already-collapsed channels are accepted.
For singleton TIME, default synthesis is numerically identical to the old sum.
`time_index=None` returns `[RX, RX_ANT, TX, TX_ANT, TIME]`.

A nonzero frequency offset uses `exp(-j*2*pi*offset*tau)` and requires delays;
it does not reapply the carrier phase. This helper's within-band offset holds
the traced antenna response fixed and does not model beam squint. The notebook
currently evaluates TIME=0 at each carrier, with bandwidth used for noise and
band separation. It does not simulate OFDM symbols, ISI, CP violation, or Doppler
time evolution. These require a separate wideband/time-domain experiment.

### Coherent-path estimation and full-array nulling

1. Form each UE's channel by coherently adding **all** its paths. In sample mode,
   one waveform per UE/RX antenna drives the combined channel; no independent
   random waveform is fabricated for each ray.
2. Form the full UL covariance. Extract translated 2D rectangular subarrays from
   the actual wavelength-normalized Sionna coordinates and average their
   covariances. Optional forward/backward averaging uses subarray centrosymmetry.
3. Estimate the number of spatial modes and MUSIC peaks on the smoothed
   covariance. This can resolve coherent paths under the usual array aperture,
   angular separation, SNR and subarray-rank conditions. It does not guarantee
   recovery of every ray, and does not remove spatial aliasing at higher UL
   frequencies. Set `music_spatial_smoothing=False` for the unsmoothed comparison.
   `detect_num_sources`, if set, means spatial modes here, not the number of UEs.
4. Rebuild each peak's **full-array** UL steering vector and fit
   `R_UL = A_UL Q_UL A_UL^H + sigma^2 I`, allowing a full correlated PSD `Q_UL`.
   A pseudoinverse followed by PSD projection estimates `Q_UL`; `diag(Q_UL)`
   provides full-array power weights. Conditioning and residuals are saved.
   This replaces diagonal/noncoherent power fitting; it does not perform the old
   diagonal-model joint angle refinement. MUSIC peak refinement remains active.
5. Select estimated directions per BS sector by power, optionally applying
   `multipath_top_k` and `multipath_energy_fraction`. When both are set, top-K can
   prevent reaching the requested energy fraction. The existing
   `max_detected_b_terms` can impose a further beamformer-side cap.
6. Transfer angles onto the full DL array for `ul_to_dl_mode="angle"`, retaining
   estimated UL weights. The UL correlation phases are **not** copied into DL.
   `raw` mode retains the existing raw-vector mismatch control. Evaluate beams
   on the complete coherent DL channel, including every omitted/missed path.

The diagonal sum of selected path outer products is a direction-leakage penalty,
not the instantaneous coherent DL covariance. UL weights also need not equal DL
weights. In multipath mode, the optional `music_real_*` oracle uses all **true DL
path vectors and powers** for the same directional penalty, with full coherent
DL evaluation. It is a perfect-path-information comparison, not a guaranteed
pointwise performance bound. Disabled mode retains the original effective-channel
oracle. Subarray smoothing reduces estimation aperture; transmission still uses
all 64 physical elements. Source counts must fit the subarray's noise-subspace
requirements, and sample-mode MDL on overlapping smoothed data is a heuristic.

### Path diagnostics and reproducibility

Common DL caches now additionally include `channels/paths_dl_<macro>.npz` when
`max_depth > 0`. They contain full TN/NTN CIRs and delays, valid-path masks,
BS departure angles, powers, direct-path indicators and interaction types.
Per-frequency `paths_ul_<macro>.npz` files retain the corresponding UL data.

MUSIC archives additionally save raw/smoothed UL covariances, raw signal rank,
subarray shape/count, all candidate angles/powers before selection, and padded
correlated-source covariance blocks. `candidate_counts` and `candidate_t_idx`
define each block's active dimensions and ordering. Selected peaks and the
vectors actually used for DL are saved separately.

`macro_stats[*].path_metrics_ul` and `.path_metrics_dl` report all valid paths,
LOS/NLOS/no-path link counts, one-to-one angular matches and matched path-power
fractions. The archive stores these dictionaries as JSON strings, avoiding
pickle. All path truth is used only for evaluation or explicitly labeled oracle
beams, never for the blind estimator. Matching includes every valid ray rather
than only the strongest path per UE. Angular matching uses a 5-degree spherical
angular gate. Coincident directions, front/back ambiguity and spatial aliasing
can limit individual-ray matching even when a null suppresses the channel.
The existing vector-correlation coverage diagnostics provide a complementary
view. In multipath mode `noncoh_metrics` compare against the sum of true DL path
outer products, not the coherent per-UE covariance.

### Tests in this repository

| File | Purpose |
|---|---|
| `tests/test_tn_sector_drop.py` | Sector geometry, indoor/outdoor sampling, serving-power margin, retries and final path/UE alignment. |
| `tests/test_music_accuracy.py` | MUSIC angles and source counts, gain fitting, array conventions, sample noise, numerical scaling and output caching. |
| `tests/test_fdd_nulling.py` | Fixed physical geometry across frequencies, UL-to-DL transfer, zero-offset regression, Cartesian-product CDFs and archives. |
| `tests/test_multipath_support.py` | Coherent CIR/time handling, subarray smoothing, correlated powers, NLOS classification, path selection and full FDD multipath integration. |

Run all tests in the Sionna environment:

```bash
/home/shizhen/miniconda3/envs/sionna20/bin/python -m unittest discover -s tests -v
```

Multipath validation includes a two-path single-UE rank-1 covariance becoming
rank 2 after smoothing, recovery of both directions and full-array powers,
signed FDD transfer, and unchanged full-channel evaluation after top-K selection.
Pre-change CDF arrays were also saved and compared exactly at `max_depth=0`.
Real Sionna tests used a direct-plus-reflected link and its direct-path-removed
variant at three frequencies. A small Denver run exercised 12 TNs, eight NTN UEs,
an 8x8 array, three frequencies and two lambdas through saving and CDF plotting.

## Interpreting the frequency sweep

Here "frequency offset" means UL/DL carrier separation, not residual oscillator
CFO or Doppler. DL stays at 7 GHz, and UL spans 5.6, 6.3, 6.65, 7, 7.35, 7.7,
and 8.4 GHz. The estimator uses an analytic covariance by default; 800 configured
snapshots do not introduce finite-snapshot randomness in that mode.

### What the saved 10-macro experiment actually shows

The following values are computed from
`result/sector_drop_fdd_20260914_133236_039356/nulling_cdf_metrics.npz`, with
20 NTN devices, 12 TN devices, depth 3 and lambda = 1e12. They describe the
previous five-frequency, 10-macro run, not the new seven-frequency run.
Lower INR means better NTN interference suppression.

| UL offset | UL frequency (GHz) | Median INR (dB) | 90th-percentile INR (dB) |
|---|---:|---:|---:|
| -10% | 6.30 | -19.941 | -9.620 |
| -5% | 6.65 | -19.661 | -9.175 |
| 0% | 7.00 | -19.775 | -8.882 |
| +5% | 7.35 | -19.320 | -5.719 |
| +10% | 7.70 | -18.918 | -5.995 |

These data do not establish monotonic degradation with absolute frequency
separation. Positive offsets generally perform worse for this lambda, whereas
negative offsets can outperform zero. Other lambdas and CDF quantiles can cross.
All five cases have exactly the same no-nulling DL INR samples. With only ten
independent geometry draws, the curves are descriptive rather than a proof of
an asymptotic trend; UE samples within a macro are not independent macro trials.

### Angle transfer already compensates the known array frequency dependence

`transfer_music_peaks_to_dl()` rebuilds each estimated direction using DL
wavelength-normalized positions. Therefore an explanation based solely on
reusing UL steering vectors at DL applies to `raw` mode, not to the current
`angle` experiment. Transfer cannot repair a wrong angle, a missed path or an
ambiguous UL spatial peak.

For the fixed physical spacing d = c/(2 f_DL), d/lambda_UL equals 0.4, 0.45,
0.475, 0.5, 0.525, 0.55 and 0.6 for the seven offsets. Lower UL frequencies
reduce electrical aperture; higher frequencies can introduce spatial aliasing
for some directions once the spacing exceeds half a wavelength. Neither effect
implies universal monotonic angle error. Spatial smoothing further reduces the
estimation aperture to 6x6 while nulling still uses the full 8x8 array.

In the previous run, the macro-averaged `coverage95` was 0.9821 at zero offset,
0.9615 at +5%, and 0.9621 at +10%. This diagnostic measures DL path power with
a steering-vector match of at least 0.95, not a 95% confidence interval. It is
not the same as the one-to-one, 5-degree angular power coverage. These changes
support an estimation/coverage contribution but do not isolate its causal share.

### Retained UL powers change the effective nulling strength

The implemented beam maximizes, for a unit-norm v,

```math
|\widetilde{h}_0^H v|^2 - \lambda v^H \widehat B_{DL}v,
\qquad
\widehat B_{DL}=\sum_\ell \widehat g_{\ell,UL}
\widehat u_{\ell,DL}\widehat u_{\ell,DL}^H.
```

The desired effective channel is normalized. UL power weights are not normalized
to a common trace or predicted at DL, and lambda is held fixed across frequencies.
Thus identical lambda values do not imply identical effective penalty strengths.
If every power weight is multiplied by s, the objective is exactly equivalent
to replacing lambda by s*lambda, with directions held fixed.

For a free-space path with fixed antenna gains, a first-order approximation is

```math
g_{UL}/g_{DL}\approx(f_{DL}/f_{UL})^2.
```

At +10%, the ratio is about 0.826; at -10%, it is about 1.235. At +20% and -20%,
the corresponding ratios are about 0.694 and 1.563. Higher UL frequency can then
underweight the DL interference penalty, while lower UL frequency can overweight
it. This is a plausible explanation of the observed sign asymmetry, not a measured
decomposition of the result. Frequency-dependent material responses and imperfect
power fitting prevent applying this free-space approximation to every reflected
or refracted path as an exact correction.

### Coherent propagation and the noncoherent directional penalty

A narrowband channel can be written as

```math
h(f)=\sum_\ell \alpha_\ell(f)e^{-j2\pi f\tau_\ell}u_\ell(f),
\qquad
\Delta\phi_{\ell m}=-2\pi\Delta f(\tau_\ell-\tau_m).
```

Changing carrier frequency changes relative path phases and potentially path
amplitudes. UL and DL coherent covariance matrices therefore need not coincide,
even for shared geometric directions. Sionna's baseband CIR already contains
the carrier propagation phase; it must not be applied a second time.
See the [Sionna Paths documentation](https://nvlabs.github.io/sionna/rt/api/paths.html).

The estimator fits the full correlated Q_UL to obtain path powers, but the
beamformer uses only diag(Q_UL). The off-diagonal UL correlation phases are saved
as diagnostics, not transferred into DL. The directional penalty consequently
omits coherent cross terms, while the evaluated DL INR retains them.

A change of relative phase alone is not sufficient to prove nulling degradation:
if v is orthogonal to every true DL path vector, their coherent sum is also zero
for arbitrary path phases. Residual leakage arises when directions are missing
or inaccurate, or finite lambda leaves a nonzero response along them. Their DL
coherent combination then determines the actual interference. Stronger penalties
cannot reconstruct unobserved directions.

### Diagnostics and controlled comparisons

`B_nrmse` in multipath mode compares the estimated directional penalty with the
sum of true DL path outer products. It is not error against the coherent per-UE
covariance. It need not vary monotonically with frequency offset or track every
INR quantile. The unweighted relative gain error can be dominated by tiny matched
true powers; do not interpret its very large values as a uniform gain error.

To distinguish mechanisms in future experiments:

1. Apply the approximate power correction
   `g_DL_hat = g_UL_hat * (f_UL/f_DL)**2`, holding estimated directions fixed,
   and compare with uncorrected results. This isolates a simple frequency-scale
   effect; it does not compensate material-specific gains.
2. Use a common trace for the penalty matrices, or sweep lambda separately per
   frequency, to examine direction/subspace quality at comparable penalty scale.
3. Enable `plot_oracle=True` to compare against true DL path directions/powers.
   This path oracle is not a guaranteed pointwise INR bound for the coherent
   channel. Keep it distinct from an instantaneous coherent-channel oracle.
4. Compare direct-path and multipath cases with controlled geometry, and inspect
   accepted peaks, coverage, fit conditioning, and path power diagnostics.

The current 20-macro run retains the original UL weights and physical/estimation
model. It adds +/-20% and more geometry draws; none of the proposed ablations or
power corrections above is silently enabled.

## Run archive and plotting

Every execution of the experiment cell creates a fresh local-time directory
`result/YYYYMMDD_HHMMSS_microseconds/` before tracing begins. Older runs remain
intact. The run records timezone-aware start/end timestamps and its status.

- `run_config.json`: actual frequencies, lambda values, random seeds, geometry,
  propagation, MUSIC, noise/power and plotting parameters.
- `run_status.json`: running, metrics_saved, complete or failed, with timestamps.
- `environment.json`: Python/platform and relevant installed package versions.
- `sources/` and `source_manifest.json`: notebook, helper modules and documentation
  snapshots with SHA-256 hashes. The notebook snapshot is the saved file on disk;
  `run_config.json` describes the parameters actually used by the running kernel.
- `run.log`: experiment stdout/stderr, including per-macro MUSIC/drop diagnostics.
- `nulling_cdf_metrics.npz` and `ul_*/nulling_cdf_metrics.npz`: combined and separate
  raw metric arrays, oracle arrays and diagnostic statistics, without pickle.
- `summary.csv`: sample counts and 10th/50th/90th percentiles for INR, SNR and SINR,
  including the no-nulling and oracle comparisons. Quantiles use finite samples;
  total and finite counts are reported separately.
- `channels/` and `tn_drop/`: shared DL channels, per-frequency UL channels,
  path/MUSIC diagnostics and TN admission reports.
- `nulling_inr_cdf.{png,pdf}` and `nulling_tn_sinr_cdf.{png,pdf}`: result figures;
  `plot_snr=True` additionally saves `nulling_tn_snr_cdf.{png,pdf}`.
- `cdf_style_map.json`: the color/lambda and dash/frequency mapping.
- `artifacts.json`: file names and sizes at successful completion.

The color legend lists lambda values and the black no-nulling reference once.
The line-style legend lists UL offsets and frequencies once. Optional DL oracle
curves use circle markers to distinguish them from the frequency styles.

Run all notebook cells with the `sionna20` kernel, or execute:

```bash
/home/shizhen/miniconda3/envs/sionna20/bin/python run_sector_drop.py
```

The command-line runner also saves `Nulling_CDF_SectorDrop.executed.ipynb` and
`notebook_execution.log` in the run directory, and refreshes the working notebook
outputs after successful completion. On execution failure it preserves the partial
notebook and diagnostic log without replacing the working notebook.
