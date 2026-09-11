# Sector-Constrained TN Drop

## Independent Files

- `Nulling_CDF_SectorDrop.ipynb`: copy of the main notebook using the new placement.
- `SceneConfigSionnaSectorDrop.py`: independent copy of the scene class.
- `tn_sector_drop.py`: NumPy geometry sampling and channel-dominance checks.
- `tests/test_tn_sector_drop.py`: geometry, probability, association, and retry tests.

The original `Nulling_CDF.ipynb`, `SceneConfigSionna.py`, and MUSIC implementation
are unchanged. The shared nulling utility keeps its legacy single-frequency API
and additionally supports the optional FDD sweep described below.

Run the new notebook from its import cell. Its scene-class name remains
`SceneConfigSionna`, imported from the new module.

## Geometry

The variant requires four BS sites with `bs_grid=(2, 2)`, three sectors per
site, and `tn_rx=12`. Existing BS placement, heights, and sector downtilt
are retained.

Split the coverage-grid bounding rectangle at its x and y midpoints. Each
quadrant must contain exactly one BS. For the current scene, each rectangle
is about 5.040 km by 5.045 km. These dimensions are not the BS-to-BS distances.

Within each rectangle, assign every grid point to the closest horizontal
sector boresight of its own BS, using circular azimuth distance. With zero yaw
offset, centers are 0, 120, and 240 degrees; each angular interval has width
120 degrees. Sector wedges are clipped to the rectangle, so their areas need
not be equal. Boundaries have deterministic ownership; the BS's exact
horizontal position is excluded from candidate UE positions.

The partition follows the actual BS position and the configured sector yaw.
It is a horizontal partition, not a 3D boresight cone. Downtilt still affects
the ray-traced channels and therefore the acceptance test.

## Notebook Controls

```python
tn_rx = 12
tn_outdoor_probability = 1.0
tn_association_margin_db = 3.0
tn_max_attempts = 256
tn_candidates_per_batch = 16
```

- `tn_outdoor_probability`: one global probability p for every region.
  Indoor probability is exactly 1-p.
- `tn_association_margin_db`: required channel-power advantage over the
  strongest of all other eleven sectors, including the two sibling sectors
  at the same BS.
- `tn_max_attempts`: maximum distinct sampled grid candidates per region,
  including its initial candidate. Sampling also stops if its pool is exhausted.
- `tn_candidates_per_batch`: candidates traced per unresolved region on each
  retry. The first trace contains one candidate per region.
- The notebook passes `tn_min_channel_norm=h_tn_th`, preserving the original
  absolute link-admission threshold as well as the new relative margin.

The new notebook explicitly uses p=1 for a runnable outdoor baseline. It is
not an implicit fallback. Set p=0.4 for 40% outdoor / 60% indoor probability,
subject to the indoor propagation limitation below.

Heights are copied from the existing class:

```python
self.tn_height_above_ground = 1.8
self.tn_height_above_roof = -1.5
```

Outdoor candidates come from non-building grid cells at local ground +1.8 m.
Indoor candidates come from building cells at local roof -1.5 m. This is a
geometry-based indoor placement rule, not an added indoor penetration-loss model.

## Per-Macro Workflow

1. Place BS and NTN devices with the existing scene logic.
2. Independently draw indoor/outdoor type once for each of the 12 regions.
3. Randomly permute eligible grid cells for that region and type.
4. Trace candidate TN channels to all 12 sectors. Orient each UE toward its
   assigned BS, not a separately chosen nearest BS.
5. Accept the first passing candidate in sampled order for each region.
   Keep accepted positions fixed and resample only unresolved regions.
6. After all regions succeed, trace only the 12 retained UEs again and validate
   their channels. The final positions, CIR, and path object are aligned.
7. Run the unchanged strongest-sector pairing and MUSIC/nulling experiment.

The accepted UE order is TX index order: `target_tx = 3 * bs_index + sector_index`.
The dominance check guarantees that the unchanged strongest-sector pairing
chooses these targets. A successful macro therefore has one UE per sector and
`min_count=1`, yielding 12 SNR/SINR samples. INR sample count still depends on
the NTN evaluation mask, not the number of TNs.

Indoor/outdoor type is never re-drawn during retries. The number of outdoor UEs
in a macro is random, not necessarily exactly 12*p. Positions are uniform
candidate draws over the selected grid pool before channel rejection; accepted
positions are intentionally conditioned on feasible dominance and link strength.

The sampler derives its private RNG from the seeded placement context and uses
it for all later retries. It does not depend on unrelated global RNG changes
during path tracing.

## Association Test

Use the same narrowband channel collapse as the main experiment. For UE i and
sector t, define:

```text
P[i,t] = ||H[i,t]||_F^2
P_other = max(P[i,t] for t != target[i])
margin_dB = 10*log10(P[i,target[i]] / P_other)
```

Acceptance requires:

- Finite channel powers for all sectors.
- Serving channel norm strictly above both `h_tn_th` and the existing numerical
  nonzero threshold.
- Strictly greater serving power than every competing sector, including when
  the configured margin is zero.
- Achieved margin greater than or equal to the configured dB threshold.

A positive serving power with zero competing power has infinite margin.
An all-zero channel or a tie is rejected. The power ratio is not a raw amplitude
difference. At 3 dB the serving power must be approximately twice the largest
competing power. This test is pre-nulling channel association, not a post-nulling
SINR guarantee.

## Indoor Propagation Limitation

The inherited notebook uses `max_depth=0`. In the examined original cache
`result/music_improved_20260908_172943_920674/channels/channels_0000.npz`,
all 120 indoor TNs had zero channels. The geometry puts them below the roof;
the present direct-path configuration does not guarantee a usable indoor link.

A mixed indoor/outdoor run may therefore be infeasible. The new implementation
does not silently replace indoor UEs, reduce the margin, reassign their sector,
or skip a failed macro. It stops with an explicit error after the finite
candidate budget. The error identifies the failed BS/sector and sampled type.

Increasing the budget alone cannot fix an absent propagation mechanism.
Changing ray-tracing depth or material/penetration settings needs a separate
physical-model validation; this feature does not claim to solve indoor coverage.
If users later discard failed macros manually, the retained ensemble may have a
biased indoor/outdoor composition even though type draws were initially correct.

## Outputs

New runs use `result/sector_drop_<timestamp>/`, leaving old experiments intact.

In addition to the existing metrics, channels, and CDF plots:

```text
run_config.json
tn_drop/drop_0000.json
tn_drop/drop_0001.json
...
```

Each TN-drop report contains rectangle bounds, target sectors, the fixed
indoor/outdoor types, sampled candidate counts, accepted flags, and best observed
relative margins. Successful reports also contain final positions, final margins,
serving channel powers, and strongest competing powers.

Reports use exclusive creation to avoid overwriting earlier diagnostics.
Nonfinite margins are stored as JSON null; successful infinite margins can be
identified using `zero_competitor_power`. A best-margin value can describe a
candidate that failed the absolute-strength test; it is not alone proof of
acceptance. The ordinary channel caches contain only the final 12 TNs.

## Verification

The regression suite includes the original MUSIC tests plus geometry, height,
probability endpoints, seeded sampling, non-repeating bounded retries, all-eleven
comparison, zero/tie/nonfinite rejection, final channel revalidation, and failure
reporting.

```sh
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

A real Denver outdoor-only macro with a 3 dB required margin passed the entire
MUSIC/nulling/CDF pipeline:

- 12 UEs, 12 distinct serving sectors, one UE per sector.
- 188 candidate evaluations before the final 12-UE trace.
- Achieved margins from approximately 4.33 dB to 27.60 dB.
- 12 SNR/SINR samples and 16 INR samples.
- Artifacts: `result/sector_drop_20260909_161919_021588/`.

This validates the new placement/association workflow for that realization,
not universal feasibility at every probability, margin, or propagation setting.

A second real-scene check used p=0.4 and a 17-candidate budget per region.
The fixed type draw produced three outdoor and nine indoor UEs. All three
outdoor regions succeeded; all nine indoor regions exhausted their budgets
without a positive serving channel. The run stopped as intended and preserved
the original types. Diagnostic artifact:
`result/sector_drop_20260909_162030_336797/tn_drop/drop_0000.json`.


## Fixed DL array and UL frequency sweep

The parameter cell of `Nulling_CDF_SectorDrop.ipynb` now includes:

```python
f_dl = 7e9
ul_frequency_percentages = [-10.0, -5.0, 0.0, 5.0, 10.0]
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

The default settings produce 3 x 5 = 15 estimated curves plus one no-nulling
baseline per CDF. Lambda controls color; percentage controls line style and each
pair has its own legend entry. Set `plot_oracle=True` for one additional true-DL
oracle per lambda, or `plot_snr=True` for the optional SNR plot. The oracle does
not depend on UL frequency and is only drawn once per lambda.

Runs are saved under `result/sector_drop_fdd_<timestamp>/`. Access one case as:

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
The default is **`multipath = 0`**, retaining the old propagation, estimation,
and numerical CDF results. To enable the extension, change the parameter cell:

```python
multipath = 1
multipath_max_depth = 2
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
retained. Multipath options are inactive when `multipath=0`, including the NLOS
stress-test setting. Disabling multipath sets trace depth to zero and uses the
original diagonal covariance refinement. Enabling it selects the correlated
multipath pipeline and disables that incompatible diagonal refinement.

`natural` includes all valid direct and indirect paths that the environment and
selected propagation mechanisms support. A link is classified as NLOS if it has
at least one valid nonzero indirect path and no valid nonzero direct path.
`nlos_only` suppresses direct paths **only on NTN links**, at both UL and DL;
this is an artificial direct-component removal test. TN links retain natural
LOS/NLOS and gain multipath as well, keeping their serving-sector admission
rule. A link with no valid path is reported separately; it is not called NLOS.

Depth, reflection/refraction/scattering/diffraction switches, tracing budgets,
and solver seed are shared between NTN UL and DL. Direct-path mode preserves
the original solver defaults. Multipath defaults to specular reflection and
refraction, depth 2, 100,000 launched samples and a 100,000 path cap per source.
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
multipath is enabled. They contain full TN/NTN CIRs and delays, valid-path masks,
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
Pre-change CDF arrays were also saved and compared exactly with disabled mode.
Real Sionna tests used a direct-plus-reflected link and its direct-path-removed
variant at three frequencies. A small Denver run exercised 12 TNs, eight NTN UEs,
an 8x8 array, three frequencies and two lambdas through saving and CDF plotting.
