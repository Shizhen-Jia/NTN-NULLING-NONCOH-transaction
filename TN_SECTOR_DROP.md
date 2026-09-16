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

The notebook currently uses p=0.5 and `max_depth=3`, with reflection and
refraction enabled. The original direct-path validation used p=1 as an explicit
outdoor baseline. Indoor feasibility still depends on the propagation settings,
materials and geometry; see the limitation below.

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

New runs use `result/YYYYMMDD_HHMMSS_microseconds/`, leaving old experiments intact.
See [FDD_MULTIPATH_ANALYSIS.md](FDD_MULTIPATH_ANALYSIS.md#run-archive-and-plotting)
for the full archive contents, compact legends, and command-line runner.

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


## FDD and multipath processing

The frequency-sweep implementation, coherent multipath processing, NLOS controls,
diagnostics, frequency-transfer analysis and run archive details have moved to
[FDD_MULTIPATH_ANALYSIS.md](FDD_MULTIPATH_ANALYSIS.md).

The current notebook uses 20 macro realizations and UL offsets
`[-20, -10, -5, 0, 5, 10, 20]%` around a fixed 7 GHz DL carrier.
