# MUSIC snapshot experiment

Open `MUSIC_Snapshot_Experiment.ipynb` in the project root and use the existing
`sionna20` kernel. Run all cells to generate the two paper figures. All new Python
code, notebook text, comments and generated reports are in English.

Defaults: 20 frozen SectorDrop scenes with 50 NTN UEs each, five waveform/noise repeats,
eight snapshot budgets from 64 to 8192, 8 GHz UL/DL, lambda=1e13, and four CPU workers.
The CDF shows T=64, 1024 and 8192 plus one no-nulling baseline. The detector scans the
same 0.5-degree grid as the CDF notebook. No new ray tracing is performed.

Outputs are created in `result/MUSIC_SNAPSHOT/<timestamp>/`, alongside the existing
`result/map/` and `result/CDF/` directories. The main outputs are:

- `snapshot_detection_ab.pdf/png`: side-by-side UE count and coverage panels.
- `snapshot_inr_cdf.pdf/png`: one single-column INR CDF.
- `detection_summary.csv`, `nulling_summary.csv`, `RESULTS.md`: numerical results.
- `episodes/`: pickle-free checkpoints with per-user powers, detections and matches.
- `run_config.json`, `channel_manifest.json`, `sources/`: reproducibility records.
- `noise_validation.json`, `observation_validation.json`, `render_validation.json`:
  validation evidence. Source-count false alarms bound peak false alarms.
- `status.json`, `progress/`, `run.log`: progress and failure diagnostics.

Panel (b) shows path-power-weighted recall and recall among UEs whose fixed
no-nulling network INR exceeds -3 dB. Ray power excludes coherent cross terms;
matching a UE at any sector is not proof of successful interference protection.
The INR CDF still includes the complete fixed DL-linked population. Original
path precision, path recall and matched angular errors remain in the statistics.
All intervals resample scenes with their repeats and snapshot budgets kept together.

The new 64-snapshot calibration uses 2,000 pure-noise training trials, followed by
1,000 independent validation trials at each budget. Its frozen rule is
`margin(T) = 0.70 * sqrt(64 / T)`. The notebook performs a further independent
noise check. Calibration provenance is copied into `sources/rank_calibration/`.
Snapshot counts do not specify physical observation durations.

The root notebook exposes `resume_directory`; set it to an existing result folder
to reuse verified checkpoints. A missing path never triggers new ray tracing.
Each short record is a prefix of the same long waveform/noise record. All UEs
remain active, user waveforms are shared across sectors, and sector noises are
independent. Truth matching is exclusively an offline metric. The sensing-noise
normalization is read from the source CDF configuration, not silently changed to
the separate listening experiment's uplink-power setting.

The experiment reuses the existing shared MUSIC, covariance fitting, beamforming,
and observation primitives without modifying the CDF or map pipelines. This
folder contains the experiment-specific configuration, cache adapter, matching,
runner, statistics, plotting, validation, tests and optional CLI.

Optional CLI, from the project root:

```bash
/home/shizhen/miniconda3/envs/sionna20/bin/python -m snapshot_experiments
/home/shizhen/miniconda3/envs/sionna20/bin/python -m snapshot_experiments --resume result/MUSIC_SNAPSHOT/<timestamp>
OPENBLAS_NUM_THREADS=1 /home/shizhen/miniconda3/envs/sionna20/bin/python -m unittest discover -s snapshot_experiments/tests -v
```

Keep BLAS thread counts at one per worker. For notebook execution, restart the
kernel before the first cell so the thread settings apply before NumPy loads.
