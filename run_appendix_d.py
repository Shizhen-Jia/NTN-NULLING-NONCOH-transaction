#!/usr/bin/env python3
"""Run finite Appendix D experiments and optional fresh or cached Sionna RT validation."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import sys

from appendix_d_experiments.reporting import json_file, output_dir

ROOT = Path(__file__).resolve().parent


def run_suite(output, experiments=('E4', 'E5', 'E6', 'E7', 'E8'), quick=True,
              seed=20260921, gamma_db=(-15, -10, -5, 0), theta_dl=.40,
              theta_ul=.50, delta_tn=.10, delta_ntn=.08, cache_dir=None,
              cache_max_sectors=None, fresh_rt=False, rt_num_macros=None,
              rt_max_depth=None, e4_alpha=.10):
    from appendix_d_experiments.dynamic import ModelConfig
    from appendix_d_experiments.e45 import run_e4, run_e5
    from appendix_d_experiments.e6 import run as run_e6
    from appendix_d_experiments.e78 import run_e7, run_e8

    if fresh_rt and cache_dir is not None:
        raise ValueError('Choose fresh RT or an existing cache, not both.')
    if not fresh_rt and (rt_num_macros is not None or rt_max_depth is not None):
        raise ValueError('RT generation options require fresh_rt=True.')
    if not 0 < e4_alpha < 1:
        raise ValueError('e4_alpha must be strictly between zero and one.')
    experiments = tuple(experiments)
    gamma_db = tuple(gamma_db)
    if any(name not in ('E4', 'E5', 'E6', 'E7', 'E8') for name in experiments):
        raise ValueError('Unknown experiment; choose E4, E5, E6, E7, or E8.')
    if not experiments and not fresh_rt and cache_dir is None:
        raise ValueError('Select at least one experiment or a spatial data source.')
    spatial_mode = 'fresh_rt' if fresh_rt else ('cache' if cache_dir is not None else 'none')
    original = ROOT / 'Nulling_CDF_SectorDrop.ipynb'
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    out = output_dir(output)
    if (out / 'manifest.json').exists():
        raise FileExistsError(f'{out} already has a run manifest; choose a new output directory.')
    cfg = ModelConfig(theta_dl=theta_dl, theta_ul=theta_ul,
                      delta_tn=delta_tn, delta_ntn=delta_ntn)
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        profile='quick' if quick else 'full', seed=seed,
        experiments=list(experiments), gamma_db=list(gamma_db), model=asdict(cfg),
        e4_calibration_alpha=e4_alpha,
        spatial=dict(mode=spatial_mode, source_cache=str(cache_dir) if cache_dir is not None else None,
                     max_evaluated_sectors=cache_max_sectors,
                     requested_num_macros=rt_num_macros, requested_max_depth=rt_max_depth),
        original_notebook_sha256=before, python=sys.version,
        packages={p: importlib.metadata.version(p)
                  for p in ('numpy', 'scipy', 'matplotlib', 'cvxpy', 'clarabel')},
        scope='E4-E8: synthetic finite models. Optional fresh/cached Sionna RT: static spatial validation.',
        status='running', completed=[])
    archive = output_dir(out / 'source')
    sources = list((ROOT / 'appendix_d_experiments').glob('*.py')) + [
        Path(__file__).resolve(), ROOT / 'requirements-appendix-d.txt',
        ROOT / 'APPENDIX_D_EXPERIMENTS.md', ROOT / 'Nulling_CDF_SectorDrop_AppendixD.ipynb']
    if fresh_rt:
        sources += [ROOT / name for name in (
            'SceneConfigSionnaSectorDrop.py', 'ntn_music_detection.py', 'multipath_support.py',
            'nulling_cdf_utils.py', 'tn_sector_drop.py', 'sector_drop_reporting.py',
            'satellite_projection.py', 'vsat_dish_3gpp.py', 'BeamformingCalc.py')]
        sources += list((ROOT / 'sionnautils').rglob('*.py'))
    manifest['source_sha256'] = {}
    for source in sources:
        if source.is_file():
            relative = source.relative_to(ROOT)
            target = archive / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest['source_sha256'][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    json_file(out / 'manifest.json', manifest)
    try:
        for name in experiments:
            print(f'Running {name} -> {out / name}', flush=True)
            if name == 'E4':
                run_e4(out / name, seed=seed + 4, quick=quick, alpha=e4_alpha)
            elif name == 'E5':
                run_e5(out / name, seed=seed + 5, quick=quick)
            elif name == 'E6':
                run_e6(out / name, seed=seed + 6)
            elif name == 'E7':
                run_e7(out / name, seed=seed, quick=quick, gamma_db=gamma_db, config=cfg)
            elif name == 'E8':
                run_e8(out / name, seed=seed, quick=quick, config=replace(cfg, gamma_db=-10))
            manifest['completed'].append(name)
            json_file(out / 'manifest.json', manifest)
        if fresh_rt:
            from appendix_d_experiments.rt_source import generate_fresh_rt_cache
            print(f'Generating new UE drops and Sionna RT channels -> {out / "fresh_rt_cache"}', flush=True)
            generated = generate_fresh_rt_cache(
                out / 'fresh_rt_cache', quick=quick, seed=seed,
                num_macros=rt_num_macros, max_depth=rt_max_depth)
            cache_dir = generated['cache_dir']
            manifest['spatial']['generation'] = generated
            manifest['spatial']['source_cache'] = str(cache_dir)
            manifest['completed'].append('fresh_rt_generation')
            json_file(out / 'manifest.json', manifest)
        if cache_dir is not None:
            from appendix_d_experiments.cache_adapter import run_cached_spatial
            destination = out / ('fresh_rt_spatial' if fresh_rt else 'cached_spatial')
            run_cached_spatial(destination, cache_dir, gamma_db=gamma_db,
                               seed=seed, max_sectors=cache_max_sectors)
            manifest['completed'].append(destination.name)
        after = hashlib.sha256(original.read_bytes()).hexdigest()
        manifest['original_notebook_unchanged'] = before == after
        if before != after:
            raise RuntimeError('Original notebook changed during this run; investigate concurrent edits.')
        manifest['status'] = 'complete'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        json_file(out / 'manifest.json', manifest)
    print(f'Completed. Figures/tables/raw data: {out.resolve()}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=['E4', 'E5', 'E6', 'E7', 'E8'],
                        default=['E4', 'E5', 'E6', 'E7', 'E8'])
    parser.add_argument('--profile', choices=['quick', 'full'], default='quick')
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--gamma-db', type=float, nargs='+', default=[-15, -10, -5, 0])
    parser.add_argument('--theta-dl', type=float, default=.40)
    parser.add_argument('--theta-ul', type=float, default=.50)
    parser.add_argument('--delta-tn', type=float, default=.10)
    parser.add_argument('--delta-ntn', type=float, default=.08)
    parser.add_argument('--e4-alpha', type=float, default=.10,
                        help='E4 scene miscoverage target, held fixed across quick/full profiles.')
    spatial = parser.add_mutually_exclusive_group()
    spatial.add_argument('--cache-dir', type=Path, help='Evaluate a preserved SectorDrop cache.')
    spatial.add_argument('--fresh-rt', action='store_true',
                         help='Generate new UE drops and trace DL/UL channels before spatial evaluation.')
    parser.add_argument('--rt-macros', type=int, default=None,
                        help='Fresh independent macro drops: at least 6; default quick=6, full=20.')
    parser.add_argument('--rt-max-depth', type=int, default=None,
                        help='Fresh RT path interaction depth; see the recorded generation configuration.')
    parser.add_argument('--cache-max-sectors', type=int,
                        help='Limit evaluated held-out sectors for either fresh or existing caches.')
    parser.add_argument('--spatial-only', action='store_true',
                        help='Skip synthetic E4-E8; requires --fresh-rt or --cache-dir.')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if args.spatial_only and not (args.fresh_rt or args.cache_dir is not None):
        parser.error('--spatial-only requires --fresh-rt or --cache-dir.')
    if not args.fresh_rt and (args.rt_macros is not None or args.rt_max_depth is not None):
        parser.error('--rt-macros and --rt-max-depth require --fresh-rt.')
    path = args.output_dir or ROOT / 'result' / ('appendix_d_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_suite(path, experiments=() if args.spatial_only else args.experiments,
              quick=args.profile == 'quick', seed=args.seed, gamma_db=args.gamma_db,
              theta_dl=args.theta_dl, theta_ul=args.theta_ul,
              delta_tn=args.delta_tn, delta_ntn=args.delta_ntn, e4_alpha=args.e4_alpha,
              cache_dir=args.cache_dir, cache_max_sectors=args.cache_max_sectors,
              fresh_rt=args.fresh_rt, rt_num_macros=args.rt_macros, rt_max_depth=args.rt_max_depth)


if __name__ == '__main__':
    main()
