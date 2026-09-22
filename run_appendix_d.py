#!/usr/bin/env python3
"""Run E4--E8 without loading or modifying the original SectorDrop notebook."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import sys
from appendix_d_experiments.reporting import json_file, output_dir

ROOT=Path(__file__).resolve().parent


def run_suite(output,experiments=('E4','E5','E6','E7','E8'),quick=True,seed=20260921,
              gamma_db=(-15,-10,-5,0),theta_dl=.40,theta_ul=.50,delta_tn=.10,delta_ntn=.08,
              cache_dir=None,cache_max_sectors=None):
    from appendix_d_experiments.dynamic import ModelConfig
    from appendix_d_experiments.e45 import run_e4,run_e5
    from appendix_d_experiments.e6 import run as run_e6
    from appendix_d_experiments.e78 import run_e7,run_e8
    original=ROOT/'Nulling_CDF_SectorDrop.ipynb'
    before=hashlib.sha256(original.read_bytes()).hexdigest()
    out=output_dir(output)
    if (out/'manifest.json').exists():
        raise FileExistsError(f'{out} already has a run manifest; choose a new output directory.')
    cfg=ModelConfig(theta_dl=theta_dl,theta_ul=theta_ul,delta_tn=delta_tn,delta_ntn=delta_ntn)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),profile='quick' if quick else 'full',
                  seed=seed,experiments=list(experiments),gamma_db=list(gamma_db),model=asdict(cfg),
                  original_notebook_sha256=before,python=sys.version,
                  packages={p:importlib.metadata.version(p) for p in
                            ('numpy','scipy','matplotlib','cvxpy','clarabel')},
                  scope='Synthetic finite-model validation; optional cache adapter is static empirical only.',
                  status='running',completed=[])
    archive=output_dir(out/'source')
    sources=list((ROOT/'appendix_d_experiments').glob('*.py'))+[Path(__file__).resolve(),
            ROOT/'requirements-appendix-d.txt',ROOT/'APPENDIX_D_EXPERIMENTS.md',
            ROOT/'Nulling_CDF_SectorDrop_AppendixD.ipynb']
    manifest['source_sha256']={}
    for source in sources:
        if source.is_file():
            relative=source.relative_to(ROOT)
            target=archive/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
            manifest['source_sha256'][str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    json_file(out/'manifest.json',manifest)
    try:
        for name in experiments:
            print(f'Running {name} -> {out/name}',flush=True)
            if name=='E4': run_e4(out/name,seed=seed+4,quick=quick)
            elif name=='E5': run_e5(out/name,seed=seed+5,quick=quick)
            elif name=='E6': run_e6(out/name,seed=seed+6)
            elif name=='E7': run_e7(out/name,seed=seed,quick=quick,gamma_db=gamma_db,config=cfg)
            elif name=='E8':
                from dataclasses import replace
                run_e8(out/name,seed=seed,quick=quick,config=replace(cfg,gamma_db=-10))
            else: raise ValueError(f'Unknown experiment {name}')
            manifest['completed'].append(name)
            json_file(out/'manifest.json',manifest)
        if cache_dir:
            from appendix_d_experiments.cache_adapter import run_cached_spatial
            run_cached_spatial(out/'cached_spatial',cache_dir,gamma_db=gamma_db,seed=seed,
                               max_sectors=cache_max_sectors)
            manifest['completed'].append('cached_spatial')
        after=hashlib.sha256(original.read_bytes()).hexdigest()
        manifest['original_notebook_unchanged']=before==after
        if before!=after:
            raise RuntimeError('Original notebook changed during this run; investigate concurrent edits.')
        manifest['status']='complete'
    except Exception as error:
        manifest['status']='failed'
        manifest['error']=str(error)
        raise
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        json_file(out/'manifest.json',manifest)
    print(f'Completed. Figures/tables/raw data: {out.resolve()}',flush=True)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',choices=['E4','E5','E6','E7','E8'],default=['E4','E5','E6','E7','E8'])
    parser.add_argument('--profile',choices=['quick','full'],default='quick')
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--gamma-db',type=float,nargs='+',default=[-15,-10,-5,0])
    parser.add_argument('--theta-dl',type=float,default=.40)
    parser.add_argument('--theta-ul',type=float,default=.50)
    parser.add_argument('--delta-tn',type=float,default=.10)
    parser.add_argument('--delta-ntn',type=float,default=.08)
    parser.add_argument('--cache-dir',type=Path)
    parser.add_argument('--cache-max-sectors',type=int)
    parser.add_argument('--output-dir',type=Path,default=None)
    args=parser.parse_args()
    path=args.output_dir or ROOT/'result'/('appendix_d_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run_suite(path,experiments=args.experiments,quick=args.profile=='quick',seed=args.seed,
              gamma_db=args.gamma_db,theta_dl=args.theta_dl,theta_ul=args.theta_ul,
              delta_tn=args.delta_tn,delta_ntn=args.delta_ntn,
              cache_dir=args.cache_dir,cache_max_sectors=args.cache_max_sectors)


if __name__=='__main__':
    main()
