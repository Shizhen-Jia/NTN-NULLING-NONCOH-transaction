"""Paired activity sweeps with anonymous sensing and offline active-path scoring."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import json
import multiprocessing as mp
import shutil
import time
import numpy as np

import ntn_music_detection as nmd
from listening_experiments.detection import detect_directions
from listening_experiments.evaluation import evaluate_beams, final_beams
from listening_experiments.observations import pooled_observations, window_observations
from listening_experiments.storage import atomic_json, atomic_npz, file_hash, fingerprint, read_npz
from snapshot_experiments.cache import load_inputs
from snapshot_experiments.config import SnapshotConfig, reference_music_config
from snapshot_experiments.metrics import match_directions, coverage_metrics
from snapshot_experiments.storage import SHARED_FILES, computation_signature, run_log
from snapshot_experiments.validation import validate_noise
from .config import ActivityConfig, ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def inputs(config):
    config.validate()
    matched = SnapshotConfig(source_run=config.source_run, num_scenes=config.num_scenes,
        num_ntn_ues=config.num_ntn_ues, repeats=config.repeats, snapshots=config.snapshots,
        cdf_snapshots=(config.snapshots[0], config.snapshots[-1]), seed=config.seed,
        workers=config.workers, match_angle_deg=config.match_angle_deg)
    cases, physical, observation, channels = load_inputs(matched)
    observation = SimpleNamespace(**vars(observation))
    observation.p_values = config.p_values
    observation.listening_windows = config.windows
    music = reference_music_config()
    previous = json.loads((config.path(config.continuous_run) / 'run_config.json').read_text())
    assert previous['config']['seed'] == config.seed
    assert previous['config']['block_size'] == config.snapshots_per_window
    assert fingerprint(previous['identity']['music']) == fingerprint(asdict(music))
    assert previous['identity']['channels']['sensing_noise_normalized'] == observation.sensing_noise_normalized
    assert previous['identity']['channels']['source_run'] == channels['source_run']
    for case, old in zip(cases, previous['identity']['channels']['scenes']):
        assert case.channel.source_id == old['source_id']
    channels['continuous_archive'] = str(config.path(config.continuous_run))
    channels['continuous_config_sha256'] = file_hash(config.path(config.continuous_run) / 'run_config.json')
    channels['continuous_episodes'] = [dict(path=str(p), sha256=file_hash(p))
        for case in cases for repeat in range(config.repeats)
        for p in [config.path(config.continuous_run) / 'episodes' /
                  f'macro_{case.channel.macro_id:04d}_repeat_{repeat:03d}.npz']]
    return matched, cases, physical, observation, music, channels


def archive(directory, config):
    files = [ROOT / name for name in SHARED_FILES]
    files += list((ROOT / 'snapshot_experiments').glob('*.py'))
    files += list((ROOT / 'activity_experiments').rglob('*.py'))
    for path in files:
        target = directory / 'sources' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in (config.path(config.continuous_run) / 'sources/fonts').glob('*.TTF'):
        target = directory / 'sources/fonts' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    notebook = ROOT / 'Intermittent_Activity_Experiment.ipynb'
    if notebook.exists():
        shutil.copy2(notebook, directory / 'sources' / notebook.name)


def create_run(config, channels, music, physical, observation, resume=None):
    computation = computation_signature()
    for name in ['config.py', 'runner.py']:
        computation['activity_experiments/' + name] = file_hash(ROOT / 'activity_experiments' / name)
    settings = {k: v for k, v in asdict(config).items() if k not in ['workers', 'output_root', 'bootstrap_draws', 'coverage_trials', 'conditional_trials']}
    identity = dict(settings=settings, channels=channels, music=asdict(music),
        physical=asdict(physical), noise=observation.sensing_noise_normalized, code=computation)
    key = fingerprint(identity)
    directory = config.path(resume) if resume else config.path(config.output_root) / datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d_%H%M%S_%f')
    if resume:
        assert json.loads((directory / 'run_config.json').read_text())['run_id'] == key, 'Resume identity mismatch'
    else:
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'run_config.json', dict(run_id=key, config=asdict(config), identity=identity,
            created=now(), estimator='Finite-snapshot rank-MUSIC with spatial smoothing and correlated power fitting',
            assumptions='Independent Bernoulli activity per UE/window; shared UE activity and symbols across sectors; static channels; independent white receiver noise; no truth in beam design',
            matching='One-to-one 5-degree matches against paths of UEs with J>0; silent UEs remain in INR evaluation',
            continuous_reuse='Identical p=1 prefixes are replayed from the validated snapshot archive and checked numerically at lambda=1e13'))
        atomic_json(directory / 'channel_manifest.json', channels)
    atomic_json(directory.parent / 'latest_run.json', dict(directory=str(directory), run_id=key))
    archive(directory, config)
    return directory, key


def check_observations(case, observation):
    short = SimpleNamespace(**vars(observation))
    short.p_values, short.listening_windows = (0.0, .1, .3, 1.0), (1, 2)
    blocks = [window_observations(case.channel, short, 0, w) for w in (0, 1)]
    for L, covariance, counts in pooled_observations(case.channel, short, 0):
        y = np.concatenate([b[0] for b in blocks[:L]], axis=-1)
        np.testing.assert_allclose(covariance, y @ y.swapaxes(-1, -2).conj() / y.shape[-1], rtol=1e-12, atol=1e-25)
        np.testing.assert_array_equal(counts, np.sum([b[1] for b in blocks[:L]], axis=0))
        assert np.all(counts[0] == 0) and np.all(counts[-1] == L)
        assert np.all(np.diff(counts, axis=0) >= 0)
    return dict(passed=True, explicit_covariance_agrees=True, prefix_activity_agrees=True,
                nested_p_values=True, silent_and_continuous_endpoints=True)


def cached_estimates(case, record, budget):
    index = int(np.flatnonzero(record['snapshots'] == budget)[0])
    peaks = record['peaks'][record['peaks'][:, 0] == budget]
    directions, diagnostics = [], []
    for b in range(len(case.pairs)):
        group = peaks[peaks[:, 1] == b]
        vectors = np.asarray([nmd.array_position_steering_global(phi, theta,
            case.channel.array_positions_dl, orientation_rad=tuple(case.channel.orientations[b])).ravel()
            for phi, theta in group[:, 2:4]], complex).reshape(-1, case.channel.h_ntn_ul.shape[-1])
        directions.append(dict(u=vectors, phi=group[:, 2], theta=group[:, 3], g=group[:, 4]))
        diagnostics.append(dict(num_sources=int(record['num_sources'][index, b]),
            accepted_peaks=int(record['accepted_peaks'][index, b]),
            noise_estimate=float(record['noise_estimate'][index, b]), covariance_fit=float(record['covariance_fit'][index, b])))
    return directions, diagnostics, index


def episode(case, config, music, physical, observation, directory, key, repeat):
    start = time.perf_counter()
    P, K, W = len(config.p_values), len(config.windows), len(config.lambdas)
    B, N, U = len(case.pairs), len(case.ue_ids), config.num_ntn_ues
    inr, sinr = np.empty((P, K, W, N)), np.empty((P, K, W, B))
    activity = np.empty((P, K, U), np.int64)
    detected = np.zeros((P, K, U), bool)
    confusion = np.empty((P, K, B, 3), np.int64)
    sources, accepted = np.empty((P, K, B), np.int64), np.empty((P, K, B), np.int64)
    noise, fit = np.empty((P, K, B)), np.empty((P, K, B))
    path_power, high_counts = np.empty((P, K, B, 2)), np.empty((P, K, 2), np.int64)
    reused = np.zeros((P, K), bool)
    peak_rows, match_rows = [], []
    previous = read_npz(config.path(config.continuous_run) / 'episodes' / f'macro_{case.channel.macro_id:04d}_repeat_{repeat:03d}.npz')
    np.testing.assert_array_equal(previous['ue_ids'], case.ue_ids)
    np.testing.assert_array_equal(previous['raw_inr_linear'], case.baseline_inr)
    for li, (windows, covariance, counts) in enumerate(pooled_observations(case.channel, observation, repeat)):
        assert windows == config.windows[li]
        budget = windows * config.snapshots_per_window
        for pi, probability in enumerate(config.p_values):
            old_index = None
            if probability == 1.0 and budget in previous['snapshots']:
                estimates, diagnostics, old_index = cached_estimates(case, previous, budget)
                reused[pi, li] = True
            else:
                estimates, diagnostics = detect_directions(covariance[pi],
                    positions_dl=case.channel.array_positions_dl, orientations=case.channel.orientations,
                    f_ul=case.channel.f_ul, f_dl=case.channel.f_dl, snapshots=budget,
                    noise_normalized=observation.sensing_noise_normalized, config=music, power_correction=True)
            activity[pi, li] = counts[pi]
            active_truth = dict(case.truth)
            active_truth['valid'] = case.truth['valid'] & (counts[pi] > 0)[:, None, None]
            evaluation = match_directions(active_truth, estimates, config.match_angle_deg)
            assert not np.any(evaluation['detected_mask'] & (counts[pi] == 0))
            detected[pi, li], confusion[pi, li] = evaluation['detected_mask'], evaluation['counts']
            coverage = coverage_metrics(case.truth, evaluation, case.ue_ids, case.baseline_inr, config.threshold_db)
            path_power[pi, li], high_counts[pi, li] = coverage['path_power_by_sector'], coverage['high_inr_ue_counts']
            peak_rows.extend((probability, windows, *row) for row in evaluation['peaks'])
            match_rows.extend((probability, windows, *row) for row in evaluation['matches'])
            for wi, weight in enumerate(config.lambdas):
                setting = replace(physical, lambda_weight=weight)
                inr[pi, li, wi], sinr[pi, li, wi] = evaluate_beams(case.channel, case.pairs,
                    final_beams(case.pairs, estimates, setting), case.ue_ids, setting)
                if old_index is not None and weight == 1e13:
                    np.testing.assert_allclose(inr[pi, li, wi], previous['inr_linear'][old_index], rtol=1e-9, atol=1e-15)
                    np.testing.assert_allclose(sinr[pi, li, wi], previous['sinr_linear'][old_index], rtol=1e-9, atol=1e-15)
                    np.testing.assert_array_equal(detected[pi, li], previous['detected_ue_mask'][old_index])
            sources[pi, li] = [d['num_sources'] for d in diagnostics]
            accepted[pi, li] = [d['accepted_peaks'] for d in diagnostics]
            noise[pi, li] = [d['noise_estimate'] for d in diagnostics]
            fit[pi, li] = [d['covariance_fit'] for d in diagnostics]
            atomic_json(directory / 'progress' / f'macro_{case.channel.macro_id:04d}_repeat_{repeat:03d}.json',
                dict(macro_id=case.channel.macro_id, repeat=repeat, windows=windows, p=probability,
                     finished_network_evaluations=li*P+pi+1, total_network_evaluations=P*K, updated=now()))
    for array in [inr, sinr]:
        assert np.all(np.isfinite(array)) and np.all(array >= 0)
    elapsed = time.perf_counter()-start
    atomic_npz(directory / 'episodes' / f'macro_{case.channel.macro_id:04d}_repeat_{repeat:03d}.npz',
        run_id=np.array(key), macro_id=np.array(case.channel.macro_id), repeat_id=np.array(repeat),
        p_values=np.asarray(config.p_values), windows=np.asarray(config.windows), lambdas=np.asarray(config.lambdas),
        ue_ids=case.ue_ids, high_inr_ue_ids=coverage['high_inr_ue_ids'],
        tn_user_ids=np.asarray([p['tn_idx'] for p in case.pairs]),
        raw_inr_linear=case.baseline_inr, raw_sinr_linear=case.baseline_sinr,
        inr_linear=inr, sinr_linear=sinr, J=activity, detected_ue_mask=detected,
        active_tp_fp_fn=confusion, path_power_coverage=path_power, high_inr_ue_counts=high_counts,
        num_sources=sources, accepted_peaks=accepted, noise_estimate=noise, covariance_fit=fit,
        reused_continuous=reused, peaks=np.asarray(peak_rows).reshape(-1, 6),
        matches=np.asarray(match_rows).reshape(-1, 7), runtime_seconds=np.array(elapsed))
    return case.channel.macro_id, repeat, elapsed, int(reused.sum())


def run(config=ActivityConfig(), resume=None):
    matched, cases, physical, observation, music, channels = inputs(config)
    directory, key = create_run(config, channels, music, physical, observation, resume)
    with run_log(directory):
        print('Activity output directory:', directory, flush=True)
        atomic_json(directory / 'observation_validation.json', check_observations(cases[0], observation))
        reference = json.loads((config.path(config.continuous_run) / 'noise_validation.json').read_text())
        assert reference['passed']
        covered = {r['T'] for r in reference['rows']}
        new_budgets = tuple(t for t in config.snapshots if t not in covered)
        atomic_json(directory / 'noise_validation_reference.json', reference)
        if new_budgets:
            noise_config = replace(matched, snapshots=new_budgets, cdf_snapshots=new_budgets, seed=config.validation_seed)
            validate_noise(directory, cases[0], noise_config, music)
        tasks, completed = [], 0
        for case in cases:
            for repeat in range(config.repeats):
                path = directory / 'episodes' / f'macro_{case.channel.macro_id:04d}_repeat_{repeat:03d}.npz'
                if path.exists():
                    record = read_npz(path)
                    assert str(record['run_id']) == key
                    np.testing.assert_array_equal(record['ue_ids'], case.ue_ids)
                    completed += 1
                else:
                    tasks.append((case, repeat))
        total = config.num_scenes*config.repeats
        def status(state, **extra):
            atomic_json(directory / 'status.json', dict(status=state, completed_episodes=completed,
                total_episodes=total, updated=now(), **extra))
        status('running')
        try:
            with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp.get_context('spawn')) as pool:
                futures = [pool.submit(episode, case, config, music, physical, observation, directory, key, repeat)
                           for case, repeat in tasks]
                for future in as_completed(futures):
                    macro, repeat, seconds, reused = future.result()
                    completed += 1
                    status('running')
                    print(f'[{completed}/{total}] scene={macro}, repeat={repeat}, {seconds:.1f}s, reused={reused}', flush=True)
            status('simulation_complete')
        except BaseException as error:
            status('failed', error=f'{type(error).__name__}: {error}')
            raise
    return directory


def records(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'run_config.json').read_text())
    config = ActivityConfig(**manifest['config'])
    loaded = []
    for macro in range(config.num_scenes):
        for repeat in range(config.repeats):
            record = read_npz(directory / 'episodes' / f'macro_{macro:04d}_repeat_{repeat:03d}.npz')
            assert str(record['run_id']) == manifest['run_id']
            np.testing.assert_array_equal(record['p_values'], config.p_values)
            np.testing.assert_array_equal(record['windows'], config.windows)
            np.testing.assert_array_equal(record['lambdas'], config.lambdas)
            loaded.append(record)
    return config, loaded
