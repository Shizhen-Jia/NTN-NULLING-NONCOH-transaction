"""Independent aggregate checks and a complete provenance manifest."""
from pathlib import Path
import csv
import json
import numpy as np
from listening_experiments.storage import atomic_json, file_hash
from .runner import records, archive, now


def finish(directory):
    directory=Path(directory)
    config,loaded=records(directory)
    summary=json.loads((directory/'results_summary.json').read_text())
    threshold=10**(config.threshold_db/10)
    reference_checks=0
    for record in loaded:
        assert np.all(record['J']>=0)
        assert np.all(record['J']<=np.asarray(config.windows)[None,:,None])
        assert np.all(np.diff(record['J'],axis=1)>=0)
        assert np.all(np.diff(record['J'],axis=0)>=0)
        assert not np.any(record['detected_ue_mask'] & (record['J']==0))
        np.testing.assert_array_equal(record['J'][-1],np.broadcast_to(np.asarray(config.windows)[:,None],record['J'][-1].shape))
        previous_path=config.path(config.continuous_run)/'episodes'/f"macro_{int(record['macro_id']):04d}_repeat_{int(record['repeat_id']):03d}.npz"
        with np.load(previous_path) as previous:
            for li,budget in enumerate(config.snapshots):
                if budget in previous['snapshots']:
                    ti=int(np.flatnonzero(previous['snapshots']==budget)[0])
                    wi=config.lambdas.index(1e13)
                    np.testing.assert_allclose(record['inr_linear'][-1,li,wi],previous['inr_linear'][ti],rtol=1e-9,atol=1e-15)
                    reference_checks+=1
    for row in summary['rt']:
        pi=config.p_values.index(row['p']);li=config.windows.index(row['L']);wi=config.lambdas.index(row['lambda_value'])
        L=row['L'];rf=1.1*L;mrt=L;final=256-2.1*L
        q=[];qd=[];rate=[]
        for record in loaded:
            after=np.mean(record['inr_linear'][pi,li,wi]>threshold)
            before=np.mean(record['raw_inr_linear']>threshold)
            q.append(after);qd.append((mrt*before+final*after)/(256-rf))
            rate.append((mrt*np.log2(1+record['raw_sinr_linear']).mean()+final*np.log2(1+record['sinr_linear'][pi,li,wi]).mean())/256)
        np.testing.assert_allclose([row['q_out'],row['q_data'],row['r_eff']],
                                  [np.mean(q),np.mean(qd),np.mean(rate)],rtol=1e-12,atol=1e-14)
    with (directory/'failure_decomposition.csv').open() as stream:groups=list(csv.DictReader(stream))
    for row in summary['rt']:
        selected=[r for r in groups if float(r['p'])==row['p'] and int(r['L'])==row['L'] and float(r['lambda_value'])==row['lambda_value']]
        assert len(selected)==3
        assert sum(int(r['user_records']) for r in selected)==sum(len(r['ue_ids']) for r in loaded)
        np.testing.assert_allclose(sum(float(r['outage_contribution_scene_mean']) for r in selected),row['q_out'],atol=1e-14)
    noise=json.loads((directory/'noise_validation.json').read_text())
    reference=json.loads((directory/'noise_validation_reference.json').read_text())
    assert noise['passed'] and reference['passed']
    assert set(config.snapshots).issubset({r['T'] for r in noise['rows']+reference['rows']})
    assert json.loads((directory/'synthetic/covariance_sampler_validation.json').read_text())['passed']
    with (directory/'synthetic/coverage_50ue.csv').open() as stream:coverage=list(csv.DictReader(stream))
    for row in coverage:
        p=float(row['p']);L=int(row['L']);actual=float(row['q_miss_mc']);expected=float(row['q_miss_exact'])
        np.testing.assert_allclose(expected,1-(1-(1-p)**L)**50,atol=1e-12)
        assert abs(actual-expected)<=5*np.sqrt(expected*(1-expected)/config.coverage_trials)+1/config.coverage_trials
    with np.load(directory/'protection_minus6db_arrays.npz') as paper:
        assert float(paper['threshold_db']) == -6.0
        threshold_paper = 10**(float(paper['threshold_db'])/10)
        expected=np.mean([(r['inr_linear']>threshold_paper).mean(axis=-1) for r in loaded],axis=0)
        baseline=np.mean([np.mean(r['raw_inr_linear']>threshold_paper) for r in loaded])
        np.testing.assert_allclose(paper['q_out'],expected,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(paper['q_mrt'],baseline,rtol=1e-12,atol=1e-14)
    stems=['activity_coverage_tradeoff_ab','activity_coverage_protection_ab_minus6db','activity_rt_diagnostics','activity_failure_breakdown','synthetic_conditional_inr_ab','activity_inr_cdf']
    for stem in stems:
        for ext in ['pdf','png']:
            assert (directory/f'{stem}.{ext}').stat().st_size>1000
    atomic_json(directory/'independent_validation.json',dict(passed=True,episodes=len(loaded),
        continuous_prefix_comparisons=reference_checks,summary_rows_checked=len(summary['rt']),
        activity_prefixes_and_matching=True,fixed_linked_population=True,failure_decomposition_sums=True,
        direct_timing_formula_agreement=True,all_snapshot_budgets_noise_validated=True,
        independent_activity_mc_agrees=True,synthetic_sampler_validated=True,
        paper_protection_threshold_db=-6.0,paper_protection_direct_counts_agree=True,
        synthetic_theory_scope='Approximation discrepancies are reported, not treated as generator failures'))
    archive(directory,config)
    atomic_json(directory/'status.json',dict(status='complete',completed_episodes=len(loaded),
        total_episodes=config.num_scenes*config.repeats,total_network_evaluations=len(loaded)*len(config.p_values)*len(config.windows),
        reused_network_detections=reference_checks,new_network_detections=len(loaded)*len(config.p_values)*len(config.windows)-reference_checks,
        beam_configurations=len(loaded)*len(config.p_values)*len(config.windows)*len(config.lambdas),updated=now()))
    files=[dict(path=str(p.relative_to(directory)),bytes=p.stat().st_size,sha256=file_hash(p))
        for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='artifacts.json']
    atomic_json(directory/'artifacts.json',files)
    print('Completed and independently checked:',directory,flush=True)
