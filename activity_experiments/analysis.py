"""Scene-cluster intervals, activity failure decomposition and timing sensitivity."""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json
import numpy as np

from listening_experiments.config import TimingConfig
from listening_experiments.storage import atomic_json, atomic_npz
from .runner import records


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode_metrics(record, timing, threshold):
    durations = [timing.durations(int(L)) for L in record['windows']]
    mrt = np.array([d['mrt'] for d in durations])[None, :, None]
    final = np.array([d['final'] for d in durations])[None, :, None]
    data_time = np.array([d['data'] for d in durations])[None, :, None]
    out = record['inr_linear'] > threshold
    q_out = out.mean(axis=-1)
    q_mrt = float(np.mean(record['raw_inr_linear'] > threshold))
    r_final = np.log2(1+record['sinr_linear']).mean(axis=-1)
    r_mrt = float(np.log2(1+record['raw_sinr_linear']).mean())
    linked_activity = record['J'][:, :, record['ue_ids']]
    linked_detection = record['detected_ue_mask'][:, :, record['ue_ids']]
    groups = np.stack([linked_activity == 0,
        (linked_activity > 0) & ~linked_detection,
        (linked_activity > 0) & linked_detection], axis=-1)
    assert np.all(groups.sum(axis=-1) == 1)
    group_counts = groups.sum(axis=-2)
    group_exceed = np.einsum('pkwu,pkug->pkwg', out.astype(int), groups.astype(int))
    contributions = group_exceed / out.shape[-1]
    np.testing.assert_allclose(contributions.sum(axis=-1), q_out, atol=1e-15)
    high = record['raw_inr_linear'] > threshold
    high_counts = groups[:, :, high].sum(axis=-2)
    high_exceed = np.einsum('pkwu,pkug->pkwg', out[:, :, :, high].astype(int), groups[:, :, high].astype(int))
    return dict(q_out=q_out, q_any=out.any(axis=-1).astype(float),
        q_mrt=np.full(q_out.shape, q_mrt), r_mrt=np.full(q_out.shape, r_mrt),
        r_final=r_final,
        q_data=(mrt*q_mrt+final*q_out)/data_time,
        r_eff=(mrt*r_mrt+final*r_final)/timing.horizon,
        r_mrt_scheduled=data_time/timing.horizon*np.full(q_out.shape, r_mrt),
        q_miss_all=np.any(record['J']==0, axis=-1).astype(float),
        q_miss_linked=np.any(linked_activity==0, axis=-1).astype(float),
        matched_ues=record['detected_ue_mask'].sum(axis=-1),
        heard_ues=(record['J']>0).sum(axis=-1),
        group_fraction=group_counts/out.shape[-1], group_contributions=contributions,
        group_counts=group_counts, group_exceed=group_exceed,
        high_group_counts=high_counts, high_group_exceed=high_exceed)


def analyze(directory):
    directory = Path(directory)
    config, loaded = records(directory)
    timing = TimingConfig()
    threshold = 10**(config.threshold_db/10)
    values = [episode_metrics(r, timing, threshold) for r in loaded]
    by_scene = {name: np.stack([np.mean([v[name] for r,v in zip(loaded,values)
        if int(r['macro_id'])==scene], axis=0) for scene in range(config.num_scenes)])
        for name in values[0]}
    rng=np.random.default_rng(config.validation_seed+100)
    draws=rng.integers(0,config.num_scenes,size=(config.bootstrap_draws,config.num_scenes))
    stats={}
    for name, array in by_scene.items():
        lo, hi=np.quantile(array[draws].mean(axis=1),[.025,.975],axis=0)
        stats[name]=(array.mean(axis=0),lo,hi)
    summary=[]
    metric_names=['q_out','q_any','q_mrt','q_data','r_final','r_eff','r_mrt','r_mrt_scheduled']
    for pi,p in enumerate(config.p_values):
        for li,L in enumerate(config.windows):
            d=timing.durations(L)
            for wi,weight in enumerate(config.lambdas):
                row=dict(p=p,L=L,T=L*config.snapshots_per_window,lambda_value=weight,
                    independent_scenes=config.num_scenes,repeats_per_scene=config.repeats,
                    rf_fraction=d['rf']/d['horizon'],ready_fraction=d['ready']/d['horizon'])
                for name in metric_names:
                    for suffix,array in zip(['','_lower','_upper'],stats[name]):
                        row[name+suffix]=float(array[pi,li,wi])
                summary.append(row)
    write_csv(directory/'rt_summary.csv',summary)
    counts=np.sum([r['active_tp_fp_fn'].sum(axis=2) for r in loaded],axis=0)
    power=np.sum([r['path_power_coverage'].sum(axis=2) for r in loaded],axis=0)
    high=np.sum([r['high_inr_ue_counts'] for r in loaded],axis=0)
    detection=[]
    for pi,p in enumerate(config.p_values):
        for li,L in enumerate(config.windows):
            tp,fp,fn=counts[pi,li]
            row=dict(p=p,L=L,T=L*config.snapshots_per_window,
                active_path_precision=float(tp/(tp+fp)) if tp+fp else None,
                active_path_recall=float(tp/(tp+fn)) if tp+fn else None,
                persistent_path_power_recall=float(power[pi,li,0]/power[pi,li,1]),
                high_inr_ue_recall=float(high[pi,li,0]/high[pi,li,1]))
            for name in ['q_miss_all','q_miss_linked','heard_ues','matched_ues']:
                for suffix,array in zip(['','_lower','_upper'],stats[name]):
                    row[name+suffix]=float(array[pi,li])
            detection.append(row)
    write_csv(directory/'detection_summary.csv',detection)
    groups=['never_transmitted','transmitted_not_matched','transmitted_matched']
    decomposition=[]
    for pi,p in enumerate(config.p_values):
        for li,L in enumerate(config.windows):
            for wi,weight in enumerate(config.lambdas):
                for gi,group in enumerate(groups):
                    n=int(sum(v['group_counts'][pi,li,gi] for v in values))
                    k=int(sum(v['group_exceed'][pi,li,wi,gi] for v in values))
                    hn=int(sum(v['high_group_counts'][pi,li,gi] for v in values))
                    hk=int(sum(v['high_group_exceed'][pi,li,wi,gi] for v in values))
                    decomposition.append(dict(p=p,L=L,lambda_value=weight,group=group,
                        user_records=n,exceedance_records=k,
                        conditional_outage_pooled=k/n if n else None,
                        high_inr_user_records=hn,high_inr_exceedance_records=hk,
                        high_inr_conditional_outage_pooled=hk/hn if hn else None,
                        population_fraction_scene_mean=float(stats['group_fraction'][0][pi,li,gi]),
                        outage_contribution_scene_mean=float(stats['group_contributions'][0][pi,li,wi,gi])))
    write_csv(directory/'failure_decomposition.csv',decomposition)
    sensitivity=[]
    for horizon in [256.,512.,1024.]:
        alternative=replace(timing,horizon=horizon)
        metrics=[episode_metrics(r,alternative,threshold) for r in loaded]
        for pi,p in enumerate(config.p_values):
            for li,L in enumerate(config.windows):
                for wi,weight in enumerate(config.lambdas):
                    sensitivity.append(dict(horizon=horizon,p=p,L=L,lambda_value=weight,
                        q_data=float(np.mean([m['q_data'][pi,li,wi] for m in metrics])),
                        r_eff=float(np.mean([m['r_eff'][pi,li,wi] for m in metrics]))))
    write_csv(directory/'timing_sensitivity.csv',sensitivity)
    baseline=np.concatenate([r['raw_inr_linear'] for r in loaded if int(r['repeat_id'])==0])
    arrays=dict(p_values=np.asarray(config.p_values),windows=np.asarray(config.windows),lambdas=np.asarray(config.lambdas),raw_inr_linear=baseline)
    for name,(mean,lo,hi) in stats.items():
        arrays[name],arrays[name+'_lower'],arrays[name+'_upper']=mean,lo,hi
    atomic_npz(directory/'analysis_arrays.npz',**arrays)
    atomic_npz(directory/'scene_metrics.npz',**by_scene)
    report=dict(timing=asdict(timing),threshold_db=config.threshold_db,
        aggregation='Equal deployment, equal repetition, then equal linked UE or served TN link; compute log2(1+SINR) before averaging',
        intervals='Pointwise 95% deployment-cluster bootstrap; repetitions, p, L and lambda remain paired',
        detection_ratios='Pooled active-path counts, persistent path powers, and fixed high-INR UE counts',
        decomposition='Mutually exclusive activity/detection groups; scene-mean contributions sum to final outage. Conditional subgroup risks use pooled record counts.',
        time_units='One 64-snapshot listening window; stipulated protocol costs, not measured hardware delay',
        timing_policy='MRT in gaps and processing; final estimated beam used only after all requested windows',
        scope='No adaptive interim beam updates or oracle activity-based power correction',
        baseline_samples=len(baseline),total_episodes=len(loaded),
        linked_ues_per_scene=[len(r['ue_ids']) for r in loaded if int(r['repeat_id'])==0],
        reused_continuous_evaluations=int(sum(r['reused_continuous'].sum() for r in loaded)),
        total_network_evaluations=len(loaded)*len(config.p_values)*len(config.windows),
        caveat='Zero observed events or bootstrap widths do not certify zero risk; activity-only MC is independently sampled.')
    atomic_json(directory/'analysis_config.json',report)
    atomic_json(directory/'results_summary.json',dict(config=asdict(config),analysis=report,rt=summary,detection=detection))
    print('Analyzed',len(loaded),'episodes:',directory,flush=True)
    return summary
