"""E7/E8 paper artifacts for the explicitly declared finite model."""
from dataclasses import asdict, replace
import numpy as np
from scipy.stats import beta
from .dynamic import ModelConfig, JointModel
from .reporting import output_dir, rows_csv, json_file, save_figure, ecdf, latex_table, plt


def summarize(model, result, episodes, label, tuning=None):
    row = dict(strategy=label, gamma_db=model.config.gamma_db, status=result.status,
               theta_dl=model.config.theta_dl, theta_ul=model.config.theta_ul,
               delta_tn=model.config.delta_tn, delta_ntn=model.config.delta_ntn,
               expected_bits_per_hz=result.objective, lp_time_s=result.solve_time_s,
               decision_states=len(model.graph.actions), terminal_histories=len(model.graph.terminals),
               lp_variables=result.n_variables, socp_cache_entries=len(model.beams))
    if tuning:
        row.update(tuning_candidates=tuning['tuning_candidates'], fixed_duration=tuning['duration'],
                   fixed_schedule=str(tuning['schedule']))
    row.update({f'lp_{key}': value for key, value in result.residuals.items()})
    if result.feasible:
        row.update({f'predicted_{key}': value for key, value in result.costs.items()})
    if episodes:
        row['episodes'] = len(episodes)
        for key in ['total_bits_per_hz', 'dl_bits_per_hz', 'ul_bits_per_hz']:
            x = np.array([e[key] for e in episodes])
            row[f'empirical_{key}'] = float(x.mean())
            row[f'se_{key}'] = float(x.std(ddof=1)/np.sqrt(len(x)))
        for key in model.reference:
            k, n = sum(e[f'fail_{key}'] for e in episodes), len(episodes)
            row[f'empirical_fail_{key}'] = k/n
            row[f'fail_{key}_ci_low'] = 0. if k==0 else float(beta.ppf(.025,k,n-k+1))
            row[f'fail_{key}_ci_high'] = 1. if k==n else float(beta.ppf(.975,k+1,n-k))
        for i in range(2):
            num = np.array([e[f'exceed_{i}'] for e in episodes],float)
            den = np.array([e[f'active_{i}'] for e in episodes],float)
            row[f'outage_{i}'] = float(num.sum()/den.sum())
            # Whole trajectories, never correlated slots, are resampling units.
            rng = np.random.default_rng(202601+i)
            ratios = []
            for _ in range(400):
                ix = rng.integers(0,len(num),len(num))
                ratios.append(num[ix].sum()/den[ix].sum())
            lo,hi=np.quantile(ratios,[.025,.975])
            row[f'outage_{i}_bootstrap_low'],row[f'outage_{i}_bootstrap_high']=float(lo),float(hi)
            # All-zero bootstrap is uninformative. Constant per-episode denominator:
            row[f'outage_{i}_hoeffding_upper']=float(min(1,row[f'outage_{i}']+np.sqrt(np.log(20)/(2*len(num)))))
            row[f'certified_outage_{i}']=result.costs[f'certified_{i}']/result.costs[f'active_{i}']
    return row


def plot_trace(trace,path,gamma):
    if not trace:
        return
    t=[s['t'] for s in trace]
    fig,ax=plt.subplots(4,1,figsize=(9,8),sharex=True)
    for key,label in [('rf_gap','RF unavailable'),('processing','processing'),('tx','DL transmit')]:
        ax[0].step(t,[s[key] for s in trace],where='post',label=label)
    ax[0].legend(ncol=3,fontsize=8)
    ax[1].step(t,[s['age'] for s in trace],where='post')
    ax[1].set_ylabel('Available-record age')
    ax[2].step(t,[s['power'] for s in trace],where='post')
    ax[2].set_ylabel('Power / Pmax')
    for key,label in [('cumulative_dl','DL'),('cumulative_ul','UL')]:
        ax[3].step(t,[s[key] for s in trace],where='post',label=label)
    ax[3].set(xlabel='Tick',ylabel='Delivered bits/Hz')
    ax[3].legend()
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle(f'Declared finite model: J trajectory, Gamma={gamma:g} dB')
    save_figure(fig,path)


def run_e7(path,seed=20260921,quick=True,gamma_db=(-15,-10,-5,0),config=None):
    path=output_dir(path)
    config=config or ModelConfig()
    n=400 if quick else 10000
    rows,all_episodes,cdf=[],[],{}
    trace_saved=False
    for gamma in gamma_db:
        print(f'E7: complete-history Gamma={gamma:g} dB',flush=True)
        model=JointModel(replace(config,gamma_db=float(gamma)))
        model.build()
        rows_csv(path/f'socp_diagnostics_gamma_{gamma:g}.csv',[
            dict(record=record,age=age,status=b.status,amplitude=b.amplitude,power_fraction=b.power_fraction,
                 max_violation=b.max_violation,raw_max_violation=b.raw_max_violation,
                 raw_max_relative_violation=b.raw_max_relative_violation,feasibility_scale=b.feasibility_scale,
                 solve_time_s=b.solve_time) for (record,age),b in model.beams.items()])
        solutions={}
        for mode in ('F','T','L','J'):
            result,tune=model.optimize(mode)
            solutions[mode]=result
            ep,samples,trace=model.simulate(result,episodes=n,seed=seed)
            rows.append(summarize(model,result,ep,mode,tune))
            all_episodes.extend([dict(strategy=mode,gamma_db=gamma,**e) for e in ep])
            if result.feasible:
                json_file(path/f'policy_{mode}_gamma_{gamma:g}.json',
                          dict(tuning=tune,policy=result.policy,predicted_costs=result.costs,
                               objective=result.objective,residuals=result.residuals))
            if mode=='J':
                cdf[gamma]=dict(
                    foreground_all=np.array([max(-60,s['inr_db']) for s in samples if s['receiver']==0]),
                    foreground_tx=np.array([s['inr_db'] for s in samples if s['receiver']==0 and s['bs_transmitting']]),
                    background=np.array([max(-60,s['inr_db']) for s in samples if s['receiver']==1 and s['dl_active']]),
                    tn_snr=np.array([10*np.log10(max(s['tn_snr_linear'],1e-6)) for s in samples
                                     if s['receiver']==0 and s['tn_dl_slot']]))
                rows_csv(path/f'inr_tn_samples_J_gamma_{gamma:g}.csv',samples)
                rows_csv(path/f'timeline_J_gamma_{gamma:g}.csv',trace)
                if not trace_saved and trace:
                    plot_trace(trace,path/'timeline',gamma)
                    trace_saved=True
        if solutions['J'].feasible:
            for mode,r in solutions.items():
                if r.feasible and r.objective>solutions['J'].objective+1e-6:
                    raise AssertionError(f'Nested class {mode} exceeds J')
    rows_csv(path/'strategy_summary.csv',rows)
    rows_csv(path/'episode_metrics.csv',all_episodes)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for gamma,samples in cdf.items():
        ecdf(ax[0],samples['foreground_all'],f'{gamma:g} dB')
        ecdf(ax[1],samples['foreground_tx'],f'{gamma:g} dB')
    ax[0].set_title('All DL-active ticks; zero shown at -60 dB')
    ax[1].set_title('Conditional on controlled BS transmitting')
    for a in ax:
        a.set(xlabel='Foreground INR [dB]',ylabel='CDF',ylim=(0,1))
        if a.lines: a.legend(title='Gamma')
        a.grid(alpha=.25)
    fig.suptitle('Finite model: J; fixed TN demand and failure budgets')
    save_figure(fig,path/'gamma_inr_cdf')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for gamma,samples in cdf.items():
        ecdf(ax[0],samples['background'],f'{gamma:g} dB')
        ecdf(ax[1],samples['tn_snr'],f'{gamma:g} dB')
    ax[0].set(xlabel='UL-silent receiver INR [dB]',ylabel='CDF',title='All DL-active ticks')
    ax[1].set(xlabel='TN SNR [dB]',ylabel='CDF',title='All TN DL slots; zero at -60 dB')
    for a in ax:
        if a.lines: a.legend(title='Gamma')
        a.grid(alpha=.25)
    save_figure(fig,path/'silent_ntn_and_tn_cdf')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for mode in ('F','T','L','J'):
        rr=[r for r in rows if r['strategy']==mode and r['status']=='optimal']
        ax[0].plot([r['gamma_db'] for r in rr],[r['expected_bits_per_hz'] for r in rr],'o-',label=mode)
        ax[1].plot([r['gamma_db'] for r in rr],[r['predicted_gap_ticks']/config.epochs/4 for r in rr],'o-',label=mode)
    ax[0].set(xlabel='Gamma [dB]',ylabel='Expected delivered bits/Hz')
    ax[1].set(xlabel='Gamma [dB]',ylabel='RF/blocked-processing fraction')
    for a in ax: a.legend(); a.grid(alpha=.25)
    save_figure(fig,path/'service_protection_tradeoff')
    latex_table(path/'strategy_table.tex',['Gamma','Policy','Status','Bits/Hz','TN worst fail','NTN bound'],
                [[r['gamma_db'],r['strategy'],r['status'],
                  f"{r['expected_bits_per_hz']:.3f}" if r['expected_bits_per_hz'] is not None else '--',
                  f"{max((v for k,v in r.items() if k.startswith('predicted_fail_')),default=0):.3f}" if r['status']=='optimal' else '--',
                  f"{r.get('certified_outage_0',0):.3f}" if r['status']=='optimal' else '--'] for r in rows],
                'Exact finite-model policies at fixed TN constraints; infeasible points retained.')
    sensitivity=[]
    for theta in (.25,.40,.60,.80):
        for gamma in gamma_db:
            model=JointModel(replace(config,gamma_db=float(gamma),theta_dl=theta))
            model.build()
            result,_=model.optimize('J')
            sensitivity.append(dict(theta_dl=theta,gamma_db=gamma,status=result.status,expected_bits_per_hz=result.objective))
    rows_csv(path/'tn_demand_sensitivity.csv',sensitivity)
    matrix=np.full((4,len(gamma_db)),np.nan)
    for i,theta in enumerate((.25,.40,.60,.80)):
        for j,gamma in enumerate(gamma_db):
            r=next(r for r in sensitivity if r['theta_dl']==theta and r['gamma_db']==gamma)
            if r['status']=='optimal': matrix[i,j]=r['expected_bits_per_hz']
    fig,ax=plt.subplots(figsize=(6,4))
    im=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',origin='lower')
    for i in range(4):
        for j in range(len(gamma_db)):
            ax.text(j,i,'infeasible' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=8,
                    color='white' if np.isfinite(matrix[i,j]) and matrix[i,j]<(np.nanmin(matrix)+np.nanmax(matrix))/2 else 'black')
    ax.set_xticks(range(len(gamma_db)),labels=gamma_db)
    ax.set_yticks(range(4),labels=(.25,.40,.60,.80))
    ax.set(xlabel='Gamma [dB]',ylabel='TN DL demand / fixed no-gap reference')
    fig.colorbar(im,ax=ax,label='Expected bits/Hz')
    save_figure(fig,path/'tn_demand_feasibility')
    summary=dict(experiment='E7',scope='declared synthetic finite hidden-Markov model',config=asdict(config),
                 gamma_db=list(gamma_db),episodes=n,units='bits/Hz per tick; multiply by B and tick seconds for bits',
                 fixed_tn='thetaDL/thetaUL/deltaTN/deadlines/reference fixed across main Gamma scan',
                 ci='TN Clopper-Pearson; NTN episode bootstrap plus Hoeffding upper bound',
                 optimality='Exact declared finite model only',infeasible=sum(r['status']!='optimal' for r in rows))
    json_file(path/'summary.json',summary)
    return summary


def run_e8(path,seed=20260921,quick=True,config=None):
    path=output_dir(path)
    config=config or ModelConfig(gamma_db=-10)
    n=400 if quick else 10000
    cases=[
        ('nominal',config,{},'within model'),
        ('no_processing_delay',replace(config,processing_ticks=0),{},'within reoptimized model'),
        ('blocking_processing',replace(config,blocking_processing=True),{},'within reoptimized model'),
        ('frequent_failed_refresh',replace(config,acceptance_short=.15,acceptance_long=.40),{},'within reoptimized model'),
        ('refresh_unavailable',replace(config,listening_epochs=()),{},'within reoptimized model'),
        ('modeled_new_arrival',replace(config,background_arrival_tick=6),{},'within reoptimized model'),
        ('larger_silent_background',replace(config,background_bound=.8),{},'within reoptimized model'),
        ('missing_background_ablation',replace(config,background_bound=.8,include_background=False),{},'deliberately omitted envelope'),
        ('gain_bound_exceeded',config,{'physical_gain':3.,'background_gain':3.},'outside gain bounds'),
        ('bursty_same_marginals',config,{'burst_success':True},'outside independent observation kernel'),
    ]
    rows=[]
    for label,cfg,stress,scope in cases:
        print(f'E8: {label}',flush=True)
        model=JointModel(cfg)
        model.build()
        result,tune=model.optimize('J')
        ep,samples,trace=model.simulate(result,episodes=n,seed=seed,**stress)
        row=summarize(model,result,ep,label,tune)
        row['scope']=scope
        row['bound_valid_for_this_scenario']=scope.startswith('within')
        rows.append(row)
        rows_csv(path/f'{label}_episodes.csv',ep)
        rows_csv(path/f'{label}_timeline.csv',trace)
        rows_csv(path/f'{label}_inr.csv',samples)
    rows_csv(path/'stress_summary.csv',rows)
    fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
    x=np.arange(len(rows))
    bars=ax[0].bar(x,[r.get('empirical_total_bits_per_hz',0) for r in rows],
                   yerr=[1.96*r.get('se_total_bits_per_hz',0) for r in rows],capsize=2)
    for bar,r in zip(bars,rows):
        if not r['bound_valid_for_this_scenario']: bar.set_hatch('//')
    ax[0].set_ylabel('Delivered bits/Hz')
    for i,offset in [(0,-.15),(1,.15)]:
        bars=ax[1].bar(x+offset,[r.get(f'outage_{i}',0) for r in rows],width=.3,label=f'NTN {i}')
        for bar,r in zip(bars,rows):
            if not r['bound_valid_for_this_scenario']: bar.set_hatch('//')
    ax[1].axhline(config.delta_ntn,color='k',ls='--',label='risk target')
    ax[1].set_ylabel('Active-time INR exceedance')
    ax[1].set_xticks(x,labels=[r['strategy'] for r in rows],rotation=35,ha='right',fontsize=8)
    ax[1].legend()
    for i,r in enumerate(rows):
        if r['status']!='optimal':
            ax[0].text(i,0,'infeasible',rotation=90,va='bottom',ha='center')
            ax[1].text(i,0,'N/A',rotation=90,va='bottom',ha='center')
    fig.suptitle('Synthetic finite model; hatched cases deliberately violate assumptions')
    save_figure(fig,path/'stress_comparison')
    short_names=['Nominal','Zero delay','Blocking','Low detection','No refresh','Arrival',
                 'Large BG','No BG','Gain x3','Burst']
    latex_table(path/'stress_table.tex',['Case','Model valid','Status','Bits/Hz','NTN0 outage','Silent NTN outage'],
                [[short_names[index],'yes' if r['bound_valid_for_this_scenario'] else 'no',r['status'],f"{r['empirical_total_bits_per_hz']:.3f}" if r['status']=='optimal' else '--',
                  f"{r['outage_0']:.4f}" if r['status']=='optimal' else '--',f"{r['outage_1']:.4f}" if r['status']=='optimal' else '--'] for index,r in enumerate(rows)],
                'Finite-model failure scenarios; deliberate model violations have no guarantee.')
    summary=dict(experiment='E8',scope='synthetic; each scenario labels model validity',
                 config=asdict(config),episodes=n,cases=len(cases))
    json_file(path/'summary.json',summary)
    return summary
