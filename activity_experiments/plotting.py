"""Compact paper figures and supplementary diagnostic plots from saved results."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator
from listening_experiments.storage import atomic_json
from .protection import summarize_protection

COLORS=['#0072B2','#E69F00','#009E73']


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def save(fig,directory,stem):
    fig.canvas.draw()
    for extension in ['pdf','png']:
        fig.savefig(directory/f'{stem}.{extension}',dpi=600)
    plt.close(fig)


def panel_labels(fig,axes,texts):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    top=min(ax.xaxis.label.get_window_extent(renderer).y0 for ax in axes)-2*fig.dpi/72
    y=fig.transFigure.inverted().transform((0,top))[1]
    for ax,text in zip(axes,texts):
        x=(ax.get_position().x0+ax.get_position().x1)/2
        caption=fig.text(x,y,text,ha='center',va='top',fontsize=7)
        fig.canvas.draw()
        box=caption.get_window_extent(fig.canvas.get_renderer())
        assert fig.bbox.contains(box.x0,box.y0) and fig.bbox.contains(box.x1,box.y1)
        assert not box.overlaps(ax.xaxis.label.get_window_extent(fig.canvas.get_renderer()))


def plot(directory):
    directory=Path(directory)
    config=json.loads((directory/'run_config.json').read_text())['config']
    for path in (directory/'sources/fonts').glob('*.TTF'):
        font_manager.fontManager.addfont(str(path))
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    style={'font.family':'Times New Roman','font.size':8,'axes.labelsize':7.5,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,
        'axes.linewidth':.6,'axes.labelpad':2,'xtick.major.pad':2,'ytick.major.pad':2,
        'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'custom',
        'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
        'mathtext.bf':'Times New Roman:bold','mathtext.cal':'Times New Roman:italic',
        'mathtext.sf':'Times New Roman','mathtext.tt':'Times New Roman',
        'mathtext.fallback':None,'savefig.bbox':None}
    with np.load(directory/'analysis_arrays.npz') as loaded:
        data={k:loaded[k] for k in loaded.files}
    ps,Ls,weights=data['p_values'],data['windows'],data['lambdas']
    protection=summarize_protection(directory,threshold_db=-6.0)
    coverage=csv_rows(directory/'synthetic/coverage_50ue.csv')
    dense=csv_rows(directory/'synthetic/coverage_dense.csv')
    validation={}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(3.5,2.2))
        fig.subplots_adjust(left=.13,right=.985,top=.96,bottom=.34,wspace=.52)
        for p,color in zip(ps,COLORS):
            exact=[r for r in dense if float(r['p'])==p]
            mc=[r for r in coverage if float(r['p'])==p]
            x=np.array([int(r['L']) for r in mc])
            y=np.array([float(r['q_miss_mc']) for r in mc])
            lo=np.array([float(r['lower']) for r in mc]);hi=np.array([float(r['upper']) for r in mc])
            axes[0].plot([int(r['L']) for r in exact],[float(r['q_exact']) for r in exact],color=color,label=f'$p={p:g}$',linewidth=1)
            axes[0].errorbar(x,y,yerr=np.stack([y-lo,hi-y]),color=color,fmt='o',markersize=2,linewidth=.5,capsize=1)
        axes[0].axhline(.05,color='.5',linestyle=':',linewidth=.7)
        axes[0].set_xscale('log',base=2)
        axes[0].set_xticks([1,8,32,96],['1','8','32','96'])
        axes[0].xaxis.set_minor_locator(NullLocator())
        axes[0].set(xlabel='Listening windows, $L$',ylabel='Coverage failure probability',ylim=(-.025,1.025))
        axes[0].set_yticks([0,.5,1])
        axes[0].legend(loc='center left',handlelength=1.1,borderpad=.25,labelspacing=.2,framealpha=.92)
        for pi,p in enumerate(ps):
            for wi,weight in enumerate(weights):
                axes[1].plot(data['q_data'][pi,:,wi],data['r_eff'][pi,:,wi],color=COLORS[pi],
                    linestyle='--' if weight==1e11 else '-',linewidth=1,marker='o',markersize=1.6)
                axes[1].plot(data['q_data'][pi,0,wi],data['r_eff'][pi,0,wi],color=COLORS[pi],marker='^',markersize=3)
                axes[1].plot(data['q_data'][pi,-1,wi],data['r_eff'][pi,-1,wi],color=COLORS[pi],marker='s',markersize=3)
        axes[1].set(xlabel='Data-time INR risk',ylabel='TN rate (bit/s/Hz)')
        axes[1].set_xlim(left=0)
        handles=[Line2D([0],[0],color='.25',linestyle=s,label=label) for s,label in [('--',r'$\lambda=10^{11}$'),('-',r'$\lambda=10^{13}$')]]
        handles += [Line2D([0],[0],color='.25',linestyle='None',marker=m,markersize=3,label=label) for m,label in [('^','$L=1$'),('s','$L=96$')]]
        axes[1].legend(handles=handles,loc='lower left',handlelength=1.2,borderpad=.25,labelspacing=.15,framealpha=.92,fontsize=6)
        for ax in axes:ax.grid(alpha=.22,linewidth=.4)
        panel_labels(fig,axes,['(a) Activity coverage','(b) Protection and service'])
        save(fig,directory,'activity_coverage_tradeoff_ab')
        validation['activity_coverage_tradeoff_ab']=dict(size_inches=[3.5,2.2],N=50,coverage_trials=config['coverage_trials'],
            coverage_curves='Exact activity probability plus independent MC and Wilson intervals; dotted line is 0.05',
            tradeoff='Scene-mean data-time risk versus effective TN rate, same p colors in both panels; final-only beam update policy')

        fig,axes=plt.subplots(1,2,figsize=(3.5,2.05))
        fig.subplots_adjust(left=.12,right=.985,top=.97,bottom=.30,wspace=.55)
        for p,color in zip(ps,COLORS):
            exact=[r for r in dense if float(r['p'])==p]
            mc=[r for r in coverage if float(r['p'])==p]
            x=np.array([int(r['L']) for r in mc])
            y=np.array([float(r['q_miss_mc']) for r in mc])
            lo=np.array([float(r['lower']) for r in mc])
            hi=np.array([float(r['upper']) for r in mc])
            axes[0].plot([int(r['L']) for r in exact],
                [float(r['q_exact']) for r in exact],color=color,label=f'$p={p:g}$',linewidth=1)
            axes[0].errorbar(x,y,yerr=np.stack([y-lo,hi-y]),color=color,
                fmt='o',markersize=2,linewidth=.5,capsize=1)
        axes[0].axhline(.05,color='.5',linestyle=':',linewidth=.7)
        axes[0].set(ylabel='Coverage failure probability',ylim=(-.025,1.025))
        axes[0].set_yticks([0,.5,1])
        axes[0].legend(loc='center left',handlelength=1.1,borderpad=.25,
            labelspacing=.2,framealpha=.92)
        wi=int(np.flatnonzero(weights==1e13)[0])
        for pi,p in enumerate(ps):
            axes[1].plot(Ls,protection['q_out'][pi,:,wi],color=COLORS[pi],
                marker='o',markersize=2,linewidth=1)
            axes[1].fill_between(Ls,protection['q_out_lower'][pi,:,wi],
                protection['q_out_upper'][pi,:,wi],color=COLORS[pi],alpha=.12,linewidth=0)
        baseline=float(protection['q_mrt'][0,0,wi])
        axes[1].axhline(baseline,color='black',linestyle='--',linewidth=.85,label='No nulling')
        axes[1].set(ylabel='INR exceedance fraction',ylim=(.05,.52))
        axes[1].set_yticks([.1,.2,.3,.4,.5])
        axes[1].legend(loc='upper right',fontsize=6,handlelength=1.4,
            borderpad=.25,framealpha=.92)
        for ax in axes:
            ax.set_xscale('log',base=2)
            ax.set_xticks([1,8,32,96],['1','8','32','96'])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel('Listening windows, $L_{\\rm s}$')
            ax.grid(alpha=.22,linewidth=.4)
        panel_labels(fig,axes,['(a) Activity coverage','(b) Final-beam protection'])
        save(fig,directory,'activity_coverage_protection_ab_minus6db')
        validation['activity_coverage_protection_ab_minus6db']=dict(size_inches=[3.5,2.05],N=50,
            coverage_trials=config['coverage_trials'],lambda_value=1e13,
            threshold_db=float(protection['threshold_db']),baseline=baseline,
            coverage='At least one of all 50 UEs never transmits; exact probability, independent MC, Wilson intervals',
            protection='Equal deployment/repetition means of linked-UE final-beam INR exceedance fractions',
            intervals='Pointwise 95% deployment-cluster bootstrap; paired p and L retained',
            scope='Final service phase; includes never-transmitted and undetected linked UEs; no timing cost')

        fig,axes=plt.subplots(1,3,figsize=(10.5,3.0))
        for ax,name,label in zip(axes,['q_out','q_data','r_eff'],['Final-beam INR exceedance','Data-time INR exceedance','Effective TN rate (bit/s/Hz)']):
            for pi,p in enumerate(ps):
                for wi,weight in enumerate(weights):
                    ax.plot(Ls,data[name][pi,:,wi],color=COLORS[pi],linestyle='--' if weight==1e11 else '-',marker='o',markersize=2.5,linewidth=1.2)
                    if weight==1e13:
                        ax.fill_between(Ls,data[name+'_lower'][pi,:,wi],data[name+'_upper'][pi,:,wi],color=COLORS[pi],alpha=.10,linewidth=0)
            baseline=float(data['r_mrt'][0,0,0] if name=='r_eff' else data['q_mrt'][0,0,0])
            ax.axhline(baseline,color='black',linestyle=':',linewidth=1)
            ax.set_xscale('log',base=2);ax.set_xticks([1,4,16,64,96],['1','4','16','64','96'])
            ax.set(xlabel='Listening windows, $L$',ylabel=label);ax.grid(alpha=.22,linewidth=.4)
        handles=[Line2D([0],[0],color=c,label=f'$p={p:g}$') for c,p in zip(COLORS,ps)]
        handles += [Line2D([0],[0],color='.25',linestyle=s,label=label) for s,label in [('--',r'$\lambda=10^{11}$'),('-',r'$\lambda=10^{13}$'),(':','MRT, no sensing')]]
        fig.legend(handles=handles,ncol=6,loc='upper center',frameon=False)
        fig.text(.5,.01,'20 frozen 50-UE deployments; 5 repeats; shaded 95% deployment-cluster intervals for lambda = 1e13.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.91));save(fig,directory,'activity_rt_diagnostics')

        fig,axes=plt.subplots(1,3,figsize=(9,2.65),sharey=True)
        wi=int(np.flatnonzero(weights==1e13)[0])
        group_colors=['#E69F00','#7A7A7A','#CC79A7']
        group_labels=['Never transmitted','Transmitted, not matched','Transmitted, matched']
        for pi,ax in enumerate(axes):
            bottom=np.zeros(len(Ls))
            for gi,(color,label) in enumerate(zip(group_colors,group_labels)):
                y=data['group_contributions'][pi,:,wi,gi]
                ax.bar(np.arange(len(Ls)),y,bottom=bottom,color=color,label=label,width=.75)
                bottom+=y
            np.testing.assert_allclose(bottom,data['q_out'][pi,:,wi])
            ax.set_xticks(np.arange(len(Ls)),[str(L) for L in Ls]);ax.set_xlabel('Listening windows, $L$')
            ax.set_title(f'$p={ps[pi]:g}$',fontsize=9);ax.grid(axis='y',alpha=.2,linewidth=.4)
        axes[0].set_ylabel('Contribution to INR exceedance')
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=3,loc='lower center',frameon=False)
        fig.tight_layout(rect=(0,.13,1,1));save(fig,directory,'activity_failure_breakdown')

        conditional=csv_rows(directory/'synthetic/conditional_inr.csv')
        fig,axes=plt.subplots(1,2,figsize=(3.5,2.05),sharey=True)
        fig.subplots_adjust(left=.14,right=.985,top=.97,bottom=.30,wspace=.35)
        for ax,snr in zip(axes,[0.,10.]):
            for fraction,color in zip([.125,.5,1.],COLORS):
                rows=[r for r in conditional if r['mode']=='pooled' and float(r['gamma_s_db'])==snr and float(r['activity_fraction'])==fraction]
                x=np.array([int(r['L']) for r in rows]);y=np.array([float(r['mean_inr_linear']) for r in rows])
                lo=np.array([float(r['lower']) for r in rows]);hi=np.array([float(r['upper']) for r in rows])
                predicted=np.array([float(r['predicted_inr_linear']) for r in rows])
                ax.plot(x,10*np.log10(predicted),color=color,linestyle='--',linewidth=.9,label=f'$J/L_{{\\rm s}}={fraction:g}$')
                ax.errorbar(x,10*np.log10(y),yerr=np.stack([10*np.log10(y/lo),10*np.log10(hi/y)]),color=color,fmt='o',markersize=2.5,linewidth=.6,capsize=1)
            ax.set_xscale('log',base=2);ax.set_xticks([8,32,128,512],['8','32','128','512'])
            ax.set_xlabel('Listening windows, $L_{\\rm s}$');ax.grid(alpha=.22,linewidth=.4)
        axes[0].set_ylabel('Mean INR (dB)')
        axes[0].legend(loc='lower left',fontsize=6,handlelength=1.1,borderpad=.2,labelspacing=.2)
        axes[1].legend(handles=[Line2D([0],[0],color='.3',linestyle='--',label='Approximation'),Line2D([0],[0],color='.3',marker='o',markersize=2.5,linestyle='None',label='Monte Carlo')],loc='upper right',fontsize=6,handlelength=1.2,borderpad=.2)
        panel_labels(fig,axes,[r'(a) $\gamma_{\mathrm{S}}=0$ dB',r'(b) $\gamma_{\mathrm{S}}=10$ dB'])
        save(fig,directory,'synthetic_conditional_inr_ab')
        validation['synthetic_conditional_inr_ab']=dict(size_inches=[3.5,2.05],antennas=8,sources=1,
            statistic='10 log10 of the Monte Carlo linear mean INR; 95% independent-MC mean intervals',
            scope='Conditional positive activity count; separate from the 50-UE MUSIC network')

        fig,ax=plt.subplots(figsize=(3.5,2.05));fig.subplots_adjust(left=.15,right=.98,top=.965,bottom=.20)
        curves=[('No nulling',data['raw_inr_linear'],'black','-')]
        records=[]
        for path in sorted((directory/'episodes').glob('*.npz')):
            with np.load(path) as z:records.append(z['inr_linear'][0,:,wi])
        for L,color,style in [(1,'#0072B2','--'),(16,'#E69F00','-.'),(66,'#009E73',':'),(96,'#CC79A7','-')]:
            li=int(np.flatnonzero(Ls==L)[0]);values=np.concatenate([r[li] for r in records])
            curves.append((f'$L={L}$',values,color,style))
        below={}
        for label,power,color,style in curves:
            x=np.sort(10*np.log10(np.maximum(power,1e-12)));y=np.arange(1,len(x)+1)/len(x)
            ax.plot(x,y,color=color,linestyle=style,linewidth=1.1,label=label)
            below[label]=int(np.sum(x < -60))
        ax.set(xlabel='NTN-UE INR (dB)',ylabel='CDF',ylim=(0,1));ax.set_xlim(left=-60)
        ax.xaxis.set_major_locator(MultipleLocator(20));ax.grid(alpha=.22,linewidth=.4)
        ax.legend(loc='upper left',title=r'$p=0.1,\ \lambda=10^{13}$',title_fontsize=7,framealpha=.92,handlelength=2.3,labelspacing=.2,borderpad=.3)
        save(fig,directory,'activity_inr_cdf')
        validation['activity_inr_cdf']=dict(size_inches=[3.5,2.05],baseline_samples=len(data['raw_inr_linear']),
            samples_per_nulling_curve=sum(r.shape[-1] for r in records),all_samples_used=True,
            samples_below_display_min=below,display_min_db=-60,
            aggregation='Pooled fixed linked UE population across repeats; deterministic baseline counted once')
    atomic_json(directory/'render_validation.json',validation)
    print('Figures generated:',directory,flush=True)
