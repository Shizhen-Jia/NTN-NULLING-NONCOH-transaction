"""Activity-only MC and exact covariance simulation for the conditional INR law."""
from pathlib import Path
import json
import numpy as np
from listening_experiments.analysis import coverage_experiment
from listening_experiments.storage import atomic_json, atomic_npz
from .analysis import write_csv
from .config import ActivityConfig


def complex_normal(rng, shape):
    return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)


def covariance_trials(rng, trials, antennas, snapshots, signal_energy):
    """Exact noncentral complex Wishart law for a fixed-energy rank-one waveform.

    Temporal unitary invariance concentrates the deterministic waveform into
    one sample; the remaining white-noise covariance has a Bartlett factor.
    No asymptotic approximation is used to generate these covariances.
    """
    df=snapshots-1
    if df < antennas:
        residual=complex_normal(rng,(trials,antennas,df))
        noise=residual @ residual.swapaxes(-1,-2).conj()
    else:
        factor=np.tril(complex_normal(rng,(trials,antennas,antennas)),-1)
        index=np.arange(antennas)
        factor[:,index,index]=np.sqrt(rng.gamma(df-index,1,size=(trials,antennas)))
        noise=factor @ factor.swapaxes(-1,-2).conj()
    first=complex_normal(rng,(trials,antennas))
    first[:,0]+=np.sqrt(signal_energy)
    return (noise+first[:,:,None]*first[:,None,:].conj())/snapshots


def leakage(covariance, rho=.5, gamma_i=10.):
    _,vectors=np.linalg.eigh(covariance)
    estimated=vectors[:,:,-1]
    desired=np.zeros(estimated.shape[1],complex)
    desired[0],desired[1]=rho,np.sqrt(1-rho*rho)
    beam=desired-estimated*np.einsum('bi,i->b',estimated.conj(),desired)[:,None]
    beam/=np.linalg.norm(beam,axis=1)[:,None]
    return gamma_i*np.abs(beam[:,0])**2


def validate_covariance_sampler(seed):
    rng=np.random.default_rng(seed)
    trials,antennas,snapshots,energy=5000,8,128,256.
    direct=complex_normal(rng,(trials,antennas,snapshots))
    # Fixed waveform energy, with a spread waveform rather than one occupied sample.
    direct[:,0,:]+=np.sqrt(energy/snapshots)
    direct=direct @ direct.swapaxes(-1,-2).conj()/snapshots
    sampled=covariance_trials(rng,trials,antennas,snapshots,energy)
    expected=np.eye(antennas);expected[0,0]+=energy/snapshots
    errors=[float(np.linalg.norm(x.mean(axis=0)-expected)/np.linalg.norm(expected)) for x in [direct,sampled]]
    a,b=leakage(direct),leakage(sampled)
    se=np.sqrt(a.var(ddof=1)/trials+b.var(ddof=1)/trials)
    z=float(abs(a.mean()-b.mean())/se)
    assert max(errors)<.02 and z<5
    return dict(passed=True,direct_observation_trials=trials,relative_covariance_errors=errors,
        direct_mean_inr=float(a.mean()),sampled_mean_inr=float(b.mean()),mean_difference_standard_errors=z,
        meaning='Independent direct Y Y^H/T simulation agrees with the exact distributional covariance sampler')


def run_synthetic(directory):
    directory=Path(directory)
    config=ActivityConfig(**json.loads((directory/'run_config.json').read_text())['config'])
    output=directory/'synthetic';output.mkdir(exist_ok=True)
    validation=validate_covariance_sampler(config.synthetic_seed)
    atomic_json(output/'covariance_sampler_validation.json',validation)
    coverage=coverage_experiment(config.p_values,config.windows,[config.num_ntn_ues],
        samples=config.coverage_trials,seed=config.synthetic_seed+1)
    write_csv(output/'coverage_50ue.csv',coverage)
    dense=[]
    for p in config.p_values:
        for L in range(1,max(config.windows)+1):
            zero=(1-p)**L
            exact=1. if zero==1 else float(-np.expm1(config.num_ntn_ues*np.log1p(-zero)))
            dense.append(dict(p=p,L=L,N=config.num_ntn_ues,q_exact=exact,union_bound=min(1.,config.num_ntn_ues*zero)))
    write_csv(output/'coverage_dense.csv',dense)
    rows=[];samples={}
    antennas,ns,rho,gamma_i=8,64,.5,10.
    for gi,gamma_s in enumerate([1.,10.]):
        for L in [8,32,128,512]:
            for fi,fraction in enumerate([.125,.5,1.]):
                J=int(L*fraction)
                for mode in ['pooled','active_only_oracle']:
                    T=(L if mode=='pooled' else J)*ns
                    rng=np.random.default_rng(np.random.SeedSequence([config.synthetic_seed,100,gi,L,fi,int(mode!='pooled')]))
                    observed=[]
                    for start in range(0,config.conditional_trials,128):
                        cov=covariance_trials(rng,min(128,config.conditional_trials-start),antennas,T,gamma_s*J*ns)
                        observed.append(leakage(cov,rho,gamma_i))
                    observed=np.concatenate(observed)
                    alpha=1/gamma_s
                    prediction=gamma_i*(alpha/(J*ns)+(L if mode=='pooled' else J)*alpha**2/(J*J*ns))
                    mean=float(observed.mean());se=float(observed.std(ddof=1)/np.sqrt(len(observed)))
                    rows.append(dict(mode=mode,antennas=antennas,gamma_s_db=10*np.log10(gamma_s),
                        gamma_i_db=10.,rho=rho,ns=ns,L=L,J=J,activity_fraction=fraction,T=T,
                        trials=len(observed),mean_inr_linear=mean,lower=max(0.,mean-1.96*se),upper=mean+1.96*se,
                        predicted_inr_linear=prediction,ratio_to_prediction=mean/prediction))
                    samples[f'g{gi}_L{L}_f{fi}_{mode}']=observed
                print(f'Synthetic conditional INR: gamma_s={gamma_s:g}, L={L}, J={J}',flush=True)
    write_csv(output/'conditional_inr.csv',rows)
    atomic_npz(output/'conditional_samples.npz',**samples)
    atomic_json(output/'config.json',dict(seed=config.synthetic_seed,coverage_trials=config.coverage_trials,
        coverage_population=config.num_ntn_ues,conditional_trials=config.conditional_trials,
        conditional_antennas=antennas,conditional_sources=1,snapshots_per_window=ns,
        conditional_model='Fixed active-block energy; iid complex white noise; unconstrained leading-eigenvector estimate and hard null; fixed nonparallel desired/victim signatures',
        covariance_sampler='Exact temporal-unitary reduction and complex Bartlett factor; separately checked against direct sample matrices',
        active_only_oracle='Diagnostic reference using true active windows, not an implementable blind policy',
        scope='Conditional j>0 single-source asymptotic formula validation; does not predict 50-UE MUSIC performance or model the j=0 branch',
        zero_activity='All j=0 users and episodes are retained in the separate full RT network experiment',
        interpretation='Finite-T discrepancy is reported rather than tuning the generator to match the asymptote'))
    print('Synthetic studies complete:',output,flush=True)
