"""Causal stress diagnostics separate from the optimized E7/E8 finite model.

The repeated-observation policy is fixed in advance and need not satisfy the
main experiment's TN constraints. Random-arrival controls are fixed causal
comparators, not an LP solution for an unknown-arrival decision process.
"""
from dataclasses import asdict, dataclass, replace
import numpy as np
from scipy.stats import beta

from .dynamic import JointModel, ModelConfig
from .policy import evaluate_policy
from .reporting import output_dir, rows_csv, json_file, save_figure, plt


def _binomial_interval(k, n):
    if not n:
        return None, None
    return (0.0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
            1.0 if k == n else float(beta.ppf(.975, k+1, n-k)))


def _correlation(pair):
    pair = np.asarray(pair, dtype=float)
    if len(pair) < 3 or np.any(np.std(pair, axis=0) < 1e-12):
        return None
    return float(np.corrcoef(pair.T)[0, 1])


def observation_diagnostics(episodes, *, seed=2048, bootstrap=500):
    """Describe actual executed observations, not unreachable policy entries.

    Correlation is descriptive among trajectories with at least two listens.
    Under adaptive policies that subset can itself be selected by observations;
    the separately declared fixed policy avoids that selection effect.
    """
    counts = np.array([r['sensing_count'] for r in episodes], dtype=int)
    pairs = []
    for row in episodes:
        accepted = [row[k] for k in sorted(row, key=lambda x: int(x.rsplit('_', 1)[-1])
                                          if x.startswith('observation_accepted_') else -1)
                    if k.startswith('observation_accepted_') and row[k] is not None]
        if len(accepted) >= 2:
            pairs.append(accepted[:2])
    n, repeated = len(counts), int(np.sum(counts >= 2))
    lo, hi = _binomial_interval(repeated, n)
    corr, low, high = _correlation(pairs), None, None
    if corr is not None:
        rng = np.random.default_rng(seed)
        a = np.asarray(pairs, dtype=float)
        values = [_correlation(a[rng.integers(0, len(a), len(a))]) for _ in range(bootstrap)]
        values = [v for v in values if v is not None]
        if values:
            low, high = map(float, np.quantile(values, [.025, .975]))
    return dict(mean_sensing_count=float(counts.mean()) if n else None,
                min_sensing_count=int(counts.min()) if n else None,
                max_sensing_count=int(counts.max()) if n else None,
                repeated_listening_episodes=repeated,
                repeated_listening_fraction=repeated/n if n else None,
                repeated_listening_ci_low=lo, repeated_listening_ci_high=hi,
                observation_pair_count=len(pairs), accepted_indicator_correlation=corr,
                accepted_correlation_ci_low=low, accepted_correlation_ci_high=high,
                temporal_correlation_stress_effective=bool(repeated),
                correlation_ci_method='episode percentile bootstrap; undefined for constant indicators')


def _ratio_summary(num, den, *, seed=718, bootstrap=500, horizon=12):
    """Ratio of summed active times with independent-episode uncertainty.

    The extra Hoeffding ratio bound treats both numerator and denominator as
    random bounded episode totals. It remains informative when bootstrap is
    degenerate at zero, and never treats within-episode ticks as independent.
    """
    num, den = np.asarray(num, float), np.asarray(den, float)
    if not len(num) or den.sum() == 0:
        return dict(exceed_ticks=int(num.sum()), active_ticks=int(den.sum()), outage=None,
                    outage_ci_low=None, outage_ci_high=None, outage_hoeffding_upper=None)
    ratio = float(num.sum()/den.sum())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstrap):
        ix = rng.integers(0, len(num), len(num))
        if den[ix].sum():
            values.append(float(num[ix].sum()/den[ix].sum()))
    lo, hi = map(float, np.quantile(values, [.025, .975]))
    radius = np.sqrt(np.log(40)/(2*len(num)))
    lower_den = den.mean()/horizon-radius
    upper = min(1.0, (num.mean()/horizon+radius)/lower_den) if lower_den > 0 else 1.0
    return dict(exceed_ticks=int(num.sum()), active_ticks=int(den.sum()), outage=ratio,
                outage_ci_low=lo, outage_ci_high=hi, outage_hoeffding_upper=float(upper))


def fixed_repeated_policy(model, schedule=(0, 1), duration=1):
    """A declared two-listen causal comparator; no optimization or retuning."""
    if len(set(schedule)) < 2 or any(k not in model.config.listening_epochs for k in schedule):
        raise ValueError('The diagnostic needs at least two legal listening epochs.')
    if model.graph is None:
        model.build()
    return {state: {f'd{duration if model.state_info[state]["epoch"] in schedule else 0}_serve': 1.0}
            for state in model.graph.actions}


def run_repeated_observation(path, *, episodes=400, seed=20260921, config=None):
    path = output_dir(path)
    cfg = replace(config or ModelConfig(gamma_db=-10), listening_epochs=(0, 1))
    model = JointModel(cfg)
    policy = fixed_repeated_policy(model)
    prediction = evaluate_policy(model.graph, policy)
    feasible = all(prediction.costs[k] <= bound+1e-8 for k, bound in model.bounds.items())
    runs, summaries = {}, []
    for label, correlated in [('independent', False), ('temporally_correlated', True)]:
        rows, _, trace = model.simulate_policy(policy, episodes=episodes, seed=seed,
                                               burst_success=correlated)
        runs[label] = rows
        summary = dict(case=label, policy='fixed_two_listens_epochs_0_1_duration_1',
                       episodes=episodes, optimized=False,
                       independent_model_budget_feasible=feasible,
                       certificate_applies=False,
                       empirical_total_bits_per_hz=float(np.mean([r['total_bits_per_hz'] for r in rows])),
                       expected_bits_independent_model=prediction.objective,
                       predicted_worst_tn_fail_independent_model=max(v for k, v in prediction.costs.items()
                                                                     if k.startswith('fail_')),
                       empirical_worst_tn_fail=max(np.mean([r[k] for r in rows])
                                                  for k in model.bounds if k.startswith('fail_')))
        summary.update(observation_diagnostics(rows, seed=seed+31))
        for i in (0, 1):
            summary.update({f'ntn{i}_{k}': v for k, v in _ratio_summary(
                [r[f'exceed_{i}'] for r in rows], [r[f'active_{i}'] for r in rows],
                seed=seed+41+i, horizon=model.H).items()})
        summaries.append(summary)
        rows_csv(path/f'{label}_episodes.csv', rows)
        rows_csv(path/f'{label}_first_episode_timeline.csv', trace)
    differences = [dict(episode=a['episode'],
                        total_bits_difference=b['total_bits_per_hz']-a['total_bits_per_hz'],
                        foreground_outage_difference=b['exceed_0']/b['active_0']-a['exceed_0']/a['active_0'])
                   for a, b in zip(runs['independent'], runs['temporally_correlated'])]
    paired = {}
    rng = np.random.default_rng(seed+51)
    for key in ('total_bits_difference', 'foreground_outage_difference'):
        a = np.array([r[key] for r in differences])
        means = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(500)]
        lo, hi = map(float, np.quantile(means, [.025, .975]))
        paired[key] = dict(mean=float(a.mean()), ci_low=lo, ci_high=hi,
                           direction='temporally_correlated minus independent',
                           ci_method='paired episode percentile bootstrap')
    rows_csv(path/'summary.csv', summaries)
    rows_csv(path/'paired_episode_differences.csv', differences)
    meta = dict(scope='Fixed causal repeated-observation diagnostic; not optimized J',
                config=asdict(cfg), episodes=episodes, seed=seed,
                units='cumulative normalized bits/Hz per episode',
                policy_schedule_epochs=[0, 1], policy_duration_ticks=1,
                physical_paths_paired=True, observation_marginals='P(accept)=acceptance_short in each epoch',
                correlated_kernel='One uniform draw reused in both listening epochs',
                policy_budget_feasible_in_independent_model=feasible,
                warning='The fixed comparator may violate TN budgets; it demonstrates kernel sensitivity, not a feasible J gain.',
                paired_differences=paired, cases=summaries)
    json_file(path/'summary.json', meta)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for i, row in enumerate(summaries):
        ax[0].bar(i, row['accepted_indicator_correlation'] or 0)
        ax[1].bar(i, row['ntn0_outage'])
    ax[0].set(ylabel='First/second acceptance correlation', ylim=(-.2, 1.1))
    ax[1].set(ylabel='Foreground active-time INR exceedance')
    for a in ax:
        a.set_xticks([0, 1], labels=['Independent', 'Correlated'])
        a.grid(axis='y', alpha=.25)
    fig.suptitle('Fixed two-listen diagnostic; TN-budget feasibility reported separately')
    save_figure(fig, path/'repeated_observation_comparison')
    return meta


@dataclass(frozen=True)
class ArrivalDiagnosticConfig:
    horizon: int = 12
    arrival_last_tick: int = 10
    scan_starts: tuple[int, ...] = (2, 6)
    detection_probability: float = .8
    processing_ticks: int = 1
    background_bound: float = .8
    trace_episodes: int = 25


class ArrivalController:
    """Online controller sees only time and released detection notifications."""
    def __init__(self, mode):
        if mode not in ('prior_background', 'no_prior_no_response', 'delayed_detection'):
            raise ValueError(mode)
        self.mode = mode
        self.protection_effective_tick = 0 if mode == 'prior_background' else None

    def receive_released_detection(self, tick, detected):
        if detected and self.mode == 'delayed_detection' and self.protection_effective_tick is None:
            self.protection_effective_tick = tick

    @property
    def protect_background(self):
        return self.protection_effective_tick is not None


def simulate_random_arrival(*, episodes=400, seed=20260921, config=None,
                            arrival_config=None, receiver_kind='ul_active',
                            mode='prior_background', arrival_ticks=None, detection_uniforms=None):
    """Independent random arrivals are hidden from the controller.

    The simulator alone gates observations on physical existence and UL activity.
    A successful one-tick scan samples at start+1, completes at start+2, and is
    usable after the configured processing delay. All three comparators pay the
    same fixed scan costs. The norm envelope is a declared class assumption,
    never inferred from the hidden true channel or the realized arrival time.
    """
    cfg, ac = config or ModelConfig(gamma_db=-10), arrival_config or ArrivalDiagnosticConfig()
    if receiver_kind not in ('ul_active', 'ul_silent'):
        raise ValueError(receiver_kind)
    if ac.horizon != cfg.epochs*cfg.block or not 0 <= ac.arrival_last_tick < ac.horizon:
        raise ValueError('Arrival diagnostic horizon must match the finite TN service reference.')
    if not 0 <= ac.detection_probability <= 1 or ac.processing_ticks < 0:
        raise ValueError('Invalid observation law or processing delay.')
    if any(s < 0 or s+2+ac.processing_ticks >= ac.horizon for s in ac.scan_starts):
        raise ValueError('Scans and releases must lie inside the diagnostic horizon.')
    rng = np.random.default_rng(seed)
    arrivals = rng.integers(0, ac.arrival_last_tick+1, episodes)
    uniforms = rng.random((episodes, len(ac.scan_starts)))
    if arrival_ticks is not None:
        arrivals = np.asarray(arrival_ticks, dtype=int)
    if detection_uniforms is not None:
        uniforms = np.asarray(detection_uniforms, dtype=float)
    if arrivals.shape != (episodes,) or uniforms.shape != (episodes, len(ac.scan_starts)):
        raise ValueError('Invalid paired diagnostic input shape.')
    if np.any(arrivals < 0) or np.any(arrivals >= ac.horizon):
        raise ValueError('Arrival lies outside the episode.')
    models = {protected: JointModel(replace(cfg, include_background=protected,
                                            background_bound=ac.background_bound))
              for protected in (False, True)}
    # Beam lookup depends on released protection state and time only.
    beams = {(protected, t): models[protected].beam(0, t+1).v
             for protected in (False, True) for t in range(ac.horizon)}
    physical_background = np.array([0, 0, 0, ac.background_bound], complex)
    threshold = 10**(cfg.gamma_db/10)
    rows, traces = [], []
    buckets = ('all', 'before_detection', 'after_detection', 'pending_release',
               'before_protection', 'after_protection')
    for ep, arrival in enumerate(arrivals):
        controller = ArrivalController(mode)
        detections = {s+2: s+1 for j, s in enumerate(ac.scan_starts)
                      if receiver_kind == 'ul_active' and arrival <= s+1
                      and uniforms[ep, j] < ac.detection_probability}
        releases = {t+ac.processing_ticks for t in detections}
        first_detection = min(detections) if detections else None
        first_release = min(releases) if releases else None
        row = dict(episode=ep, receiver_kind=receiver_kind, policy=mode,
                   first_appearance_tick=int(arrival), first_detection_tick=first_detection,
                   first_detection_reference_tick=detections[first_detection] if detections else None,
                   first_release_tick=first_release, first_protected_transmission_tick=None,
                   sensing_count=len(ac.scan_starts), dl_bits_per_hz=0.0, ul_bits_per_hz=0.0)
        for bucket in buckets:
            row[f'{bucket}_active_ticks'] = row[f'{bucket}_exceed_ticks'] = 0
        for t in range(ac.horizon):
            # No arrival or physical-channel argument is passed to the controller.
            controller.receive_released_detection(t, t in releases)
            protected = controller.protect_background
            rf_gap = any(s <= t < s+2 for s in ac.scan_starts)
            tx = models[False].is_dl(t) and not rf_gap
            v = beams[protected, t] if tx else np.zeros(4, complex)
            inr = float(abs(np.vdot(physical_background, v))**2)
            active, exceed = t >= arrival, inr > threshold*(1+1e-7)
            if active and protected and tx and row['first_protected_transmission_tick'] is None:
                row['first_protected_transmission_tick'] = t
            conditions = dict(all=True,
                              before_detection=first_detection is None or t < first_detection,
                              after_detection=first_detection is not None and t >= first_detection,
                              pending_release=first_detection is not None and first_detection <= t < first_release,
                              before_protection=not protected, after_protection=protected)
            for bucket, condition in conditions.items():
                row[f'{bucket}_active_ticks'] += int(active and condition)
                row[f'{bucket}_exceed_ticks'] += int(active and condition and exceed)
            dl = float(np.log2(1+cfg.tn_snr_linear*abs(np.vdot(models[False].h, v))**2))
            ul = cfg.ul_rate if not models[False].is_dl(t) and not rf_gap else 0.0
            row['dl_bits_per_hz'] += dl
            row['ul_bits_per_hz'] += ul
            if t+1 in models[False].deadlines:
                for direction, theta in [('DL', cfg.theta_dl), ('UL', cfg.theta_ul)]:
                    row[f'fail_{direction}_{t+1}'] = int(row[f'{direction.lower()}_bits_per_hz'] <
                        theta*models[False].reference[f'{direction}_{t+1}']-1e-9)
            if ep < ac.trace_episodes:
                traces.append(dict(episode=ep, t=t, receiver_kind=receiver_kind, policy=mode,
                                   first_appearance_tick=int(arrival), dl_active=int(active),
                                   detection_completed=int(t in detections), detection_released=int(t in releases),
                                   protected=int(protected), rf_gap=int(rf_gap), bs_transmitting=int(tx),
                                   power=float(np.vdot(v, v).real), inr_linear=inr,
                                   exceed=int(active and exceed), dl_bits_per_hz=dl, ul_bits_per_hz=ul))
        row['protection_effective_tick'] = controller.protection_effective_tick
        row['total_bits_per_hz'] = row['dl_bits_per_hz']+row['ul_bits_per_hz']
        rows.append(row)
    return rows, traces


def run_random_arrival(path, *, episodes=400, seed=20260921, config=None, arrival_config=None):
    path = output_dir(path)
    cfg = config or ModelConfig(gamma_db=-10)
    horizon = cfg.epochs*cfg.block
    ac = arrival_config or ArrivalDiagnosticConfig(horizon=horizon, arrival_last_tick=min(10, horizon-2),
                scan_starts=tuple(s for s in (2, 6) if s+3 < horizon))
    summaries, all_rows, all_traces = [], [], []
    for kind in ('ul_active', 'ul_silent'):
        for mode in ('prior_background', 'no_prior_no_response', 'delayed_detection'):
            rows, traces = simulate_random_arrival(episodes=episodes, seed=seed, config=cfg,
                arrival_config=ac, receiver_kind=kind, mode=mode)
            all_rows.extend(rows)
            all_traces.extend(traces)
            detected = sum(r['first_detection_tick'] is not None for r in rows)
            lo, hi = _binomial_interval(detected, episodes)
            summary = dict(receiver_kind=kind, policy=mode, episodes=episodes,
                           detected_fraction=detected/episodes, detected_ci_low=lo, detected_ci_high=hi,
                           mean_sensing_count=float(np.mean([r['sensing_count'] for r in rows])),
                           mean_bits_per_hz=float(np.mean([r['total_bits_per_hz'] for r in rows])),
                           worst_tn_failure_rate=max(np.mean([r[k] for r in rows]) for k in rows[0]
                                                     if k.startswith('fail_')),
                           guarantee_scope={'prior_background': 'all active times if prior envelope is valid',
                                            'no_prior_no_response': 'none',
                                            'delayed_detection': 'after release only; no pre-discovery guarantee'}[mode],
                           optimized=False, unknown_arrival_visible_to_controller=False)
            if kind == 'ul_silent' and mode == 'delayed_detection':
                summary['guarantee_scope'] = 'none: permanently UL-silent receiver cannot be discovered'
            summary['global_ntn_budget_certificate'] = mode == 'prior_background'
            for bucket in ('all', 'before_detection', 'after_detection', 'pending_release',
                           'before_protection', 'after_protection'):
                summary.update({f'{bucket}_{k}': v for k, v in _ratio_summary(
                    [r[f'{bucket}_exceed_ticks'] for r in rows],
                    [r[f'{bucket}_active_ticks'] for r in rows],
                    seed=seed+71, horizon=ac.horizon).items()})
            summaries.append(summary)
    rows_csv(path/'summary.csv', summaries)
    rows_csv(path/'episode_events.csv', all_rows)
    rows_csv(path/'tick_traces.csv', all_traces)
    meta = dict(scope='Random unknown-arrival fixed causal policy diagnostic; no global LP optimality claim',
                episodes_per_case=episodes, seed=seed, model=asdict(cfg), arrival_config=asdict(ac),
                arrival_law=f'Uniform integer tick in [0, {ac.arrival_last_tick}], independently per episode',
                hidden_from_controller=['realized arrival time', 'true channel', 'receiver UL-silent status'],
                prior_information='Common receiver-class norm bound; enforced always, never, or only after released discovery',
                observation_law='UL-active receiver can be detected at a scan after arrival; UL-silent receiver is never detected',
                causal_timing='scan sample=start+1; detection complete=start+2; release=start+2+processing_ticks',
                paired_comparators='Same arrival times and detection uniforms; same RF monitoring costs',
                uncertainty='Independent-episode bootstrap ratios; conservative Hoeffding ratio upper bounds handle random denominators',
                trace_episodes_per_case=min(episodes, ac.trace_episodes),
                units='cumulative normalized bits/Hz per episode', cases=summaries)
    json_file(path/'summary.json', meta)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labels = ['Prior envelope', 'No response', 'Delayed discovery']
    for a, kind in zip(ax, ('ul_active', 'ul_silent')):
        subset = [r for r in summaries if r['receiver_kind'] == kind]
        a.bar(range(3), [r['all_outage'] for r in subset])
        a.set_xticks(range(3), labels=labels, rotation=20, ha='right')
        a.set_title('UL-active new receiver' if kind == 'ul_active' else 'UL-silent new receiver')
        a.set_ylabel('New receiver active-time INR exceedance')
        a.grid(axis='y', alpha=.25)
    fig.suptitle('Random hidden arrival; fixed causal comparators, shared sensing overhead')
    save_figure(fig, path/'random_arrival_comparison')
    return meta
