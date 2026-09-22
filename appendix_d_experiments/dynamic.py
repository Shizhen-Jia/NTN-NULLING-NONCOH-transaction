"""Exact finite-history experiment model for E7/E8.

This is a declared synthetic hidden-Markov model, not a temporal Sionna replay.
The physical channel process is exogenous. The controller sees only accepted
delayed observations and its own delivered service. No online true NTN CSI.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from .robust import ProtectionSet, solve_robust_beam
from .policy import Action, Graph, solve_occupancy


@dataclass(frozen=True)
class ModelConfig:
    epochs: int = 3
    block: int = 4
    antennas: int = 4
    gamma_db: float = -5.0
    theta_dl: float = 0.40
    theta_ul: float = 0.50
    delta_tn: float = 0.10
    delta_ntn: float = 0.08
    switch_probability: float = 0.025
    processing_ticks: int = 1
    blocking_processing: bool = False
    acceptance_short: float = 0.48
    acceptance_long: float = 0.88
    rejection_short: float = 0.12
    rejection_long: float = 0.06
    rho0: float = 0.025
    rho_per_tick: float = 0.012
    background_bound: float = 0.28
    include_background: bool = True
    listening_epochs: tuple[int, ...] = (0, 1, 2)
    background_arrival_tick: int = 0
    tn_snr_linear: float = 20.0
    ul_rate: float = 2.0


class JointModel:
    def __init__(self, config=ModelConfig()):
        self.config = config
        if config.antennas != 4 or config.epochs < 2 or config.block != 4:
            raise ValueError('This declared model uses M=4, block=4, and at least two epochs.')
        if config.processing_ticks not in (0, 1):
            raise ValueError('Use processing_ticks=0 or1; pending jobs may not cross this decision grid.')
        for a, r in [(config.acceptance_short, config.rejection_short),
                     (config.acceptance_long, config.rejection_long)]:
            if min(a, r) < 0 or a+r > 1:
                raise ValueError('Invalid observation probability.')
        q = config.switch_probability
        if not 0 <= q <= 1:
            raise ValueError('Invalid Markov switching probability.')
        if not 0 <= config.background_arrival_tick < config.epochs*config.block:
            raise ValueError('Background arrival must fall within the episode.')
        if min(config.theta_dl, config.theta_ul) < 0 or not 0 <= config.delta_tn <= 1 or not 0 <= config.delta_ntn <= 1:
            raise ValueError('Invalid service demand or probability budget.')
        self.P = np.array([[1-q, q], [q, 1-q]])
        self.directions = np.array([[1, 0, 0, 0], [0.15, 0.98, 0.10, 0]], dtype=complex)
        self.directions /= np.linalg.norm(self.directions, axis=1)[:, None]
        self.foreground = 2.0*self.directions
        self.background = np.array([0, 0, 0, config.background_bound], dtype=complex)
        self.h = np.array([0.78, 0.38, 0.39, 0.30], dtype=complex)
        self.h /= np.linalg.norm(self.h)
        self.H = config.epochs*config.block
        self.deadlines = (2*config.block, self.H) if config.epochs > 2 else (self.H,)
        self.reference = {}
        for end in self.deadlines:
            self.reference[f'DL_{end}'] = sum(self.is_dl(t) for t in range(end))*np.log2(1+config.tn_snr_linear)
            self.reference[f'UL_{end}'] = sum(not self.is_dl(t) for t in range(end))*config.ul_rate
        self.bounds = {f'fail_{key}': config.delta_tn for key in self.reference}
        self.bounds.update({'risk_0': 0.0, 'risk_1': 0.0})
        self.state_info = {}
        self.branch_info = {}
        self.beams = {}
        self.graph = None

    @staticmethod
    def is_dl(t):
        return t % 4 != 1  # one fixed UL opportunity per block

    def sets(self, record, age):
        c = self.config
        sets = [ProtectionSet(self.directions[record, :, None], 2.0,
                              c.rho0+c.rho_per_tick*age, 'anonymous_foreground')]
        if c.include_background:
            sets.append(ProtectionSet(np.zeros((4, 0), complex), 0.0,
                                      c.background_bound, 'UL_silent_background'))
        return sets

    def beam(self, record, age):
        key = (record, age)
        if key not in self.beams:
            self.beams[key] = solve_robust_beam(self.h, self.sets(record, age),
                                              10**(self.config.gamma_db/10))
        return self.beams[key]

    def failure_probability(self, belief, record, age):
        a = self.directions[record]
        c = self.config
        failures = []
        for f in self.foreground:
            coeff = np.vdot(a, f)
            if abs(coeff) > 2:
                coeff *= 2/abs(coeff)
            failures.append(np.linalg.norm(f-a*coeff) > c.rho0+c.rho_per_tick*age+1e-10)
        return float(np.dot(belief, failures))

    def outcomes(self, belief, duration):
        if duration == 0:
            return [('none', 1.0)]
        c = self.config
        accept = c.acceptance_short if duration == 1 else c.acceptance_long
        reject = c.rejection_short if duration == 1 else c.rejection_long
        at_reference = belief @ self.P  # g_in=1; reference timestamp s+1
        return [(f'accept{j}', float(accept*at_reference[j])) for j in range(2)] + [
            ('reject', reject), ('no_detection', 1-accept-reject)]

    def segment(self, info, duration, serving, outcome):
        c = self.config
        start = info['epoch']*c.block
        release = start+1+duration+c.processing_ticks if duration else self.H+1
        accepted = outcome.startswith('accept')
        new_record = int(outcome[-1]) if accepted else info['record']
        steps, dl, ul, risk, gap = [], 0.0, 0.0, 0.0, 0
        for t in range(start, start+c.block):
            rf = bool(duration and t < start+1+duration)
            processing = bool(duration and start+1+duration <= t < release)
            blocked = rf or (processing and c.blocking_processing)
            released = accepted and t >= release
            record = new_record if released else info['record']
            stamp = start+1 if released else info['stamp']
            age = t-stamp
            belief = (np.eye(2)[record] @ np.linalg.matrix_power(self.P, age)
                      if released else info['belief'] @ np.linalg.matrix_power(self.P, t-start))
            v = np.zeros(4, complex)
            rd, ru, eta = 0.0, 0.0, 0.0
            tx = self.is_dl(t) and not blocked and serving
            if tx:
                v = self.beam(record, age).v
                rd = float(np.log2(1+c.tn_snr_linear*abs(np.vdot(self.h, v))**2))
                eta = self.failure_probability(belief, record, age)
            elif not self.is_dl(t) and not blocked:
                ru = c.ul_rate  # DL mute does not cancel the scheduled TN uplink
            dl += rd
            ul += ru
            risk += eta
            gap += int(blocked)
            steps.append(dict(t=t, rf_gap=int(rf), processing=int(processing),
                              blocked=int(blocked), released=int(accepted and t == release),
                              age=age, record=record, tx=int(tx), power=float(np.vdot(v,v).real),
                              dl_bits_per_hz=rd, ul_bits_per_hz=ru, eta=eta, v=v))
        next_belief = (np.eye(2)[new_record] @ np.linalg.matrix_power(self.P, c.block-1)
                       if accepted else info['belief'] @ np.linalg.matrix_power(self.P, c.block))
        next_info = dict(epoch=info['epoch']+1, record=new_record,
                         stamp=start+1 if accepted else info['stamp'], belief=next_belief,
                         dl=info['dl']+dl, ul=info['ul']+ul)
        active_background = float(sum(t >= c.background_arrival_tick for t in range(start,start+c.block)))
        costs = {'risk_0': risk-c.delta_ntn*c.block,
                 # Both receivers are always DL-active, also during TN UL/gaps/mute.
                 'risk_1': -c.delta_ntn*active_background,
                 'active_0': float(c.block), 'active_1': active_background,
                 'certified_0': risk, 'certified_1': 0.0,
                 'dl_bits': dl, 'ul_bits': ul,
                 'dl_reference_loss': 3*np.log2(1+c.tn_snr_linear)-dl,
                 'ul_reference_loss': c.ul_rate-ul,
                 'power_sum': float(sum(s['power'] for s in steps)),
                 'gap_ticks': float(gap), 'sensing_count': float(duration>0),
                 'sensing_ticks': float(duration),
                 'mute_dl_ticks': float(sum(self.is_dl(s['t']) and not s['blocked'] and not serving for s in steps))}
        if next_info['epoch']*c.block in self.deadlines:
            end = next_info['epoch']*c.block
            for direction, theta in [('DL', c.theta_dl), ('UL', c.theta_ul)]:
                costs[f'fail_{direction}_{end}'] = float(next_info[direction.lower()] < theta*self.reference[f'{direction}_{end}']-1e-9)
        return next_info, steps, float(dl+ul), costs

    def build(self):
        actions, terminals = {}, set()
        initial = dict(epoch=0, record=0, stamp=-1, belief=np.eye(2)[0]@self.P, dl=0.0, ul=0.0)
        self.state_info = {'s': initial}
        queue = ['s']
        for sid in queue:
            info = self.state_info[sid]
            if info['epoch'] == self.config.epochs:
                terminals.add(sid)
                continue
            aa = []
            durations = (0,1,2) if info['epoch'] in self.config.listening_epochs else (0,)
            for duration, serving in product(durations, (False, True)):
                name = f'd{duration}_{"serve" if serving else "mute"}'
                transitions, reward, expected_costs = [], 0.0, {}
                for outcome, prob in self.outcomes(info['belief'], duration):
                    if prob <= 1e-15:
                        continue
                    nid = f'{sid}/{name}:{outcome}'
                    nxt, steps, r, costs = self.segment(info, duration, serving, outcome)
                    self.state_info[nid] = nxt
                    self.branch_info[(sid, name, outcome)] = (nid, steps)
                    queue.append(nid)
                    transitions.append((nid, prob))
                    reward += prob*r
                    for key, val in costs.items():
                        expected_costs[key] = expected_costs.get(key, 0.0)+prob*val
                aa.append(Action(name=name, reward=reward, transitions=transitions, costs=expected_costs))
            actions[sid] = aa
        self.graph = Graph(actions=actions, initial={'s': 1.0}, terminals=terminals)
        return self.graph

    def restrict(self, mode='J', fixed_duration=1, schedule=(0, 1, 2)):
        if self.graph is None:
            self.build()
        allowed, terminal, queue = {}, set(), ['s']
        for sid in queue:
            if sid in self.graph.terminals:
                terminal.add(sid)
                continue
            k = self.state_info[sid]['epoch']
            if mode == 'J':
                durations = {0, 1, 2}
            elif mode == 'T':
                durations = {0, fixed_duration}
            elif mode == 'L':
                durations = {1, 2} if k in schedule else {0}
            elif mode == 'F':
                durations = {fixed_duration} if k in schedule else {0}
            else:
                raise ValueError(mode)
            aa = [a for a in self.graph.actions[sid] if int(a.name[1]) in durations]
            allowed[sid] = aa
            queue.extend(n for a in aa for n, p in a.transitions if p > 0)
        return Graph(actions=allowed, initial={'s': 1.0}, terminals=terminal)

    def optimize(self, mode='J'):
        # Offline tuning is exact in the SAME declared model, never on test trajectories.
        legal_epochs = tuple(k for k in range(self.config.epochs) if k in self.config.listening_epochs)
        schedules = [tuple(k for k, yes in zip(legal_epochs, mask) if yes)
                     for mask in product((False, True), repeat=len(legal_epochs))]
        menus = [('J', 1, ())] if mode == 'J' else (
            [('T', d, ()) for d in (1, 2)] if mode == 'T' else (
                [('L', 1, sched) for sched in schedules] if mode == 'L' else
                [('F', d, sched) for sched in schedules for d in (1, 2)]))
        candidates = []
        for kind, duration, schedule in menus:
            graph = self.restrict(kind, duration, schedule)
            result = solve_occupancy(graph, self.bounds)
            candidates.append((result, dict(mode=kind, duration=duration, schedule=schedule)))
        feasible = [x for x in candidates if x[0].feasible]
        chosen = max(feasible, key=lambda x: x[0].objective) if feasible else candidates[0]
        chosen[1]['tuning_candidates'] = len(candidates)
        return chosen

    def simulate(self, result, episodes=1000, seed=7001, physical_gain=1.0,
                 background_gain=1.0, burst_success=False):
        """Common random streams across policies; physical states never used by policy."""
        if not result.feasible:
            return [], [], []
        return self.simulate_policy(result.policy, episodes=episodes, seed=seed,
                                    physical_gain=physical_gain, background_gain=background_gain,
                                    burst_success=burst_success)

    def simulate_policy(self, policy, episodes=1000, seed=7001, physical_gain=1.0,
                        background_gain=1.0, burst_success=False):
        """Execute a causal policy without claiming TN/NTN budget feasibility."""
        c = self.config
        rng = np.random.default_rng(seed)
        episode_rows, samples, trace = [], [], []
        for ep in range(episodes):
            # Exogenous physical channel is generated BEFORE selecting sensing/beam actions.
            latent = np.zeros(self.H+1, int)
            latent[0] = int(rng.random() < c.switch_probability)
            physical_u = rng.random(self.H)
            for t in range(self.H):
                latent[t+1] = 1-latent[t] if physical_u[t] < c.switch_probability else latent[t]
            observation_u = rng.random(c.epochs)
            if burst_success:
                observation_u[:] = observation_u[0]  # same marginal, deliberately mismatched temporal correlation
            action_u = rng.random(c.epochs)
            sid, totals = 's', {'dl': 0.0, 'ul': 0.0}
            counts = [0, 0]
            row = dict(episode=ep, sensing_count=0)
            for k in range(c.epochs):
                distribution = policy[sid]
                names = list(distribution)
                cumulative = np.cumsum([distribution[n] for n in names])
                name = names[min(int(np.searchsorted(cumulative, action_u[k], side='right')), len(names)-1)]
                duration = int(name[1])
                if duration:
                    a = c.acceptance_short if duration == 1 else c.acceptance_long
                    r = c.rejection_short if duration == 1 else c.rejection_long
                    u = observation_u[k]
                    outcome = f'accept{latent[k*c.block+1]}' if u < a else ('reject' if u < a+r else 'no_detection')
                else:
                    outcome = 'none'
                row['sensing_count'] += int(duration > 0)
                row[f'observation_outcome_{k}'] = outcome
                row[f'observation_accepted_{k}'] = int(outcome.startswith('accept')) if duration else None
                nid, steps = self.branch_info[(sid, name, outcome)]
                for st in steps:
                    t, v = st['t'], st['v']
                    values = [float(abs(np.vdot(physical_gain*self.foreground[latent[t]], v))**2),
                              float(abs(np.vdot(background_gain*self.background, v))**2)]
                    for i, val in enumerate(values):
                        active = i == 0 or t >= c.background_arrival_tick
                        counts[i] += int(active and val > 10**(c.gamma_db/10)*(1+1e-7))
                        samples.append(dict(episode=ep, t=t, receiver=i, dl_active=int(active),
                                            bs_transmitting=st['tx'], tn_dl_slot=int(self.is_dl(t)),
                                            tn_snr_linear=float(c.tn_snr_linear*abs(np.vdot(self.h, v))**2), inr_linear=val,
                                            inr_db=10*np.log10(max(val, 1e-16))))
                    totals['dl'] += st['dl_bits_per_hz']
                    totals['ul'] += st['ul_bits_per_hz']
                    if ep == 0:
                        trace.append({kk: vv for kk, vv in st.items() if kk != 'v'} |
                                     dict(action=name, outcome=outcome,
                                          cumulative_dl=totals['dl'], cumulative_ul=totals['ul'],
                                          inr_0_db=10*np.log10(max(values[0], 1e-16)),
                                          inr_1_db=10*np.log10(max(values[1], 1e-16))))
                sid = nid
                end = (k+1)*c.block
                if end in self.deadlines:
                    for direction, theta in [('DL', c.theta_dl), ('UL', c.theta_ul)]:
                        row[f'fail_{direction}_{end}'] = int(totals[direction.lower()] < theta*self.reference[f'{direction}_{end}']-1e-9)
            row.update(dl_bits_per_hz=totals['dl'], ul_bits_per_hz=totals['ul'],
                       total_bits_per_hz=totals['dl']+totals['ul'],
                       exceed_0=counts[0], exceed_1=counts[1], active_0=self.H, active_1=self.H-c.background_arrival_tick)
            episode_rows.append(row)
        return episode_rows, samples, trace
