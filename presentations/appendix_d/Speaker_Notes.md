# Appendix D — English speaker notes

Main presentation: slides 1–25, approximately 25–30 minutes. Backup: slides 26–30.
The editable PowerPoint uses native text and diagram shapes; equations and scientific plots are high-resolution images.
Paper source: `overleaf_ntn_paper/direction.tex`. Pinned run: `result/appendix_d_20260921_235448_775259`.

## 01. Service-Constrained Joint Sensing and Robust Transmission

Opening: The main paper tells us where to suppress interference. Appendix D asks the next question: when should a base station pay the cost of sensing, and how should it transmit until the next useful observation? The aim is to maximize delivered terrestrial service while controlling deadline failures and non-terrestrial interference exposure. This presentation follows the five parts of the appendix, then adds explicitly labeled results from our current implementation. The theory is conditional on its observation and protection models. Current fresh ray tracing validates the spatial component, not the complete physical dynamic controller. Suggested duration: 25–30 minutes for slides 1–25; slides 26–30 are backups.

Source: Paper: Appendix D · direction.tex | 22 September 2026

## 02. A better estimate is useful only if it earns back its cost

Explain the three competing actions. Listening is not free: reusing the front end removes scheduled downlink and uplink opportunities. Continuing service uses old, potentially less informative protection sets. Silence controls this base station’s interference contribution but sacrifices downlink bits. A received estimate is also already aged by the time processing finishes. The desired optimization compares the complete service and risk consequences of all three choices. This is why a static INR curve, or a smaller direction-estimation error, is not sufficient evidence for the joint controller.

Source: Appendix D

## 03. One controlled sector, two networks, partial information

TN means terrestrial network; NTN means non-terrestrial network. The controlled object is one TN sector with one scheduled stream per resource. Its downlink can interfere with an NTN downlink receiver. The base station listens to an NTN uplink band, which can differ from the protected downlink frequency. An uplink-silent terminal can still be receiving its satellite downlink and therefore require protection. The controller has its own current TN channel in the declared model, but it does not receive the true victim channel or the future physical state. Other interference is fixed or separately budgeted in the exact core.

Source: Appendix D.A–B · eq:joint_exposure

## 04. The appendix connects sensing to service through two optimizers

Walk left to right. Released sensing information is mapped to a catalog of uncertainty sets and conditional coverage-risk bounds. For each usable state, a robust second-order cone program computes the best desired-channel amplitude and the corresponding power. The outer linear program uses those service values, sensing outcomes and deadline events to choose a randomized causal policy. Observation quality, delay and unsuccessful sensing enter the transition model. These are layers of one design, not three unrelated algorithms. This slide is the roadmap: timing, sets, service and risk, SOCP, then LP.

Source: Appendix D

## 05. A new estimate is unavailable until its release time

Use the example rather than starting with symbols. Entry takes one time unit, collection two, return one, and processing three. Samples are referenced to time one, not time seven. The front end is unavailable until time four. With nonblocking processing, the base station can resume calendar-permitted service at time four, but it must use the old record until time seven. Acceptance at time seven gives a record aged six units. Rejection preserves the old reference time. This is an illustrative appendix timeline, not the shorter guard values in the current 12-tick simulator. Blocking processing would extend unavailability through release.

Source: Appendix D.A · eq:joint_timing · illustrative 1/2/1/3 time units

## 06. Longer listening trades accuracy against usable service time

This plot is deliberately schematic. It visualizes uncertainty growing while the previous record is used; after an accepted observation, the reference changes to the start of that listening window, not to its release time. A refreshed record can therefore start with substantial age. Better observation quality can reduce an estimation component, while age, calibration error and missing multipath remain. The appendix does not assume a universal inverse-square-root law in listening duration. For a fixed ground BS-to-VSAT link, satellite orbital speed must not be substituted for ground-link direction change.

Source: Appendix D.A–B · schematic, not fitted experimental data

## 07. Protect a set of complete channels, not individual peaks

Explain A, C and rho separately. Columns of A are downlink array responses reconstructed from observed directions. The complex coefficient vector permits arbitrary coherent relative phases within its norm bound. The residual ball covers unmodeled or nontransferable paths and other errors. A group does not require identification of a specific receiver. A complete victim channel must belong to at least one imposed group; independently protecting individual peaks is insufficient when their coherent sum is the actual channel. The two-dimensional figure is a conceptual projection of a complex high-dimensional set, not an exact physical channel plot.

Source: Appendix D.B · eq:joint_multipath_set

## 08. The worst-case leakage has an exact closed form

The upper bound comes from Cauchy–Schwarz and the triangle inequality. It is attainable: choose the coefficient vector aligned with A Hermitian v and the residual aligned with v, with phases that add rather than cancel. That is why the support expression is exact for this uncertainty set. The two terms explain the engineering tradeoff. Steering away from the estimated subspace reduces the first term, but a residual ball generally leaves a term proportional to total beam norm. The INR threshold Gamma is a power ratio; therefore the amplitude constraint uses square root Gamma.

Source: Appendix D.B · Proposition: exact set-wise protection

## 09. A silent receiver can still force power backoff

The normalized channel f includes receiver noise, spectral coupling and the power limit. The vector v combines beam shape and power fraction: physical power is Pmax times p. A background norm ball can be represented either with A equal to the identity and coefficient bound beta, or as a pure residual ball of radius beta; both give beta times beam norm. A receiver with no observable uplink requires prior coverage, a valid arrival/activity model, or side information. The method cannot infer an unknown downlink antenna gain from a MUSIC weight alone. Zero downlink transmission protects the controlled contribution, but not unrelated external interference.

Source: Appendix D.B–C · eq:joint_exposure · background catalog

## 10. Service reliability and interference exposure use different metrics

The TN constraint is a chance constraint on deadline service, independently for each applicable user, direction and window. The reference and demand cannot be recalculated after seeing a favorable Gamma or future channel. The NTN metric is the expected exceeding active duration divided by expected active duration. In measurements, estimate it by total exceeding duration divided by total active duration across complete independent episodes, not an unweighted average of episode ratios with varying denominators. BS silence makes the controlled interference zero but does not erase the receiver’s active time. Gamma controls interference magnitude; delta NTN controls the allowable active-time fraction.

Source: Appendix D.C · eq:joint_service_target · eq:joint_activity_outage

## 11. Coverage risk bridges robust constraints and actual outage

On valid catalog coverage, the robust beam constraint makes the exceedance event impossible for the controlled contribution. Exceedance can therefore occur only through a coverage or bound failure. Use component upper bounds under the same history, action and receiver-activity conditioning; their union bound does not require independence. Multiplying a valid throughout-segment conditional probability bound by expected active duration yields a risk-duration coefficient. Overall average calibration coverage does not automatically remain valid after a policy selects particular histories. Sparse histories require conservative fallback bounds or a narrower model. Statistical confidence in an estimated bound is separate from the physical outage budget.

Source: Appendix D.C/E · eq:joint_risk_union · eq:joint_risk_cost

## 12. The joint problem maximizes net service under explicit budgets

The decision variable is a causal policy, not just a beam vector. It chooses legal sensing starts, listening durations and templates, plus transmission or silence. C is a computable upper bound on expected exceeding active duration; D is expected active duration. For valid positive activity, C less than delta times D is sufficient for the actual outage target. The objective directly rewards bits delivered after overhead, so a common arbitrary penalty lambda is unnecessary in this core. Infeasibility is a legitimate result: do not relax the TN requirement silently just to draw a curve.

Source: Appendix D.C · eq:joint_policy_problem

## 13. At one state, a convex program selects beam and power

With fixed current TN channel and protection catalog, all constraints depend on phase-invariant magnitudes. Rotate the beam so the desired inner product is real and nonnegative, then maximize its real part. This gives an SOCP. The optimum can use full power, partial nulling, or power backoff. When the solution is nonzero, p is its squared norm and w is the unit-normalized direction. Its rate is B log2(1 + Pmax V squared divided by noise plus external interference). The feasible beam set always contains zero; positive service feasibility is the real issue. Age expansion of fixed-center uncertainty sets can only shrink the feasible set.

Source: Appendix D.D · eq:joint_value_socp

## 14. Why the outer controller does not search every beam

This is the key reduction linking continuous beam design to a finite policy problem. Consider a virtual copy of any original policy with its original service counters. On the same exogenous trajectory and random seed, keep the sensing and mute choices, but use the SOCP maximizing beam whenever transmitting. Actual service only increases, while the certified risks and future observation laws remain unchanged under the stated assumptions. Minimum-service failures cannot become worse. A virtual copy is needed because later decisions may depend on the original service counters. This theorem does not cover beam-dependent actual outage costs, intertemporal energy, coupled network SINR, or uncertain current TN CSI without further analysis.

Source: Appendix D.D · Proposition: lossless beam elimination

## 15. A causal state contains what is known—not the hidden channel

The complete observable history supports a belief over hidden physical states without revealing them to the controller. The state also carries timing, pending-release metadata, delivered service and timestamps, because those affect legal actions and deadline events. The graph branches on observations actually released. Rejection and no detection remain explicit branches; pruning them and renormalizing changes the declared model. Time increases along each edge, so the graph is acyclic. It can nevertheless grow exponentially with horizon and observation alphabet, which is why the exact solver is a small-instance benchmark rather than an automatic solution to arbitrary large deployments.

Source: Appendix D.E · eq:joint_history_state · eq:joint_belief_update

## 16. Occupation measures turn policy optimization into an LP

Read the four lines as reward, probability flow, TN deadline failures and NTN risk-duration budgets. The LP variables are probabilities of state-action visits. Flow conservation ensures they correspond to a causal policy. The coefficient g-bar is a deadline-failure event cost, charged once after crediting the just-completed resource. Coefficients c and d are expected exceeding-duration bounds and activity durations under the declared model. All SOCP values and coefficients are fixed before solving the LP. At positive-mass states, divide each action mass by the sum over actions to get action probabilities. Terminal states absorb mass and have no outgoing actions.

Source: Appendix D.E · eq:joint_occupancy_objective through eq:joint_policy_extraction

## 17. Randomization is sometimes required, not merely convenient

The top example uses deliberately simple illustrative numbers, not measured results. Treat the NTN cost as a normalized risk cost with a common fixed active denominator. Policy A is safe but has excessive TN failures; policy B has better service reliability but excessive NTN risk. The equal mixture is feasible in both coordinates. A deterministic choice of either whole policy fails a requirement. In the actual E6 finite test, all 128 deterministic policies are infeasible for the randomization-required configuration, but their randomized mixture and the occupation LP achieve the same feasible objective 1.125. The extracted probabilities must be executed as probabilities; taking an argmax is not an equivalent implementation.

Source: Appendix D.E + E6/E6_validation.csv · illustrative mixture below

## 18. There are three distinct levels of guarantee

Do not merge these claims. The support-function proposition is exact for the declared uncertainty set. The beam-elimination proposition and occupation-measure theorem establish optimality over randomized causal policies in the declared finite certified problem, assuming exact transitions, coefficients and solutions. A physical outage guarantee additionally requires that the implemented model and conditional certificates remain valid for the physical system. A favorable static CDF alone does not establish that final condition. Similarly, approximate history compression or numerical solutions require their approximation and residual errors to be addressed. State these distinctions explicitly during the talk.

Source: Appendix D.B/D/E · robust support, beam elimination, finite optimality theorem

## 19. Most optimization is offline; execution follows released events

The algorithm freezes model inputs, expands all reachable histories, builds catalogs and risk bounds, solves or caches state-specific SOCPs, computes transition rewards and costs, and solves the LP. Online operation follows the conditional policy and only updates on events that have actually been released. A beam cache key must include every quantity affecting the solution, including current CSI, sets, radii, interference and Gamma. Deadline boundaries do not justify resetting ages, pending jobs or unfinished requirements. If a zero-probability history is physically reached, that signals model mismatch: a silent fallback controls the BS contribution, but its service loss must still be counted.

Source: Appendix D.E · finite-model construction and execution

## 20. What the current experiments validate

This table prevents a common overclaim. E4 verifies a synthetic whole-scene calibration procedure; it does not supply a physical conditional risk table for arbitrary adaptive histories. E5 and E6 provide solver and reduction checks. E7 is the dynamic finite-model service comparison. E8 includes both optimized-model stress cases and separate fixed-policy mechanism diagnostics. Fresh ray tracing now really redraws user positions and recomputes propagation, but it validates the static spatial component. The 34 regression tests and full notebook execution support implementation consistency, not a universal physical deployment guarantee. The pinned run is September 21 at 23:54:48 local time.

Source: Run 20260921_235448_775259 · E4–E8 + fresh_rt_spatial

## 21. Finite-model gains are positive, but small

All four classes use the same robust transmission subproblem and can serve or mute; fixed baselines are optimized within their declared classes. These are exact model expectations accumulated over 12 ticks, not per-tick throughput or noisy simulation means. At minus 15 dB, J yields 23.0089 bits/Hz versus F at 22.4713, a 2.39% gain. Its advantage over T is only 0.247%. At the other three Gamma values J and T coincide; F and L coincide throughout. Increasing the Monte Carlo sample count will tighten uncertainty but will not separate identical model optima. Sixteen main configurations were feasible. Confidence on actual simulated risk is discussed in a backup slide.

Source: E7/strategy_summary.csv · exact model expectations; Monte Carlo: 400 episodes per setting

## 22. Fresh ray tracing reveals a substantial service cost

Unlike the earlier cache-only path, this run creates six fresh TN/NTN drops and retraces both bands. Each drop contains 12 TN users, 20 NTN receivers and 12 sectors. Two complete drops each are used for training, calibration and test, with all 24 test sectors evaluated. The plotted INR is one controlled sector’s contribution, not the network sum. At Gamma minus 15 dB, only 58.33% of sectors meet the static minus 6 dB TN SNR floor; median power is just 0.032% of Pmax. The broad empirical residual bound strongly limits power. Zero observed exceedances in this small test do not certify a tail probability. This is physical static validation, not physical dynamic E7/E8. Quick RT also uses a coarser search and ray budget than full.

Source: fresh_rt_spatial/gamma_summary.csv · 6 drops, 2/2/2 split; 24 test sectors

## 23. Unknown arrivals expose the gap before usable information

Arrival times are drawn by the evaluator and hidden from the controller. All three comparison policies share arrival trajectories, sensing random numbers and RF costs. The prior-envelope policy protects the newcomer class from the outset. No-response ignores detection. Delayed-discovery starts protection only after detection has completed and its processing delay has elapsed. For an UL-active newcomer, exposure falls from 47.53% with no response to 24.45% with delayed discovery; detection probability is 67.5%. The permanently UL-silent newcomer is never detected and receives no discovery-based benefit. These are mechanism diagnostics using fixed policies, not an optimal random-arrival LP solution. Their TN deadline failures violate the main budget, so they cannot be presented as feasible service improvements.

Source: E8/random_arrival/summary.csv · fixed policies, 400 paired episodes per case

## 24. Repeated observations now genuinely test temporal correlation

The original optimized J policy listened at most once along each reachable trajectory, so making latent observation uniforms correlated did not actually test repeated observations. The new diagnostic fixes two listening occasions before observing any outcomes. Independent and correlated cases use paired physical trajectories and the same observation marginal probability. The first/second acceptance correlation changes from about minus 0.029, with a confidence interval spanning zero, to one. Foreground exposure is about 7.46% versus 7.38%; the paired difference confidence interval spans zero. Do not force a negative result into a robustness claim: the mechanism is now exercised, but these parameters do not show a significant loss. The fixed schedule is explicitly not feasible under the original TN constraints.

Source: E8/repeated_observation/summary.json · fixed two-listen policy, 400 paired episodes

## 25. The contribution is a causal connection from sensing to service

Close with three points. First, complete-channel uncertainty sets unify dominant-path and coherent-multipath protection, including unobserved background. Second, under explicit assumptions, beam elimination lets a finite causal policy combine sensing decisions, power, service and silence. Third, exact finite-model optimality and physical-system protection are different claims. We have verified numerical identities and policy consistency, exercised the previously missing causal mechanisms, and connected fresh physical ray tracing for spatial evaluation. The next research step is an independently calibrated physical dynamic model with repeated observations, user activity and timing, together with less conservative but valid observable-condition protection sets.

Source: Appendix D + current implementation scope

## 26. How listening quality enters the multipath residual

Derive the residual bound by writing the error as delta-A times c plus the missing tail. The spectral norm of delta-A is no larger than its Frobenius norm, which is bounded by the root-sum-square of per-path response errors. Multiplying by C and adding the tail gives a sufficient radius. This is a constructive sufficient bound, not a claim that every path can be resolved. Total channel power cannot in general bound path coefficient norm, because coherent cancellation can hide large coefficients. The illustrative age envelope can be replaced by independently calibrated tables, and its validity must cover the intended service interval.

Source: Appendix D.B · eq:joint_error_envelope · eq:joint_multipath_radius

## 27. Silence is feasible; positive deadline service may not be

For a fixed channel and catalog over remaining usable downlink time, completing d bits is equivalent to the amplitude threshold on the first line. The second pair gives a necessary screening bound using the smallest eigenvalue of the protected subspace Gram matrix and the residual radius. A rank-deficient span can leave a spatial null space, but nonzero residual still caps power. A full span can cap power even when residual is zero. The screening bound is not sufficient, because it ignores alignment of the desired TN channel with the protected subspace. Do not use a current-catalog bound to discard a full sensing plan whose future accepted observation can change that catalog.

Source: Appendix D.D · eq:joint_service_feasibility · eq:joint_power_cap

## 28. A network extension needs local budgets and an external reserve

For independent BS data streams, symbol-averaged interference powers add. Assign local Gamma budgets whose sum is within the network budget and local risk allowances whose sum is within the per-receiver risk target. If every local contribution is below its budget, the total is below the summed budget, so network exceedance requires at least one local exceedance. A union bound works under common receiver-activity conditioning without independent certificate failures. Uncontrolled TN uplink or other sources require a valid reserved external contribution, including failure risk if the reserve is uncertain. Endogenous inter-sector TN SINR coupling is outside the exact separable theorem. A negative remaining budget cannot be fixed by silencing these base stations alone.

Source: Appendix D.E · eq:joint_network_allocation · eq:joint_external_budget

## 29. A target, an estimate and a confidence bound are different

The E4 interval is Clopper–Pearson for independent scene failure indicators. There are 128 independent scenes even though the stratified CSV has many more rows. The E7 interval resamples complete episodes because time samples within an episode are dependent. These are different experiments: E4’s scene marginal coverage does not certify E7’s conditional state risks. A measured value just above a budget is not by itself evidence of a solver or theorem failure when the confidence interval includes the target. Conversely, a mean below budget is not a high-confidence certification if its upper confidence bound exceeds that budget. Formal simultaneous or one-sided risk certification needs an explicitly designed statistical procedure.

Source: E4/scene_coverage.csv · E7/strategy_summary.csv

## 30. Notation and source map

This is a reference slide for questions. Sources are local and pinned to the current paper and the September 21 23:54:48 experiment archive. All chart values are replotted from saved CSV or JSON without re-running or modifying the experiments. Mathematical illustrations are explicitly marked schematic. The deck distinguishes current implementation details from the broader Appendix D theorem. Speaker notes cite the relevant local source and explain qualifications that would overload the visible slides. The source manifest records SHA256 hashes so the deck can be reproduced and checked after future changes.

Source: Local paper and pinned result archive; full paths and hashes in sources.json

