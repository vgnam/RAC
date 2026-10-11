# Tremble-Robust IQL (TR-IQL)

TR-IQL is a fully decentralised, independent Q-learning method for cooperative
games in which agents **choose** an action but the environment sometimes
**executes** a different one (an *execution tremble*). Agents never observe
teammates' actions or intentions. In this setting, an action that is optimal
without tremble can become too risky with it. TR-IQL learns the best joint
intention *under* tremble. Plain IQL falls into relative overgeneralisation,
and optimistic learners (hysteretic Q, distributed Q, BQL) ignore the risk that
tremble adds.

## Setting

- Each agent `i` picks an **intended** action `a_i`. Independently, with probability
  `eps`, it **slips** to one of the other `A-1` actions chosen uniformly at random.
  `K_eps(executed | intended)` is defined in `src/tremble/games.py:tremble_kernel`.
- The common payoff of an intended joint action is the tremble-averaged payoff
  `r_eps(a) = E_{a' ~ K_eps(.|a)} [r(a')]` (`noisy_payoff`).
- The goal is a greedy joint intention that attains `max_a r_eps(a)`. This optimum
  can differ from the noise-free optimum. In the climbing game, the optimum is
  `(0,0)` up to `eps = 0.15` and becomes `(2,2)` at `eps = 0.2`, because `(0,0)`
  sits next to `-30` payoffs.

## Method

An agent faces two sources of noise from its teammates, and TR-IQL treats them
differently:

| Source | Nature | Operator |
|---|---|---|
| Teammates' **intentions** (exploration, learning) | transient, cooperative | **max** (optimism) |
| Teammates' **execution** (tremble) | persists at deployment, a real risk | **expectation** (nominal) or **worst case over an uncertainty set** (robust) |

```
Q_i(a_i) = max_{a_-i}  E_{K_eps}[ r | a_i, a_-i ]                  (nominal)
Q_i(a_i) = max_{a_-i}  min_{K in U}  E_K[ r | a_i, a_-i ]          (robust, see below)
```

Plain IQL averages over both sources, so it suffers from relative
overgeneralisation. Optimistic learners (distributed Q, hysteretic Q, BQL) take
the max over both, so they ignore the risk tremble adds. TR-IQL is optimistic
about teammates and nominal or pessimistic about nature. Each agent's greedy
action attains the tremble-robust optimum, provided the maximising joint
intention is unique.

Agents cannot see teammates' intentions, so TR-IQL uses **commitment blocks**:

1. Training is split into blocks of `block_len` episodes. All agents know where
   the block boundaries are (a shared clock), but they share no other information.
2. At the start of a block, each agent commits to one intended action for the
   whole block: a random action if it explores, its greedy action otherwise.
   Teammates' intentions are therefore fixed within the block, and only execution
   trembles.
3. The block's mean reward is an unbiased estimate of `r_eps(a_i, a_-i)` for the
   (unknown) fixed `a_-i`.
4. At the end of the block, `Q_i(a_i)` is updated towards the best block seen so
   far for that intention. Across blocks this approaches the max over teammates'
   intentions.

Q is always indexed by the **intended** action, never the executed one.
(Indexing by the executed action is kept as the ablation `ours, exec-keyed Q`.)

### Estimators

Maximising noisy block means is biased upwards, and the bias is largest for
high-variance (risky) intentions, which are exactly the ones tremble penalises.
Three estimators are implemented:

| Estimator | Where | Rule |
|---|---|---|
| `double` | tabular default, deep option | Select the best block using the mean of the block's first half; write the mean of its second half into Q. |
| `pooled` | deep default, tabular `TR-IQL pooled` | Keep the best *group* of blocks as (mean, variance). A new block replaces the group if it is higher by more than `z` standard errors, is merged into it (inverse-variance weighting) if not significantly different, and is ignored if lower. Q can move down, and recurring teammate intentions are pooled without being observed. |
| `lcb` | tabular `TR-IQL lcb` | Score each block by `mean - kappa * std_err` and keep the max. |

### Robustness to the tremble rate (TR-IQL-DRO)

The deployment tremble rate is rarely known exactly. If the true rate is
`eps' > eps`, the nominal target can pick a joint that is no longer optimal. In
climbing trained at `eps = 0.1`, TR-IQL keeps `(0,0)`, which is wrong at
`eps' = 0.2`.

TR-IQL-DRO scores each block by

```
mean - kappa * std
```

This is the value of a **chi-square-ball distributionally robust** problem. By
the robustness <-> regularisation equivalence (Derman, Geist & Mannor, 2021), it
equals **variance regularisation**. The radius `kappa` is not a tuned
hyperparameter. It is derived from the range of tremble rates to be covered,
`[eps, eps_max]` (`src/tremble/games.py:chi2_kappa`):

- Per agent, `chi2(K_eps' || K_eps) = (1-eps')^2/(1-eps) + eps'^2/eps - 1`. This
  holds for every intention (`A` cancels out) and increases in `eps' > eps`.
- Agents tremble independently, so `1 + chi2` multiplies:
  `kappa = sqrt((1 + c)^n - 1)`.
- Cauchy-Schwarz then gives `r_eps'(a) >= r_eps(a) - kappa * std_eps(a)` for
  every intended joint `a` and every `eps' in [eps, eps_max]`.

The robust block score is therefore a certified lower bound on the
intention's value under any tremble rate up to `eps_max`.

Two related estimators are distinct and should not be confused:

- Contamination `(1-rho)*mean + rho*min` is degenerate under uniform tremble.
  Every intention can reach the worst outcome, so `min` is the same for all of
  them, and in exact computation it never changes a decision.
- The `lcb` estimator penalises the *standard error*. That is a statistical
  bound which vanishes as `block_len` grows. The DRO score penalises the reward
  *std*, which does not vanish.

Robustness can only cover risk that is observed during training. Training at
`eps = 0` gives `kappa = inf`. A shift towards *less* tremble (`eps' < eps`) is
outside the uncertainty set.

## Code map

| File | Contents |
|---|---|
| `src/tremble/games.py` | Matrix games (climbing, penalty, `matrix_game_3`, stochastic climbing variants, asymmetric 2x2, n-agent coordination), the tremble kernel, `r_eps`, and the exact individual optimum. |
| `src/tremble/learners.py` | Tabular learners: `tremble_robust_iql`, `tremble_robust_iql_stats` (pooled / lcb), `iql` (IQL, or hysteretic Q when `beta` is set), `optimistic_sample_max`, `hurwicz_per_agent`, and `exact_fixed_points`. |
| `run_tremble_matrix.py` | Tabular benchmark: trains every method at each `eps` and reports how often the greedy joint reaches `max r_eps`. |
| `run_tremble_shift.py` | Tremble-rate shift: trains at `eps` and evaluates at `eps` and `eps_max`, plus the max regret over `[eps, eps_max]`, for TR-IQL, TR-IQL-DRO, IQL and hysteretic Q. |
| `src/learners/tremble_iql_learner.py` | Deep TR-IQL (PyMARL). Q is fitted by full-batch regression at each block end, with the `pooled` or `double` estimator. |
| `src/controllers/tremble_block_controller.py` | `TrembleBlockMAC`: per-block commitment exploration. `explore_policy: action` is for matrix games; `random_linear` commits to a random deterministic linear policy, for multi-step games. |
| `src/envs/tremble_wrapper.py` | Applies execution tremble to any env (enabled by `env_args.tremble_eps`). For matrix games it also logs the exact `intended_value` and `intended_optimal`. |
| `src/config/algs/tr_iql.yaml`, `tr_iql_lbf.yaml` | Deep presets for matrix games and Level-Based Foraging. |

## Usage

Tabular benchmark (fast, exact evaluation):

```bash
# All methods, all games, default budget (250+ blocks x 200 episodes)
py -3.11 run_tremble_matrix.py --seeds 10

# TR-IQL vs IQL vs hysteretic Q with longer blocks and an equal episode budget
py -3.11 run_tremble_matrix.py --seeds 3 --block-len 1000 --max-td-episodes 10000000 \
  --methods "TR-IQL (ours)" IQL hysteretic
```

IQL and hysteretic Q get `n_blocks x block_len` episodes, capped by
`--max-td-episodes`. Raise the cap so every method gets the same budget. The
results are written to `results/tremble_matrix/summary.json`.

Deep version (PyMARL):

```bash
python src/main.py --config tr_iql --env-config matrix_game_climbing with env_args.tremble_eps=0.1
python src/main.py --config tr_iql_lbf --env-config lbf with env_args.tremble_eps=0.1
```

Key hyperparameters: `block_len` (episodes per block), `estimator`
(`pooled` | `double`), `pool_z`, `fit_iters`, `fit_lr`, and `init_log_var`.

## Results (tabular)

Setup: 3 seeds, `block_len = 1000`, and the same episode budget for every
method. Each cell is the % of seeds whose greedy joint intention attains
`max r_eps`. `*` marks the cases where tremble changes the optimal joint.

| Game | eps | Optimal joint | TR-IQL | IQL | Hysteretic (beta=0.01) |
|---|---|---|---|---|---|
| climbing | 0 / .05 / .1 / .15 | (0,0) | 100 | 0 | 100 |
| climbing | 0.2* | (2,2) | **100** | 100 | 0 |
| penalty | 0 | (0,0) or (2,2) | 33 | 100 | 100 |
| penalty | 0.1 / 0.2 | (0,0) or (2,2) | 100 | 33 / 0 | 100 |
| matrix_game_3 | 0 / 0.1 | (0,0) | 100 / 67 | 0 | 100 |
| matrix_game_3 | 0.2* | (1,1) or (2,2) | **100** | 67 | 0 |
| climbing partial-stoch | 0 / 0.1 | (0,0) | 100 | 0 | 100 |
| climbing partial-stoch | 0.2* | (2,2) | 67 | 100 | 0 |
| climbing full-stoch | 0 / 0.1 | (0,0) | **100** | 0 | 0 |
| climbing full-stoch | 0.2* | (2,2) | 67 | 100 | 33 |
| asymmetric 2x2 | 0.25 | (0,0) | 100 | 33 | 100 |
| n-agent (n=2,3) | 0.05 / 0.1 | all 0 | 100 | 33-100 | 100 |
| n-agent (n=4) | 0.05 | all 0 | 100 | 33 | 100 |
| n-agent (n=4) | 0.1* | all 1 | 67 | 100 | 0 |

Averaged over all 24 (game, eps) cells: **TR-IQL 92%**, hysteretic 72%, IQL 42%.

- IQL fails wherever the optimum is a risky joint (relative overgeneralisation).
  It wins the `*` cells only because the "safe" joint is optimal there.
- Hysteretic Q fails every `*` cell: its optimism keeps the noise-free optimum
  after tremble has made that joint too risky. It also fails in full-stochastic
  climbing, where it trusts the occasional 14 payoff of `(1,1)`. More data does
  not fix either failure.
- TR-IQL is the only method that tracks the optimum in both regimes.

On climbing, the remaining TR-IQL misses were estimation noise. With
`block_len = 200`, 1 of 3 seeds failed at `eps >= 0.1`; with
`block_len >= 1000`, every `eps` reaches 100%.

## Results: tremble-rate shift

Command: `py -3.11 run_tremble_shift.py --seeds 3 --block-len 1000`. Setup: 7
games (climbing, penalty, matrix_game_3, both stochastic climbing variants,
n-agent with n=3 and n=4) x 5 (eps -> eps_max) pairs, giving 35 cells. All
methods use the same episode budget.

| Method | optimal at train `eps` | optimal at `eps_max` | gap to max-min oracle | cells at oracle | max regret over `[eps, eps_max]` |
|---|---|---|---|---|---|
| TR-IQL | **91%** | 52% | 0.49 | 14/35 | **0.51** |
| TR-IQL-DRO | 45% | **90%** | **0.04** | **29/35** | 0.95 |
| IQL | 30% | 55% | 0.70 | 13/35 | 1.86 |
| Hysteretic | 74% | 34% | 0.94 | 12/35 | 1.01 |

The max-min oracle is `max_a min_{eps' in [eps, eps_max]} r_eps'(a)`, the
standard robust-MDP objective.

- **TR-IQL-DRO solves the max-min problem it is built for.** It is within 0.04 of
  the oracle on average, at no extra sample cost and with no tuned
  hyperparameter, because `kappa` comes from `(eps, eps_max)`.
- **Robustness has a price.** At the training rate itself, DRO is optimal only 45%
  of the time, because it switches early to the safe joint.
- **Max-min value and min-max regret disagree.** Under the regret criterion,
  nominal TR-IQL is better (0.51 vs 0.95). Keeping the high-payoff joint loses
  little at `eps_max`, while switching early loses a lot at `eps`. State which
  criterion is meant: DRO is the right choice when the worst-case *value* must
  be guaranteed, as in safety-critical deployment.
- Hysteretic Q is worst under the max-min criterion, because it ignores the
  tremble risk entirely.

**Deep version (PyMARL).** Climbing, trained at `eps = 0.1`, evaluated at
`eps = 0.2`. Setup: `t_max = 50k`, `block_len = 200`, 3 seeds. `kappa = 0.48`
is derived from `(0.1, 0.2)`.

```bash
python src/main.py --config=tr_iql_dro --env-config=matrix_game_climbing \
  with env_args.tremble_eps=0.1 env_args.tremble_eps_eval=0.2 dro_eps_max=0.2
```

| Method | optimal at 0.1 | value at 0.1 | optimal at 0.2 | value at 0.2 (= worst case over [0.1, 0.2]) |
|---|---|---|---|---|
| `tr_iql` | 3/3 | **6.25** | 0/3 | 2.42 |
| `tr_iql_dro` | 0/3 | 4.21 | **3/3** | **3.26** (max-min oracle) |

Every seed converges to the joint predicted by the exact fixed point:
`(0,0)` for the nominal method and `(2,2)` for DRO.

## Limitations

- **Shared clock and commitment.** Agents must agree on where blocks start and
  must keep one intention for a whole block. IQL and hysteretic Q learn online
  and need neither.
- **Sample cost.** The number of blocks grows as `A^n`
  (`blocks_per_joint * A^n`), and each block needs enough episodes to average
  out tremble. Climbing (3x3) needs about 250k episodes for 100% at `eps = 0.2`.
- **Near-ties at high `eps`.** When the gap between the best and second-best
  intention is small (0.6-1.3 in the failing cells above) and rewards have high
  variance, a seed can still pick the wrong intention. The exact fixed point is
  correct in all of these cells, so longer blocks or more blocks fix this.
- **Equilibrium selection.** If several joint intentions are equally optimal
  (penalty at `eps = 0`), `Q_i` ties exactly and independent tie-breaking can
  miscoordinate. A tie-break order known to every agent (for example, the lowest
  index) avoids this.
- The results above use 3 seeds and an untuned hysteretic `beta`. Use 10+ seeds
  and a `beta` sweep before drawing final conclusions.
