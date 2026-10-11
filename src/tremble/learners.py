"""Tabular independent learners for single-state common-payoff matrix games.

Every learner returns Q of shape (n_agents, n_actions) indexed by the agent's
*intended* action, plus the greedy joint intention. Agents never observe
teammates' actions.

Tremble-Robust IQL ("Cach A" of the proposal): optimistic (max) over teammates'
intentions, nominal (mean) or robust (mean - kappa * std) over execution tremble.
Training is split into blocks that all agents
know (a shared clock, no shared information). In each block every agent commits
to one intended action, so teammates' intentions are fixed and only execution
trembles. The block mean therefore estimates r_eps(a_i, a_-i); taking the max
over blocks keyed by a_i gives max_{a_-i} r_eps(a_i, a_-i), the fixed point of
T_eps. A double estimator (select on one half of the block, evaluate on the
other) limits the upward bias of maximising over noisy means.
"""
import numpy as np

from .games import individual_optimum, noisy_payoff, tremble, tremble_kernel


def greedy_joint(q, rng):
    """Per-agent argmax with uniform tie-breaking among exact ties."""
    joint = []
    for row in q:
        best = np.flatnonzero(row == row.max())
        joint.append(int(rng.choice(best)))
    return tuple(joint)


def tremble_robust_iql(game, eps, rng, n_blocks, block_len, executed_keys=False,
                       estimator="double", kappa=0.0):
    """Proposed method. ``eps=0`` is BQL; ``executed_keys`` is the ablation that
    keys Q by the executed action and averages over the own tremble afterwards.
    ``kappa > 0`` scores each block by mean - kappa * std (chi^2-ball DRO, see
    ``games.chi2_kappa``), which lower-bounds its value under any tremble rate up to the
    eps_max that kappa was derived from."""
    score = (lambda r: r.mean() - kappa * r.std()) if kappa else (lambda r: r.mean())
    n, A = game.n_agents, game.n_actions
    sel = np.full((n, A), -np.inf)
    q = np.full((n, A), -np.inf)
    half = block_len // 2
    for _ in range(n_blocks):
        intents = rng.integers(A, size=n)
        executed = tremble(np.tile(intents, (block_len, 1)), eps, A, rng)
        rewards = game.sample_rewards(executed, rng)
        for i in range(n):
            if executed_keys:
                keys = np.unique(executed[:, i])
            else:
                keys = [intents[i]]
            for a in keys:
                if executed_keys:
                    mask = executed[:, i] == a
                    r_a, r_b = rewards[:half][mask[:half]], rewards[half:][mask[half:]]
                    if len(r_a) == 0 or len(r_b) == 0:
                        continue
                else:
                    r_a, r_b = rewards[:half], rewards[half:]
                if estimator == "double":
                    m_sel, m_val = score(r_a), score(r_b)
                else:
                    m_sel = m_val = score(np.concatenate([r_a, r_b]))
                if m_sel > sel[i, a]:
                    sel[i, a] = m_sel
                    q[i, a] = m_val
    q[np.isinf(q)] = q[np.isfinite(q)].min()  # never-tried actions rank last
    if executed_keys:
        q = q @ tremble_kernel(A, eps).T  # Q(intent) = sum_a K(a | intent) Q_exec(a)
    return q


def tremble_robust_iql_stats(game, eps, rng, n_blocks, block_len, mode="pooled", kappa=1.0, z=2.0,
                             rank_kappa=0.0):
    """Variance-aware variants of the best-block estimator.

    ``lcb``: score each block by mean - kappa * standard error and keep the max, which
    penalises high-variance (risky) intentions whose noisy maxima are inflated most.
    This is a statistical bound that vanishes as block_len grows; robustness to the
    tremble rate itself uses the reward std instead (``tremble_robust_iql(kappa=...)``).
    ``pooled``: keep, per own intention, the best group as (mean, var, n). A new block
    replaces it when significantly higher (z-test), is merged into it when not
    significantly different, and is ignored when significantly lower. ``rank_kappa``
    ranks the pooled groups by their lower confidence bound instead of the mean. Recurring
    teammate intentions are thus pooled without being observed, the estimate of the
    best group sharpens over time, and Q can move down as well as up.
    """
    n, A = game.n_agents, game.n_actions
    mean = np.full((n, A), -np.inf)
    var = np.zeros((n, A))  # variance of the stored estimate of the mean
    count = np.zeros((n, A))
    for _ in range(n_blocks):
        intents = rng.integers(A, size=n)
        executed = tremble(np.tile(intents, (block_len, 1)), eps, A, rng)
        rewards = game.sample_rewards(executed, rng)
        m, v = rewards.mean(), rewards.var(ddof=1) / block_len
        for i in range(n):
            a = intents[i]
            if mode == "lcb":
                mean[i, a] = max(mean[i, a], m - kappa * np.sqrt(v))
                continue
            if count[i, a] == 0:
                mean[i, a], var[i, a], count[i, a] = m, v, block_len
                continue
            score = (m - mean[i, a]) / np.sqrt(v + var[i, a] + 1e-12)
            if score > z:
                mean[i, a], var[i, a], count[i, a] = m, v, block_len
            elif score >= -z:
                total = count[i, a] + block_len
                mean[i, a] = (count[i, a] * mean[i, a] + block_len * m) / total
                var[i, a] = (count[i, a] ** 2 * var[i, a] + block_len ** 2 * v) / total ** 2
                count[i, a] = total
    mean = mean - rank_kappa * np.sqrt(var)
    mean[np.isinf(mean)] = mean[np.isfinite(mean)].min()
    return mean


def optimistic_sample_max(game, eps, rng, n_blocks, block_len):
    """Distributed Q-learning / deterministic BQL: Q_i(a_i) = max of observed rewards.
    Mixes tremble with intention, so it over-estimates in stochastic G_eps."""
    n, A = game.n_agents, game.n_actions
    q = np.full((n, A), -np.inf)
    for _ in range(n_blocks):
        intents = rng.integers(A, size=n)
        executed = tremble(np.tile(intents, (block_len, 1)), eps, A, rng)
        r_max = game.sample_rewards(executed, rng).max()
        for i in range(n):
            q[i, intents[i]] = max(q[i, intents[i]], r_max)
    return np.where(np.isinf(q), q[np.isfinite(q)].min(), q)


def hurwicz_per_agent(game, eta, rng, n_blocks, block_len):
    """Original proposal: eta * max + (1 - eta) * min over samples, per agent, in the
    noise-free game (deterministic rewards, so sample max/min are exact envelopes)."""
    n, A = game.n_agents, game.n_actions
    hi = np.full((n, A), -np.inf)
    lo = np.full((n, A), np.inf)
    for _ in range(n_blocks):
        intents = rng.integers(A, size=n)
        r = game.sample_rewards(np.tile(intents, (block_len, 1)), rng)
        for i in range(n):
            hi[i, intents[i]] = max(hi[i, intents[i]], r.max())
            lo[i, intents[i]] = min(lo[i, intents[i]], r.min())
    return eta * hi + (1 - eta) * lo


def iql(game, eps, rng, n_episodes, n_runs=1, lr=0.1, explore_start=1.0, explore_end=0.05,
        anneal_frac=0.5, beta=None):
    """Tabular independent Q-learning with epsilon-greedy over intentions, for
    ``n_runs`` independent runs at once -> (n_runs, n, A).
    ``beta`` turns it into hysteretic Q-learning (lr for positive TD, beta for negative)."""
    n, A = game.n_agents, game.n_actions
    q = np.zeros((n_runs, n, A))
    runs, agents = np.arange(n_runs)[:, None], np.arange(n)[None]
    anneal = max(1, int(anneal_frac * n_episodes))
    for t in range(n_episodes):
        e = explore_start + (explore_end - explore_start) * min(1.0, t / anneal)
        greedy = q.argmax(axis=2)
        intents = np.where(rng.random((n_runs, n)) < e, rng.integers(A, size=(n_runs, n)), greedy)
        r = game.sample_rewards(tremble(intents, eps, A, rng), rng)[:, None]
        td = r - q[runs, agents, intents]
        step = lr if beta is None else np.where(td > 0, lr, beta)
        q[runs, agents, intents] += step * td
    return q


def exact_fixed_points(game, eps):
    """Analytic fixed points of each operator (what the learners estimate)."""
    n, A = game.n_agents, game.n_actions
    r_eps, r_0 = noisy_payoff(game, eps), game.mean_payoff
    k = tremble_kernel(A, eps)
    exec_keyed = []
    for i in range(n):
        # Teammates tremble, agent i's executed action is known: max, then own tremble.
        t = r_0
        for j in range(n):
            if j != i:
                t = np.moveaxis(np.tensordot(k, t, axes=([1], [j])), 0, j)
        q_exec = individual_optimum(t)[i]
        exec_keyed.append(k @ q_exec)
    eta = (1 - eps) ** (n - 1)
    hi, lo = individual_optimum(r_0), individual_optimum(-r_0)
    return {
        "TR-IQL (ours)": individual_optimum(r_eps),
        "BQL (eps=0 train)": individual_optimum(r_0),
        "ours, exec-keyed Q": np.stack(exec_keyed),
        "Hurwicz per agent (eta=(1-eps)^(n-1))": eta * hi - (1 - eta) * lo,
    }
