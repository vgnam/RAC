# Method (Detailed): Belief-RAC — Belief-Conditional Return-Aware Context for Fully Decentralized Cooperative MARL

> Paper-ready Method section, written directly from the reference implementation
> (`src/modules/context_belief.py`, `src/controllers/dual_ree_controller.py`,
> `src/learners/dual_episode_ree_q_learner.py`, `src/modules/agents/new_ree_agent.py`,
> `src/modules/agents/mate_entity_gru_agent.py` (MATE twin backbone),
> `src/config/algs/belief_dual_iql_ree.yaml`).
> Supersedes / extends `docs/methods_belief_rac.md`. Two implementation-accurate
> corrections relative to that draft: (i) the shift signal is the base-2
> Jensen–Shannon divergence, already normalized to $[0,1]$ (no extra factor of 2);
> (ii) the dynamics-reconstruction term is weighted by the *hard straight-through
> Gumbel context assignment*, not by the soft posterior (the soft posterior is
> used only by the two KL regularizers).

---

## 1 Notation and problem setting

We consider a fully cooperative multi-agent task modeled as a Dec-POMDP
$\langle \mathcal{N}, \mathcal{S}, \{\mathcal{O}^i\}, \mathcal{A}, p, r, \gamma \rangle$
with $n$ agents. At time $t$, agent $i$ receives a local observation
$o^i_t \in \mathcal{O}^i$, takes an action $a^i_t \in \mathcal{A}$, and the team
receives a **shared** scalar reward $r_t$. Agent $i$ may only use its
action–observation history

$$
\tau^i_t \;=\; \big\{(o^i_1, a^i_1, r_1), \dots, (o^i_{t-1}, a^i_{t-1}, r_{t-1})\big\}
$$

(note $\tau^i_t$ excludes the pair at time $t$). During training we exploit
centralized data (the full episode, including the realized episodic return
$R = \sum_t r_t$), but at **execution** each agent acts with its own information
only: its current observation, its own previous action, and the previously
shared team reward. No mixer, no communication, no central critic is used at
any point (the learner is fully decentralized, `mixer: None`).

We posit that the *mode of cooperative play* being experienced by the team —
e.g., which joint payoff level the team is currently achieving — behaves as a
latent, slowly switching **context** $z \in \{1, \dots, K\}$ ($K$ =
`slot_number`, default $K=4$). RAC (Sec. 2) ties this latent to the realized
episodic return; Belief-RAC (Sec. 3) infers it causally, before acting.

---

## 2 Preliminaries: RAC (Return-Aware Context)

RAC (implemented as `dual_iql_ree`) trains, per agent $i$ and with parameters
shared across agents, two value heads over a common recurrent encoder:

- a **behavior Q-function** $Q^i(\tau^i_t, a)$ — the light, deployed policy; and
- a **context-conditioned twin Q-function** $Q^i_{\mathrm{twin}}(\tau^i_t, z, a)$
  that additionally conditions on a categorical context $z$.

**Return-aware context (training only).** Because the episodic return $R$ is
revealed in the replay data, RAC maps $R$ onto one of $K$ ordered slots by
quantizing against environment-specific thresholds (`possible_returns`):

- Matrix game: $K = 4$ with thresholds $\{-12, 0, 6, 8\}$ (exact match);
- Continuous-return domains (Predator–Prey, SMAC, MATE): $K+1$ evenly spaced
  thresholds over the return range, e.g. $[0, R_{\max}]$ with
  $R_{\max}{=}1000$ for MATE, and $z = \mathrm{bin}(R)$.

The twin is trained by TD learning conditioned on the one-hot $z=\mathrm{bin}(R)$.

**Hyper-network context conditioning.** $Q^i_{\mathrm{twin}}$ factorizes over
contexts without re-encoding the history: a hyper-network maps the one-hot
context to the weights and bias of the final linear layer,

$$
\big(W(z),\, b(z)\big) = \mathrm{hyper}(z), \qquad
Q^i_{\mathrm{twin}}(\tau^i, z, a) = \big(h_{\tau}\, W(z)\big)_a + b(z)_a ,
$$

where $h_\tau \in \mathbb{R}^{H}$ is the backbone's representation of the local
history (a recurrent encoder in the default agent; a typed entity-attention
encoder in the MATE variant). Evaluating at the $K$ one-hot vertices of the simplex yields
$Q$ for *all* contexts in one forward pass (`_all_context_q`).

**Teacher distillation.** A detached teacher built from
$\max_z Q^i_{\mathrm{twin}}$ distills into the behavior head through a KL term
(weight `kl_weight = 10`), so the deployed policy imitates the best
context-conditioned action distribution.

**The limitation.** At execution the true return-context is unavailable (the
episode has not ended), so RAC must act with the context-agnostic optimistic
rule $Q_{\mathrm{dec}}(a) = \max_z Q^i_{\mathrm{twin}}(\tau^i, z, a)$, applied
uniformly at *every* step regardless of how informative the local history is.
Belief-RAC removes exactly this blind spot.

---

## 3 Belief-RAC

Belief-RAC (`belief_dual_iql_ree`) keeps RAC's twin, hyper-network and
distillation machinery, and replaces the post-hoc return context with an
**online, causal categorical belief filter**. Each agent maintains a posterior

$$
b^i_t(z) \;\triangleq\; \mathbb{P}\!\big(z^i_t = z \,\big|\, o^i_t,\ a^i_{t-1},\ r_{t-1}\big),
$$

over contexts, using only information available *before* acting at time $t$.
The belief is produced by a standalone module (`ContextBeliefModel`) that sits
**in parallel with** the value backbone — it never modifies the backbone's
input or hidden state — which makes the method backbone-agnostic (MLP, recurrent, or entity-attention
policies all work unchanged).

### 3.1 Causal belief filter

**Filter input.** At time $t$ the per-agent filter input is the concatenation

$$
\xi^i_t \;=\; \big[\, o^i_t,\; \mathbf{1}(a^i_{t-1}),\; r_{t-1},\; \mathrm{id}_i \,\big]
\in \mathbb{R}^{|\mathcal{O}^i| + |\mathcal{A}| + 1 + n},
$$

with $\mathbf{1}(a^i_{t-1})$ the one-hot previous *own* action, $r_{t-1}$ the
scalar previous team reward (broadcast to all agents), and $\mathrm{id}_i$ the
one-hot agent identity (`belief_include_agent_id=true`). At $t=0$ the previous
action and reward are zero.

**Encoding.** A two-layer evidence encoder produces per-agent hidden states

$$
e^i_t = \mathrm{ReLU}\!\big(W_e\, \xi^i_t + c_e\big) \in \mathbb{R}^{H_b},
\qquad
h^i_t = f_{b}\big(e^i_t,\, h^i_{t-1}\big) \in \mathbb{R}^{H_b},
$$

where $f_b$ is a generic recurrent state update — the method only requires
that it map $(e^i_t, h^i_{t-1}) \mapsto h^i_t$ and is otherwise agnostic to its
realization — with $H_b = 64$ (`belief_hidden_dim`). Evidence logits are read out by a linear
head with temperature $\varsigma$ (`belief_temperature`):

$$
\ell^i_t(z) = \frac{W_{eh} h^i_t + c_{eh}}{\varsigma} .
$$

The evidence head is **zero-initialized**, so at the start of training (and at
the start of every episode, $h^i_0 = 0$) the filter emits exactly the uniform
belief rather than arbitrary confidence.

**Bayes-like update with a drifting prior.** The previous belief is turned
into a *predictive prior* that admits regime switches via a self-transition
probability $\rho$ (`belief_transition_stay`):

$$
\bar b_t(z) \;=\; \rho\, b_{t-1}(z) \;+\; (1-\rho)\,\frac{1}{K},
$$

i.e. with probability $1-\rho$ the agent allows itself to re-consider any
context, preventing lock-in to one slot. The posterior combines evidence and
prior in log space:

$$
b_t \;=\; \mathrm{softmax}_z\!\Big( \ell_t(z) + \beta \log \bar b_t(z) \Big),
$$

with prior strength $\beta$ (`belief_prior_strength`). As $\beta \to \infty$
this approaches exact posterior updates under a Markov-chain context model;
$\beta = 1$ keeps the update robust to poorly calibrated evidence early in
training.

**Regime-shift signal.** We define a per-step shift score as the
Jensen–Shannon divergence between posterior and predictive prior, computed
with base-2 logarithms so that

$$
s_t \;=\; \mathrm{JSD}_2\!\big(b_t \,\Vert\, \bar b_t\big) \;\in\; [0, 1],
\qquad
\mathrm{JSD}_2(p \Vert q) = \tfrac12 \mathrm{KL}_2(p \Vert m) + \tfrac12 \mathrm{KL}_2(q \Vert m),
\;\; m = \tfrac12(p+q).
$$

$s_t$ is large exactly when the local transition is surprising relative to the
agent's current regime hypothesis — the signature of a context switch.

### 3.2 Self-supervised context learning via mixture dynamics

No ground-truth context label exists at execution time, so the filter is
trained **self-supervised**: each latent context $z$ is interpreted as a
distinct local transition regime, and contexts are scored by how well they
explain the observed transition.

**Projected-delta targets.** Predicting raw observation deltas is wasteful in
high dimensions, so the target is the observation delta pushed through a fixed
random projection $P \in \mathbb{R}^{d \times |\mathcal{O}^i|}$
($d = \min(|\mathcal{O}^i|, 16)$ = `belief_projection_dim`; rows $\ell_2$-normalized;
generated once from seed $\texttt{seed} + 7919$):

$$
\Delta\tilde o_t \;=\; P\,(o^i_{t+1} - o^i_t) \in \mathbb{R}^{d}.
$$

**Mixture decoder.** A decoder $g_\phi$ takes only *pre-decision* information
(current observation, taken action, agent id) and emits, for every context
simultaneously, a predicted projected delta and reward:

$$
g_\phi\big(o^i_t, a^i_t, \mathrm{id}_i\big) \;=\;
\Big\{ \big(\widehat{\Delta o}[z], \widehat r[z]\big) \Big\}_{z=1}^{K},
\qquad
\widehat{\Delta o}[z] \in \mathbb{R}^{d}.
$$

**Belief-weighted reconstruction with hard assignments.** Let
$z^*_t \sim \mathrm{GumbelSoftmax}(\ell_t + \beta\log\bar b_t,\ \tau_g)$ be the
straight-through hard sample of the posterior ($\tau_g$ = `belief_gumbel_tau`,
default $1.0$). The per-context dynamics error is

$$
\epsilon_t(z) \;=\; \mathrm{smooth}_{L1}\big(\widehat{\Delta o}_t[z],\, \Delta\tilde o_t\big)
\;+\; \lambda_r\, \mathrm{smooth}_{L1}\big(\widehat r_t[z],\, r_t\big),
$$

with $\lambda_r$ = `belief_reward_loss_weight` $= 1$. The reconstruction loss
weights each context's error by the hard assignment $z^*_t(z)$ (gradients
reach the evidence head through the straight-through estimator):

$$
\mathcal{L}_{\text{recon}}
\;=\; \frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}}
\sum_{i} \sum_{z} z^*_t(z)\, \epsilon_t(z),
$$

where $\mathcal{M}$ is the set of valid (non-padded, non-post-terminal)
transitions. Intuitively: whichever context the agent currently believes in
must *predict the transition*; if it consistently fails, the evidence logits
shift toward a context that does.

**Regularizers against collapse.** Two KL terms to the uniform distribution
$\mathcal{U}_K$ keep the belief informative:

$$
\mathcal{L}_{\text{belief}}
\;=\; \mathcal{L}_{\text{recon}}
\;+\; \lambda_{\text{kl}}\; \underbrace{\frac{1}{|\mathcal{M}|}\sum_{t\in\mathcal{M}} \mathrm{KL}\big(b_t \Vert \mathcal{U}_K\big)}_{\text{per-step prior KL}}
\;+\; \lambda_{\text{bal}}\; \underbrace{\mathrm{KL}\Big( \tfrac{1}{|\mathcal{M}|}\textstyle\sum_{t\in\mathcal{M}} b_t \,\Big\Vert\, \mathcal{U}_K\Big)}_{\text{marginal-balance KL}} ,
$$

with $\lambda_{\text{kl}} = 10^{-3}$ (`belief_kl_weight`) and
$\lambda_{\text{bal}} = 10^{-2}$ (`belief_balance_weight`). The prior KL keeps
individual posteriors from becoming spuriously overconfident; the marginal
balance term prevents *collapse* where all agents dump their mass on a single
context slot regardless of the regime.

### 3.3 Training the twin with inferred contexts

During replay training, the one-hot episodic return context of RAC is
**replaced** by the inferred context: the twin's TD update is conditioned on
the Gumbel sample $z^*_t$ from Sec. 3.2. (The realized-return slot is still
computed and logged — `return_slot_{k}_fraction` — for analysis, but plays no
role in the update.) This is the key training-time change: the twin now learns
$Q^i_{\mathrm{twin}}(\tau^i, z, a)$ for the contexts the *causal filter
actually infers*, so training and execution agree on the semantics of $z$.

### 3.4 Belief-merged optimistic decision rule

Given the posterior $b_t$ and shift $s_t$, the agent evaluates the twin at all
$K$ simplex vertices (`_all_context_q`, one vectorized pass) and forms

$$
Q_{\mathrm{dec}}(a)
\;=\;
(1-\alpha_t)\,
\underbrace{\sum_{z} b_t(z)\, Q^z_{\mathrm{twin}}(\tau^i, a)}_{\text{posterior mean}}
\;+\;
\alpha_t\, \underbrace{\max_{z} Q^z_{\mathrm{twin}}(\tau^i, a)}_{\text{optimistic max}},
$$

where the optimism coefficient interpolates between the two extremes according
to two complementary uncertainty signals:

$$
\alpha_t \;=\; \alpha_{\min} + (\alpha_{\max} - \alpha_{\min})\, u_t,
\qquad
u_t \;=\; 1 - \big(1 - \mathcal{H}_{\text{norm}}(b_t)\big)\big(1 - s_t\big),
\qquad
\mathcal{H}_{\text{norm}}(b_t) = \frac{\mathcal{H}(b_t)}{\log K}.
$$

$u_t$ is the *probabilistic union* of ambiguity (normalized posterior entropy)
and regime shift: either signal alone is sufficient to restore strong
optimism. Two properties are by construction:

1. **RAC is the degenerate case.** $\alpha_{\min} > 0$ is enforced
   (`0 < belief_optimism_min <= belief_optimism_max <= 1`); with a uniform
   belief and $\alpha_{\max} = 1$, $\alpha_t = 1$ and the rule *exactly
   recovers* RAC's $\max_z$. Belief-RAC therefore degrades gracefully to RAC
   whenever the filter knows nothing.
2. **Optimism is never fully removed.** Even a maximally confident posterior
   keeps weight $\alpha_{\min} = 0.25$ on the cooperative-optimistic max,
   preserving RAC's mechanism for escaping coordination equilibria.

The same rule is used to build the **KL teacher during training** (Sec. 3.5),
so the behavior policy is distilled toward the *belief-aware* decision
distribution, not merely the max-over-contexts teacher of RAC.

### 3.5 Total training objective

Per replay batch (32 episodes by default; episode-level replay, buffer of 5000
episodes), with double-Q one-step targets ($\gamma = 0.99$), the loss is

$$
\mathcal{L}
\;=\;
\underbrace{\mathcal{L}_{\text{TD}}\big(Q\big) + \mathcal{L}_{\text{TD}}\big(Q_{\text{twin}}\big)}_{\text{double-Q TD, behavior + twin}}
\;+\; \lambda_{\text{distill}}\,
\underbrace{\mathbb{E}_t\Big[ \mathrm{KL}\big(\pi_{\text{main}}(\cdot|\tau^i_t) \,\Vert\, \pi_{\text{teacher}}(\cdot|\tau^i_t, b_t, s_t)\big) \Big]}_{\text{belief-aware teacher distillation}}
\;+\; \lambda_{\text{belief}}\, \mathcal{L}_{\text{belief}},
$$

with $\lambda_{\text{distill}} = $ `kl_weight` $= 10$,
$\lambda_{\text{belief}} = $ `belief_dynamics_weight` $= 1$, and

$$
\pi_{\text{main}} = \mathrm{softmax}_a\, Q(\tau^i, a),
\qquad
\pi_{\text{teacher}} = \mathrm{softmax}_a\, Q_{\mathrm{dec}}(a)\ \ \text{(detached)} .
$$

Both TD heads use standard double-Q target selection (online network picks the
argmax action, target network evaluates it); the twin's target is conditioned
on $z^*_t$ detached. The teacher $Q_{\mathrm{dec}}$ is computed from the
*detached* counterfactual twin outputs aggregated with the replayed belief
trajectory, so the KL gradient flows only into the behavior head.
Optimization is RMSprop (lr $5{\times}10^{-4}$, $\alpha = 0.99$,
$\varepsilon = 10^{-5}$), gradient-norm clip $10$, target networks synced
every 200 episodes (or every `target_update_interval_steps` environment steps
in the MATE entity-attention setup).

**Efficient replay inference.** Over a replayed episode the evidence encoder
and recurrent core run as one fused batched sequence across all timesteps and
agents; only the Bayesian prior→posterior scan remains sequential over $T$ (it
is a cheap closed-form softmax recursion with no sampling). The same
fused-sequence evaluation applies to the value backbone (`forward_sequence`)
and to the all-context hyper-head (`counterfactual_sequence`).

### 3.6 Decentralized execution

Algorithm 1 summarizes deployment. Per step and per agent, the filter consumes
$(o^i_t, a^i_{t-1}, r_{t-1})$, emits $(b_t, s_t)$; the twin encoder runs once
and the hyper-head evaluates all $K$ contexts; $Q_{\mathrm{dec}}$ is formed by
Eq. (3.4) and actions chosen greedily ($\varepsilon$-greedy during training
exploration, $\varepsilon{:}\ 1.0 \to 0.05$ over 50k steps). The belief is a
sequential state — it is carried across steps through the drifting prior and
the filter's recurrent state — even when the value backbone is feed-forward.

```text
Algorithm 1  Belief-RAC execution (per agent i, fully decentralized)
 1: h^i_0 ← 0;  b^i_0 ← (1/K, …, 1/K);  a^i_0 ← 0;  r_0 ← 0
 2: for t = 1, 2, … do
 3:    receive o^i_t, r_{t-1}
 4:    ξ^i_t ← [o^i_t, 1(a^i_{t-1}), r_{t-1}, id_i]
 5:    h^i_t ← f_b(ReLU(W_e ξ^i_t), h^i_{t-1})                   ▹ evidence
 6:    b̄_t ← ρ b^i_{t-1} + (1−ρ)/K                              ▹ predictive prior
 7:    b^i_t ← softmax( (W_eh h^i_t)/ς + β log b̄_t )             ▹ posterior
 8:    s_t ← JSD_2(b^i_t ∥ b̄_t)                                  ▹ shift signal
 9:    h_τ ← value backbone(τ^i_t)                               ▹ one encode
10:    Q^z ← h_τ W(z) + b(z) for all z = 1…K                     ▹ one hyper pass
11:    u_t ← 1 − (1 − H(b^i_t)/log K)(1 − s_t)
12:    α_t ← α_min + (α_max − α_min) u_t
13:    Q_dec ← (1−α_t) Σ_z b^i_t(z) Q^z + α_t max_z Q^z
14:    a^i_t ~ ε-greedy(Q_dec)   (greedy at evaluation)
15: end for
```

**Complexity.** The filter adds, per agent per step, one
$(|\mathcal{O}|{+}|\mathcal{A}|{+}1{+}n) \to 64$ linear layer, one recurrent
state update ($64 \to 64$), and one $64 \to K$ head — negligible next to the
value backbone. All-context evaluation costs $O(K \cdot H \cdot A)$ via the
hyper-head, versus $K$ full backbone passes in naive counterfactual evaluation.

### 3.7 Backbone-agnostic instantiation

Because the belief module is parallel to the value network and the hyper-head
only needs the penultimate representation $h_\tau$, Belief-RAC instantiates
unchanged on:

| Backbone | Used for | Context conditioning |
|---|---|---|
| MLP + hyper head (`agent_recurrent=false`) | low-dim domains (matrix game) | `HyperREEAgent` hyper layer |
| Recurrent encoder + hyper head (default `hyper_ree_agent`) | Predator–Prey, SMAC | `HyperREEAgent` recurrent encoder |
| Typed entity-attention encoder + recurrent core (MATE variant) | MATE | `MateEntityGRUTwinAgent`: shared context-independent recurrent core, vectorized `einsum` all-context head |

In the entity-attention twin the recurrent core is declared
`context_independent_recurrent_core = True`: the context enters only through
the hyper-head, so the counterfactual all-context evaluation shares one
recurrent encoding across the $K$ contexts. The belief filter likewise runs
fused over replay sequences; only the causal posterior scan is sequential.

---

## 4 Diagnostics (logged quantities)

The implementation logs the internal signals that the method's claims rest
on, enabling direct empirical verification:

| Metric | Meaning |
|---|---|
| `context_uncertainty_mean` | mean normalized posterior entropy $\mathcal{H}_{\text{norm}}(b_t)$ |
| `context_shift_mean` / `_max` | mean/max of the shift signal $s_t$ (spikes should align with regime switches) |
| `context_optimism_weight` | mean realized $\alpha_t$ |
| `context_posterior_q_mean`, `context_optimistic_q_mean`, `context_optimism_gap` | posterior-mean vs. optimistic-max Q and their gap |
| `context_reconstruction_loss`, `context_prior_kl`, `context_balance_kl` | self-supervision health |
| `belief_slot_{k}_mass` | average posterior mass per context (slot usage) |
| `return_slot_{k}_fraction` | realized-return slot distribution (ties inferred contexts to realized returns for analysis) |

---

## 5 Default hyper-parameters

**Belief module and decision rule** (`belief_dual_iql_ree.yaml`):

| Parameter | Symbol | Default |
|---|---|---|
| `slot_number` | $K$ | 4 |
| `belief_hidden_dim` | $H_b$ | 64 |
| `belief_decoder_hidden_dim` | — | 128 |
| `belief_projection_dim` | $d$ | 16 |
| `belief_include_agent_id` | — | true |
| `belief_temperature` | $\varsigma$ | 1.0 |
| `belief_transition_stay` | $\rho$ | 0.95 |
| `belief_prior_strength` | $\beta$ | 1.0 |
| `belief_gumbel_tau` | $\tau_g$ | 1.0 |
| `belief_dynamics_weight` | $\lambda_{\text{belief}}$ | 1.0 |
| `belief_reward_loss_weight` | $\lambda_r$ | 1.0 |
| `belief_kl_weight` | $\lambda_{\text{kl}}$ | $10^{-3}$ |
| `belief_balance_weight` | $\lambda_{\text{bal}}$ | $10^{-2}$ |
| `belief_optimism_min` / `belief_optimism_max` | $\alpha_{\min}, \alpha_{\max}$ | 0.25 / 1.0 |
| `kl_weight` | $\lambda_{\text{distill}}$ | 10.0 |

**Shared training setup** (RAC and Belief-RAC unless noted):

| Parameter | Value |
|---|---|
| Optimizer | RMSprop, lr $5{\times}10^{-4}$, $\alpha=0.99$, $\varepsilon=10^{-5}$ |
| $\gamma$ | 0.99 |
| Replay buffer | 5000 episodes, batch 32 episodes |
| Target update | every 200 episodes (MATE entity-attention variant: every 1000 steps) |
| $\varepsilon$ schedule | 1.0 → 0.05 over 50k steps (MATE entity-attention variant: 150k) |
| Grad-norm clip | 10 |
| $K$ (matrix game) | 4, thresholds $\{-12, 0, 6, 8\}$ |
| $K$ (MATE) | 10, returns quantized on $[0, 1000]$ |

**MATE entity-attention variant overrides** (`mate_belief_entity_gru` preset):
batch 8 episodes, buffer 1000, lr $10^{-4}$, `kl_weight` 1.0,
`target_update_interval_steps` 1000, $\varepsilon$ anneal 150k,
`test_interval` 10k, 20 test episodes, `save_model_interval` 10k.

---

## 6 Experimental methodology (summary)

**Environments.** (i) Matrix game with payoffs inducing four distinct
coordination levels (contexts map 1:1 to returns $\{-12, 0, 6, 8\}$);
(ii) Predator–Prey with punishment (`pred_prey_punish`), $10^6$ steps,
test every 5k steps over 32 episodes; (iii) MATE multi-agent tracking
(scenarios `MATE-4v8-0`, `MATE-4v2-9`, `MATE-8v8-9`), 0.5–1M steps with the
entity-attention backbone, test coverage rate and return reported;
(iv) SMAC (`sc2`). All runs use 5 seeds for the matrix-game benchmark and
multiple seeds elsewhere; curves plot mean ± one standard deviation over seeds
with individual runs faint.

**Baselines.** RAC (`dual_iql_ree`) — the direct ablation isolating the belief
contribution; IQL; Hysteretic Q; IPPO (on-policy baseline, same environment
and evaluation budget; PPO-specific rollout/update schedule retained).
All off-policy methods share identical lr, replay, target-update, $\varepsilon$
schedule, discount, and one-step targets for fairness.

**Ablations supported by the implementation.** (a) belief off
(`use_context_belief=false`) → exact RAC; (b) belief inference on but the
decision rule reduced to posterior mean only ($\alpha_{\min}=\alpha_{\max}=0$)
→ tests whether optimism is needed; (c) fixed uniform belief → tests whether
context *inference* (vs. mere context-conditioning) matters; (d) removing the
balance KL (`belief_balance_weight=0`) → tests collapse behavior;
(e) MLP vs. recurrent vs. entity-attention backbones → tests backbone-agnosticism.
Metrics in Sec. 4 additionally allow qualitative verification that
`s_t` spikes at regime transitions and that `α_t` rises with uncertainty.
