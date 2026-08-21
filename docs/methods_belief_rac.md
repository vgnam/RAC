# Methods: Belief-RAC (Causal Categorical Context Inference for Decentralized Cooperative MARL)

> Written from the reference implementation (`src/modules/context_belief.py`,
> `src/controllers/dual_ree_controller.py`,
> `src/learners/dual_episode_ree_q_learner.py`,
> `src/modules/agents/mate_entity_gru_agent.py`).
> Use this as the core "Method" content for the paper.

---

## Notation

We consider a decentralized cooperative multi-agent setting with $n$ agents, a
joint observation space and a shared team reward $r$. Agent $i$ maintains an
action--observation history $\tau^i_t = \{(o^i_{1}, a^i_{1}, r_1), \dots,
(o^i_{t-1}, a^i_{t-1}, r_{t-1})\}$ up to (but not including) the action at
time $t$. We use $o^i_t$ for the local observation, $a^i_t$ for the action, and
$K$ for the number of latent categorical contexts (slots), with the context
indexed by $z \in \{1,\dots,K\}$.

---

## 3 Method

### 3.1 Setting the context: RAC as a return-aware-context policy

The base algorithm, *Return-Aware Context* (RAC) learning (referred to as
`dual_iql_ree` in our reference implementation), trains two per-agent value
functions in parallel:

- a **behavior policy Q-function** $Q^i(\tau^i_t, a^i)$ that depends only on
  the local history; and
- a **context-conditioned (twin) Q-function** $Q^i_{\text{twin}}(\tau^i_t, z, a)$
  that further conditions on a categorical latent variable $z$ whose role is
  to index the *level of cooperative play / expected joint payoff*.

The latent context is *return-aware*: it is obtained by exploiting a special
signal that RAC makes available — the episodic team return. During offline
training, because the full episode return $R = \sum_t r_t$ is revealed, the
algorithm maps the episode return onto one of $K$ context slots by quantizing
$R$ against a set of thresholds `possible_returns`:

$$z \ =\ \mathrm{bin}(R) \in \{1,\dots,K\}.$$

The twin network is trained on this episodic context, and it is used to
produce a **counterfactual teacher**: for each time step the network evaluates
$Q^i_{\text{twin}}$ for *all* contexts (i.e. $\max_z Q^i_{\text{twin}}(\tau^i, z, a)$),
and a KL constraint pushes the behavior policy $Q^i$ toward the teacher's
action distribution. At execution the agent cannot observe the final return
before acting, so the original RAC policy is a *max-over-contexts* optimistic
decision rule $Q_{\text{dec}}(a) = \max_z Q_{\text{twin}}(\tau, z, a)$.

The limitation we address is that the return-aware context is *blind at
decision time*: the true return-context is only ever revealed post-episode
during training, so at execution RAC must resort to a context-agnostic
max-over-contexts rule applied uniformly at every step, regardless of how
confident the agent is about the current cooperation regime.

### 3.2 Causal categorical context belief

Belief-RAC replaces the post-hoc episodic context with an **online, causal
categorical belief filter** that infers $z_t$ *before* acting, using only
information available at time $t$. Formally each agent conditions its decision
on the posterior

$$b^i_t(z) \ \triangleq\ \mathbb{P}\!\big(z^i_t = z \mid o^i_t,\ a^i_{t-1},\ r_{t-1}\big),$$

i.e. on the current local observation together with the previous action and
the (shared) previous reward. No future information enters the inference.

**Filter architecture.** A key design property is that the belief inference is
*backbone-agnostic*: the belief filter is an independent network module
(`ContextBeliefModel`) that sits *in parallel with* the agent's value network
rather than inside it, and it is agnostic to whether the underlying agent is a
feed-forward (MLP) or a recurrent (GRU/LSTM) policy. Everything in this section
— the evidence encoding, the prior update, and the dynamics self-supervision —
applies unchanged to both backbones, so Belief-RAC is a drop-in extension of
RAC for any differentiable local-policy family.

The filter's inputs at time $t$ are simply concatenated as

$$\xi^i_t \ =\ \big[\,o^i_t,\ \mathbf{1}(a^{i}_{t-1}),\ r_{t-1},\ \mathrm{id}_i\,\big],$$

where $\mathbf{1}(a^{i}_{t-1})$ is the one-hot previous-action embedding,
$r_{t-1}$ is the scalar previous reward broadcast to all agents, and
$\mathrm{id}_i$ is the agent identity. An encoder maps the (history-extended)
filterset of these features into an evidence vector $h^i_t$ — a nonlinear layer
perceiving the current observation in the MLP case, or a recurrent state such
as a GRU in the memory-backed case — from which categorical evidence logits
are read out:

$$h^i_t \ =\ \mathrm{Enc}\big(\xi^i_t,\ h^i_{t-1}\big)\in\mathbb{R}^{H},
\qquad
\ell^i_t(z) \ =\ \frac{W_{\text{eh}}\, h^i_t + b_{\text{eh}}}{\varsigma},$$

where $\varsigma$ is a temperature (`belief_temperature`) and
$\mathrm{Enc}(\cdot)$ abstracts over the choice of backbone (identity/MLP with
the observation only, or a recurrent core in the memory-backed case). The
evidence head is initialized to zero weights and bias so that the filter
*starts from an exactly uniform belief* rather than from arbitrary confidence.

**Bayes-like update with a drifting prior.** The previous belief is first
transformed into a *predictive prior* that models the possibility of a regime
switch via a self-transition probability $\rho$ (`belief_transition_stay`):

$$\bar b^{\,}_{t}(z) \ =\ \rho\, b_{t-1}(z) + (1-\rho)\,\frac{1}{K},$$

so that with probability $(1-\rho)$ the agent is willing to consider any
context (preventing it from getting stuck in one slot). The posterior is then

$$b_{t}(z) \ =\ \mathrm{softmax}_{z}\Big( \ell_{t}(z) + \beta\,
\log \bar b^{\,}_{t}(z) \Big),$$

with prior strength $\beta$ (`belief_prior_strength`). In addition we compute a
**regime-shift signal**

$$s_{t} \ =\ 2\,\mathrm{JSD}\!\big(b_{t} \,\Vert\, \bar b^{\,}_{t}\big) \in [0,1],$$

the normalized Jensen--Shannon divergence between the posterior and its
predictive prior, which is high exactly when the environment regime is
surprising or transitioning.

### 3.3 Self-supervised learning of the belief filter

The belief cannot be supervised by a ground-truth context label, so it is
trained with a **mixture-of-contexts dynamics decoder** in a self-supervised
fashion. The assumption is that each latent context $z$ describes a distinct
local transition regime. A decoder predicts, per context, a *projected
observation delta* and the *reward*:

$$\Big(\widehat{\Delta o},\ \widehat r\Big)[z]
\ =\ \mathrm{decoder}\big(o^i_t,\ a^i_t,\ \mathrm{id}_i\big)[z].$$

The projected delta is obtained by measuring the change in observation through
a fixed random projection matrix $P$ (normalized rows, fixed seed), i.e.
$P\,(o_{t+1}-o_t)$, so the decoder has a low-dimensional self-supervised
target that does not require reconstructing the full observation.

Training then minimizes the belief-weighted reconstruction error

$$\mathcal{L}_{\text{recon}} \ =\ \sum_{z} b_t(z)\,
\underbrace{\Big(\mathrm{smooth}_{L1}\big(\widehat{\Delta o}[z],\
P(o_{t+1}-o_t)\big) \;+\; \lambda_r\,
\mathrm{smooth}_{L1}\big(\widehat r[z],\ r_t\big)\Big)}_{\text{per-context
dynamics error}},$$

which acts as a soft assignment: each context is allocated credit proportional
to how well it explains the observed local transition. Two regularizers keep
the filter useful:
a prior term $\mathrm{KL}(b_t \,\Vert\, \mathcal{U})$ and a balance term
$\mathrm{KL}(\bar b \,\Vert\, \mathcal{U})$ over the marginal belief, the
latter preventing collapse where all agents assign mass to a single slot.
The total self-supervised objective is

$$\mathcal{L}_{\text{belief}}
\ =\ \mathcal{L}_{\text{recon}}
\;+\; \lambda_{\text{kl}}\; \mathrm{KL}(b_t \,\Vert\, \mathcal{U})
\;+\; \lambda_{\text{bal}}\; \mathrm{KL}(\bar b \,\Vert\, \mathcal{U}).$$

### 3.4 Context-conditioned Q and belief-merged decision rule

The twin Q-function is implemented as a **hyper-network** so that
$Q^i_{\text{twin}}(\tau^i, z, a)$ factorizes over contexts without re-encoding
the state: a hyper-network maps the context $z$ to the weights and bias of the
final linear layer,

$$\big(W(z),\, b(z)\big) \ =\ \mathrm{hyper}(z),\qquad
Q^i_{\text{twin}}(\tau^i, z, a) \ =\ \big(h_{\tau}\, W(z)\big)_a + b(z)_a,$$

where $h_{\tau}$ is the agent's feature representation of the local history
$\tau^i$ produced by its value backbone (an MLP or, in the memory-backed case,
a recurrent core). This context-conditioning is again *backbone-agnostic*: the
hyper-network only needs access to the backbone's penultimate representation
$h_{\tau}$, so the same construction serves both MLP and GRU/LSTM value
networks. Evaluating the network at the $K$ vertices of the probability
simplex (one-hot vectors for each $z$) yields $Q$ for all contexts in a single
forward pass, which is used both at training time (counterfactual teacher) and
at execution time (decision).

Given the posterior belief $b_t$ and shift signal $s_t$, the agent forms its
decision Q-values as a **belief-augmented optimistic rule**:

$$Q_{\text{dec}}(a)
\ =\ (1-\alpha_t)\;
\underbrace{\sum_z b_t(z)\, Q_{\text{twin}}(\tau, z, a)}_{\text{posterior mean}}
\;+\; \alpha_t\;
\underbrace{\max_z Q_{\text{twin}}(\tau, z, a)}_{\text{optimistic max}},$$

where the optimism coefficient $\alpha_t \in [\alpha_{\min}, \alpha_{\max}]$ is
driven by belief uncertainty:

$$\alpha_t \ =\ \alpha_{\min} + (\alpha_{\max}-\alpha_{\min})\, u_t,
\qquad
u_t \ =\ \underbrace{\frac{\mathcal{H}(b_t)}{\log K}}_{\text{normalized entropy}}
\ \sqcup\ s_t,$$

with $\sqcup$ the probabilistic union of the ambiguity and regime-shift
signals: $u_t = 1 - (1-\mathcal{H}_{\text{norm}})(1-s_t)$. Because
$\alpha_{\min} > 0$, the optimistic $\max_z$ component is **never fully
removed**; when the belief is maximally uncertain (uniform posterior),
$\alpha_t = \alpha_{\max}$ and, with $\alpha_{\max}=1$, the rule *exactly
recovers* the original RAC max-over-contexts. When the agent is confident,
$\alpha_t$ falls to (but not below) $\alpha_{\min}$ and the decision is
dominated by the probabilistically-weighted posterior mean.

### 3.5 Training objective

Let $\hat z$ be the episodic context derived from the revealed return
(Sec. 3.1) and let $z^*_t \sim \mathrm{GumbelSoftmax}(\ell_t, \tau)$ be the
differentiable hard sample of the belief posterior used to condition the twin
during TD updates. With `use_context_belief` enabled the learner is trained
with the total loss

$$\mathcal{L} \ =\ \underbrace{\mathcal{L}_{\text{TD}}\big(Q\big)
+ \mathcal{L}_{\text{TD}}\big(Q_{\text{twin}}\big)}_{\text{double-Q temporal-difference}}
\;+\; \lambda_{\text{kl}}\;
\underbrace{\mathrm{KL}\big(\pi_{\text{main}} \,\Vert\,
\pi_{\text{teacher}}\big)}_{\text{behavior-to-teacher distillation}}
\;+\; \lambda_{\text{belief}}\; \mathcal{L}_{\text{belief}}.$$

Here $\mathcal{L}_{\text{TD}}$ are the standard one-step double-Q-learning
targets, and the teacher distribution $\pi_{\text{teacher}}$ is built from
$Q_{\text{twin}}$ evaluated at *all* contexts and merged with the belief
posterior and shift signal (Sec. 3.4). The KL term makes the light behavior
policy $Q$ imitate the richer, belief-aware context-conditioned teacher, and
is distinct from a plain return-aware-context baseline because the teacher
incorporates the *estimated* (not revealed) context.

### 3.6 Execution

At execution the belief filter is advanced one causal step per time step,
using only $(o^i_t, a^i_{t-1}, r_{t-1})$ to produce $(b_t, s_t)$; the belief
itself is a sequential state even when the underlying value backbone is an MLP,
since the prior update (Sec. 3.2) carries $b_{t-1}$ forward through the
regime-switching transition. The agent evaluates $Q_{\text{twin}}$ over all $K$
contexts, forms $Q_{\text{dec}}$ via Sec. 3.4, and selects greedy
/$\varepsilon$-greedy actions. The overall procedure is fully decentralized:
each agent needs only its own observation, its previous action, and the shared
reward — no episode-return reveal and no central reasoning.

---

## Compact key equations (for a figure/table)

| Quantity | Definition | Code symbol |
|---|---|---|
| Predictive prior | $\bar b_t = \rho\, b_{t-1} + (1-\rho)\,\frac{1}{K}$ | `transition_stay` |
| Posterior | $b_t = \mathrm{softmax}\big(\ell_t + \beta \log \bar b_t\big)$ | `filter_step` |
| Shift | $s_t = 2\,\mathrm{JSD}(b_t \Vert \bar b_t)$ | `normalized_js_divergence` |
| Context-Q | $Q_{\text{twin}}(\tau,z,a) = (h_\tau W(z))_a + b(z)_a$ | `_all_context_q` |
| Decision | $Q_{\text{dec}} = (1-\alpha_t)\sum_z b_t Q^z + \alpha_t \max_z Q^z$ | `posterior_optimistic_q` |
| Optimism | $\alpha_t \in [\alpha_{\min},\alpha_{\max}]$, from $u_t=\mathcal{H}_{\text{norm}}(b_t)\sqcup s_t$ | `optimism_weight` |
| Belief loss | $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{kl}}\mathrm{KL} + \lambda_{\text{bal}}\mathrm{KL}_{\text{marg}}$ | `dynamics_loss` |

---

## Default hyper-parameters (belief config)

| Parameter | Meaning | Default |
|---|---|---|
| `slot_number` | Number of latent contexts $K$ | 4 |
| `use_context_belief` | Enable causal belief inference | `true` |
| `belief_direct_action` | Use belief in the execution decision | `true` |
| `belief_transition_stay` | Context self-transition $\rho$ | 0.95 |
| `belief_prior_strength` | Prior strength $\beta$ | 1.0 |
| `belief_temperature` | Evidence temperature $\varsigma$ | 1.0 |
| `belief_optimism_min` / `max` | Optimism bounds $\alpha_{\min},\alpha_{\max}$ | 0.25 / 1.0 |
| `belief_dynamics_weight` | Self-supervision weight $\lambda_{\text{belief}}$ | 1.0 |
| `belief_kl_weight` | Prior-KL weight $\lambda_{\text{kl}}$ | 1e-3 |
| `belief_balance_weight` | Marginal-balance weight $\lambda_{\text{bal}}$ | 1e-2 |
| `belief_include_agent_id` | Add agent identity to filter input | `true` |
| `kl_weight` | Behavior→teacher distillation weight | 10.0 |
