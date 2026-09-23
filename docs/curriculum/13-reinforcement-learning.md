# 13 — Reinforcement Learning

> Back to [index](../../README.md) · Prev: [12 Generative Models](12-generative-models.md) · Next: [14 Large Language Models](14-large-language-models.md)

RL: an **agent** takes **actions** in an **environment**, receiving **rewards** — learning a **policy** to maximize cumulative reward. No labeled data; delayed, aggregate feedback.

## 1. Formalism — MDPs

A Markov Decision Process = (S, A, P, R, γ):
- States S, actions A, transition P(s'|s,a) (Markov: future ⊥ past | present), reward R, **discount γ ∈ [0,1)** (near-future worth more; also bounds returns for γ<1).
- **Return:** `G_t = Σ γ^k r_{t+k+1}`; **value** `V(s) = E[G|s]`; **Q(s,a)** = value of acting.
- **Policy π(a|s):** stochastic mapping; optimal π* maximizes V.

**Credit assignment** is the core difficulty: which of the 10,000 moves won the game?

## 2. Value-based methods

- **Dynamic programming:** policy evaluation & improvement when P is known (rarely is).
- **Monte Carlo:** wait for episode end, average actual returns — unbiased, high variance.
- **TD learning (temporal difference):** bootstrap from current estimate:
  `Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]`
  The bracket = **TD error**.
- **Q-learning:** off-policy, learns optimal Q regardless of behavior policy; **SARSA:** on-policy.
- **Deep Q-Network (DQN):** Q-function = neural net; key stabilizers: **experience replay** (break correlations) + **target network** (freeze targets periodically). Extensions: Double DQN, Dueling, Prioritized replay, Rainbow. Atari-to-human (Mnih 2015).

## 3. Policy gradient methods

- Directly parametrize π_θ, ascend `∇_θ E[G]` — REINFORCE: `∇J = E[Σ ∇log π(a_t|s_t)·G_t]`.
- **Baselines/advantages** reduce variance without bias: `A = Q − V`.
- **Actor-Critic:** actor (policy) + critic (value) networks; A2C/A3C; **GAE** for advantage estimation.
- **PPO (Proximal Policy Optimization):** clip policy-ratio updates (`clip(ratio, 1−ε, 1+ε)`) — stable, the **workhorse of RLHF** and robotics. Its sibling TRPO uses KL constraints.
- **DDPG/TD3/SAC:** continuous action spaces (deterministic policies + replay, clipped double-Q, entropy regularization).

## 4. Model-based RL & beyond

- **Learn/known dynamics** → planning: Dyna, MuZero (learns latent model + MCTS — AlphaGo lineage), Monte-Carlo tree search.
- **Imitation learning / inverse RL:** behavior cloning (supervised on demos), GAIL — for when reward is undefined.
- **Exploration:** ε-greedy, UCB/Thompson (bandits = 1-state MDPs), intrinsic/reward curiosity.
- **Multi-agent RL (MARL):** independent learners, CTDE (centralized training/decentralized execution), self-play (AlphaZero).
- **Safety in RL:** constrained MDPs, reward hacking/specification gaming — a warning repeated in [14](14-large-language-models.md)/[22](22-ai-safety-security-ethics.md).

## 5. Why RL matters in modern AI (the RLHF bridge)

1. **RLHF for LLMs:** preference data → **reward model** → **PPO** optimizes the LLM under a **KL penalty** to the fine-tuned policy (keeps it from degenerating into reward-hacking gibberish). Full pipeline in [14](14-large-language-models.md).
2. **DPO:** skips the RL loop — direct preference optimization as a closed-form loss (see 14).
3. **RL for vision:** RL-based detection (RAP), active learning policies.
4. **Robotics:** locomotion/manipulation (Sim2Real, domain randomization); warehouse/autonomy.
5. **AlphaGo/AlphaZero, systems tuning, LLM agent trajectories** (agents scored by outcome reward — [18](18-ai-agents-and-tool-use.md)).

## 6. Classic pitfalls

- **Reward hacking:** optimize the proxy, destroy the goal (robot flips instead of runs; LLM pads answers to satisfy judge). Design reward + human oversight ([22](22-ai-safety-security-ethics.md)).
- **Sparse rewards:** shaping, curriculum, hindsight replay, exploration bonuses.
- **Non-stationarity in multi-agent:** moving targets.
- **Off-policy evaluation** before deploying (importance sampling, model-based eval).

## Mastery Checklist

- [ ] Defines MDP, value, Q, policy, return, and discounting with formulas
- [ ] Contrasts Q-learning (value-based, off-policy) vs PPO (policy gradient) with use cases
- [ ] Explains DQN's replay buffer + target network and why each is needed
- [ ] Sketches the RLHF pipeline: preferences → reward model → PPO + KL
- [ ] Names at least 3 real places RL appears in today's AI products
- [ ] Explains reward hacking with a concrete example and one mitigation
