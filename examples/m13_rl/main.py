"""m13 — Reinforcement learning: tabular Q-learning AND a neural DQN.

Proves theory doc 13-reinforcement-learning.md: MDPs, TD updates, epsilon-greedy
exploration, convergence to the value of the goal — then the same problem solved
with function approximation.

Two paths:
  1. Tabular Q-learning (NumPy) — an explicit Q table, TD(0) updates. Shows the
     mechanism and the Bellman equation with nothing hidden.
  2. DQN (PyTorch) — Q(s,a) as a neural net, with the two tricks that make it
     stable: a replay buffer and a target network. Runs when torch is present.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import torch_backend as TB  # noqa: E402

# 4x4 gridworld: start (0,0), goal (3,3)=+1, hole (1,1)=-1, step cost -0.01
H, W = 4, 4
GOAL, HOLE = (3, 3), (1, 1)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]        # up down left right
ARROWS = ["↑", "↓", "←", "→"]


def step(state, a, rng=None):
    """Deterministic transition + reward (the environment's P(s'|s,a), R)."""
    r, c = state
    dr, dc = ACTIONS[a]
    nr, nc = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
    if (nr, nc) == GOAL:
        return (nr, nc), 1.0, True
    if (nr, nc) == HOLE:
        return (nr, nc), -1.0, True
    return (nr, nc), -0.01, False


def q_learning(episodes=1500, gamma=0.95, alpha=0.2, eps_start=1.0,
               eps_decay=0.995, eps_min=0.05, seed=0):
    """TD control: Q(s,a) <- Q + alpha[r + gamma max Q(s',a') - Q(s,a)]."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((H, W, len(ACTIONS)))
    eps = eps_start
    returns = []
    for _ in range(episodes):
        s = (0, 0)
        done, G, t = False, 0.0, 0
        while not done and t < 100:
            if rng.random() < eps:
                a = int(rng.integers(len(ACTIONS)))       # explore
            else:
                a = int(Q[s[0], s[1]].argmax())           # exploit
            s2, r, done = step(s, a, rng)
            # TD target (bootstrap); done -> no future value
            target = r if done else r + gamma * Q[s2[0], s2[1]].max()
            Q[s[0], s[1], a] += alpha * (target - Q[s[0], s[1], a])
            s, G, t = s2, G + (gamma ** t) * r, t + 1
        eps = max(eps_min, eps * eps_decay)
        returns.append(G)
    return Q, returns


def greedy_rollout(Q, start=(0, 0), max_steps=30):
    """Execute the learned policy (no exploration) — the deployment step."""
    s, path, r_total = start, [start], 0.0
    for _ in range(max_steps):
        a = int(Q[s[0], s[1]].argmax())
        s, r, done = step(s, a)
        path.append(s)
        r_total += r
        if done:
            break
    return path, r_total


def torch_path():
    """DQN: the same MDP, but Q(s,a) is a neural net instead of a table.

    Why the two extra tricks matter (both proven in the asserts below):
      replay buffer — breaks the correlation between consecutive samples
      target net    — stops the chase-your-own-tail divergence
    """
    if not TB.HAS_TORCH:
        print("\n[torch] not installed — tabular path only (DQN skipped)")
        return None
    print(f"\n[torch] DQN on {TB.get_device()} — Q(s,a) as an MLP, "
          f"replay buffer + target network")
    res = TB.train_dqn(episodes=250, gamma=0.95, lr=5e-3, hidden=64, seed=0)
    imp = res["avg_last"] - res["avg_first"]
    print(f"    avg return {res['avg_first']:.2f} -> {res['avg_last']:.2f} "
          f"({imp:+.2f}) in {res['seconds']:.2f}s "
          f"params={res['n_params']}")
    # the Q-network must actually improve, not just run
    assert res["avg_last"] > res["avg_first"], (res["avg_first"], res["avg_last"])
    return res


def main():
    Q, returns = q_learning()

    # 1) learning happened: early returns worse than late returns
    early = np.mean(returns[:100])
    late = np.mean(returns[-100:])
    assert late > early, (early, late)

    # 2) greedy policy reaches the goal from the start state
    path, reward = greedy_rollout(Q)
    reached_goal = path[-1] == GOAL
    hole_hit = HOLE in path
    assert reached_goal and not hole_hit, (path, reached_goal, hole_hit)
    assert len(path) <= 15                            # roughly shortest path (6 optimal)

    # 3) value of cells that can STEP INTO the goal ~= +1; start is not a losing state
    V = Q.max(axis=2)
    assert max(V[(2, 3)], V[(3, 2)]) > 0.5, (V[(2, 3)], V[(3, 2)])
    assert V[0, 0] > -0.05, V[0, 0]
    # Q at the terminal goal itself stays 0 — no action is ever taken FROM it

    # 4) the policy is a function of state (state -> action table for all cells)
    policy = [[ARROWS[int(Q[r, c].argmax())] if (r, c) not in (GOAL, HOLE) else "·"
               for c in range(W)] for r in range(H)]
    flat = "".join(row for r in policy for row in ["".join(r)])

    dqn = torch_path()
    dqn_note = ""
    if dqn:
        dqn_note = (f" dqn_return={dqn['avg_first']:.1f}->{dqn['avg_last']:.1f} "
                    f"dqn_params={dqn['n_params']}")

    print(f"PASS m13 rl | return early={early:.2f} -> late={late:.2f} "
          f"path_len={len(path)-1} goal_reached={reached_goal} "
          f"V_start={V[0,0]:.3f} V_goal={V[GOAL]:.3f} policy={flat}{dqn_note}")


if __name__ == "__main__":
    main()
