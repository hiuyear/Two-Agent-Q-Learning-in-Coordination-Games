** this is part of my application to the ROP course: ro243 - Building a Theoretical Foundation for Multi-Agent Reinforcement Learning **
# Two-Agent Q-Learning in Coordination Games

**Empirical study of Q-learning convergence under noisy reward signals in two-agent coordination games.**

This project implements independent Q-learning from scratch for two agents playing repeated normal-form games, and studies how convergence properties change with reward noise, learning rate, and game structure.

## Motivation

Multi-agent reinforcement learning (MARL) presents a key theoretical challenge: agents learning simultaneously create a non-stationary environment for one another, and convergence guarantees that hold in single-agent settings break down. This project explores an accessible version of this problem — two agents playing a repeated coordination game — and empirically studies the factors that affect whether agents converge to the Pareto-optimal equilibrium.

## Games Studied

**Stag Hunt** — coordination game with two Nash equilibria:
- Pareto-optimal: (Stag, Stag) → payoff (4, 4)  
- Risk-dominant: (Hare, Hare) → payoff (3, 3)

**Matching Pennies** — zero-sum game with *no* pure-strategy Nash equilibrium, used as a baseline where convergence should not occur.

## Experiments

1. **Noise effect**: How does reward noise (σ) affect convergence to the Pareto-optimal equilibrium?
2. **Learning rate**: Does a higher α speed convergence at the cost of stability?
3. **Game structure**: Do agents converge differently in games with vs. without pure NE?

## Results

- Reward noise significantly delays Pareto coordination (agents defect to the risk-dominant equilibrium more often under noise)
- Moderate learning rates (α ≈ 0.1) balance convergence speed and stability; very high α causes oscillation
- Stag Hunt reliably converges to the Pareto equilibrium under low noise; Matching Pennies oscillates near the random baseline (0.5), consistent with its mixed-strategy NE

## Structure

```
marl_coordination.py   # Core implementation + experiments
README.md
```

## How to Run

```bash
pip install numpy matplotlib
python marl_coordination.py
# Outputs: marl_convergence_analysis.png
```

## Theoretical Context

This connects to open questions in MARL theory:
- Under what conditions does independent Q-learning converge in multi-agent settings?
- How does reward noise interact with the equilibrium selection problem?
- Can learning dynamics be tuned to prefer Pareto-optimal equilibria?

These questions motivate more rigorous analysis using stochastic approximation theory and game-theoretic tools — exactly the kind of theoretical foundation I hope to build through the ROP.
