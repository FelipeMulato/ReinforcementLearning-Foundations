# ReinforcementLearning-Foundations

This repository contains from-scratch implementations of fundamental algorithms in Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL).  
The project is organized as a progressive study, starting from classical and tabular RL methods and advancing toward deep and multi-agent value decomposition approaches.

The focus is on understanding value functions, learning dynamics, and coordination in cooperative multi-agent settings.

---

## Repository Structure

- `Foundation/`  
  Classical and tabular reinforcement learning algorithms, including Monte Carlo methods, Temporal-Difference learning, n-step methods, and control algorithms.  
  This folder serves as the theoretical and algorithmic foundation of the project.

- `deep_rl/`  
  Single-agent deep reinforcement learning algorithms, including DQN and supporting components such as neural networks, replay buffers, and training loops.

- `Multi_Agent/`  
  Multi-agent reinforcement learning algorithms and experiments:
  - `VDN/`: Value Decomposition Networks (additive decomposition of value functions)
  - `QMIX/`: QMIX with monotonic mixing networks and hypernetworks for cooperative coordination

- `utils/`  
  Shared utilities used across different modules, such as decay schedules, trajectory generation, and helper functions.

---

## Motivation

Many reinforcement learning problems, especially in multi-agent settings, involve global reward signals that depend on the joint actions of multiple agents.  
Independent learning approaches often fail in such scenarios due to non-stationarity and credit assignment issues.

This project investigates value-based methods and value decomposition techniques as a way to address these challenges. The work is motivated by the theoretical framework introduced in *Q-Decomposition for Reinforcement Learning Agents* (Russell & Zimdars, 2003), and explores how these ideas extend to deep and multi-agent settings.

---

## Current Status

- From-scratch implementations of:
  - Tabular RL algorithms (MC, TD, n-step methods)
  - Deep Q-Networks (DQN)
  - Value Decomposition Networks (VDN)
  - QMIX with monotonic mixing networks
- Preliminary experiments in simplified coordination environments
- Empirical observation of the limitations of additive value decomposition (VDN) and the advantages of QMIX in tasks requiring coordinated joint actions

---

## Ongoing and Future Work

- Extension to more complex single-agent and multi-agent environments
- Quantitative evaluation with multiple seeds and metrics
- Integration of replay buffers and target networks for QMIX
- Exploration of recurrent architectures (e.g., DRQN) for partial observability
- Study of advanced multi-agent reinforcement learning methods

---

## References

- Russell, S., & Zimdars, A. (2003). *Q-Decomposition for Reinforcement Learning Agents*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*.
- Sunehag, P. et al. (2017). *Value-Decomposition Networks for Cooperative Multi-Agent Learning*.
- Rashid, T. et al. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*.

---

## Disclaimer

This repository is intended for research and educational purposes.  
The implementations prioritize clarity and conceptual correctness over performance optimizations.
