# Markov Decision Processes – Dynamic Programming in Reinforcement Learning

This repository contains clean and explicit implementations of core Dynamic Programming algorithms for solving Markov Decision Processes (MDPs), including **Policy Evaluation**, **Policy Iteration**, and **Value Iteration**.  
The project is designed to build strong intuition about Bellman equations, optimal policies, and the effect of stochasticity and discounting in Reinforcement Learning.
---

## Implemented Algorithms

- **Policy Evaluation** (Iterative)
- **Policy Improvement**
- **Policy Iteration**
- **Value Iteration**

All algorithms are implemented from scratch using NumPy, following the standard formulation from Sutton & Barto.

---

## Environments (MDPs)

The project includes handcrafted toy MDPs designed to highlight different aspects of decision making:

- **MDP1** – Deterministic environment  
- **MDP2** – Stochastic environment  
- **MDP3** – Corridor with trap and delayed reward (custom designed)

Each environment follows the transition structure:

P[s][a] -> list of (probability, next_state, reward, done)


This format is compatible with the classical OpenAI Gym specification.

---

##  Experiments

The file `experiments.py` contains structured experiments for:

### 1. Policy Evaluation
- Comparison of different fixed policies on the same MDP
- Analysis of state-value functions under each policy
- Study of the effect of the discount factor (γ)

### 2. Policy Iteration
- Convergence from a random policy to the optimal policy
- Inspection of the final policy and value function

### 3. Value Iteration
- Direct solution of the Bellman optimality equation
- Comparison with Policy Iteration results

---


## Learning Objectives
This project aims to:

Develop deep intuition about Bellman equations

Understand how optimal policies emerge from value functions

Compare deterministic vs stochastic decision processes

Analyze the influence of the discount factor (γ)

Build a strong foundation for advanced RL topics and research


## Notes
This repository is intentionally kept minimal and explicit to emphasize clarity and understanding over abstraction.
The goal is not performance, but conceptual mastery.
