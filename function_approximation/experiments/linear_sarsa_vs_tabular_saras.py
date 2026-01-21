import numpy as np
import  gymnasium as gym
import matplotlib.pyplot as plt 

from tabular_rl.sarsa import sarsa
from function_approximation.sarsa import sarsa_linear

env = gym.make("FrozenLake-v1",is_slippery=True)

def smooth(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

EPISODES =2000
MAX_STEPS = 200

params = dict(
    gamma=0.99,
    init_alpha=0.3,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    init_epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay_ratio=0.9,
    n_episodes=EPISODES,
    max_steps=MAX_STEPS,
)

print("Running Tabular SARSA")
Q_tab, Q_track_tab, V_tab, pi_tab = sarsa(env,**params)
print("Running Sarsa linear")
w_lin, Q_track_lin, V_lin, pi_lin, _ = sarsa_linear(env, **params)

V0_tab = np.max(Q_track_tab[:, 0, :], axis=1)
V0_lin = np.max(Q_track_lin[:, 0, :], axis=1)

V0_tab_s = smooth(V0_tab)
V0_lin_s = smooth(V0_lin)


plt.figure(figsize=(10,5))
plt.plot(V0_tab_s, label="SARSA Tabular")
plt.plot(V0_lin_s, label="SARSA Linear (FA)")
plt.title("FrozenLake — SARSA Tabular vs SARSA Linear")
plt.xlabel("Episodes")
plt.ylabel("V(start)")
plt.legend()
plt.grid(True)
plt.show()

