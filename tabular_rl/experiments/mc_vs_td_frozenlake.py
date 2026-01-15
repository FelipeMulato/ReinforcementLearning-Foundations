import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tabular_rl.mc import mc_prediciton
from tabular_rl.td import td_prediction

def random_policy(state):
    return env.action_space.sample()
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)

    n_episodes = 1000
    gamma = 0.9

    V_mc, V_mc_track = mc_prediciton(
        pi= random_policy,
        env = env,
        gamma= gamma,
        n_episodes=n_episodes
    )

    V_td, V_td_track = td_prediction(
        pi=  random_policy,
        env = env,
        gamma= gamma,
        n_episodes= n_episodes
    )

    mc_mean = np.mean(V_mc_track, axis=1)
    td_mean = np.mean(V_td_track, axis=1)

    plt.plot(mc_mean, label="Monte Carlo")
    plt.plot(td_mean, label="TD(0)")
    plt.xlabel("Episodes")
    plt.ylabel("Mean V")
    plt.title("MC vs TD on FrozenLake (deterministic)")
    plt.legend()
    plt.grid()
    plt.show()