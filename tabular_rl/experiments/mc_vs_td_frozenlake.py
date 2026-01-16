import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from tabular_rl.mc import mc_prediciton
from tabular_rl.td import td_prediction

def moving_average(x, window=100):
    return np.convolve(x, np.ones(window)/window, mode="valid")

def safe_policy(state):
    #Mapeamento manual para FrozenLake
    policy_map = {
        0: 2,  
        1: 2,  
        2: 1,  

        4: 1,  
        6: 1,  

        8: 2,  
        10: 1, 

        13: 2, 
        14: 2 
    }

    return policy_map.get(state, 1)  # default: Down
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_episodes = 10000
    gamma = 0.9

    V_mc, V_mc_track = mc_prediciton(
        pi= safe_policy,
        env = env,
        gamma= gamma,
        n_episodes=n_episodes
    )

    V_td, V_td_track = td_prediction(
        pi=  safe_policy,
        env = env,
        gamma= gamma,
        n_episodes= n_episodes
    )

    mc_mean = np.mean(V_mc_track, axis=1)
    td_mean = np.mean(V_td_track, axis=1)
    
    mc_smooth = moving_average(mc_mean,200)
    td_smooth = moving_average(td_mean,200)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(mc_smooth, label="MC")
    axs[0].plot(td_smooth, label="TD")
    axs[0].set_title("Mean V (smoothed)")
    axs[0].legend()

    axs[1].plot(V_mc_track[:, 0], label="MC start")
    axs[1].plot(V_td_track[:, 0], label="TD start")
    axs[1].set_title("Start state value")
    axs[1].legend()

    plt.xlabel("Episodes")
    plt.tight_layout()
    plt.show()