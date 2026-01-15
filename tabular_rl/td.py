import numpy as np
from utils import decay_schedule

def td_prediction(pi,env,
        gamma=1.0,
        init_alpha=0.5,
        min_alpha=0.01,
        alpha_decay_ratio=0.3,
        n_episodes=500,
        max_steps=100):
    
    n_states = env.observation_space.n 


    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

    V = np.zeros(n_states)
    V_track = np.zeros((n_episodes,n_states))

    for e in range(n_episodes):
        state, info  = env.reset()
        for t in range(max_steps):

            action = pi[state]

            next_state,reward, terminated, truncated, info = env.step(action)

            if terminated:
                TD_target = reward
            else:
                TD_target = reward + gamma * V[next_state]

            TD_error = V[state]

            V[state] += alphas[e]*(TD_target-TD_error)

            if truncated or terminated:
                break

            state = next_state
        V_track[e] =V
    
    return V, V_track

