import numpy as np
from tabular_rl.utils import decay_schedule

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

            action = pi(state)

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


def n_step_td_prediction(pi,env,
        gamma=1.0,
        init_alpha=0.5,
        min_alpha=0.01,
        alpha_decay_ratio=0.3,
        n_steps=3,
        n_episodes=500,
        max_steps=100):
    
    n_states = env.observation_space.n
    
    V = np.zeros(n_states)
    V_track = np.zeros((n_episodes,n_states))

    alphas = decay_schedule(init_alpha,min_alpha, alpha_decay_ratio)

    

    for e in range(n_episodes):
        state, info = env.reset()
        states = [state]
        rewards = []
        T = np.inf
        t = 0
        while True:
            if t < T:
                action = pi(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                rewards.append(reward)
                states.append(state)

                if terminated or truncated:
                    T = t+1
                else:
                    state = next_state
            tau = t -n_steps+1

            if tau>=0:
                G =0.0
                for i in range(tau,min(tau+n_steps,T)):
                    G += (gamma**(i-tau))*rewards[i]
                if tau +n_steps <T:
                    G += (gamma ** n_steps) * V[states[tau + n_steps]]

                s_tau = states[tau]
                V[s_tau] += alphas[e]*(G-V[s_tau])
            if tau == T-1:
                break
            t+=1 

            if t>=max_steps:
                break 
        V_track[e] = V
    return V,V_track



