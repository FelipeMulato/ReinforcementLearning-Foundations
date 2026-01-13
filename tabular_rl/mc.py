import numpy as np
from utils import decay_schedule,generate_trajectory
def mc_prediciton(pi,env,
                  gamma=1.0,
                  init_alpha=0.5,
                  min_alpha=0.01,
                  alpha_decay_ratio=0.3,
                  n_episodes=500,
                  max_steps=100,
                  first_visit =True):
    n_States = env.observation_space.n

    discounts = gamma ** np.arange(max_steps)

        
    alphas = decay_schedule(init_alpha,
                         min_alpha,alpha_decay_ratio,n_episodes)
    
    V = np.zeros(n_States)
    V_track = np.zeros((n_episodes,n_States))

    for e in range(n_episodes):
        trajectory = generate_trajectory(pi,env,max_steps)
        visited = np.zeros(n_States,dtype=np.bool)

        for t, (state,_,reward,_,_) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True
            
            rewards = [step[2] for step in trajectory[t:]]
            n_steps = len(rewards)

            G = np.sum(discounts[:n_steps] * rewards)

            V[state] = V[state] + alphas[e]*(G-V[state])

        V_track[e] = V

    return V.copy(),V_track
