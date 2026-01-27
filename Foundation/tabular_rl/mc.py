import numpy as np
from tabular_rl.utils import decay_schedule,generate_trajectory_control,generate_trajectory_prediction
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
        trajectory = generate_trajectory_prediction(pi,env,max_steps)
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


def mc_control(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000,
               max_steps=200,
               first_visit=True):
        
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    discounts = gamma**np.arange(max_steps)

    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)

    Q = np.zeros((n_states,n_actions), dtype=np.float64)
    Q_track = np.zeros((n_episodes,n_states,n_actions), dtype = np.float64)
    pi_track = []

    def select_action(state,Q,epsilon):
        if np.random.random() > epsilon:
            return np.argmax(Q[state])
        return np.random.randint(len(Q[state]))
    
    for e in range(n_episodes):
        trajectory = generate_trajectory_control(select_action,Q,epsilons[e],env,max_steps)
        visited = np.zeros((n_states,n_actions), dtype = bool)

        for t,(state,action,reward,_,_) in enumerate(trajectory):
            
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            rewards = [r[2] for r in trajectory[t:]]

            steps = len(rewards)    
            G = np.sum(rewards * discounts[:steps])

            Q[state][action] = Q[state][action] + alphas[e]*\
                                (G-Q[state][action])

        Q_track[e] = Q
        pi_track.append(np.argmax(Q,axis=1))

    V = np.max(Q,axis=1)
    pi = np.argmax(Q,axis=1)

    return Q,V ,pi, Q_track, pi_track


