import numpy as np 
from tabular_rl.utils import decay_schedule
def q_learning(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.3,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=500,
         max_steps=100):
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)

    Q = np.zeros((n_states,n_actions))
    Q_track = np.zeros((n_episodes,n_states,n_actions))

    def select_action(Q,state, epsilon):
        if np.random.random() > epsilon:
            return np.argmax(Q[state])
        else:
            return np.random.randint(len(Q[state]))

    for e in range(n_episodes):
        state, _ = env.reset()

        for s in range(max_steps):

            action = select_action(Q,state,epsilons[e])

            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                q_target = reward
                q_error = q_target-Q[state][action]
                Q[state][action] += alphas[e]*(q_error)
                break

            q_target = reward + gamma*np.max(Q[next_state])
            q_error = q_target - Q[state][action]
            Q[state][action] += alphas[e]*(q_error)

            state = next_state
        Q_track[e] = Q
    pi = np.argmax((Q),axis=1)
    V = np.max((Q),axis=1)

    return Q,Q_track, pi,V

                       


