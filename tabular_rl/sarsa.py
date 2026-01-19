import numpy as np
from tabular_rl.utils import decay_schedule
def sarsa(env,
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
    pi_track = []

    def select_action(Q,state,epsilon):
        if np.random.random()>epsilon:
            return np.argmax(Q[state])
        else:
            return np.random.randint(len(Q[state]))

    for e in range(n_episodes):
        state, _ = env.reset()
        action = select_action(Q,state,epsilons[e])
        for s in range(max_steps):
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                td_target = reward
                td_error  = td_target - Q[state][action]
                Q[state][action] += alphas[e]*td_error
                break

            next_action = select_action(Q,next_state,epsilons[e])

            td_target = reward + gamma*Q[next_state][next_action]
            td_error = td_target-Q[state][action]

            Q[state][action] += alphas[e]*td_error

            action = next_action
            state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q,axis=1))

    V = np.max(Q,axis=1)
    pi = np.argmax(Q,axis=1)
    return Q,Q_track,V, pi



def lambda_sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.3,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          lambda_=0.5,
          replacing_traces=True,
          n_episodes=500,
         max_steps=100):
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    

    Q = np.zeros((n_states,n_actions))
    Q_track = np.zeros((n_episodes,n_states,n_actions))
    E = np.zeros((n_states,n_actions))
    pi_track = []

    def select_action(Q,state,epsilon):
        if np.random.random()>epsilon:
            return np.argmax(Q[state])
        else:
            return np.random.randint(len(Q[state]))

    for e in range(n_episodes):
        state, _ = env.reset()
        E.fill(0)
        action = select_action(Q,state,epsilons[e])
        for s in range(max_steps):
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done =  terminated or truncated

            if done:
                td_target = reward
            else:
                next_action = select_action(next_state, epsilons[e])
                td_target = reward + gamma * Q[next_state, next_action]

            td_error = td_target - Q[state, action]

            if replacing_traces:
                E[state, action] = 1
            else:
                E[state, action] += 1

            Q += alphas[e] * td_error * E
            E *= gamma * lambda_

            if done:
                break

            state = next_state
            action = next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q,axis=1))

    V = np.max(Q,axis=1)
    pi = np.argmax(Q,axis=1)
    return Q,Q_track,V, pi

    