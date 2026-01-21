import numpy as np
from tabular_rl.utils import decay_schedule
from function_approximation.utils import phi_sa
def sarsa_linear(env,
                 gamma=1.0,
                 init_alpha=0.5,
                 min_alpha=0.01,
                 alpha_decay_ratio=0.3,
                 init_epsilon=1.0,
                 min_epsilon=0.1,
                 epsilon_decay_ratio=0.9,
                 n_episodes=500,
                 max_steps=100):
        """
        SARSA with linear function approximation:
           Q(s,a;w) = w^T  φ(s,a)
        """
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
        epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)

        feat_dim = n_states*n_actions

        w = np.zeros(feat_dim, dtype = np.float64)

        Q_track = np.zeros((n_episodes,n_states,n_actions),dtype = np.float64)
        pi_track = []

        def Q(state,action):
                return np.dot(w, phi_sa(state,action,n_states,n_actions))

        def select_action(state, epsilon):
                if epsilon < np.random.random():
                        return np.argmax(Q(state,a) for a in range(n_actions))
                
                return np.random.randint(n_actions)
        
        for e in range(n_episodes):
            state, _ = env.reset()
            action = select_action(state,epsilons[e])

            for s in range(max_steps):
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_action = select_action(next_state,epsilons[e])

                    td_target = reward if done else reward + gamma*Q(next_state,next_action)

                    td_error = td_target - Q(state,action)

                    w += alphas[e]*td_error*phi_sa(state,action,n_states,n_actions)

                    state = next_state
                    action = next_action

                    if done:
                        break
            for s in range(n_states):
                for a in range(n_actions):
                    Q_track[e,s,a] = Q(s,a)
        
            pi_track.append(np.argmax(Q_track[e],axis=1))
            
        V = np.max(Q_track[-1], axis=1)
        pi = np.argmax(Q_track[-1], axis=1)

        return w, Q_track, V, pi, pi_track

                    
                                