import numpy as np
from tabular_rl.utils import decay_schedule

def td_0_linear_prediction(env,
                           pi,
                           phi_fn,
                           feat_dim:int,
                           gamma =0.99,
                           init_alpha=0.2,
                           min_alpha=0.01,
                           alpha_decay_ratio=0.5,
                           n_episodes=2000,
                           max_steps=200):

        alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

        w = np.zeros(feat_dim,dtype=np.float64)

        V_track = np.zeros((n_episodes,),dtype=np.float64)

        def V(s:int) ->float:
                return float(np.dot(w,phi_fn(s)))
        
        for e in range(n_episodes):
                state, _ = env.reset()

                for s in range(max_steps):
                    a = pi(s)

                    next_state, reward, terminated, truncated, _ = env.step(a)
                    done = terminated or truncated

                    td_target = r if done else (r + gamma*V(next_state))
                    td_error = td_target - V(state)

                    w+= alphas[e]*td_error*phi_fn(state)

                    if done:
                           break
                V_track[e] = V(0)
        return w, V_track
                    