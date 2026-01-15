import numpy as np
"""this funciton allows you to calculate 
 all the values for alpha for the full tranning process"""
def decay_schedule(init_value,min_value,
                    decay_ratio, max_steps,
                    log_start = -2, log_base =10):
    decay_steps = int(max_steps*decay_ratio)
    rem_steps = max_steps-decay_steps

    values = np.logspace(log_start, 0, decay_steps,
                         base = log_base, endpoint =True)
    values = (values-values.min())

    values = (init_value-min_value)*values +min_value

    values = np.pad(values,(0, rem_steps), 'edge')

    return values

""" This fuction allows to run a policy and extrat 
    the trajectory
"""

def generate_trajectory(policy, env, max_steps=20):
    trajectory = []
    state, info = env.reset()
    
    for t in range(max_steps):
        action = policy[state]
        next_state, reward, terminated, truncated, info = env.step(action)

        trajectory.append((state,action,reward,next_state,(terminated or truncated)))
        

        state = next_state

        if terminated or truncated:
            break
    
    return trajectory
