import numpy as np
def phi_sa(state, action, n_states, n_actions):
    """
    Feature vector φ(s,a) one-hot.
    Shape: (n_states * n_actions,)
    """
    x = np.zeros(n_states * n_actions, dtype=np.float64)
    x[state * n_actions + action] = 1.0
    return x