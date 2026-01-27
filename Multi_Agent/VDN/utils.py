import numpy as np

def epsilon_greedy(Q):
    if np.random.random()<0.3:
        return np.random.randint(0,2)
    return np.argmax(Q)