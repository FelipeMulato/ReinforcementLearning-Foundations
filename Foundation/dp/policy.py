import numpy as np
#Given a MDP we use this function to estimate how good a policy will be
def policy_evaluation(pi,P, gamma=1.0, theta =1e-10):
    prev_V = np.zeros(len(P))

    while True:
        V = np.zeros((len(P)))

        for s in range(len(P)):
            for next_state, prob,reward, done in P[s][pi[s]]:
                V[s] += prob*(reward+gamma*\
                              prev_V[next_state]*(not done))
        
        if np.max(np.abs(prev_V-V)) < theta:
            break


        prev_V = V.copy()
    return V
#Given a MDP we use this function to improve a given policy
def policy_improvement(V,P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])),dtype = np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            for next_state,prob, reward, done in P[s][a]:
                Q[s][a] += prob*(reward +gamma*\
                                 V[next_state]*(not done))
    
    new_pi = np.argmax(Q, axis=1)
    
    return new_pi 

def policy_iteration(P,gamma=1.0,theta=1e-10):
    n_states = len(P)
    n_actions = len(P[0])

    pi = np.random.randint(0, n_actions, size=n_states)
    while True:
        old_pi = pi.copy()

        V = policy_evaluation(pi,P,gamma,theta)

        pi = policy_improvement(V,P,gamma)

        if np.array_equal(old_pi, pi):
            break
    return V,pi


def value_iteration(P, gamma=1.0, theta=1e-10):
    n_states = len(P)
    n_actions = len(P[0])

    V = np.zeros(n_states)

    while True:
        Q = np.zeros((n_states, n_actions), dtype=np.float64)
        for s in range(n_states):
            for a in range(n_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s, a] += prob * (reward + gamma * V[next_state] * (not done))

        new_V = np.max(Q, axis=1)

        if np.max(np.abs(V - new_V)) < theta:
            break

        V = new_V.copy()

    pi = np.argmax(Q, axis=1)
    return V, pi