#Given a MDP we use this function to estimate how good a policy will be
def policy_evaluation(pi,P, gamma=1.0, theta =1e-10):
    prev_V = np.zeros(len(P))

    while True:
        V = np.zeros((len(P)))

        for s in range(len(P)):

            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob*(reward+gamma*\
                              prev_V[next_state]*(not done))
        
        if np.max(np.abs(prev_V-V)) < theta:
            break


        prev_V = V.copy()
    return V
#Given a MDP we use this function to improve a given policy
def policy_improvement(V,P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])),dytpe = np.float64)

    for s in range(len(p)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob*(reward +gamma*\
                                 V[next_state]*(not done))
    
    new_pi = lambda s: {s: a for s,a in enumerate(
                                    np.argmax(Q,axis=1))}[s]
    
    return new_pi 

def policy_iteration(P,gamma=1.0,theta=1e-10):
    random_actions = np.random.choice(
                                tuple(P[0].keys()),len(P))
    pi = lambda s:{s:a for s, in enumerate(
                             random_actions)}[s]
    while True:
        old_pi = {s:pi(s) for s in range(len(P))}

        V = policy_evaluation(pi,p,gamma,theta)

        pi = policy_improvement(V,P,gamma)

        if old_pi == {s:pi(s) for s in range(len(P))}:
            break
    return V,pi