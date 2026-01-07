import numpy as np
from policy import policy_evaluation, policy_iteration
from mdp import MDP3

pi_0= np.zeros(len(MDP3),dtype=int)
pi_1= np.ones(len(MDP3),dtype=int)
V0 =policy_evaluation(pi_0,MDP3,gamma=0.9)
V1 =policy_evaluation(pi_1,MDP3,gamma=0.9)

print("State | V(pi_0) | V(pi_1)") 
for s in range(len(V1)):
    print(f"{s:5d} | {V0[s]:7.4f} | {V1[s]:7.4f}")
print("-=-=-=-=-After Policy Iterative-=-=-=-=-")
v,pi =policy_iteration(MDP3,gamma=0.9)
print("State | Pi(S) | V(S)") 
for s in range(len(v)):
    print(f"{s:5d}| {pi[s]:7.4f} | {v[s]:7.4f}")
    
