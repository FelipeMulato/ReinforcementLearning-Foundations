import numpy as np
from Multi_Agent.VDN.env import CordenateEnv
from Multi_Agent.VDN.utils import epsilon_greedy
alpha = 0.2

Q1 = np.zeros(2)
Q2 = np.zeros(2)

env = CordenateEnv()

for e in range(1000):
    state = env.reset()

    a1 = epsilon_greedy(Q1)
    a2 = epsilon_greedy(Q2)

    _,reward,_ =  env.step(a1,a2)

    td_target = reward
    td_error = reward - (Q1[a1]+Q2[a2])

    Q1[a1] += alpha*td_error
    Q2[a2] += alpha*td_error
   

print(f"Action 0: {Q1[0]}--{Q2[0]}")
print(f"Action 1: {Q1[1]}--{Q2[1]}")