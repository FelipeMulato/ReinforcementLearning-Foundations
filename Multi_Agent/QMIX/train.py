import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from Multi_Agent.QMIX import config
from Multi_Agent.QMIX.Environments.cordenate import CordenateEnv
from Multi_Agent.QMIX.Model.dqn import DQN
from Multi_Agent.QMIX.Model.mixer import MIXER
from Multi_Agent.QMIX.utils import decay_schedule,epsilon_greedy

n_agents = 2
n_actions = 2
obs_dim = 1
state_dim = 1

agent_nets = nn.ModuleList([
    DQN(obs_dim,config.hidden_dim,n_actions),
    DQN(obs_dim,config.hidden_dim,n_actions)
])

mixer_net = MIXER(n_agents,state_dim,config.mixing_hidden_dim)

env = CordenateEnv()

params = list(agent_nets.parameters()) + list(mixer_net.parameters())
opt = optim.Adam(params, lr=config.lr)

epsilons = decay_schedule(config.epsilon_start,config.epsilon_end,config.epsilon_decay, config.n_episodes)

sucess =0
for e in range(1, config.n_episodes):
    state = env.reset()

    obs1 = env.to_obs(state)
    obs2 = env.to_obs(state)
    svec = env.to_state_vec(state)

    q1 = agent_nets[0](obs1)
    q2 = agent_nets[1](obs2)

    a1 = epsilon_greedy(q1,epsilons[e])
    a2 = epsilon_greedy(q2,epsilons[e])

    next_state, reward, done = env.step(a1,a2)

    
    states = svec.view(1, -1)

    agent_qs = torch.stack([q1[a1], q2[a2]]).unsqueeze(0)
    q_tot = mixer_net(agent_qs, states).squeeze(-1)

        
    target = torch.tensor([reward], dtype=torch.float32)

    loss = F.mse_loss(q_tot, target)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if reward>0:
        sucess +=1
   
    if (e+1) % config.print_every==0:
        print(f"current sucess rate:{sucess/config.print_every}")
        sucess=0




