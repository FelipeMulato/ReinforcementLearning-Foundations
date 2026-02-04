import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from pettingzoo.mpe.simple_spread_v2 import env

from Multi_Agent.QMIX import config
from Multi_Agent.QMIX.Model.dqn import DQN
from Multi_Agent.QMIX.Model.mixer import MIXER
from Multi_Agent.QMIX.utils import decay_schedule,epsilon_greedy
env = env(continuous_actions=False)

n_agents = 3
n_actions = 5
obs_dim = 18
state_dim = 54

agent_nets = nn.ModuleList([
    DQN(obs_dim,config.hidden_dim,n_actions),
    DQN(obs_dim,config.hidden_dim,n_actions),
    DQN(obs_dim,config.hidden_dim,n_actions)
])

mixer_net = MIXER(n_agents,state_dim,config.mixing_hidden_dim)



params = list(agent_nets.parameters()) + list(mixer_net.parameters())
opt = optim.Adam(params, lr=config.lr)

epsilons = decay_schedule(config.epsilon_start,config.epsilon_end,config.epsilon_decay, config.n_episodes)


agent_id_map = {'agent_0':0, 'agent_1':1,'agent_2':2}
reward_track = []
for e in range(config.n_episodes):
    episode_return = 0.0
    env.reset()
    for s in range(config.max_steps):
        obs_buffer = [None]*n_agents
        reward_buffer = []
        action_buffer = [None]*n_agents
        q_buffer = [None]*n_agents
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            idx = agent_id_map[agent]
            
            if reward is not None:
                episode_return +=reward
                reward_buffer.append(reward)
            if termination or truncation:
                env.step(None)
                continue

            obs_t = torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
            q = agent_nets[idx](obs_t)
            action = epsilon_greedy(q,epsilons[e])

            obs_buffer[idx] = obs_t
            action_buffer[idx] = action
            q_buffer[idx] = q.gather(1,torch.tensor([[action]]))

            env.step(action)

            if env.agent_selection == env.agents[0] and None not in q_buffer:
                q_values = torch.cat(q_buffer, dim=1)

                state = torch.cat(obs_buffer, dim=1)

                Q_tot = mixer_net(q_values,state)

                target = sum(reward_buffer)

                loss = F.mse_loss(Q_tot, torch.tensor([[target]],dtype=torch.float32))

                opt.zero_grad()
                loss.backward()
                opt.step()

                obs_buffer = [None]*n_agents
                reward_buffer = []
                action_buffer = [None]*n_agents
                q_buffer = [None]*n_agents
    
    reward_track.append(episode_return)

import matplotlib.pyplot as plt

plt.plot(reward_track)
plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("QMIX on Simple Spread")
plt.show()


        

    


   




