import torch
import torch.nn.functional as F
from torch import nn as nn
def optimize_batch(buffer, agent_nets, mixer, agent_targets, mixer_target, optimizer, cfg):
    obs, state, actions, reward, next_obs, next_state, done = buffer.sample(cfg.batch_size)

    

    q_agents = []
    for i in range(cfg.n_agents):
        q = agent_nets[i](obs[:, i, :]).gather(1, actions[:, i:i+1])
        q_agents.append(q)
    q_agents = torch.cat(q_agents, dim=1)  
    q_tot = mixer(q_agents, state)

    
    with torch.no_grad():
        next_actions = []
        for i in range(cfg.n_agents):
            a = agent_nets[i](next_obs[:, i, :]).argmax(dim=-1, keepdim=True)
            next_actions.append(a)
        next_actions = torch.cat(next_actions, dim=1)

        next_q_agents = []
        for i in range(cfg.n_agents):
            q = agent_targets[i](next_obs[:, i, :]).gather(1, next_actions[:, i:i+1])
            next_q_agents.append(q)
        next_q_agents = torch.cat(next_q_agents, dim=1)

        q_tot_target = mixer_target(next_q_agents, next_state)
        y = reward + cfg.gamma * (1 - done) * q_tot_target

    loss = F.mse_loss(q_tot, y)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(list(mixer.parameters()), 10.0)
    optimizer.step()

    return loss.item()