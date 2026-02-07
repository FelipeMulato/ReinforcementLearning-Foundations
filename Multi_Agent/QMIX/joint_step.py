import torch
import numpy as np
from Multi_Agent.QMIX.utils import epsilon_greedy
def collect_joint_step(env, agent_nets, epsilon, agent_id_map, cfg):

    obs_all = [env.observe(a) for a in env.agents]
    obs_all = torch.from_numpy(np.array(obs_all)).float()


    actions =[]
    for i in range(cfg.n_agents):
        with torch.no_grad():
            Q = agent_nets[i](obs_all[i].unsqueeze(0))
            actions.append(epsilon_greedy(Q,epsilon))
    
    actions = torch.tensor(actions,dtype=torch.long, device=cfg.device)


    total_reward = 0
    done = False
    for agent in env.agent_iter():
        obs, reward, term, trunc, _ = env.last()
        idx =  agent_id_map[agent]

        if reward is not None:
            total_reward+=reward
        if term or trunc:
            env.step(None)
            done = True
        else:
            env.step(actions[idx].item())

    state = obs_all.view(1, -1)
    reward = torch.tensor([[total_reward]], dtype=torch.float32, device=cfg.device)
    done = torch.tensor([[done]], dtype=torch.float32, device=cfg.device)
    if done:
        next_obs_all = torch.zeros_like(obs_all)
        next_state = torch.zeros_like(state)
    else:
        next_obs_all = [env.observe(a) for a in env.agents]
        next_obs_all = torch.from_numpy(np.array(next_obs_all)).float().to(cfg.device)
        next_state = next_obs_all.view(1, -1)

    return obs_all, state, actions, reward, next_obs_all, next_state, done
    

