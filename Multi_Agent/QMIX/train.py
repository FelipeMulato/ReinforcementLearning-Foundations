import torch
import copy
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from pettingzoo.mpe.simple_spread_v2 import env as mpe_env
import copy

from Multi_Agent.QMIX import config
from Multi_Agent.QMIX.Model.dqn import DQN
from Multi_Agent.QMIX.Model.mixer import MIXER
from Multi_Agent.QMIX.Model.replaybuffer import ReplayBuffer
from Multi_Agent.QMIX.utils import decay_schedule
from Multi_Agent.QMIX.batch import optimize_batch
from Multi_Agent.QMIX.joint_step import collect_joint_step

def train():
    cfg = config
    env = mpe_env(continuous_actions=False)
    env.reset()
    agent_id_map = {a: i for i, a in enumerate(env.agents)}

    agent_nets = nn.ModuleList([DQN(cfg.obs_dim, cfg.hidden_dim, cfg.n_actions) for _ in range(cfg.n_agents)])
    mixer = MIXER(cfg.n_agents, cfg.state_dim, cfg.mixing_hidden_dim)

    agent_targets = copy.deepcopy(agent_nets)
    mixer_target = copy.deepcopy(mixer)

    optimizer = optim.Adam(list(agent_nets.parameters()) + list(mixer.parameters()), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size)

    step_count = 0
    returns = []

    epsilons = decay_schedule(cfg.epsilon_start,cfg.epsilon_end,cfg.epsilon_decay,cfg.n_episodes)

    for ep in range(cfg.n_episodes):
        env.reset()
        ep_ret = 0.0
        

        for _ in range(cfg.max_steps):
            tr = collect_joint_step(env, agent_nets, epsilons[ep], agent_id_map, cfg)
            buffer.push(tr)
            ep_ret += tr[3].item()

            if len(buffer) >= cfg.batch_size:
                optimize_batch(buffer, agent_nets, mixer, agent_targets, mixer_target, optimizer, cfg)

            step_count += 1
            if step_count % cfg.target_update_interval == 0:
                agent_targets = copy.deepcopy(agent_nets)
                mixer_target = copy.deepcopy(mixer)

            if tr[-1].item():
                break

        returns.append(ep_ret)
        if ep % 50 == 0:
            print(f"Episode {ep}, Return {np.mean(returns[-10:]):.2f}")

    return returns

train()