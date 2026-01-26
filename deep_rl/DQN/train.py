import gymnasium as gym
import torch
import numpy as np

from deep_rl.DQN.models.dqn import DQN
from deep_rl.DQN.utils import decay_schedule
from deep_rl.DQN.agents.dqn_agent import DQNAgent
from deep_rl.DQN.memory.replay_buffer import ReplayMemory
from deep_rl.DQN import config

env = gym.make(config.ENV_NAME)  
render_env = gym.make(config.ENV_NAME, render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_states,config.HIDDEN_SIZE,n_actions).to(device)
target_net = DQN(n_states,config.HIDDEN_SIZE,n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(config.BATCH_SIZE)

optimizer = torch.optim.Adam(policy_net.parameters())
epsilons = decay_schedule(config.EPS_START,config.EPS_END,config.EPS_DECAY,config.N_EPISODES)

agent = DQNAgent(policy_net,target_net, memory,config.BATCH_SIZE,optimizer ,config.GAMMA,n_actions,device)


for e in range(config.N_EPISODES):
    if e % config.RENDER_EVERY==0:
        curr_env = render_env
        render = True
    else:
        curr_env = env
        render = False

    state, _ = curr_env.reset()
    state = torch.tensor(state, dtype =torch.float32, device=device).unsqueeze(0)

    total_reward =0

    for t in range(config.MAX_STEPS):

        if render:
            curr_env.render()

        action = agent.select_action(state,epsilons[e])

        next_state, reward, terminated, truncated, _ =  curr_env.step(action)
        done = terminated or truncated

        next_state_tensor = torch.tensor(next_state, dtype = torch.float32,
                                         device =device).unsqueeze(0)
        agent.store_transition(state,action, reward,next_state_tensor,done)
        agent.optimize_model()
        
        state = next_state_tensor
        total_reward += reward

        if (t+1) % config.TARGET_UPDATE==0:
            agent.update_target_network()
        if done:
            break
    if (e+1)%50 == 0:
        print(f"Episode {e+1}, reward: {total_reward}")