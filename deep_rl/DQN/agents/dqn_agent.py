import random
import torch
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, policy_net, target_net, memory, optimizer,
                 gamma, n_actions, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.optimizer = optimizer
        self.gamma = gamma
        self.n_actions = n_actions
        self.device = device
    
    def select_action(self,state,epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            return self.policy_net(state).argmax(dim=1).item()
    
    def store_transition(self, state, action,reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
            