import random
import torch
import torch.nn.functional as F

class DQNAgent:
    def __init__(self, policy_net, target_net, memory,batch_size, optimizer,
                 gamma, n_actions, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.batch_size = batch_size
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
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        states, actions, rewards, next_states, dones = zip(*transitions)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, device=self.device,dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device,dtype=torch.float32)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, device =self.device, dtype =torch.float32)

        q_values =self.policy_net(states).gather(1,actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            td_target = rewards + self.gamma*next_q_values*(1-dones)
        
        loss = F.mse_loss(q_values,td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())



            