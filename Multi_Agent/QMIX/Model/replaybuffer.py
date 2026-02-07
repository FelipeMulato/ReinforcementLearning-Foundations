from collections import deque
import torch
import random
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        obs, state, actions, reward, next_obs, next_state, done = zip(*batch)
        obs = torch.stack(obs)        
        state = torch.stack(state)     
        actions = torch.stack(actions)    
        reward = torch.stack(reward)    
        next_obs  = torch.stack(next_obs)   
        next_state = torch.stack(next_state) 
        done = torch.stack(done)       

        return obs, state, actions, reward, next_obs, next_state, done
        
    def __len__(self):
        return len(self.buffer)