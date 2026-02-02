import torch
class CordenateEnv:
    def reset(self):
        state = 0
        return state
    def step(self,action1, action2):
        reward = 10 if action1==0 and action2==0 else 0
        state = 0
        done  = True
        return state, reward, done
    def to_obs(self,state):
        return torch.tensor([1.0], dtype=torch.float32)

    def to_state_vec(self,state):
        return torch.tensor([1.0], dtype=torch.float32)