from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_input,h1_nodes,n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_input,h1_nodes)
        self.out = nn.Linear(h1_nodes,n_actions)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.out(x)

        return x
    
