from torch import nn

class DQN(nn.Module):
    def __init__(self,ninput,h1_nodes,out_actions):
        super().__init__()

        self.fc1 = nn.Linear(ninput,h1_nodes)
        self.out = nn.Linear(h1_nodes,out_actions)
    
    def forward(self,x):
        x = nn.functional.relu(self.fc1(x))
        x = self.out(x)
        return x