from torch import nn
import torch.nn.functional as F
import torch
class MIXER(nn.Module):
    def __init__(self,n_agents,state_dim,hidden):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden = hidden

        #hypernetwork
        self.hyper_w1 = nn.Linear(state_dim,n_agents*hidden)
        self.hyper_w2 = nn.Linear(state_dim,hidden)

        self.hyper_b1 = nn.Linear(state_dim,hidden)
        self.hyper_b2 = nn.Linear(state_dim,1)

    def forward(self, agent_qs, states):

        batch_size = agent_qs.size(0)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(batch_size, self.n_agents, self.hidden)
        b1 = b1.view(batch_size, 1, self.hidden)


        agent_qs = agent_qs.view(batch_size,1,self.n_agents)

        hidden = F.elu(torch.bmm(agent_qs,w1)+b1)

        
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(batch_size, self.hidden, 1)
        b2 = b2.view(batch_size, 1, 1)

        q_total = torch.bmm(hidden,w2)+b2

        return q_total.squeeze(-1)




