n_episodes=200
n_agents = 3
n_actions = 5
obs_dim = 18
state_dim = 54
gamma=0.99       
lr=1e-3
epsilon_start=1.0
epsilon_end=0.2
epsilon_decay=0.999
hidden_dim=16
mixing_hidden_dim=16
print_every=500
max_steps=25

batch_size = 32
buffer_size = 50000
target_update_interval = 200

device = "cpu"