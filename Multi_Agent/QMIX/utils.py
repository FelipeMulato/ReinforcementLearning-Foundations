import torch
import random

def epsilon_greedy(q_values,epsilon):
    if random.random()<epsilon:
        return random.randrange(q_values[0].shape)
    return int(torch.argmax(q_values).item())
