from collections import deque
import random
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(lambda x: torch.stack(x), zip(*batch))
    
    def __len__(self):
        return len(self.buffer)