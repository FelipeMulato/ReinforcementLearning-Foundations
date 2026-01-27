class CordenateEnv:
    def reset(self):
        return None
    def step(self,a1,a2):
        reward = 10 if a1==1 and a2 == 1 else 0
        done = True
        return None, reward, done
    