class CordenateEnv:
    def reset(self):
        state = 0
        return state
    def step(self,action1, action2):
        reward = 10 if action1==1 and action2==2 else 0
        state = 0
        done  = True
        return state, reward, done