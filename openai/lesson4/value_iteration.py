
# coding: utf-8

# In[1]:


import numpy as np
from Env import GridWorld
# import gym

DISCOUNT_FACTOR = 0.9


class Agent:
    def __init__(self, env):
        self.env = env

    def value_evaluation(self, policy, V):
        V_new = np.copy(V)
        for s in range(self.env.nS):
            expected_value = 0
            for action, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in self.env.P[s][action]:
                    tempV= prob * (reward + DISCOUNT_FACTOR * V[next_state])
                    if tempV > V[s]:
                        V_new[s] = tempV
        
        return V_new

    
    def next_best_action(self, s, V):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + DISCOUNT_FACTOR * V[next_state])
        return np.argmax(action_values), np.max(action_values)
    
    def optimize(self):
        V = np.zeros(self.env.nS)
        policy = np.tile(np.eye(self.env.nA)[1], (self.env.nS, 1))

        THETA = 0.0001
        delta = float("inf")
        
        round_num = 0
        
        while delta>THETA:
            is_stable = True
            
            print("\nRound Number:" + str(round_num))
            round_num += 1
            
            print("Current Policy")
            print(np.reshape([env.get_action_name(entry) for entry in [np.argmax(policy[s]) for s in range(self.env.nS)]], self.env.shape))
            
            V_new = self.value_evaluation(policy,V)
            print("Expected Value accoridng to Value Evaluation")
            print(np.reshape(V_new, self.env.shape))

            diff = np.abs(V_new - V)

            delta =  np.max(diff)

            V=V_new
            
        policy = [self.next_best_action(self, s, V)[0] for s in range(self.env.nS)]
        return policy


env = GridWorld()
# env=gym.make('GridWorld-v0')
agent = Agent(env)
policy = agent.optimize()
print("\nBest Policy")
print(np.reshape([env.get_action_name(entry) for entry in policy], env.shape))


