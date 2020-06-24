import gym
import time
env = gym.make('GridWorld-v0')
env.reset()
env.render()
time.sleep(10)