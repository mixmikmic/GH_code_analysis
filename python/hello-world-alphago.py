import gym
import numpy as np
env = gym.make('Go9x9-v0')

env.action_space

env.reset()
env.render()
game_done = False
while(not game_done):
    this_action = env.action_space.sample()
    observation, reward, game_done, info = env.step(this_action)        
    print info
    print game_done,this_action,reward

print 'hello world'



