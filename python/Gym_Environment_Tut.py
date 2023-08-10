import gym  # open AI gym

env = gym.make('CartPole-v0')
# ignore the following warning. It is an existing issue on github for the environment

# reset the environment
env.reset()

box = env.observation_space
box

for low, high in zip(box.low, box.high):
    print("Range of observation value:", low, " <-> ", high)

actions = env.action_space
actions

import time  # for slowing down the render of the environment

env = gym.make('CartPole-v0')  # this is done again to make the cell re-runable
env.reset()
for _ in range(500):
    env.render(mode='rgb_array')
    env.step(actions.sample())
env.close()

