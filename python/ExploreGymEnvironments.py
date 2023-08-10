get_ipython().magic('matplotlib inline')

import gym
from matplotlib import pyplot as plt

breakout_env = gym.envs.make('Breakout-v0')

getEnvInfo('Breakout' , breakout_env)

invader_env = gym.envs.make('SpaceInvaders-v0')

getEnvInfo('SpaceInvaders', invader_env)

seaquest_env = gym.envs.make('Seaquest-v0')

getEnvInfo('Seaquest', seaquest_env)

frozen_env = gym.envs.make('FrozenLake-v0')

print('FrozenLake' + " Action space: {}".format(frozen_env.action_space.n))
observation = frozen_env.reset()
frozen_env.render()

def getEnvInfo(env_name, env):
    
    print(env_name + " Action space: {}".format(env.action_space.n))
    print(env.get_action_meanings())

    observation = env.reset()
    print("Image dimension: {}".format(observation.shape))

    plt.figure()
    plt.imshow(env.render(mode='rgb_array'))

    env.render(close=True)



