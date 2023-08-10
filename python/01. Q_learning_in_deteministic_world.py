import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

# is_slippery가 False면 미끄럽지 않은 환경이 된다(Deterministic)
# 즉 움직이는 그대로 다음 state로 가게 됌.
register(
        id='FrozenLake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4',
                       'is_slippery' : False} 
)
env = gym.make('FrozenLake-v3')

get_ipython().magic('matplotlib inline')

Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 2000
dis = .99
rList = [] # create lists to contain total rewards and steps per episode
EGREEDY = True # True if e-greedy , False if add random_noise

for i_episode in range(num_episodes):
    state = env.reset()
    rAll = 0 # 모든 리워드?
    done = False
    e = 1. / ((i_episode//100)+1)
    
    while not done: # 한판 끝날 때 까지 해봐서 Q-value를 업데이트를 해나간다
        
        if EGREEDY:
            # e-greedy
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
        else:
            # add random_noise
            action =np.argmax(Q[state] + np.random.rand(1,env.action_space.n)/(i_episode+1))
        
        new_state, reward, done, _ = env.step(action)
        Q[state,action] = reward + dis*np.max(Q[new_state])
        
        rAll += reward
        state = new_state
    rList.append(rAll)

print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()



