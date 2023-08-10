get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
import pandas as pd
import numpy as np
import gym
from agents.agent import DDPG

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# This is nice if you interupt the learning while rendering.
# This will automatically close the previous session.
try:
    agent.task.close()
except:
    pass
from ounoise import OUNoise
noise = OUNoise(size=1, mu=0.0, theta=0.15, sigma=0.2)

# env = gym.make('MountainCarContinuous-v0')
env = gym.make('Pendulum-v0')
agent = DDPG(env, gym=True) 

state = agent.task.reset()

scores = []
best_score = -np.inf
best_traj = []
step_count = 0
for i_episode in range(1000):
    # Learning
    state = agent.reset_episode() # start a new episode
    noise.reset()
    score = 0.0
    for _ in range(500):
        step_count += 1
        action = agent.act(state) + noise()
        next_state, reward, done, _ = agent.task.step(action)
        score += reward
        agent.step(action, reward, next_state, done)
        state = next_state
        agent.task.render()
        if done:
            break
    
    # Evaluate
#     if i_episode%1 == 0:
#         state = agent.reset_episode() # start a new episode
#         score = 0.0
#         for _ in range(300):
#             step_count += 1
#             action = agent.act(state) 
#             state, reward, done, _ = env.step(action)
#             score += reward
#             env.render()
#             if done:
#                 break

    scores.append(score)
    if score > best_score:
        best_score = agent.score
    print("\rEpisode: {:4d}, score: {:.3f}, best: {:.3f}, memory: {}".format(
        i_episode, score, best_score, len(agent.memory)), end="   ")

#     if score > 95:
#         break
    sys.stdout.flush()
env.close()

smooth = 21
plt.figure(figsize=(15,5))
plt.plot(np.convolve(scores, np.ones(smooth)/smooth)[(smooth-1)//2:-smooth], color='xkcd:azure')
plt.plot(scores, alpha=0.5, color='xkcd:cornflower')
plt.xlabel("Episode")
plt.ylabel('Total Reward')
plt.xlim(0, len(scores))
plt.grid(True)

ts = [scores[0]]
gamma = 0.95
alpha = 0.75
max_score = [scores[0]]
min_scores = [scores[0]]

def soft_update(new, old, gamma):
    return gamma*old + (1-gamma)*new

for s in scores:
    ts.append(gamma*ts[-1]+(1-gamma)*s)
    count.append(count[-1]+int(s>ts[-1])-alpha)
    max_score.append(soft_update(max(ts[-1], s), max_score[-1], gamma))
    min_scores.append(soft_update(min(ts[-1], s), min_scores[-1], gamma))

plt.plot(scores)
plt.plot(ts)
# plt.plot(max_score)
# plt.plot(min_scores)
plt.axvline(np.argmax(ts))
plt.grid(True)
plt.show()

# plt.plot((np.asarray(ts)-np.asarray(min_scores))/np.clip(np.asarray(max_score) - np.asarray(min_scores), 1, None))
plt.plot(np.asarray(scores)/np.asarray(ts[:-1]))
plt.ylim(-1,1)
plt.show()

np.argwhere(np.any(np.array(count)<0))

print(np.shape(agent.actor_local.model.get_weights()))
print(np.shape(agent.critic_local.model.get_weights()))

# env.close()
# env = gym.make('MountainCarContinuous-v0')
# state = env.reset()
# print(state)
state = agent.reset_episode()
Q = []
rewards = []
for i in range(300):
    action = agent.act(state) 
#     action = agent.actor_local.model.predict_on_batch([[state]])[0]
    Q.append(agent.critic_target.model.predict_on_batch([[state], [action]])[0])
    state, reward, done, _ = agent.task.step(action)
    
    rewards.append(reward)
#     agent.task.render()
    if done:
        break;
# agent.task.close()

env.close()

plt.plot(rewards)
# plt.plot(np.cumsum(rewards[::-1])[::-1])
plt.plot(Q)
plt.show()

len(agent.memory)

actions = np.linspace(-1,1, 100).reshape(-1,1)
states = np.ones([100,3])*[1, 0.00, 0.0]
Q = agent.critic_target.model.predict_on_batch([states, actions])
plt.plot(actions, Q)
plt.show()

agent.critic_target.model.predict_on_batch([[state], [action]])

smooth = 21
plt.figure(figsize=(15,5))
plt.plot(np.convolve(score, np.ones(smooth)/smooth)[(smooth-1)//2:-smooth], color='xkcd:azure')
plt.plot(score, alpha=0.5, color='xkcd:cornflower')
plt.xlabel("Episode")
plt.ylabel('Total Reward')
plt.xlim(0, len(score))
plt.grid(True)
plt.show()

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from ounoise import OUNoise
import matplotlib.pyplot as plt

noise = OUNoise(4, 0.0, 0.15, 1.0)

plt.plot([noise.state]+[noise() for _ in range(10)])
plt.show()

plt.plot([noise.state]+[noise() for _ in range(10)])
plt.show()

noise.reset()
plt.plot([noise.state]+[noise() for _ in range(10)])
plt.show()


env = gym.make('Pendulum-v0')
env.observation_space.low

import numpy as np

def ranked_policy(scores, epsilon=0.1):
    alpha = len(scores)/(1+epsilon)
    p = np.exp(np.linspace(alpha, 0.0, len(scores), endpoint=True))
#     score_range = np.abs(np.max(scores) - np.min(scores))
#     p = np.exp(, 0.0, len(scores), endpoint=True))
#     p = np.exp(np.linspace(np.min(scores), np.max(scores), len(scores), endpoint=True))
#     p = np.exp(np.linspace(epsilon+score_range, 0.0, len(scores), endpoint=True))
    p = p/np.sum(p)
    return p[np.argsort(scores)]
    

print(ranked_policy(np.array([-250, -150, -50])))
print(ranked_policy(np.array([-1, -1, -1, ])))
print(ranked_policy(np.array([-100, -100, -100])))
print(ranked_policy(np.array([-232, -233, -50])))

test = np.array([-150, -150, -23])

p = np.max(test) - test + 0.1*(np.max(test) - np.min(test))
p = p**2

p = p/np.sum(p)
p
# -np.logspace(np.log10(250), np.log10(100), 5)
trial_mu = np.linspace(-5, 5, 4)
trial_sigma = np.linspace(1e-6, 5.0, 6)

print(trial_mu)
print(trial_sigma)

test=np.array([1.1])

a = 0.0

a+= test
a

from collections import deque
from collections import namedtuple

memory = deque(maxlen=10)
experience = namedtuple("Experience", field_names=["x", "Q"])

for i in range(10):
    memory.append(experience(*np.random.normal(0, 1, 2)))

np.random.choice(np.arange(10), p=None)

import numpy as np

test = np.arange(10)
get_ipython().run_line_magic('timeit', 'np.square(test)')

get_ipython().run_line_magic('timeit', 'test ** 2')

get_ipython().run_line_magic('timeit', 'test * test')

from memory import ReplayBuffer, RingBuffer

batch_size = 256
memory_size = 1000000

std_mem = ReplayBuffer(memory_size, batch_size)
ring_mem = RingBuffer(memory_size, batch_size)

memory = []
for i in range(50000):
    state = np.random.normal(0.0, 1.0, 12)
    action = np.random.normal(0.0, 1.0, 4)
    reward = np.random.normal(0.0, 1.0)
    next_state = np.random.normal(0.0, 1.0, 12)
    done = np.random.randint(0,2)
    memory.append([state, action, reward, next_state, done])

get_ipython().run_cell_magic('timeit', '', 'for e in memory:   \n    std_mem.add(*e) ')

get_ipython().run_line_magic('timeit', '_ = std_mem.sample()')

get_ipython().run_cell_magic('timeit', '', 'for e in memory:    \n    ring_mem.add(*e)    ')

get_ipython().run_line_magic('timeit', '_ = ring_mem.sample(normalize=False)')
get_ipython().run_line_magic('timeit', '_ = ring_mem.sample(normalize=True)')

print(ring_mem.state_norm.mean)
print(ring_mem.state_norm.std)

hasattr(ring_mem, 'state_norm')
hasattr(std_mem, 'state_norm')

get_ipython().run_cell_magic('timeit', '', 'test = []\nfor e in memory:\n    test.append(e)')

get_ipython().run_cell_magic('timeit', '', 'test = [None for _ in range(memory_size)]\ni = 0\nfor e in memory:\n    test[i%memory_size] = e\n    i = i+1\n    if i > memory_size:\n        i=0')

get_ipython().run_cell_magic('timeit', '', 'size = 50000\ni = 0\nn = 0\ntest = []\nfor data in range(100000):\n    if n > i:\n        test[i] = data\n    else:\n        test.append(data)\n    i = i + 1\n    n = max(n, i)\n    if i >= size:\n        i = 0')

get_ipython().run_cell_magic('timeit', '', 'size = 50000\ni = 0\nn = 0\ntest = [None]*size\nfor data in range(100000):\n    test[i] = data\n    i = i + 1\n    n = max(n, i)\n    if i >= size:\n        i = 0\n# print(len(test), n)')

def reward_from_huber_loss(x, delta, max_reward=1, min_reward=0):
    return np.maximum(max_reward - delta*delta*(np.sqrt(1+(x/delta)**2) - 1), min_reward)

x = np.linspace(0, 3, 100)
plt.plot(x, reward_from_huber_loss(x, 1, 1, 0))
plt.show()
    

import numpy as np
np.tanh(100)



