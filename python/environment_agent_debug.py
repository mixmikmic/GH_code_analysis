import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import datetime

import gym
import trading_env

import os
import agent 
from os import __file__

#env = gym.make('trading-v0')
env_trading = gym.make('test_trading-v0')

date = datetime.datetime(2017, 5, 1, 0, 0)
env_trading.reset(date=date)
rewards = []
portfolio = []
while True:
    action = 1.0 #Holding
    s, r, done, _ = env_trading.step(action)
    rewards.append(r)
    portfolio.append(env_trading.portfolio_value)
    if done:
        break

plt.plot(portfolio)
plt.show()

agentSPG = agent.StochasticPolicyGradientAgent(env_trading, learning_rate = 1e-3, batch_size = 64, quiet = False) #Do not run this twice without reseting the Kernel!

#As a sanity check I try to overfit over the same step
#A profitable action in this case would be to buy (action=1)

date = datetime.datetime(2017, 5, 1, 0, 0)
rewards = []
losses = []
env_trading.start_fiat = 100
env_trading.start_crypto = 1

for i in range(5000):
    start = env_trading.reset(date=date)
    start = np.reshape(start,200)
    action = agentSPG.act([start])
    s, r, done, _ = env_trading.step(action)
    s = np.reshape(s,200)
    agentSPG.store_step(action, s, r)
    rewards.append(r)
    if i % 100 == 0:
        
        agentSPG.train()

#Mu tends to 1 and sigma to zero as expected
plt.plot(rewards)
plt.show()

agentDQN = agent.DQNAgent(env_trading)

date = datetime.datetime(2017, 5, 1, 0, 0)
rewards = []
losses = []
env_trading.start_fiat = 100
env_trading.start_crypto = 1

for i in range(10000):
    start = env_trading.reset(date=date)
    start = np.reshape(start,[1,200])
    action = agentDQN.act(start, i)
    next_state, reward, done, _ = env_trading.step(action - 1) #Converting class to action
    next_state = np.reshape(next_state, [1, 200])

    agentDQN.store_step(start, action, reward, next_state, done)
    
    state = next_state
    rewards.append(reward)
    if i % 100:
        agentDQN.train()

plt.plot(rewards)
plt.show()





