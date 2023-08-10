import gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mdptoolbox
import mdptoolbox.example

get_ipython().run_line_magic('matplotlib', 'inline')

#create CartPole Env
env = gym.envs.make("Acrobot-v1")

env.action_space.n

env.observation_space

def q_learning(n_s, n_obs, discount_factor):
    #create transition probability matrix and reward matrix 
    
    global P, R
    P, R = mdptoolbox.example.forest(S = 3)
    
    
    #Q Learning using mdptoolbox
    #parameters : transiton probability matrix,
    #rewards matrix ,discount factor
    global ql
    ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
    
    #run the policy iteration
    ql.run()
    
    global policy, Q, V
    policy = ql.policy
    Q = ql.Q
    V = ql.V
    q_md = ql.mean_discrepancy
    print(Q)
    print(V)
    print(policy)
    
    return policy, Q

rl_models = [
    ('Q Learning', q_learning(3,6,0.96))
 ]

