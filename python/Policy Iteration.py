get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

def init(env):
    """
    env: OpenAI Gym Environment
    """
    values = np.zeros(env.nS)
    policy = np.zeros((env.nS, env.nA))
    return values, policy

def policy_evaluation(env, values, policy, discount, theta):
    while True:
        delta = 0
        for s in range(env.nS): # For every state
            value = values[s]
            new_value = 0
            for a in range(env.nA): # For every action
                for transition, nextstate, reward, done in env.P[s][a]: # For every next state when action a is taken
                    transition *= policy[s][a] # Don't forget we need p(s',r|s,pi(s))
                    new_value += transition * (reward + discount * values[nextstate]) # Bellman optimality equation
            delta = max(delta, np.abs(value-new_value))
            values[s] = new_value
        if delta < theta:
            break
    return values

def policy_improvement(env, values, policy, discount):
    policy_stable = True
    for s in range(env.nS):
        old_action = np.argmax(policy[s])
        action_values = []
        for a in range(env.nA):
            action_value = 0
            for transition, nextstate, reward, done in env.P[s][a]:
                action_value += transition * (reward + discount * values[nextstate]) # Bellman optimality
            action_values.append(action_value)
        new_action = np.argmax(action_values) # Since we are dealing with policy take max action instead of summing them
        new_probs = np.zeros(env.nA)
        new_probs[new_action] += 1.0
        policy[s] = new_probs
        if old_action != new_action:
            policy_stable = False
    return policy_stable, policy

def policy_iteration(env, discount=0.9, theta=0.0001):
    policy_stable = False
    values, policy = init(env)
    while not policy_stable:
        values = policy_evaluation(env, values, policy, discount, theta)
        policy_stable, policy = policy_improvement(env, values, policy, discount)
    return policy

env = FrozenLakeEnv()
policy = policy_iteration(env)
done = False
state = env.reset()
env.render()
rewards = []
while not done:
    state, reward, done, _ = env.step(np.argmax(policy[state]))
    rewards.append(reward)
    env.render()

plt.plot(rewards, label="Reward")
plt.legend()



