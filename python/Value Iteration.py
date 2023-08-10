get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

def init(env):
    """
    env: OpenAI Gym Environment
    """
    values = np.zeros(env.nS)
    policy = np.zeros(env.nS) # This time policy is deterministic
    return values, policy

def value_iteration(env, discount=0.8, theta=0.00001):
    policy_stable = False
    values, policy = init(env)
    
    while True:
        delta = 0
        for s in range(env.nS): # For every state
            value = values[s]
            actions = []
            for a in range(env.nA): # For every action
                action_value = 0
                for transition, nextstate, reward, done in env.P[s][a]: # For every next state when action a is taken
                    action_value += transition * (reward + discount * values[nextstate]) # Bellman optimality equation
                actions.append(action_value)
            policy[s] = np.argmax(actions)
            new_value = max(actions)
            delta = max(delta, np.abs(value-new_value))
            values[s] = new_value
        if delta < theta:
            break
    policy = policy.astype(np.int8)
    return policy

env = FrozenLakeEnv()
policy = value_iteration(env)
done = False
state = env.reset()
env.render()
rewards = []
while not done:
    state, reward, done, _ = env.step(policy[state])
    rewards.append(reward)
    env.render()

plt.plot(rewards, label="Reward")
plt.legend()



