import gym
env = gym.make('CartPole-v0')
env.reset()

for ep in range(10):
    env.reset()
    for _ in range(150):
        env.render()
        obv, reward, done, info = env.step(env.action_space.sample()) # take a random action

import gym
env = gym.make('CartPole-v0')
print ("Action Space: ", env.action_space)
print("Observation Space: ", env.observation_space)

# Input a cartPole observation state
# A box number ranging from 0-162
def get_Box(obv):
	x, x_dot, theta, theta_dot = obv
    
	if x < -.8:
		box_number = 0
	elif x < .8:
		box_number = 1
	else:
		box_number = 2

	if x_dot < -.5:
		pass
	elif x_dot < .5:
		box_number += 3
	else:
		box_number += 6

	if theta < np.radians(-12):
		pass
	elif theta < np.radians(-1.5):
		box_number += 9
	elif theta < np.radians(0):
		box_number += 18
	elif theta < np.radians(1.5):
		box_number += 27
	elif theta < np.radians(12):
		box_number += 36
	else:
		box_number += 45

	if theta_dot < np.radians(-50):
		pass
	elif theta_dot < np.radians(50):
		box_number += 54
	else:
		box_number += 108

	return box_number



# Explore Rate Decay Function, Converges to MIN_EXPLORE_RATE
def update_explore_rate(episode):
	return max(MIN_EXPLORE_RATE, min(1, 1.0 - np.log10((episode + 1) / 25)))

# Explore Rate Decay Function, Converges to MIN_LEARNING_RATE
def update_learning_rate(episode):
	return max(MIN_LEARNING_RATE, (min(0.5, 1.0 - np.log10((episode + 1) / 50))))

def update_action(state, explore_rate):
	if random.random() < explore_rate:
		return env.action_space.sample()
	else:
		return np.argmax(Q[state])

def q_learn():
	total_reward = 0
	total_completions = 0
	explore_rate = update_explore_rate(0)
	learning_rate = update_learning_rate(0)

	for i in range(NUM_EPISODES):
		observation = env.reset()
		state_0 = get_Box(observation)
		for _ in range(250):
			env.render()
			action = update_action(state_0, explore_rate)
			obv, reward, done, info = env.step(action)
			state_1 = get_Box(obv)

			q_max = np.max(Q[state_0])
			Q[state_0, action] += learning_rate*(reward + GAMMA*np.amax(Q[state_1])- Q[state_0, action])
			state_0 = state_1

			total_reward += reward

			if done:
				if _ > 192:
					total_completions += 1
				break

		learning_rate = update_learning_rate(i)
		explore_rate = update_explore_rate(i)
		print("Completions : ", total_completions)
		print("REWARD/TIME: ", total_reward/(i+1))
		print("Trial: ", i)
		#print("Final Q values: ", Q)

import numpy as np
import random
import gym

"""
lets define some constants first
"""
GAMMA = 0.99
NUM_EPISODES = 2000
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.05
Q = np.random.rand(162, env.action_space.n)  # 162 x 2 noisey matrix


q_learn()

# note "esc r y"  to clear output



