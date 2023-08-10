get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import random
import csv
import numpy as np
from tasks.task import Task
import sys
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# logging and plotting utilities
import util

from ounoise import OUNoise

class Basic_Agent():
    def __init__(self, task):
        self.task = task
        self.noise = OUNoise(size=1, mu=0.0, sigma=0.01, theta=0.15)
        
    def act(self):
        return self.noise.sample()

# Modify the values below to give the quadcopter a different starting position.
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
target_position = np.array([0.0, 0.0, 10.0])
file_output = 'demo_data.txt'                    # file name for saved results

# Setup
# Setup task and agent
task = Task(init_pose=init_pose, 
            target_pos=target_position,
            pos_noise=0.25,
            ang_noise=0.02, # about 6.8 deg at 6 sigma.
            vel_noise=0.25,
            ang_vel_noise=0.1
           )
basic_agent = Basic_Agent(task)

done = False
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 
          'reward']
results = {x : [] for x in labels}
states = []
# Run the simulation, and save the results.
task.reset()
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    score = 0.0
    while True:
        rotor_speeds = basic_agent.act()
        state, reward, done = task.step(rotor_speeds)
        score += reward
        to_write = [task.sim.time] 
        to_write += list(task.sim.pose) 
        to_write += list(task.sim.v) 
        to_write += list(task.sim.angular_v) 
        to_write += list(task.sim.rotor_speeds) 
        to_write += [reward]
        
        states.append(state)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        writer.writerow(to_write)
        if done:
            break
    print(score)
            
util.plot_run(results)

# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print("Position:    ", task.sim.pose[:3])
print("Orientation: ", task.sim.pose[3:])
print("Velocity:    ", task.sim.v)
print("Angular vel: ", task.sim.angular_v)

from agents.policy_search import PolicySearch_Agent

num_episodes = 1000

init_pose = np.array([0.0, 0.0, 10.0, 
                      0.0, 0.0, 0.0])
target_pos = np.array([0., 0., 10.])
task = Task(init_pose=init_pose, target_pos=target_pos)
agent = PolicySearch_Agent(task) 

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state) 
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
            break
    sys.stdout.flush()

results = util.log_run(agent, 'policy_search.csv')
util.plot_run(results)

from agents.agent import DDPG
from ounoise import OUNoise

# Settings
num_episodes = 100
runtime = 5.0
init_pose = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
target_position = np.array([0.0, 0.0, 10.0])

# Setup task and agent
task = Task(init_pose=init_pose, 
            target_pos=target_position,
            pos_noise=0.25,
            ang_noise=None, # 0.02 about 6.8 deg at 6 sigma.
            vel_noise=0.1,
            ang_vel_noise=None
           )



noise_annealing = 0.002
noise_min_sigma = 0.01

# Restart training?
if False:
    agent = DDPG(task)
    scores = []
    grades = []
    avg_reward = []
    best_score = -np.inf
    noise = OUNoise(task.action_size, mu=0.0, theta=0.15, sigma=0.1)
    
# Give agent control
for i_episode in range(1, num_episodes+1):
    # Run with added noise
    state = agent.reset_episode() # start a new episode
    noise.reset(noise_annealing, noise_min_sigma)
    score = 0.0
    steps = 0
    while True:
        action = agent.act(state) 
        action += noise.sample()
        action = np.clip(action, -1, 1)

        next_state, reward, done = agent.task.step(action)
        agent.step(action, reward, next_state, done)
        
        state = next_state
        score += reward
        steps += 1
        if done:
            break
            
    avg_reward.append(score/max(1, steps))
    scores.append(score)
    if score > best_score:
        best_score = score 

    text = "\r"
    text += "Episode: {:4d}, ".format(len(scores))
    text += "score: {:.1f}, ".format(score)
    text += "avg_score: {:.1f}, ".format(np.mean(scores[-25:]))
    text += "best: {:.1f}, ".format(best_score)
    text += "avg_reward: {:.1f}, ".format(avg_reward[-1])
    text += "memory: {}, ".format(len(agent.memory))
    text += "sigma: {:.3f}, ".format(noise.sigma)
    text += "  "
    print(text, end="")
    sys.stdout.flush()

smooth = 21
plt.figure(figsize=(15,5))
plt.plot(scores, '.', alpha=0.25, color='xkcd:blue')
plt.plot(np.convolve(scores, np.ones(smooth)/smooth)[(smooth-1)//2:-smooth], 
         color='xkcd:blue', 
         label='Total Reward')
plt.ylabel('Total Reward')
plt.legend(loc=2)
plt.grid(True)

# plt.twinx()
# plt.plot(avg_reward, '.', alpha=0.25, color='xkcd:green')
# plt.plot(np.convolve(avg_reward, np.ones(smooth)/smooth)[(smooth-1)//2:-smooth], 
#          color='xkcd:green', 
#          label='Avg Reward per Step')
# plt.ylabel('Average Reward per Step')
# plt.legend(loc=1)

plt.xlabel("Episode")
plt.xlim(0, len(scores))
# plt.ylim(np.mean(scores)-1*np.std(scores), np.mean(scores)+1*np.std(scores))
plt.show()

results = util.log_run(agent, 'DDPG.csv')
util.plot_run(results)

from util import grade
grade(agent, 'test.csv', 10)



