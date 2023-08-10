from RL import ModelBasedRL
from RL import ModelFreeRL
from RL import SimulatedMDP
import gym

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')

env = gym.make('Taxi-v2')
mdp = SimulatedMDP(env)

# Created instance of class to run model based algorithms.
model_rl = ModelBasedRL(max_eval=100)

model_rl.value_iteration(mdp)

# Simulating using the policy that was learned and the last episode is displayed.
model_rl.simulate_policy(env)

# Plotting the returns at each episode.
model_rl.scatter_epsiode_returns()

# Changing number of iterations to speed up results. Results are still good without complete convergence.

# Number of evaluations for each state in policy evaluation.
model_rl.max_eval = 10

# Number of iterations of both policy evaluation and policy improvement.
model_rl.max_iter = 100

model_rl.policy_iteration(mdp)

# Simulating using policy that was learned and the last epsiode is displayed.
model_rl.simulate_policy(env)

# Plotting the returns at each episode.
model_rl.scatter_epsiode_returns()

model_rl.max_eval = 100

model_rl.q_value_iteration(mdp)

# Simulating using policy that was learned and the last epsiode is displayed.
model_rl.simulate_policy(env)

# Plotting the returns at each episode.
model_rl.scatter_epsiode_returns()

# Creating parameters needed for ModelFreeRL class.
n = env.observation_space.n
states = range(n)
m = env.action_space.n
actions = range(m)

# Model Free Reinforcement learning.
model_free_rl = ModelFreeRL(n=n, m=m, states=states, actions=actions)

# Running the q-learning algorithm.
model_free_rl.q_learning(env)

# Plotting the returns at each episode.
model_free_rl.plot_epsiode_returns()

# Plotting the epsilon parameters at each episode.
model_free_rl.plot_epsilon_parameters()

# Plotting the step size parameter for a state.
model_free_rl.plot_alpha_parameters(s=2)

# Simulating using policy that was learned and the last epsiode is displayed.
model_free_rl.simulate_policy(env)

# Plotting the returns at each episode.
model_free_rl.scatter_epsiode_returns()

# Running the q-learning algorithm.
model_free_rl.sarsa(env)

# Plotting the returns at each episode.
model_free_rl.plot_epsiode_returns()

# Plotting the epsilon parameters at each episode.
model_free_rl.plot_epsilon_parameters()

# Plotting the step size parameter for a state.
model_free_rl.plot_alpha_parameters(s=2)

# Simulating using policy that was learned and the last epsiode is displayed.
model_free_rl.simulate_policy(env)

# Plotting the returns at each episode.
model_free_rl.scatter_epsiode_returns()

env = gym.make('FrozenLake-v0')
# Creating parameters needed for ModelFreeRL class.
n = env.observation_space.n
states = range(n)
m = env.action_space.n
actions = range(m)

# Creating instance of the class, using epsilon greedy decay and step size decay.
model_free_rl = ModelFreeRL(n=n, m=m, states=states, actions=actions, epsilon=.3, 
                            alpha=.3, epsilon_decay_param=.0001, num_episodes=5000)

# Running the q-learning algorithm.
model_free_rl.q_learning(env)

# Plotting the returns at each episode.
model_free_rl.plot_epsiode_returns()

# Plotting the epsilon parameters at each episode.
model_free_rl.plot_epsilon_parameters()

# Plotting the step size parameter for a state.
model_free_rl.plot_alpha_parameters(s=2)

# Simulating using policy that was learned and the last epsiode is displayed.
model_free_rl.simulate_policy(env)

# Plotting the returns at each episode.
model_free_rl.scatter_epsiode_returns()

# Running the q-learning algorithm.
model_free_rl.q_learning(env)

# Plotting the returns at each episode.
model_free_rl.plot_epsiode_returns()

# Plotting the epsilon parameters at each episode.
model_free_rl.plot_epsilon_parameters()

# Plotting the step size parameter for a state.
model_free_rl.plot_alpha_parameters(s=2)

# Simulating using policy that was learned and the last epsiode is displayed.
model_free_rl.simulate_policy(env)

# Plotting the returns at each episode.
model_free_rl.scatter_epsiode_returns()

