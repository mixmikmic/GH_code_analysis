import numpy as np
import pandas as pd
from math import tanh
import matplotlib.pyplot as plt
from itertools import product
from more_itertools import unique_everseen
from multiprocessing.dummy import Pool as ThreadPool

get_ipython().magic('matplotlib inline')

get_ipython().run_cell_magic('bash', '', '# remove "__custom_run" from the filename below to overwrite logs for plots\n\nfor i in 0 1 2\ndo\n    python -m smartcab.agent 1 1 1 False default glie > data.txt\n    python log_parser.py data.txt $i >> 512_glie_random_sample__custom_run.tsv\ndone')

def plot_rewards(file, sims=None, start_from_trial=0):
    moves = pd.read_csv(file, sep='\t')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.set_size_inches(16,5)

    for ax, sim in zip([ax1, ax2, ax3], sims or range(0,3)):
        moves_sim = moves[moves['simulation'] == sim]
        reward = moves_sim['total_reward']

        # Trendline for total reward per trial
        c = np.polyfit(moves_sim['trial'], reward, 2)
        trendline = [c[0] * x * x + c[1] * x + c[2] for x in range(0,100)]

        reaches = moves_sim[moves_sim['goal_reached'] == True]
        
        ax.set_xlim(0,100)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Total Reward after Trial')
        ax.set_title('Total Reward per Trial, sim #{}'.format(sim))
        ax.plot(moves_sim['trial'], reward, label='Rewards')
        ax.plot(range(0,100), trendline, label='Trend')
        ax.scatter(reaches['trial'], [ax.get_ylim()[1]] * len(reaches['goal_reached']), label='Goal Reached')
        ax.legend(loc=4)
        successful_reaches = moves_sim[moves_sim.trial >= start_from_trial].groupby('goal_reached').count().trial
        print 'Goal reached since trial {}, sim #{}: {}\n'.format(start_from_trial, sim, successful_reaches / (100 - start_from_trial))

def plot_visited_state_actions(file, take=4, legend_loc=2, states=None, sims=None):
    
    def compress_state_actions(state_action):
        return list(unique_everseen([str(k[:take]) for k in map(list, state_action.keys())]))
    
    moves = pd.read_csv(file, sep='\t')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.set_size_inches(16,3)

    for ax, sim in zip([ax1, ax2, ax3], sims or range(0,3)):
        moves_sim = moves[moves['simulation'] == sim]
        
        visited_state_actions = moves_sim['qtable'].apply(lambda x: len(eval(x).keys()))
        visited_states = moves_sim['qtable'].apply(lambda x: len(compress_state_actions(eval(x))))
        
        ax.set_xlim(0,100)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Elements')
        ax.set_title('Unique Q-Table Elements Visited, sim #{}'.format(sim))
        ax.plot(moves_sim['trial'], visited_state_actions, 'm-', label='State-actions')
        ax.plot(moves_sim['trial'], visited_states, 'b-', label='States')
        if states:
            state_actions = states * 4
            ax.set_ylim(0, state_actions * 1.1)
            ax.plot([0,100],[states,states], 'b--')
            ax.plot([0,100],[state_actions,state_actions], 'm--')
            
        ax.legend(loc=legend_loc)

def plot_errors(file):
    moves = pd.read_csv(file, sep='\t')

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.set_size_inches(16,3)

    for ax, sim in zip([ax1, ax2, ax3], range(0,3)):
        moves_sim = moves[moves['simulation'] == sim]
        
        errors = moves_sim['errors']
        
        ax.set_xlim(0,100)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Errors')
        ax.set_title('Errors, sim #{}'.format(sim))
        ax.plot(moves_sim['trial'], errors, 'm-', label='Errors')
        ax.legend()

plot_rewards('512_glie_random_sample.tsv')

plot_visited_state_actions('512_glie_random_sample.tsv', states=128)

plot_visited_state_actions('512_glie_random_sample.tsv')

rand_moves = pd.read_csv('512_glie_random_sample.tsv', sep='\t')
rand_moves_sum = rand_moves[['forward', 'left', 'right', 'None']].sum()
plt.figure()
rand_moves_sum.plot.bar(); plt.xlabel('Actions'); plt.ylabel('Sum'); plt.title('Random choice actions')

get_ipython().run_cell_magic('bash', '', '# remove "__custom_run" from the filename to overwrite logs for plots\n# to get a greedy behavior, set self.epsilon = 0; self.epsilon_decay = False\n\nfor i in 0 1 2\ndo\n    python -m smartcab.agent 0.3 0.7 0 False default glie > data.txt\n    python log_parser.py data.txt $i >> 512_greedy_out__custom_run.tsv\ndone')

plot_rewards('512_greedy_out.tsv')

plot_visited_state_actions('512_greedy_out.tsv')

q_moves = pd.read_csv('512_greedy_out.tsv', sep='\t')

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

fig.set_size_inches(16,5)
for ax, sim in zip([ax1, ax2, ax3], range(0,3)):
    moves_sim = q_moves[q_moves['simulation'] == sim]
    q_moves_sum = moves_sim[['forward', 'left', 'right', 'None']].sum()
    q_moves_sum.plot.bar(ax=ax)

# remove "__custom_run" from the filename to overwrite logs for plots

for i in range(25):
    get_ipython().system('python -m smartcab.agent 0.3 0.7 0 False default glie > data.txt')
    get_ipython().system('python log_parser.py data.txt {i} >> 512_25s_greedy_out__custom_run.tsv')

q_moves = pd.read_csv('512_25s_greedy_out.tsv', sep='\t')

plot_rewards('512_25s_greedy_out.tsv', sims=[2,10,20])

q_moves = pd.read_csv('512_25s_greedy_out.tsv', sep='\t')

fig, (ax1, ax2, ax3) = plt.subplots(1,3)

fig.set_size_inches(16,5)
for ax, sim in zip([ax1, ax2, ax3], [2,10,20]):
    moves_sim = q_moves[q_moves['simulation'] == sim]
    q_moves_sum = moves_sim[['forward', 'left', 'right', 'None']].sum()
    q_moves_sum.plot.bar(ax=ax)

t1 = np.linspace(0.0, 20, 20)
t2 = np.linspace(0.0, 100.0, 100)

fig, (ax1, ax2) = plt.subplots(1,2)

fig.set_size_inches(12,3)
for ax, t in zip([ax1, ax2], [t1, t2]):
    ax.set_title('P(t), t in [{}, {}]'.format(t[0],t[-1]))
    ax.plot(t,[tanh(x_**2/100) for x_ in t])

plot_visited_state_actions('48_3s_greedy_out.tsv', take=2, legend_loc=1, states=12)

plot_visited_state_actions('48_3s_explore_out.tsv', take=2, legend_loc=1, states=12)

plot_errors('512_greedy_0.3_0.7.tsv')

plot_errors('512_glie_basic_0.3_0.7.tsv')

# Calculate LRE from one simulation
def lr_errors_s(simulation_errors):
    errors = []
    for t, e in zip(range(0,100), simulation_errors):
         errors.append(tanh(t**2/100) * e)
    return sum(errors)

# Find the average LRE from all simulations in the file
def lre_all_sims(file):
    errors = []
    moves = pd.read_csv(file, sep='\t')
    for sim in set(moves['simulation']):
        moves_sim = moves[moves.simulation == sim]
        errors.append(lr_errors_s(moves_sim['errors']))
    return np.mean(errors)

# This generates 49 * 2 = 98 log files for a grid search

# Sometimes it may raise I/O exception, in this case just run the rest of the notebook manually
# or delete 512_glie_grid directory and try from this point again

alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
gammas = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

def run_simulator(vals):
    alpha, gamma = vals

    for sim in range(0,3):
        get_ipython().system('python -m smartcab.agent {alpha} {gamma} 1 True default glie > 512_glie_grid/512_{alpha}_{gamma}_raw.txt')
        get_ipython().system('python log_parser.py 512_glie_grid/512_{alpha}_{gamma}_raw.txt {sim} >> 512_glie_grid/512_{alpha}_{gamma}_data.tsv')

pool = ThreadPool(8)

get_ipython().system('mkdir 512_glie_grid')
pool.map(run_simulator, list(product(alphas, gammas)))

pool.close()

grid_search = []

for alpha, gamma in product(alphas, gammas):
    file = "512_glie_grid/512_{}_{}_data.tsv".format(alpha, gamma)
    lre = lre_all_sims(file)
    grid_search.append([alpha, gamma, lre])
    
grid_errors = pd.DataFrame(grid_search, columns=['alpha', 'gamma', 'lre'])

grid_errors

plot_rewards('512_add_runs_0.1_0.1.tsv')

# From now on, I'll use my logs
# If you run a grid search, you can use yours by plotting logs from '512_glie_grid' directory 

plot_errors('512_add_runs_0.1_0.1.tsv')

plot_errors("48_glie_add_runs_0.1_0.1.tsv")

plot_rewards("48_glie_add_runs_0.1_0.1.tsv", start_from_trial=2)

sa_sample = {('forward', 'red', None, None, 'forward'): 3, ('right', 'green', None, None, 'right'): 64, ('left', 'red', None, None, 'None'): 101, ('right', 'red', None, None, 'left'): 1, ('right', 'red', 'left', None, 'right'): 2, ('forward', 'red', 'right', None, 'None'): 1, ('left', 'green', None, 'left', 'left'): 1, ('left', 'red', None, None, 'right'): 1, ('right', 'red', 'forward', None, 'forward'): 1, ('right', 'red', 'left', None, 'None'): 2, ('left', 'red', 'left', None, 'forward'): 1, ('left', 'red', 'left', None, 'right'): 1, ('left', 'green', None, None, 'forward'): 1, ('forward', 'red', 'forward', None, 'right'): 1, ('right', 'green', 'left', None, 'left'): 1, ('forward', 'red', 'left', None, 'left'): 2, ('left', 'red', 'left', None, 'None'): 2, ('right', 'green', None, None, 'None'): 2, ('left', 'red', None, 'left', 'None'): 2, ('right', 'red', 'forward', None, 'left'): 1, ('forward', 'red', None, None, 'None'): 520, ('forward', 'red', 'left', None, 'right'): 1, ('right', 'red', 'left', 'left', 'forward'): 1, ('forward', 'green', None, 'forward', 'forward'): 1, ('forward', 'green', None, 'left', 'None'): 3, ('forward', 'green', None, 'right', 'left'): 1, ('forward', 'green', None, None, 'forward'): 340, ('forward', 'green', 'left', None, 'forward'): 2, ('forward', 'green', 'left', None, 'right'): 1, ('right', 'red', 'forward', None, 'None'): 2, ('left', 'green', None, None, 'right'): 2, ('forward', 'green', 'left', 'forward', 'forward'): 1, ('right', 'green', 'left', None, 'right'): 2, ('forward', 'red', None, None, 'right'): 1, ('left', 'red', None, None, 'forward'): 2, ('left', 'green', None, 'forward', 'forward'): 1, ('left', 'green', None, None, 'left'): 56, ('forward', 'red', 'left', None, 'None'): 2, ('right', 'red', 'left', None, 'forward'): 1, ('forward', 'green', None, 'right', 'None'): 1, ('forward', 'green', 'left', None, 'left'): 1, ('forward', 'green', 'left', None, 'None'): 6, ('forward', 'green', None, 'left', 'forward'): 5, ('forward', 'red', 'right', None, 'left'): 1, ('forward', 'red', None, None, 'left'): 1, ('right', 'red', None, None, 'None'): 1, ('left', 'red', None, None, 'left'): 2, ('left', 'green', None, 'forward', 'None'): 1, ('right', 'red', None, None, 'right'): 65, ('forward', 'red', 'left', None, 'forward'): 1, ('right', 'green', None, None, 'forward'): 2, ('right', 'green', 'forward', None, 'forward'): 1, ('forward', 'green', 'right', None, 'forward'): 2, ('forward', 'red', None, 'left', 'forward'): 1, ('right', 'green', 'left', None, 'None'): 1}

df = pd.DataFrame(map(lambda x: list(x[0]) + [x[1]], sa_sample.items()), columns=['waypoint', 'light', 'oncoming_traffic', 'left_traffic', 'action', 'count'])

df.sort_values('count', ascending=False).head(7)

lre_all_sims("48_glie_add_runs_0.1_0.1.tsv")

