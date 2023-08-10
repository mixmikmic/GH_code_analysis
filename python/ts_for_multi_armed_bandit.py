# plotting inline
get_ipython().run_line_magic('matplotlib', 'inline')

# working directory
import os; os.chdir('/home/gdmarmerola/ts_demo')

# importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta as beta_dist
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm

# turning off automatic plot showing, and setting style
plt.ioff()
plt.style.use('fivethirtyeight')

# class for our row of bandits
class MAB:
    
    # initialization
    def __init__(self, bandit_probs):
        
        # storing bandit probs
        self.bandit_probs = bandit_probs
        
    # function that helps us draw from the bandits
    def draw(self, k):

        # we return the reward and the regret of the action
        return np.random.binomial(1, self.bandit_probs[k]), np.max(self.bandit_probs) - self.bandit_probs[k]

# defining a set of bandits with known probabilites
bandit_probs = [0.35, 0.40, 0.30, 0.25]

# instance of our MAB class
mab = MAB(bandit_probs)

# number of draws
N_DRAWS = 200

# number of bandits
N_BANDITS = len(mab.bandit_probs)

# numpy arrays for accumulating draws, bandit choices and rewards, more efficient calculations
k_array = np.zeros((N_BANDITS,N_DRAWS))
reward_array = np.zeros((N_BANDITS,N_DRAWS))

# lists for ease of use, visualization
k_list = []
reward_list = []

# opening figure and setting style
fig, ax = plt.subplots(figsize=(9, 3), dpi=150)
ax.set(xlim=(-1, N_DRAWS), ylim=(-0.5, N_BANDITS-0.5))

# colors for each bandit
bandit_colors = ['red', 'green', 'blue', 'purple']

# loop generating draws
for draw_number in range(N_DRAWS):
    
    # choosing arm and drawing
    k = np.random.choice(range(N_BANDITS),1)[0]
    reward, regret = mab.draw(k)
    
    # record information about this draw
    k_list.append(k)
    reward_list.append(reward)
    k_array[k, draw_number] = 1
    reward_array[k, draw_number] = reward
      
    # getting list of colors that tells us the bandit
    color_list = [bandit_colors[k] for k in k_list]
    
    # getting list of facecolors that tells us the reward
    facecolor_list = [['none', bandit_colors[k_list[i]]][r] for i, r in enumerate(reward_list)]    
    
# initializing with first data
scatter = ax.scatter(y=[k_list[0]], x=[list(range(N_DRAWS))[0]], color=[color_list[0]], linestyle='-', marker='o', s=30, facecolor=[facecolor_list[0]]);

# titles
plt.title('Random draws from the row of slot machines (MAB)', fontsize=10)
plt.xlabel('Round', fontsize=10); plt.ylabel('Bandit', fontsize=10);
ax.set_yticks([0,1,2,3])
ax.set_yticklabels(['{}\n($\\theta = {}$)'.format(i, bandit_probs[i]) for i in range(4)])
ax.tick_params(labelsize=10)
fig.tight_layout()

# function for updating
def animate(i):
    x = list(range(N_DRAWS))[:i]
    y = k_list[:i]
    scatter.set_offsets(np.c_[x, y])
    scatter.set_color(color_list[:i])
    scatter.set_facecolor(facecolor_list[:i])
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(['{}\n($\\theta = {}$)'.format(i, bandit_probs[i]) for i in range(4)])
    ax.tick_params(labelsize=10)
    return (scatter,)

# function for creating animation
anim = FuncAnimation(fig, animate, frames=N_DRAWS, interval=200, blit=True)

# showing
HTML(anim.to_html5_video())

# clearing past figures
plt.close('all')

# examples 
beta_examples = [(0,0),(2,0),(2,5),(18,9),(32,50),(130,80)]

# colors for the plots
beta_colors = ['blue','orange','green','red','purple','brown']

# opening figure
fig, ax = plt.subplots(figsize=(14, 6), dpi=150, nrows=2, ncols=3)

# loop for each 
for i, example in enumerate(beta_examples):
    
    # points to sample for drawing the curve
    X = np.linspace(0,1,1000)
    
    # generating the curve
    dist = beta_dist(1 + example[0],1 + example[1])
    curve = dist.pdf(X)
    
    # plotly data
    ax[int(i/3)][(i % 3)].fill_between(X, 0, curve, color=beta_colors[i], alpha=0.7)
    ax[int(i/3)][(i % 3)].set_title('Sucesses: {} | Failures: {}'.format(example[0],example[1]), fontsize=14)

# some adjustments
ax[0][0].set(ylim=[0,2])
plt.tight_layout()
#plt.

# showing the figure
plt.show()

# let us create a function that returns the pdf for our beta posteriors
def get_beta_pdf(alpha, beta):
    X = np.linspace(0,1,1000)
    return X, beta_dist(1 + alpha,1 + beta).pdf(X)

# for now, let us perform random draws
def random_policy(k_array, reward_array, n_bandits):
    return np.random.choice(range(n_bandits),1)[0]

# let us wrap a function that draws the draws and distributions of the bandit experiment
def plot_MAB_experiment(decision_policy, N_DRAWS, bandit_probs, plot_title):

    # clearing past figures
    plt.close('all')

    # number of bandits
    N_BANDITS = len(bandit_probs)
    
    # numpy arrays for accumulating draws, bandit choices and rewards, more efficient calculations
    k_array = np.zeros((N_BANDITS,N_DRAWS))
    reward_array = np.zeros((N_BANDITS,N_DRAWS))
    
    # lists for accumulating draws, bandit choices and rewards
    k_list = []
    reward_list = []

    # animation dict for the posteriors
    posterior_anim_dict = {i:[] for i in range(N_BANDITS)}

    # opening figure
    fig = plt.figure(figsize=(9,5), dpi=150)

    # let us position our plots in a grid, the largest being our plays
    ax1 = plt.subplot2grid((5, 4), (0, 0), colspan=4, rowspan=3)
    ax2 = plt.subplot2grid((5, 4), (3, 0), rowspan=2)
    ax3 = plt.subplot2grid((5, 4), (3, 1), rowspan=2)
    ax4 = plt.subplot2grid((5, 4), (3, 2), rowspan=2)
    ax5 = plt.subplot2grid((5, 4), (3, 3), rowspan=2)

    # loop generating draws
    for draw_number in range(N_DRAWS):

        # record information about this draw
        k = decision_policy(k_array, reward_array, N_BANDITS)
        reward, regret = mab.draw(k)

        # record information about this draw
        k_list.append(k)
        reward_list.append(reward)
        k_array[k, draw_number] = 1
        reward_array[k, draw_number] = reward
        
        # sucesses and failures for our beta distribution
        success_count = reward_array.sum(axis=1)
        failure_count = k_array.sum(axis=1) - success_count
        
        # calculating pdfs for each bandit
        for bandit_id in range(N_BANDITS):

            # pdf
            X, curve = get_beta_pdf(success_count[bandit_id], failure_count[bandit_id])

            # appending to posterior animation dict
            posterior_anim_dict[bandit_id].append({'X': X, 'curve': curve})

        # getting list of colors that tells us the bandit
        color_list = [bandit_colors[k] for k in k_list]

        # getting list of facecolors that tells us the reward
        facecolor_list = [['none', bandit_colors[k_list[i]]][r] for i, r in enumerate(reward_list)]

    # fixing properties of the plots
    ax1.set(xlim=(-1, N_DRAWS), ylim=(-0.5, N_BANDITS-0.5))
    ax1.set_title(plot_title, fontsize=10)
    ax1.set_xlabel('Round', fontsize=10); ax1.set_ylabel('Bandit', fontsize=10)
    ax1.set_yticks([0,1,2,3])
    ax1.set_yticklabels(['{}\n($\\theta = {}$)'.format(i, bandit_probs[i]) for i in range(4)])
    ax1.tick_params(labelsize=10)

    # titles of distribution plots
    ax2.set_title('Estimated $\\theta_0$', fontsize=10); ax3.set_title('Estimated $\\theta_1$', fontsize=10); 
    ax4.set_title('Estimated $\\theta_2$', fontsize=10); ax5.set_title('Estimated $\\theta_3$', fontsize=10);

    # initializing with first data
    scatter = ax1.scatter(y=[k_list[0]], x=[list(range(N_DRAWS))[0]], color=[color_list[0]], linestyle='-', marker='o', s=30, facecolor=[facecolor_list[0]]);
    dens1 = ax2.fill_between(posterior_anim_dict[0][0]['X'], 0, posterior_anim_dict[0][0]['curve'], color='red', alpha=0.7)
    dens2 = ax3.fill_between(posterior_anim_dict[1][0]['X'], 0, posterior_anim_dict[1][0]['curve'], color='green', alpha=0.7)
    dens3 = ax4.fill_between(posterior_anim_dict[2][0]['X'], 0, posterior_anim_dict[2][0]['curve'], color='blue', alpha=0.7)
    dens4 = ax5.fill_between(posterior_anim_dict[3][0]['X'], 0, posterior_anim_dict[3][0]['curve'], color='purple', alpha=0.7)

    # titles
    #plt.title('Random draws from the row of slot machines (MAB)', fontsize=10)
    #plt.xlabel('Round', fontsize=10); plt.ylabel('Bandit', fontsize=10);

    # function for updating
    def animate(i):

        # clearing axes
        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear(); ax5.clear();

        # updating game rounds
        scatter = ax1.scatter(y=k_list[:i], x=list(range(N_DRAWS))[:i], color=color_list[:i], 
                              linestyle='-', marker='o', s=30, facecolor=facecolor_list[:i]);

        # fixing properties of the plot
        ax1.set(xlim=(-1, N_DRAWS), ylim=(-0.5, N_BANDITS-0.5))
        ax1.set_title(plot_title, fontsize=10)
        ax1.set_xlabel('Round', fontsize=10); ax1.set_ylabel('Bandit', fontsize=10)
        ax1.set_yticks([0,1,2,3])
        ax1.set_yticklabels(['{}\n($\\theta = {}$)'.format(i, bandit_probs[i]) for i in range(4)])
        ax1.tick_params(labelsize=10)

        # updating distributions
        dens1 = ax2.fill_between(posterior_anim_dict[0][i]['X'], 0, posterior_anim_dict[0][i]['curve'], color='red', alpha=0.7)
        dens2 = ax3.fill_between(posterior_anim_dict[1][i]['X'], 0, posterior_anim_dict[1][i]['curve'], color='green', alpha=0.7)
        dens3 = ax4.fill_between(posterior_anim_dict[2][i]['X'], 0, posterior_anim_dict[2][i]['curve'], color='blue', alpha=0.7)
        dens4 = ax5.fill_between(posterior_anim_dict[3][i]['X'], 0, posterior_anim_dict[3][i]['curve'], color='purple', alpha=0.7)

        # titles of distribution plots
        ax2.set_title('Estimated $\\theta_0$', fontsize=10); ax3.set_title('Estimated $\\theta_1$', fontsize=10); 
        ax4.set_title('Estimated $\\theta_2$', fontsize=10); ax5.set_title('Estimated $\\theta_3$', fontsize=10);
        
        # do not need to return 
        return ()

    # function for creating animation
    anim = FuncAnimation(fig, animate, frames=N_DRAWS, interval=200, blit=True)

    # fixing the layout
    fig.tight_layout()

    # showing
    return HTML(anim.to_html5_video())

# let us plot
plot_MAB_experiment(random_policy, 200, mab.bandit_probs, 'Random draws from the row of slot machines (MAB)')

# e-greedy policy
class eGreedyPolicy:
    
    # initializing
    def __init__(self, epsilon):
        
        # saving epsilon
        self.epsilon = epsilon
    
    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):
        
        # sucesses and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)
        
        # ratio of sucesses vs total
        success_ratio = success_count/total_count
        
        # choosing best greedy action or random depending with epsilon probability
        if np.random.random() < self.epsilon:
            
            # returning random action, excluding best
            return np.random.choice(np.delete(list(range(N_BANDITS)), np.argmax(success_ratio)))
        
        # else return best
        else:
            
            # returning best greedy action
            return np.argmax(success_ratio)    

# instance of this class, let us use 0.10 for the random action probability
e_greedy_policy = eGreedyPolicy(0.10)

# let us plot
plot_MAB_experiment(e_greedy_policy.choose_bandit, 200, mab.bandit_probs, '$\epsilon$-greedy decision policy: constant exploration and exploitation')

# e-greedy policy
class UCBPolicy:
    
    # initializing
    def __init__(self):
        
        # nothing to do here
        pass
    
    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):
        
        # sucesses and total draws
        success_count = reward_array.sum(axis=1)
        total_count = k_array.sum(axis=1)
        
        # ratio of sucesses vs total
        success_ratio = success_count/total_count
        
        # computing square root term
        sqrt_term = np.sqrt(2*np.log(np.sum(total_count))/total_count)
        
        # returning best greedy action
        return np.argmax(success_ratio + sqrt_term)    

# instance of UCB class
ucb_policy = UCBPolicy()

# let us plot'
plot_MAB_experiment(ucb_policy.choose_bandit, 200, mab.bandit_probs, 'UCB decision policy: using uncertainty and optimism')

# e-greedy policy
class TSPolicy:
    
    # initializing
    def __init__(self):
        
        # nothing to do here
        pass
    
    # choice of bandit
    def choose_bandit(self, k_array, reward_array, n_bandits):

        # list of samples, for each bandit
        samples_list = []
        
        # sucesses and failures
        success_count = reward_array.sum(axis=1)
        failure_count = k_array.sum(axis=1) - success_count
                    
        # drawing a sample from each bandit distribution
        samples_list = [np.random.beta(1 + success_count[bandit_id], 1 + failure_count[bandit_id]) for bandit_id in range(n_bandits)]
                                
        # returning bandit with best sample
        return np.argmax(samples_list)    

# instance of UCB class
ts_policy = TSPolicy()

# let us plot
plot_MAB_experiment(ts_policy.choose_bandit, 200, mab.bandit_probs, 'Thompson Sampling decision policy: probability matching')

# number of rounds
N_ROUNDS = 10000

# number of simulations
N_SIMULATIONS = 1000

# dict storing each decision policy
dp_dict = {'e_greedy': e_greedy_policy.choose_bandit, 
           'ucb': ucb_policy.choose_bandit, 
           'ts': ts_policy.choose_bandit}

# dict storing results for each algorithm and simulation
results_dict = {'e_greedy': {'k_array': np.zeros((N_BANDITS,N_ROUNDS)),
                             'reward_array': np.zeros((N_BANDITS,N_ROUNDS)),
                             'regret_array': np.zeros((1,N_ROUNDS))[0]},
                'ucb':  {'k_array': np.zeros((N_BANDITS,N_ROUNDS)),
                         'reward_array': np.zeros((N_BANDITS,N_ROUNDS)),
                         'regret_array': np.zeros((1,N_ROUNDS))[0]},
                'ts':  {'k_array': np.zeros((N_BANDITS,N_ROUNDS)),
                        'reward_array': np.zeros((N_BANDITS,N_ROUNDS)),
                        'regret_array': np.zeros((1,N_ROUNDS))[0]}}

# loop for each algorithm
for key, decision_policy in dp_dict.items():
    
    # printing progress
    print(key, decision_policy)
    
    # loop for each simulation
    for simulation in tqdm(range(N_SIMULATIONS)):
        
        # numpy arrays for accumulating draws, bandit choices and rewards, more efficient calculations
        k_array = np.zeros((N_BANDITS,N_ROUNDS))
        reward_array = np.zeros((N_BANDITS,N_ROUNDS))
        regret_array = np.zeros((1,N_ROUNDS))[0]
       
        # loop for each round
        for round_id in range(N_ROUNDS):

            # choosing arm nad pulling it
            k = decision_policy(k_array, reward_array, N_BANDITS)
            reward, regret = mab.draw(k)
            
            # record information about this draw
            k_array[k, round_id] = 1
            reward_array[k, round_id] = reward
            regret_array[round_id] = regret
            
        # results for the simulation
        results_dict[key]['k_array'] += k_array
        results_dict[key]['reward_array'] += reward_array
        results_dict[key]['regret_array'] += regret_array

# closing all past figures
plt.close('all')

# opening figure to plot regret
plt.figure(figsize=(10, 3), dpi=150)

# loop for each decision policy
for policy in ['e_greedy','ucb','ts']:
    
    # plotting data
    plt.plot(np.cumsum(results_dict[policy]['regret_array']/N_SIMULATIONS), label=policy, linewidth=1.5);
    
# adding title
plt.title('Comparison of cumulative regret for each method in {} simulations'.format(N_SIMULATIONS), fontsize=10)

# adding legend
plt.legend(fontsize=8); plt.xticks(fontsize=10); plt.yticks(fontsize=10)

# showing plot
plt.show()

# closing all past figures
plt.close('all')

# opening figure to plot regret
plt.figure(figsize=(10, 3), dpi=150)

# colors for each bandit
bandit_colors = ['red', 'green', 'blue', 'purple']

# loop for each decision policy
for i, policy in enumerate(['e_greedy','ucb','ts']):
    
    # subplots
    plt.subplot(1,3,i+1)
    
    # loop for each arm
    for arm in range(N_BANDITS):
    
        # plotting data
        plt.plot(results_dict[policy]['k_array'][arm]/N_SIMULATIONS, label='Bandit {}'.format(arm), linewidth=1.5, color=bandit_colors[arm]);
    
        # adding title
        plt.title('Algorithm {} proportion of arm selection'.format(policy), fontsize=8)

        # adding legend
        plt.legend(fontsize=8); plt.xticks(fontsize=8); plt.yticks(fontsize=8); plt.ylim([-0.1,1.1])

# showing plot
plt.show()



