import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

from IPython.core.pylabtools import figsize
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')
np.random.seed(42)

# Define the multi-armed bandits
nb_bandits = 3  # Number of bandits
p_bandits = [0.45, 0.55, 0.60]  # True probability of winning for each bandit

def pull(i):
    """Pull arm of bandit with index `i` and return 1 if win, else return 0."""
    if np.random.rand() < p_bandits[i]:
        return 1
    else:
        return 0

# Run algorithim and plot priors from time to time
# Define plotting functions
plots = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000] # Plot at this iteration

plot_idx = 1
figsize(11.0, 10)
def plot(priors, step):
    global plot_idx
    plt.subplot(5, 2, plot_idx) 
    plot_x = np.linspace(0.001, .999, 100)
    for prior in priors:
        y = prior.pdf(plot_x)
        p = plt.plot(plot_x, y)
        plt.fill_between(plot_x, y, 0, alpha=0.2)
    plt.xlim([0,1])
    plt.ylim(ymin=0)
    plt.title('Priors at step {}'.format(step))
    plot_idx += 1

# Simulate multi-armed bandit process and update posterior
# The number of trials and wins will represent the prior for each
#  bandit with the help of the Beta distribution.
trials = [0, 0, 0]  # Number of times we tried each bandit
wins = [0, 0, 0]  # Number of wins for each bandit

n = 1000
# Run the trail for `n` steps
for step in range(1, n+1):
    # Define the prior based on current observations
    bandit_priors = [
        stats.beta(a=1+w, b=1+t-w) for t, w in zip(trials, wins)]
    # plot prior 
    if step in plots:
        plot(bandit_priors, step)
    # Sample a probability theta for each bandit
    theta_samples = [
        d.rvs(1) for d in bandit_priors
    ]
    # choose a bandit
    chosen_bandit = np.argmax(theta_samples)
    # Pull the bandit
    x = pull(chosen_bandit)
    # Update trials and wins (defines the posterior)
    trials[chosen_bandit] += 1
    wins[chosen_bandit] += x

plt.tight_layout()
plt.show()

