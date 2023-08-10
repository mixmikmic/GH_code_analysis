def abs_deviation_stat(observed, expected):
    """ Computes the sum of absolute deviations from
        the expected to the observed.  observed and
        expected are both dictionaries that map from
        events to counts. """
    return sum(abs(observed[k] - expected[k]) for k in expected)

def chi2_stat(observed, expected):
    """ Computes the sum of chi2 statistic between
        the expected and the observed.  observed and
        expected are both dictionaries that map from
        events to counts. """
    return sum((observed[k] - expected[k])**2/expected[k] for k in expected)

# this is the data from ThinkStats2 Chapter 9
expected = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10}
observed = {1: 8, 2: 9, 3: 19, 4: 5, 5: 8, 6: 11}

observed_abs_deviation_stat = abs_deviation_stat(expected, observed)
print "observed absolute deviation stat", observed_abs_deviation_stat

observed_chi2_stat = chi2_stat(expected, observed)
print "observed chi2 stat", observed_chi2_stat

import sys
sys.path.append('../../ThinkStats2')
import thinkstats2
import numpy as np
from numpy.random import choice

def simulate_die(probs, n):
    """ Compute the histogram of outcomes from rolling die with
        faces given by the keys of the dictionary `probs` and
        probabilities given by the value sof the dictionary `probs`
    """
    rolls = choice(probs.keys(), n, p=probs.values())
    return thinkstats2.Hist(rolls)

def get_p_vals(obs_abs_stat, obs_chi2_stat, num_rolls, n_trials):
    """ Compute the p_values for both the absolute deviation statistic
        and the chi2 statistic using `n_trials` simulations.
        The function returns the two p-values as a tuple where
        the first element the p-value as computed using the absolute
        deviation statistic and the second is the p-value using the
        chi2 statistic. """
    trials_abs_stat = []
    trials_chi2_stat = []
    null_probs = {1: 1/6.0,
                  2: 1/6.0,
                  3: 1/6.0,
                  4: 1/6.0,
                  5: 1/6.0,
                  6: 1/6.0}
    for i in range(n_trials):
        rolls = simulate_die(null_probs, num_rolls)
        trials_abs_stat.append(abs_deviation_stat(rolls, expected))
        trials_chi2_stat.append(chi2_stat(rolls, expected))

    p_val_abs = np.mean([trial >= obs_abs_stat for trial in trials_abs_stat])
    p_val_chi2 = np.mean([trial >= obs_chi2_stat for trial in trials_chi2_stat])
    return p_val_abs, p_val_chi2

n_trials_for_p_value = 2000
p_val_abs, p_val_chi2 = get_p_vals(observed_abs_deviation_stat,
                                   observed_chi2_stat,
                                   sum(observed.values()),
                                   n_trials_for_p_value)

print "p_val_abs", p_val_abs
print "p_val_chi2", p_val_chi2

get_ipython().magic('matplotlib inline')
import thinkplot
import matplotlib.pyplot as plt

thinkplot.Pmf(thinkstats2.Pmf({1: .1, 2: (1-.1)/5,
                               2: (1-.1)/5,
                               3: (1-.1)/5,
                               4: (1-.1)/5,
                               5: (1-.1)/5,
                               6:(1-.1)/5}))
thinkplot.Pmf(thinkstats2.Pmf({1: (1-.6)/5,
                               2: .6,
                               3: (1-.6)/5,
                               4: (1-.6)/5,
                               5: (1-.6)/5,
                               6:(1-.6)/5}))
thinkplot.Pmf(thinkstats2.Pmf({1: (1-.2)/5,
                               2: (1-.2)/5,
                               3: (1-.2)/5,
                               4: .2,
                               5: (1-.2)/5,
                               6:(1-.2)/5}))

plt.xlabel('die outcome')
plt.ylabel('probability')
plt.show()

# we will vary the probability of the die outcome 1 from 0 to 1 in steps of .1
prob_1 = np.linspace(0, 1, 11)

# To smooth out variability we will repeat each alternate hypothesis 100 times
n_reps = 100
p_vals_abs = np.zeros((len(prob_1), n_reps))
p_vals_chi2 = np.zeros((len(prob_1), n_reps))

for i, p_1 in enumerate(prob_1):
    print i, p_1
    die_probs = {1: p_1}
    for j in range(2, 7):
        die_probs[j] = (1 - p_1)/5.0

    for t in range(n_reps):
        rolls = simulate_die(die_probs, sum(observed.values()))
        trial_abs_stat = abs_deviation_stat(rolls, expected)
        trial_chi2_stat = chi2_stat(rolls, expected)
        p_vals_abs[i, t], p_vals_chi2[i, t] = get_p_vals(trial_abs_stat,
                                                         trial_chi2_stat,
                                                         sum(observed.values()),
                                                         n_trials_for_p_value)

plt.plot(prob_1, [np.mean(p_vals_abs[i, :] > .05) for i in range(len(prob_1))])
plt.plot(prob_1, [np.mean(p_vals_chi2[i, :] > .05) for i in range(len(prob_1))])
plt.legend(['Abs Test', 'Chi2 Test'])
plt.xlabel('probability of die outcome 1')
plt.ylabel('probability of type 2 error')
plt.show()

from scipy.stats import dirichlet

n_random = 10
probs = []
p_vals_abs = np.zeros((n_random, n_reps))
p_vals_chi2 = np.zeros((n_random, n_reps))

for i in range(n_random):
    # sample some random probabilities from a Dirichlet distribution
    # https://en.wikipedia.org/wiki/Dirichlet_distribution
    probs.append(dirichlet.rvs([1]*6))
    for t in range(n_reps):
        die_probs = dict(zip(range(1,7), probs[-1][0]))
        rolls = simulate_die(die_probs, sum(observed.values()))
        trial_abs_stat = abs_deviation_stat(rolls, expected)
        trial_chi2_stat = chi2_stat(rolls, expected)
        p_vals_abs[i, t], p_vals_chi2[i, t] = get_p_vals(trial_abs_stat,
                                                         trial_chi2_stat,
                                                         sum(observed.values()),
                                                         n_trials_for_p_value)

type2_error_abs = [np.mean(p_vals_abs[i, :] > .05) for i in range(n_random)]
type2_error_chi2 = [np.mean(p_vals_chi2[i, :] > .05) for i in range(n_random)]

difference_in_errors = np.asarray(type2_error_abs) - np.asarray(type2_error_chi2)
print difference_in_errors
best_probs = probs[np.argmin(difference_in_errors)][0]
print best_probs

p_vals_abs = np.zeros((1000,))
p_vals_chi2 = np.zeros((1000,))

for t in range(1000):
    if t % 100 == 0:
        print t/1000.0
    die_probs = dict(zip(range(1,7), best_probs))
    rolls = simulate_die(die_probs, sum(observed.values()))
    trial_abs_stat = abs_deviation_stat(rolls, expected)
    trial_chi2_stat = chi2_stat(rolls, expected)
    p_vals_abs[t], p_vals_chi2[t] = get_p_vals(trial_abs_stat,
                                               trial_chi2_stat,
                                               sum(observed.values()),
                                               n_trials_for_p_value)

print "type 2 error abs", np.mean(p_vals_abs > .05)
print "type 2 error chi2", np.mean(p_vals_chi2 > .05)

