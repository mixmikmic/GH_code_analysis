get_ipython().run_line_magic('pylab', 'inline')
import pandas as pd
import numpy as np

from scipy.stats import truncnorm

n_players = 500000
skill = np.random.beta(10, 10, size=n_players)  # called "c"
outcome = np.random.binomial(1, p=skill)
history = [outcome]
position_dependence = .3  # called "w"
num_successes = outcome
for t in range(1, 50):
    p = skill*(1-position_dependence) + position_dependence * outcome
    outcome = np.random.binomial(1, p=p)
    history.append(outcome)
    num_successes += outcome

plt.plot(num_successes, skill, '.', alpha=.01);

n_players = 500000
skill = np.random.beta(10, 10, size=n_players)  # called "c"
position_dependence = np.random.beta(1, 1, size=n_players)  # called "w"

# first round
outcome = np.random.binomial(1, p=skill)
num_successes = outcome

# subsequent rounds
for t in range(1, 50):
    p = skill*(1-position_dependence) + position_dependence * outcome
    outcome = np.random.binomial(1, p=p)
    num_successes += outcome

plt.plot(num_successes, skill, '.', alpha=.005)
plt.ylabel('Skill')
plt.xlabel('Number of Successes');

n_players = 500000
skill = np.random.beta(10, 10, size=n_players)  # called "c"
position_dependence = np.random.beta(1, 1, size=n_players)  # called "w"

# first round
outcome = np.random.binomial(1, p=skill)
num_successes = outcome

# subsequent rounds
for t in range(1, 150):
    p = skill*(1-position_dependence) + position_dependence * outcome
    outcome = np.random.binomial(1, p=p)
    num_successes += outcome

plt.plot(num_successes, skill, '.', alpha=.005)
plt.ylabel('Skill')
plt.xlabel('Number of Successes');

plt.plot(position_dependence, num_successes, '.', alpha=.002)
plt.xlabel('Position Dependence')
plt.ylabel('Number of Successes');

n_players = 500000
skill = np.random.beta(10, 10, size=n_players)  # called "c"
position_dependence = np.random.beta(1, 1, size=n_players)  # called "w"

# first round
outcome = np.random.binomial(1, p=skill)
num_successes = outcome

# subsequent rounds
for t in range(1, 5000):
    p = skill*(1-position_dependence) + position_dependence * outcome
    outcome = np.random.binomial(1, p=p)
    num_successes += outcome

plt.plot(num_successes, skill, '.', alpha=.01)
plt.ylabel('Skill')
plt.xlabel('Number of Successes');

skill = np.random.beta(10, 10, size=n_players)  # called "c"
position_dependence0 = np.random.beta(1, 1, size=n_players)  # called "w"
weight = .5
position_dependence = weight*skill + (1-weight)*position_dependence

plt.plot(skill, position_dependence, '.', alpha=.005)
plt.xlabel('skill')
plt.ylabel('position dependence');

# first round
outcome = np.random.binomial(1, p=skill)
num_successes = outcome

# subsequent rounds
for t in range(1, 150):
    p = skill*(1-position_dependence) + position_dependence * outcome
    outcome = np.random.binomial(1, p=p)
    num_successes += outcome

plt.plot(num_successes, skill, '.', alpha=.005)
plt.ylabel('Skill')
plt.xlabel('Number of Successes');



