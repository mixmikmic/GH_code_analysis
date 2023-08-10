from random import choice

def simulate_fair_coin_flips(n):
    """ Return the number of heads that occur in n flips of a
        fair coin p(heads) = 0.5 """
    return sum([choice(['H', 'T']) == 'H' for i in range(n)])

print simulate_fair_coin_flips(250)

get_ipython().magic('matplotlib inline')
import thinkstats2
import thinkplot
import matplotlib.pyplot as plt

cdf = thinkstats2.Cdf([simulate_fair_coin_flips(250) for i in range(1000)])
thinkplot.Cdf(cdf)
plt.xlabel('Number of Heads')
plt.ylabel('Cumulative Probability')

(100 - cdf.PercentileRank(139))/100.0

def simulate_fair_coin_flips_two_sided(n):
    """ Return the number of heads or tails, whichever is larger,
        that occur in n flips of a fair coin p(heads) = 0.5 """
    flips = [choice(['H', 'T']) for i in range(n)]
    n_heads = sum([flip == 'H' for flip in flips])
    n_tails = sum([flip == 'T' for flip in flips])
    return max(n_heads, n_tails)

print simulate_fair_coin_flips_two_sided(250)

cdf = thinkstats2.Cdf([simulate_fair_coin_flips_two_sided(250) for i in range(1000)])
thinkplot.Cdf(cdf)
plt.xlabel('Number of Most Common Outcome')
plt.ylabel('Cumulative Probability')

(100 - cdf.PercentileRank(139))/100.0

import pandas as pd

data = pd.read_csv('../datasets/titanic_train.csv')
data = data.dropna(subset=['Age'])
data.head()

def compute_age_diff(data):
    """ Compute the absolute value of the difference in mean age
        between men and women on the titanic """
    groups = data.groupby('Sex')
    return abs(groups.get_group('male').Age.mean() -
               groups.get_group('female').Age.mean())

observed_age_diff = compute_age_diff(data)
print "observed age difference %f" % (observed_age_diff,)

from numpy.random import permutation

def shuffle_ages(data):
    """ Return a new dataframe (don't modify the original) where
        the values in the Age column have been randomly permuted. """
    data = data.copy()
    data.Age = permutation(data.Age)
    return data

compute_age_diff(shuffle_ages(data))

cdf = thinkstats2.Cdf([compute_age_diff(shuffle_ages(data)) for i in range(1000)])
thinkplot.Cdf(cdf)
plt.xlabel('Difference in mean Age (years)')
plt.ylabel('Cumulative Probability')

print "p-value is", (100 - cdf.PercentileRank(observed_age_diff))/100.0



