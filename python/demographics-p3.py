get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shelve
import os
import scipy.stats as stats
from itertools import combinations

plt.style.use('seaborn-dark')

clean_adult = pd.read_hdf('results/df1.h5', 'clean_adult/')
clean_adult.head()

ax = sns.boxplot(x="sex", y="education.num", data=clean_adult, palette="muted")
ax.set_title("Years of Education since 4th Grade vs. Sex");
plt.savefig("fig/education_sex.png")

ax = sns.boxplot(x="race", y="education.num", data=clean_adult, palette="muted")
ax.set_title("Years of Education since 4th Grade vs Race")
plt.savefig("fig/education_race.png")

def two_sample_t_test(group1, group2, group1_name, group2_name, alpha = 0.05):
    """Performs a two-sided two sample t-test to see if there is a difference in mean between
    the value of two groups. 
    
    Parameters
    ----------
    group1: Data for the first group. Can be list or array
    group2: Data for the second group. Can be list or array
    group1_name: Name of first group
    group2_name: Name of second group
    alpha: Significance level, default of 0.05 (Although this is very arbitrary as we saw in this class)
    
    
    Return
    ------
    (t, p, reject)
    t: the t-statistic
    p: the p-value
    reject: whether we reject the null hypothesis
    
    Example
    -------
    
    >>> group1 = [1, 2, 3]
    ... group2 = [1, 2, 3]
    ... two_sample_t_test(group1, group2, "group1", "group2")
    There is no statistically significant difference between Group group1 and Group group2
    (0.0, 1.0)    
    """    
        
    n1 = len(group1)
    n2 = len(group2)
    assert(n1 > 0)
    assert(n2 > 0)
    s12 = np.var(group1)
    s22 = np.var(group2)
    m1 = np.mean(group1)
    m2 = np.mean(group2)
    se = np.sqrt((s12/n1) + (s22/n2))
    df = (np.square(s12/n1 + s22/n2) / (( np.square(s12 / n1) / (n1 - 1) ) + (np.square(s22 / n2) / (n2 - 1)))).astype(int)
    t = ((m1 - m2)) / se
    p = stats.t.sf(np.abs(t), df)*2
    if (p < alpha):
        print("The mean difference is statistically significant for Group "  + group1_name +" and Group " + group2_name)
        print("p-value is " + str(p))
        print()
    else:
        print("There is no statistically significant difference between Group " + group1_name +" and Group " + group2_name)
        print()
    return (t, p, p < alpha)

male = clean_adult[clean_adult["sex"] == "Male"]
female = clean_adult[clean_adult["sex"] == "Female"]
t, p, reject = two_sample_t_test(male["education.num"], female["education.num"], "Male", "Female")

races = clean_adult.groupby("race")
pairs = [",".join(map(str, comb)).split(",") for comb in combinations(races.groups.keys(), 2)]
for pair in pairs:
    race1_name = pair[0]
    race2_name = pair[1]
    race1 = races.get_group(pair[0])
    race2 = races.get_group(pair[1])
    two_sample_t_test(race1["education.num"], race2["education.num"], race1_name, race2_name)

import unittest

class MyTests(unittest.TestCase):
    def test_same_population(self):
        group1 = [1, 2, 3]
        group2 = group1
        t, p, reject = two_sample_t_test(group1, group2, "group1", "group2")
        self.assertAlmostEqual(0, t)
        self.assertAlmostEqual(1, p)
        self.assertTrue(not reject)
    def test_obvious_difference(self):
        group1 = [1, 2, 3]
        group2 = [1000, 1001, 1001]
        t, p, reject = two_sample_t_test(group1, group2, "group1", "group2")
        self.assertAlmostEqual(0, p)
        self.assertTrue(reject)
    def test_significance_level(self):
        t, p, reject = two_sample_t_test([1, 2, 3], [4,9, 5], "group1", "group2", 0.1)
        self.assertAlmostEqual(0.1, p, places = 1)
        self.assertTrue(reject)
        t, p, reject = two_sample_t_test([1, 2, 3], [4,9, 5], "group1", "group2")
        self.assertAlmostEqual(0.1, p, places = 1)
        self.assertTrue(not reject)
    def test_same_population_different_order(self):
        group1 = [1, 2, 4]
        group2 = [2, 4, 1]
        t, p, reject = two_sample_t_test(group1, group2, "group1", "group2")
        self.assertAlmostEqual(0, t)
        self.assertAlmostEqual(1, p)

unittest.main(argv=["foo"], exit = False, verbosity = 2)

