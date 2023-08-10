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

# %load two_sample_t_test.py
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

ax = sns.regplot(x = "age", y = "education.num", data=clean_adult);
ax.set_title("Years of Education since 4th Grade vs Age");
plt.savefig("fig/education_age.png")

ax = sns.regplot(x="age", y="hours.per.week", data=clean_adult);
ax.set_title("Hours worked per week vs Age");
plt.savefig("fig/hours_age.png")

clean_adult["hours.per.week"].mean()

overworked = clean_adult[clean_adult["hours.per.week"] > 60]
overworked.head()

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
ax = sns.boxplot(x="marital.status", y="hours.per.week", data = clean_adult, palette="muted", ax= axes[0])
ax.set_title("Hours per week worked vs Marital status (Overall)")
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

ax2 = sns.boxplot(x="marital.status", y="hours.per.week", data = overworked, palette="muted", ax = axes[1])
ax2.set_title("Hours per week worked vs Marital status (Overworked)")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 90)
plt.tight_layout()
plt.savefig("fig/marital_hours",  bbox_inches='tight')
            

print("Overall")
print()
marital = clean_adult.groupby("marital.status")
pairs = [",".join(map(str, comb)).split(",") for comb in combinations(marital.groups.keys(), 2)]
for pair in pairs:
    marital1_name = pair[0]
    marital2_name = pair[1]
    marital1 = marital.get_group(pair[0])
    marital2 = marital.get_group(pair[1])
    two_sample_t_test(marital1["education.num"], marital2["education.num"], marital1_name, marital2_name)

print("Overworked")
print()
marital = overworked.groupby("marital.status")
pairs = [",".join(map(str, comb)).split(",") for comb in combinations(marital.groups.keys(), 2)]
for pair in pairs:
    marital1_name = pair[0]
    marital2_name = pair[1]
    marital1 = marital.get_group(pair[0])
    marital2 = marital.get_group(pair[1])
    two_sample_t_test(marital1["education.num"], marital2["education.num"], marital1_name, marital2_name)

fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

ax = sns.boxplot(x="occupation", y="hours.per.week", data = clean_adult, palette="muted", ax = axes[0])
ax.set_title("Hours of work per week vs Occupation (Overall)")
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)

ax2 = sns.boxplot(x="occupation", y="hours.per.week", data = overworked, palette="muted", ax = axes[1])
ax2.set_title("Hours of work per week vs Occupation (Overworked)")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation = 90)
plt.tight_layout()
plt.savefig("fig/hours_occupation.png", bbox_inches='tight')

print("Overall")
print()
occupations = clean_adult.groupby("occupation")
pairs = [",".join(map(str, comb)).split(",") for comb in combinations(occupations.groups.keys(), 2)]
for pair in pairs:
    occ1_name = pair[0]
    occ2_name = pair[1]
    occ1 = occupations.get_group(pair[0])
    occ2 = occupations.get_group(pair[1])
    two_sample_t_test(occ1["education.num"], occ2["education.num"], occ1_name, occ2_name)

print("Overworked")
print()
occupations = overworked.groupby("occupation")
pairs = [",".join(map(str, comb)).split(",") for comb in combinations(occupations.groups.keys(), 2)]
for pair in pairs:
    occ1_name = pair[0]
    occ2_name = pair[1]
    occ1 = occupations.get_group(pair[0])
    occ2 = occupations.get_group(pair[1])
    two_sample_t_test(occ1["education.num"], occ2["education.num"], occ1_name, occ2_name)

