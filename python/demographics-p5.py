get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import shelve
import os
import scipy
import scipy.stats as stats
from itertools import combinations
# from IPython.display import display

plt.style.use('seaborn-dark')

clean_adult = pd.read_hdf('results/df1.h5', 'clean_adult/')
clean_adult.head()

get_ipython().run_line_magic('run', "-i 'functions/find_indices_with_value.py'")
get_ipython().run_line_magic('run', "-i 'functions/chi_square_test.py'")
chi_square_test(clean_adult, "income", "sex")

fig, ax = plt.subplots(nrows=1, ncols=1)

bar_width = 0.35

female = clean_adult[clean_adult["sex"] == "Female"]
male = clean_adult[clean_adult["sex"] == "Male"]

male_counts = male["income"].value_counts().sort_index()
male_percents = 100 * male_counts.values/male_counts.values.sum()
ind = np.arange(len(male_percents))
p1 = ax.bar(ind, male_percents, bar_width, color = "blue")

female_counts = female["income"].value_counts().sort_index()
female_percents = 100 * female_counts.values/female_counts.values.sum()
ind = np.arange(len(female_percents))
p2 = ax.bar(ind + bar_width, female_percents, bar_width, color = "red")

ax.set_title("Income by Sex", fontsize = 15)
ax.set_ylabel("Percentage of Group", fontsize = 15)
ax.set_xlabel("Sex", fontsize = 15)
plt.xticks(ind + bar_width/2, female_counts.index, fontsize = 12)

plt.legend((p1[0], p2[0]), ('Male', 'Female'))
plt.tight_layout()
plt.savefig("fig/income_by_sex.png")

get_ipython().run_line_magic('run', "-i 'functions/two_sample_t_test.py'")
clean_adult['income_p'] = 0
for i in range(len(clean_adult)):
    if clean_adult.iloc[i, 14] == ">50K":
        clean_adult.iloc[i, 15] = 1
race = clean_adult.groupby("race")
pairs = [",".join(map(str, comb)).split(",") for comb in combinations(race.groups.keys(), 2)]
for pair in pairs:
    race1_name = pair[0]
    race2_name = pair[1]
    race1 = race.get_group(pair[0])
    race2 = race.get_group(pair[1])
    two_sample_t_test(race1["income_p"], race2["income_p"], race1_name, race2_name)

race_groups = clean_adult.groupby("race")
nrows = round((len(race_groups) + 0.0001)/2)
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 8))
row = 0
column = 0
bar_width = 0.35
cmap = cm.get_cmap('viridis')

i = 0
for name, group in race_groups:
    if nrows > 1:
        ax = axes[row, column]
    else:
        ax = axes[column]
    counts = group["income"].value_counts().sort_index()
    percents = 100 * counts.values/counts.values.sum()
    ind = np.arange(len(percents))
    p = sns.barplot(ind, percents, ax =ax, color = cmap(i))
    ax.set_title("Income by Race", fontsize = 15)
    ax.set_ylabel("Percentage of Group", fontsize = 12)
    ax.set_xlabel("Income", fontsize = 12)
    ax.set_xticklabels(counts.index, fontsize = 8, rotation = 15)
    ax.set_title("".join(name) + " Income Percentages", y = 0.95, fontsize=12)

    column = 1 - column
    if column == 0:
        row += 1
    i += 0.2

if column == 1:
    axes[-1, -1].axis("off")
plt.tight_layout()
plt.savefig("fig/income_by_race") 

chi_square_test(clean_adult,"occupation", "income_p")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

below = clean_adult[clean_adult["income_p"] == 0]
above = clean_adult[clean_adult["income_p"] == 1]

above_counts = above["occupation"].value_counts().sort_index()
above_percents = 100 * above_counts.values/above_counts.values.sum()
ind = np.arange(len(above_percents))
p1 = ax.bar(ind, above_percents, bar_width, color = "blue")

below_counts = below["occupation"].value_counts().sort_index()
below_percents = 100 * below_counts.values/below_counts.values.sum()
ind = np.arange(len(below_percents))
p2 = ax.bar(ind + bar_width, below_percents, bar_width, color = "red")

ax.set_title("Occupation by Income", fontsize = 15)
ax.set_ylabel("Percentage of Group in Occupation", fontsize = 15)
ax.set_xlabel("Occupation", fontsize = 15)
plt.xticks(ind + bar_width/2, below_counts.index, fontsize = 12)

plt.legend((p1[0], p2[0]), ('>50k', '<=50k'))
plt.tight_layout()
plt.savefig("fig/occupation by income.png")

ax = sns.regplot(x = "hours.per.week", y = "income_p", data=clean_adult);
ax.set_title("Working hours per week vs Income");
plt.savefig("fig/hours_income.png")

#Print the summary of the linear regression method.
import statsmodels.formula.api as sm
Y = clean_adult.iloc[:,15]
X = clean_adult.iloc[:,12]
result = sm.OLS(Y, X).fit()
result.summary()

