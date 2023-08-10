get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shelve
import os
from IPython.display import display

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

adult = pd.read_csv("data/adult.csv")
adult.shape

adult.head()

for column in adult.columns:
    print("Unique Values in Column " + column)
    display(adult[column].unique())
    print("")

def find_indices_with_value(df, value):
    """Return the row indices of a Pandas Dataframe where the row contains the given value.
    
    Parameters
    ----------
    
    df: Pandas Dataframe
    value: value we want to see if row contains
    
    Return
    ------
    array
        An integer array where each value is the row index of a row that contains the value
        
    Example
    -------
    
    >>> row0 = [">", "?", "!"]
    ... df = pd.DataFrame([row0])
    ... find_indices_with_value(df, "?")
    array([0])
    """    
    
    rows = []
    for index, row in df.iterrows():
        if value in row.unique():
            rows.append(index)
    return np.array(rows)

missing_ind = find_indices_with_value(adult, "?")
len(missing_ind)

clean_adult = adult.drop(missing_ind).reset_index(drop = True)
clean_adult.head()

print("Adult")
display(adult.describe())
print("")
print("Clean Adult")
display(clean_adult.describe())

males = (adult["sex"] == "Male")
clean_males = (clean_adult["sex"] == "Male")
print("Adult Male Percentage: " + str(males.sum() / len(males)))
print("Clean Adult Male Percentage: " + str(clean_males.sum() / len(clean_males)))
white = (adult["race"] == "White")
clean_white = (clean_adult["race"] == "White")
print("Adult White Percentage: " + str(white.sum() / len(white)))
print("Clean Adult White Percentage: " + str(clean_white.sum() / len(clean_white)))

fig, axes = plt.subplots(nrows=3, ncols=2)

ax1 = axes[0, 0]
sns.distplot(clean_adult["age"], bins = 20, ax = ax1)
ax1.set_title("Distribution of Age")
ax1.set_xlabel("Age", fontsize = 8)
ax1.set_ylabel("Relative Frequency")
ax1.grid(False)

ax2 = axes[0, 1]
sns.distplot(clean_adult["education.num"], bins = 18, ax = ax2)
ax2.set_title("Distribution of Years of Education")
ax2.set_xlabel("Years of Education", fontsize = 8)
ax2.set_ylabel("Relative Frequency")
ax2.grid(False)

ax3 = axes[1, 0]
counts = clean_adult["marital.status"].value_counts()
percents = 100 * counts.values/counts.values.sum()
ind = np.arange(len(percents))  
width = 0.35  
rects = ax3.bar(ind, percents, width)
ax3.set_ylabel("Percentage of Individuals")
ax3.set_xticks(ind)
ax3.set_xticklabels(counts.index, fontsize = 8, rotation = 15)
ax3.set_title("Marital Status", y = 0.95)
ax3.grid(False)

ax4 = axes[1, 1]
counts = clean_adult["race"].value_counts()
percents = 100 * counts.values/counts.values.sum()
ind = np.arange(len(percents))  
width = 0.35  
rects = ax4.bar(ind, percents, width)
ax4.set_ylabel("Percentage of Individuals")
ax4.set_xticks(ind)
ax4.set_xticklabels(counts.index, fontsize = 8)
ax4.set_title("Race", y = 0.95)
ax4.grid(False)

ax5 = axes[2, 0]
counts = clean_adult["sex"].value_counts()
percents = 100 * counts.values/counts.values.sum()
ind = np.arange(len(percents))  
width = 0.35  
rects = ax5.bar(ind, percents, width)
ax5.set_ylabel("Percentage of Individuals")
ax5.set_xticks(ind)
ax5.set_xticklabels(counts.index, fontsize = 10)
ax5.set_title("Sex", y = 0.95)
ax5.grid(False)

ax6 = axes[2, 1]
counts = clean_adult["native.country"].value_counts()
percents = 100 * counts.values/counts.values.sum()
patches, texts = ax6.pie(percents, shadow=True, startangle=90)
ax6.axis("equal")
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(counts.index, percents)]
ax6.legend(patches, labels, loc="right", ncol = 2, bbox_to_anchor=(1.6, 0.4),
           fontsize=8)
ax6.set_title("Native Country")
ax6.grid(False)

os.makedirs("fig", exist_ok=True)
fig.savefig("fig/columns.png", bbox_inches='tight')

os.makedirs("results", exist_ok=True) #Make results directory if it does not exist yet
adult.to_hdf('results/df1.h5', 'adult')
clean_adult.to_hdf('results/df1.h5', 'clean_adult')

import unittest

class MyTests(unittest.TestCase):
    
    def test_multiple_missing_indices(self):
        row0 = [">", "", "!"]
        row1 = ["3", "1", "?"]
        row2 = ["?", "10", "hello"]
        row3 = ["my", "name", "is"]
        df = pd.DataFrame([row0, row1, row2, row3])
        missing = find_indices_with_value(df, "?")
        np.testing.assert_equal(missing, np.array([1, 2]))
        
    def test_no_missing_indices(self):
        row0 = [">", "", "!"]
        row1 = ["3", "1", "4"]
        row2 = ["6", "10", "hello"]
        row3 = ["my", "name", "is"]
        df = pd.DataFrame([row0, row1, row2, row3])
        missing = find_indices_with_value(df, "?")
        np.testing.assert_equal(missing, np.array([]))
        
    def test_one_missing_index(self):
        row0 = [">", "", "!"]
        row1 = ["3", "1", "4"]
        row2 = ["6", "10", "hello"]
        row3 = ["my", "name", "?"]
        df = pd.DataFrame([row0, row1, row2, row3])
        missing = find_indices_with_value(df, "?")
        np.testing.assert_equal(missing, np.array([3]))
        
    def test_multiple_missing_on_one_index(self):
        row0 = [">", "", "!"]
        row1 = ["3", "1", "4"]
        row2 = ["6", "10", "hello"]
        row3 = ["my", "?", "?"]
        df = pd.DataFrame([row0, row1, row2, row3])
        missing = find_indices_with_value(df, "?")
        np.testing.assert_equal(missing, np.array([3]))
        
    def test_multiple_missing_on_multiple_indices(self):
        row0 = [">", "", "!"]
        row1 = ["?", "?", "4"]
        row2 = ["6", "10", "hello"]
        row3 = ["my", "?", "?"]
        df = pd.DataFrame([row0, row1, row2, row3])
        missing = find_indices_with_value(df, "?")
        np.testing.assert_equal(missing, np.array([1, 3]))
        
    def test_one_row_df(self):
        row0 = [">", "?", "!"]
        df = pd.DataFrame([row0])
        missing = find_indices_with_value(df, "?")
        np.testing.assert_equal(missing, np.array([0]))

unittest.main(argv=["foo"], exit = False, verbosity = 2)

