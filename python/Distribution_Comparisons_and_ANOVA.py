import pandas as pd

# must specify that blank space " " is NaN
experimentDF = pd.read_csv("data_sets/parasite_data.csv", na_values=" ")

experimentDF.head()

# show all entries in the Virulence column
experimentDF.Virulence

# show the 12th row in the ShannonDiversity column
experimentDF.ShannonDiversity[12]

# show all entries in the ShannonDiversity column > 2.0
experimentDF.query("ShannonDiversity > 2.0")

experimentDF[experimentDF.Virulence.isnull()]

print("Mean virulence across all treatments:", experimentDF["Virulence"].mean())

from scipy import stats

print("Mean virulence across all treatments:", stats.sem(experimentDF["Virulence"]))

# NOTE: this drops the entire row if any of its entries are NA/NaN!
experimentDF.dropna().info()

print(experimentDF["Virulence"].dropna())

experimentDF.fillna(0.0)["Virulence"]

print("Mean virulence across all treatments w/ dropped NaN:", experimentDF["Virulence"].dropna().mean())

print("Mean virulence across all treatments w/ filled NaN:", experimentDF.fillna(0.0)["Virulence"].mean())

from pandas import *

print("Mean Shannon Diversity w/ 0.8 Parasite Virulence =", experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"].mean())

from pandas import *

print("Variance in Shannon Diversity w/ 0.8 Parasite Virulence =", experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"].var())

from pandas import *
from scipy import stats

print("SEM of Shannon Diversity w/ 0.8 Parasite Virulence =", stats.sem(experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"]))

# select two treatment data sets from the parasite data
treatment1 = experimentDF[experimentDF["Virulence"] == 0.5]["ShannonDiversity"]
treatment2 = experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"]

print("Data set 1:\n", treatment1)
print("Data set 2:\n", treatment2)

from scipy import stats

z_stat, p_val = stats.ranksums(treatment1, treatment2)

print("MWW RankSum P for treatments 1 and 2 =", p_val)

treatment3 = experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"]
treatment4 = experimentDF[experimentDF["Virulence"] == 0.9]["ShannonDiversity"]

print("Data set 3:\n", treatment3)
print("Data set 4:\n", treatment4)

z_stat, p_val = stats.ranksums(treatment3, treatment4)

print("MWW RankSum P for treatments 3 and 4 =", p_val)

treatment1 = experimentDF[experimentDF["Virulence"] == 0.7]["ShannonDiversity"]
treatment2 = experimentDF[experimentDF["Virulence"] == 0.8]["ShannonDiversity"]
treatment3 = experimentDF[experimentDF["Virulence"] == 0.9]["ShannonDiversity"]

print("Data set 1:\n", treatment1)
print("Data set 2:\n", treatment2)
print("Data set 3:\n", treatment3)

from scipy import stats
	
f_val, p_val = stats.f_oneway(treatment1, treatment2, treatment3)

print("One-way ANOVA P =", p_val)

