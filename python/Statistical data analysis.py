import pandas as pd
import numpy as np
import sklearn as sk
import scipy.stats as st
import statsmodels as stm
import statsmodels.api as stmApi

get_ipython().run_line_magic('matplotlib', 'inline')

titanic_data = pd.DataFrame.from_csv("./data/titanic_train_data.csv", index_col=None)
titanic_data.head(3)

# Excercise 1: Get the mean, variance and standard deviation of age for males and females in Titanic

# Round your results to 1 decimal place!

expected_df = pd.DataFrame({
    'a': ['Age', 'mean', 27.9, 30.7],
    'b': ["Age", "var", 199.1, 215.4],
    'c': ["Age", "std", 14.1, 14.7]
})
arrays = [expected_df.iloc[0].tolist(), expected_df.iloc[1].tolist()]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples)
expected_df.columns = index
expected_df = expected_df.iloc[2:]
expected_df.index = ["female", "male"]
expected_df.index.name = "Sex"


age_mean_var_std = '??'

print(age_mean_var_std)


assert(age_mean_var_std.to_string() == expected_df.to_string())

# Excercise 2: Perform two-sided t-test to check if mean age is equal in two groups.
# Answer a question: should you use pooled variance or not? 
# Do you consider variances amongst the groups equal?

# What is your null hypothesis? What is your alternative hypothesis?

# Round results to 3rd decimal place

# WATCH OUT FOR MISSING VALUES! Find a way to deal with them

males_age_vec = titanic_data.loc[titanic_data.Sex == "male", "Age"].dropna()
females_age_vec = titanic_data.loc[titanic_data.Sex == "female", "Age"].dropna()

expected_stat = 2.5
expected_pval = 0.01

pval = '??'
stat_val = '??'

assert(pval == expected_pval)
assert(expected_stat == stat_val)

# YOUR CODE GOES HERE!

from sklearn.linear_model import LinearRegression

no_null_age = titanic_data.Age.notnull()
titanic_data.Age.isnull().sum()
#train_df = titanic_data

clean_data = titanic_data.dropna()
lr = LinearRegression()
lr.fit(clean_data[["Pclass"]], clean_data["Age"])
X = pd.get_dummies(clean_data[["Fare", "SibSp", "Age", "Parch", "Sex"]])
X = stmApi.add_constant(X)
fit = stmApi.Logit(clean_data["Survived"], X).fit()
print(fit.summary())



