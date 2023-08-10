get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load and then clean the traning data
titanic_train = pd.read_csv('../../datasets/titanic_train.csv')
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())

titanic_train.loc[titanic_train["Sex"] == "male", "Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == "female", "Sex"] = 1

titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")
titanic_train.loc[titanic_train["Embarked"] == "S", "Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == "C", "Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == "Q", "Embarked"] = 2

titanic_train["Fare"] = titanic_train["Fare"].fillna(titanic_train["Fare"].median())


# load and then clean the testing data
titanic_test = pd.read_csv('../../datasets/titanic_test.csv')
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2

titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

def clean_data(data, age_filler):
    data.Age = data.Age.fillna(age_filler)
    data.Fare = data.Fare.fillna(data.Fare.median())
    data.loc[data.Sex == "female", "Sex"] = 1
    data.loc[data.Sex == "male", "Sex"] = 0
    data.Embarked = data.Embarked.fillna("S")
    data.loc[data.Embarked == "S", "Embarked"] = 0
    data.loc[data.Embarked == "C", "Embarked"] = 1
    data.loc[data.Embarked == "Q", "Embarked"] = 2
    return data

titanic_train = pd.read_csv('../../datasets/titanic_train.csv')
titanic_train = clean_data(titanic_train, titanic_train.Age.median())

titanic_test = pd.read_csv('../../datasets/titanic_test.csv')
titanic_test = clean_data(titanic_test, titanic_train.Age.median())

def clean_data_dummies(data, age_filler):
    data.Age = data.Age.fillna(age_filler)
    data.Fare = data.Fare.fillna(data.Fare.median())
    data = pd.concat((data, pd.get_dummies(data.Sex, prefix="sex")), axis=1)
    data.Embarked = data.Embarked.fillna("S")
    data = pd.concat((data, pd.get_dummies(data.Embarked, prefix="embarked")), axis=1)
    return data

titanic_train_dummies = pd.read_csv('../../datasets/titanic_train.csv')
titanic_train_dummies = clean_data_dummies(titanic_train_dummies,
                                           titanic_train_dummies.Age.median())

titanic_test_dummies = pd.read_csv('../../datasets/titanic_test.csv')
titanic_test_dummies = clean_data_dummies(titanic_test_dummies,
                                          titanic_train_dummies.Age.median())

titanic_train_dummies.columns

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

alg = LogisticRegression()
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

without_dummies = cross_validation.cross_val_score(alg,
                                                   titanic_train[predictors],
                                                   titanic_train.Survived)

predictors_dummies = ["Pclass",
                      "sex_female",
                      "sex_male",
                      "Age",
                      "SibSp",
                      "Parch",
                      "Fare",
                      "embarked_S",
                      "embarked_C",
                      "embarked_Q"]
with_dummies = cross_validation.cross_val_score(alg,
                                                titanic_train_dummies[predictors_dummies],
                                                titanic_train_dummies.Survived)

with_dummies.mean() - without_dummies.mean()

def do_trial(data1, predictors1, data2, predictors2):
    alg = LogisticRegression()
    # change Shuffle to False to see how this affects the conclusions
    cv = cross_validation.StratifiedKFold(data1.Survived, 3, shuffle=True)
    score1 = cross_validation.cross_val_score(alg, data1[predictors1], data1.Survived, cv=cv)
    score2 = cross_validation.cross_val_score(alg, data2[predictors2], data2.Survived, cv=cv)
    return score1.mean() - score2.mean()

results = []
n_trials = 50
for i in range(n_trials):
    results.append(do_trial(titanic_train_dummies, predictors_dummies, titanic_train, predictors))
    
plt.hist(results, bins=20)
plt.xlabel('Change in performance when using dummies')
plt.ylabel('Frequency')
plt.show()

titanic_train["is_child"] = (titanic_train.Age <= 12).astype(float)
titanic_train.is_child.value_counts()

results = []
n_trials = 50
for i in range(n_trials):
    results.append(do_trial(titanic_train,
                            predictors + ["is_child"],
                            titanic_train,
                            predictors))
    
plt.hist(results, bins=20)
plt.xlabel('Change in performance when using is_child')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns
# setting shuffle=False for repeatability
cv = cross_validation.StratifiedKFold(titanic_train.Survived, 3, shuffle=False)
correct = []
indices = []

for train, test in cv:
    model = LogisticRegression()
    X = titanic_train.iloc[train, :]
    X = X[predictors]
    y = titanic_train.iloc[train, :].Survived
    model.fit(X, y)
    
    X_test = titanic_train.iloc[test, :]
    X_test = X_test[predictors]
    y_test = titanic_train.iloc[test, :].Survived

    indices.extend(titanic_train.iloc[test,:].index)
    correct.extend(model.predict(X_test) == y_test)

is_correct = pd.Series(correct, index=indices)

# note that distplot is a density (meaning it is normalized).  This facilitates
# direct comparision between the two populations
sns.distplot(titanic_train[is_correct].Age, hist=False, label="Correct")
sns.distplot(titanic_train[~is_correct].Age, hist=False, label="Incorrect")
sns.plt.ylabel('Density of responses')
sns.plt.show()

