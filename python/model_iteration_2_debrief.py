import pandas as pd

data = pd.read_csv('../../datasets/titanic_train.csv')
data.head()

data.isnull().sum()

data.Embarked = data.Embarked.fillna(data.Embarked.mode()[0])
data.Age = data.Age.fillna(data.Age.mean())
data = data.drop(['Cabin'], axis=1, errors='ignore')

print data.isnull().sum()

# add dummy variables for Sex and Embarked
data = pd.concat((data, pd.get_dummies(data.Sex)), axis=1)
data = pd.concat((data, pd.get_dummies(data.Embarked)), axis=1)
data.columns

import re

# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Get all the titles and print how often each one occurs.
titles = data["Name"].apply(get_title)

# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# Add in the title column.
data["Title"] = titles
data["FamilySize"] = data["SibSp"] + data["Parch"]

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np

predictors = ["Pclass", "male", "Age", "SibSp", "Parch", "Fare", "Title", 'C', 'S', 'Q', 'FamilySize']

selector = SelectKBest(k=5)
selector.fit(data[predictors], data.Survived)
scores = -np.log10(selector.pvalues_)
plt.bar(range(len(scores)), scores)
plt.xticks(range(len(scores)), predictors, rotation="vertical")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1.95, 1.96, .01)
y = abs(x) > 1
x = x[:,np.newaxis]
plt.plot(x,y,'b.')
plt.ylim([-.2, 1.2])
plt.xlim([-2.5, 2.5])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y)

classif = RandomForestClassifier()
classif.fit(x_train, y_train)
print "Accuracy of classifier", classif.score(x_test, y_test)

selector = SelectKBest(k=1)
selector.fit(x, y)
print "Feature importance", -np.log(selector.pvalues_)

from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(data[predictors], data.Survived)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(predictors)),
        importances[indices],
        color="r", yerr=std[indices],
        align="center")
plt.xticks(range(len(predictors)), predictors, rotation="vertical")
plt.show()

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

def do_trial(alg, X1, X2, y):
    # change Shuffle to False to see how this affects the conclusions
    cv = cross_validation.StratifiedKFold(y, 3, shuffle=True)
    score1 = cross_validation.cross_val_score(alg, X1, y, cv=cv)
    score2 = cross_validation.cross_val_score(alg, X2, y, cv=cv)
    return score1.mean(), score2.mean()

predictors = ['Pclass',
              'male',
              'Age',
              'SibSp',
              'Parch',
              'Fare',
              'Title',
              'C',
              'S',
              'Q',
              'FamilySize']

X = np.asarray(data[predictors])
y = np.asarray(data.Survived)

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(X, y)

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
best_feature_indices = sorted(range(len(scores)), reverse=True, key=lambda i: scores[i])

alg = RandomForestClassifier(n_estimators=150, min_samples_split=8, min_samples_leaf=4)
#alg = LogisticRegression()

n_trials = 10
all_features_performances = np.zeros(n_trials,)
best_features_performances = np.zeros(n_trials,)

for i in range(n_trials):
    (all_features_performances[i], best_features_performances[i]) =             do_trial(alg, X, X[:, best_feature_indices[0:5]], y)

plt.hist(best_features_performances - all_features_performances)
plt.ylabel('number')
plt.xlabel('performance gain from feature selection')
plt.show()

X = np.asarray(data[predictors])
X = np.hstack((X, np.random.randn(X.shape[0], 10000)))

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(X, y)

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)
best_feature_indices = sorted(range(len(scores)), reverse=True, key=lambda i: scores[i])

#alg = RandomForestClassifier(n_estimators=150, min_samples_split=8, min_samples_leaf=4)
alg = LogisticRegression()

n_trials = 10
all_feats_acc = np.zeros(n_trials,)
best_feats_acc = np.zeros(n_trials,)

for i in range(n_trials):
    all_feats_acc[i], best_feats_acc[i] =             do_trial(alg, X[:,:len(predictors)], X[:, best_feature_indices[:60]], y)

print "Feature selection performance %f, original features performance %f"             % (best_feats_acc.mean(), all_feats_acc.mean())
plt.hist(best_feats_acc - all_feats_acc)
plt.xlabel('Change in accuracy when using SelectKBest')
plt.ylabel('Number')
plt.show()

from sklearn.pipeline import Pipeline

def compare_two_algs(alg1, alg2, X1, X2, y):
    # change Shuffle to False to see how this affects the conclusions
    cv = cross_validation.StratifiedKFold(y, 3, shuffle=True)
    score1 = cross_validation.cross_val_score(alg1, X1, y, cv=cv)
    score2 = cross_validation.cross_val_score(alg2, X2, y, cv=cv)
    return score1.mean(), score2.mean()

selector = SelectKBest(f_classif, k=60)
alg = LogisticRegression()
pipeline = Pipeline([('feature selection', selector), ('classifier', alg)])

all_feats_acc = np.zeros(n_trials,)
best_feats_acc = np.zeros(n_trials,)

for i in range(n_trials):
    all_feats_acc[i], best_feats_acc[i] = compare_two_algs(alg,
                                                           pipeline,
                                                           X[:,:len(predictors)],
                                                           X,
                                                           y)

print "Feature selection performance %f, original features performance %f"             % (best_feats_acc.mean(), all_feats_acc.mean())
plt.hist(best_feats_acc - all_feats_acc)
plt.xlabel('Change in accuracy when using SelectKBest')
plt.ylabel('Number')
plt.show()

def test_split_model(data, predictors):
    cv = cross_validation.StratifiedKFold(data.Survived, 3, shuffle=True)

    num_correct_split = 0
    num_correct_unsplit = 0
    num_total = 0
    for train, test in cv:
        X = data[predictors]
        y = data.Survived
        X_train, X_test, y_train, y_test = X.loc[train,:], X.loc[test,:], y.loc[train], y.loc[test]

        test_groups = X_test.groupby('male')

        for group_name, subset in X_train.groupby('male'):
            model = LogisticRegression()
            model.fit(X_train.drop('male',axis=1), y_train)

            test_transformed = test_groups.get_group(group_name).drop('male',axis=1)
            num_correct_split += (y.loc[test_transformed.index] == model.predict(test_transformed)).sum()
            num_total += len(test_transformed)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        num_correct_unsplit += (y_test == model.predict(X_test)).sum()
        
    return num_correct_split / float(num_total), num_correct_unsplit / float(num_total)

split_acc = np.zeros(n_trials,)
unsplit_acc = np.zeros(n_trials,)
for i in range(n_trials):
    split_acc[i], unsplit_acc[i] = test_split_model(data, predictors)
plt.hist(split_acc - unsplit_acc)
plt.xlabel('Change in accuracy when splitting into two models')
plt.ylabel('Number')
plt.show()

# this works because male is a binary variable
data['male_age'] = data.Age*data.male
orig_acc = np.zeros(n_trials,)
with_interaction_acc = np.zeros(n_trials,)

for i in range(n_trials):
    orig_acc[i], with_interaction_acc[i] = do_trial(LogisticRegression(),
                                                    data[predictors],
                                                    data[predictors + ["male_age"]],
                                                    data.Survived)
plt.hist(with_interaction_acc - orig_acc)
plt.xlabel('Change in accuracy when adding interaction of age and male')
plt.ylabel('Number')
plt.show()

data_titles = pd.get_dummies(data.Title)

dummies_acc = np.zeros(n_trials,)
orig_acc = np.zeros(n_trials,)

for i in range(n_trials):
    dummies_acc[i], orig_acc[i] = do_trial(RandomForestClassifier(),
                                           data_titles,
                                           data[["Title"]],
                                           data.Survived)
plt.hist(dummies_acc - orig_acc)
plt.title('Random Forest')
plt.xlabel('Change in accuracy when using Title dummies')
plt.ylabel('Number')
print "Accuracy when using dummies", dummies_acc.mean()
print "Accuracy when using ordinals", orig_acc.mean()
plt.show()

from random import shuffle

dummies_acc = np.zeros(n_trials,)
orig_acc = np.zeros(n_trials,)

title_keys = list(data.Title.value_counts().keys())
remapping = list(title_keys)
shuffle(remapping)
title_remap = dict(zip(title_keys, remapping))
data['TitleRemap'] = data.Title.apply(lambda x: title_remap[x])
print data[['Title', 'TitleRemap']].head()

for i in range(n_trials):
    dummies_acc[i], orig_acc[i] = do_trial(RandomForestClassifier(),
                                           data_titles,
                                           data[["TitleRemap"]],
                                           data.Survived)
plt.hist(dummies_acc - orig_acc)
plt.xlabel('Change in accuracy when using Title dummies')
plt.ylabel('Number')
plt.title('Random Forest with remapped titles')
print "Accuracy when using dummies", dummies_acc.mean()
print "Accuracy when using ordinals", orig_acc.mean()
plt.show()

data_titles = pd.get_dummies(data.Title)

dummies_acc = np.zeros(n_trials,)
orig_acc = np.zeros(n_trials,)

for i in range(n_trials):
    dummies_acc[i], orig_acc[i] = do_trial(LogisticRegression(),
                                           data_titles,
                                           data[["Title"]],
                                           data.Survived)

plt.hist(dummies_acc - orig_acc)
plt.title('Logistic Regression')
plt.xlabel('Change in accuracy when using Title dummies')
plt.ylabel('Number')
print "Accuracy when using dummies", dummies_acc.mean()
print "Accuracy when using ordinals", orig_acc.mean()
plt.show()

from random import shuffle

remapped_acc = np.zeros(n_trials,)
orig_acc = np.zeros(n_trials,)

title_keys = list(data.Title.value_counts().keys())
remapping = list(title_keys)
shuffle(remapping)
title_remap = dict(zip(title_keys, remapping))
data['TitleRemap'] = data.Title.apply(lambda x: title_remap[x])
print data[['Title', 'TitleRemap']].head()

for i in range(n_trials):
    remapped_acc[i], orig_acc[i] = do_trial(LogisticRegression(),
                                            data[["TitleRemap"]],
                                            data[["Title"]],
                                            data.Survived)
plt.hist(remapped_acc - orig_acc)
plt.title('Logistic Regression')
plt.xlabel('Change in accuracy when using remapped title')
plt.ylabel('Number')
plt.show()

data = pd.read_csv('../../datasets/titanic_train.csv')

print "Means before", data.groupby('Pclass').Age.mean()

for name, group in data.groupby('Pclass'):
    data.loc[group.index, 'Age'] = group.Age.fillna(group.Age.mean())

print
print "Doing imputation"    
print
print "Nulls after", data.isnull().sum()
print "Means after", data.groupby('Pclass').Age.mean()

