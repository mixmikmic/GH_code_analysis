# Import necessary packages
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Load Iris data (https://en.wikipedia.org/wiki/Iris_flower_data_set)
iris = load_iris()
# Load iris into a dataframe and set the field names
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

# add a new column to dataframe with random 75% as true, and 25% as false
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['is_train'].head()

# set a new column to specify specific epithet
# the original target is in integers, 
# pd.Categorical.from_codes change them into category names from the 2nd argument input
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print df['species'].head()

# split into two dataframes, one being just true (training), and another false (testing)
train, test = df[df['is_train']==True], df[df['is_train']==False]

print train.head(n=2)
print test.head(n=2)

# feature an index holding just the column names (1st 4 fields)
features = df.columns[:4]
features

# define the classifier
# n_job for multicore CPU; n_estimator for no. of decision trees
clf = RandomForestClassifier(n_jobs=2, n_estimators=100)
clf

# Factorizing specific epithet (basically change categories into numbers)
# creates an array of factors & an index of specific epithet
y, epithet = pd.factorize(train['species'])
print y
print epithet

# FIT THE MODEL

# set 1st arguement as training dataframe with only first 4 fields (variables that predict)
# set 2nd arguement as species factor dataframe (what u want to predict (response))
clf.fit(train[features], y)

# PREDICT THE MODEL

# predict test features
# change epithet factors to species names
preds = iris.target_names[clf.predict(test[features])]
preds

# SCORE THE MODEL

# compare actual vs predicted values, using a confusion matrix
pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])
# Every run you made yields a different result, but its clear that the prediction is very high

