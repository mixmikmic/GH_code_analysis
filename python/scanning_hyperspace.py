import wget
import pandas as pd

# Import the dataset
data_url = 'https://raw.githubusercontent.com/nslatysheva/data_science_blogging/master/datasets/wine/winequality-red.csv'
dataset = wget.download(data_url)
dataset = pd.read_csv(dataset, sep=";")

# Take a peak at the first few columns of the data
first_5_columns = dataset.columns[0:5]
dataset[first_5_columns].head()

# Examine shape of dataset and the column names
print (dataset.shape)
print (dataset.columns.values)

# Summarise feature values
dataset.describe()[first_5_columns]

# using a lambda function to bin quality scores
dataset['quality_is_high'] = dataset.quality.apply(lambda x: 1 if x >= 6 else 0)

import numpy as np

# Convert the dataframe to a numpy array and split the
# data into an input matrix X and class label vector y
npArray = np.array(dataset)
X = npArray[:,:-2].astype(float)
y = npArray[:,-1]

from sklearn.cross_validation import train_test_split

# Split into training and test sets
XTrain, XTest, yTrain, yTest = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

rf = RandomForestClassifier()
rf.fit(XTrain, yTrain)

rf_predictions = rf.predict(XTest)

print (metrics.classification_report(yTest, rf_predictions))
print ("Overall Accuracy:", round(metrics.accuracy_score(yTest, rf_predictions),4))

# Create a default random forest classifer and print its parameters
rf_default = RandomForestClassifier()
print(rf_default.get_params)

# manually specifying some HP values
hp_combinations = [
    {"n_estimators": 2, "max_features": None},  # all features are considered
    {"n_estimators": 5, "max_features": 'log2'},
    {"n_estimators": 9, "max_features": 'sqrt'} 
]

# test out different HP combinations
for hp_combn in hp_combinations:
        
    # Train and output accuracies
    rf = RandomForestClassifier(
        n_estimators=hp_combn["n_estimators"], 
        max_features=hp_combn["max_features"]
    )
    
    rf.fit(XTrain, yTrain)
    rf_predictions = rf.predict(XTest)
    print ('When n_estimators is {} and max_features is {}, test set accuracy is {}'.format(
            hp_combn["n_estimators"],
            hp_combn["max_features"], 
            round(metrics.accuracy_score(yTest, rf_predictions),2))
          )
           

import itertools

n_estimators = [2, 5, 9]
max_features = [None, 'log2', 'sqrt']

hp_combinations = list(itertools.product(n_estimators, max_features))
print (hp_combinations)
print ("The number of HP combinations is: {}".format(len(hp_combinations)))

from sklearn.grid_search import GridSearchCV

# Search for good hyperparameter values
# Specify values to grid search over
n_estimators = list(np.arange(10, 60, 20))
max_features = [None, 'sqrt', 'log2']  # we need to change the all to somthing better!

hyperparameters = {
    'n_estimators': n_estimators, 
    'max_features': max_features
}

# Grid search using cross-validation
gridCV = GridSearchCV(RandomForestClassifier(), param_grid=hyperparameters, cv=10, n_jobs=4)

# Perform grid search with 10-fold CV
gridCV.fit(XTrain, yTrain)

# Identify optimal hyperparameter values
best_n_estim      = gridCV.best_params_['n_estimators']
best_max_features = gridCV.best_params_['max_features']  

print("The best performing n_estimators value is: {}".format(best_n_estim))
print("The best performing max_features value is: {}".format(best_max_features))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# fetch scores, reshape into a grid
scores = [x[1] for x in gridCV.grid_scores_]
scores = np.array(scores).reshape(len(n_estimators), len(max_features))
scores = np.transpose(scores)

# Make heatmap from grid search results
plt.figure(figsize=(12, 6))
plt.imshow(scores, interpolation='nearest', origin='higher', cmap='jet_r')
plt.xticks(np.arange(len(n_estimators)), n_estimators)
plt.yticks(np.arange(len(max_features)), max_features)
plt.xlabel('Number of decision trees')
plt.ylabel('Max features')
plt.colorbar().set_label('Classification Accuracy', rotation=270, labelpad=20)
plt.show()

# Train classifier using optimal hyperparameter values
# We could have also gotten this model out from gridCV.best_estimator_
rf = RandomForestClassifier(n_estimators=best_n_estim,
                            max_features=best_max_features)

rf.fit(XTrain, yTrain)
rf_predictions = rf.predict(XTest)

print (metrics.classification_report(yTest, rf_predictions))
print ("Overall Accuracy:", round(metrics.accuracy_score(yTest, rf_predictions),2))

