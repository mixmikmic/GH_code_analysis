import pandas as pd
import numpy as np

# Import the dataset
dataset_path = "spam_dataset.csv"
dataset = pd.read_csv(dataset_path, sep=",")

# Take a peak at the data
dataset.head()

# Reorder the data columns and drop email_id
cols = dataset.columns.tolist()
cols = cols[2:] + [cols[1]]
dataset = dataset[cols]

# Examine shape of dataset and some column names
print dataset.shape
print dataset.columns.values[0:10]

# Summarise feature values
dataset.describe()

# Convert dataframe to numpy array and split
# data into input matrix X and class label vector y
npArray = np.array(dataset)
X = npArray[:,:-1].astype(float)
y = npArray[:,-1]

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

# Scale and split dataset
X_scaled = preprocessing.scale(X)

# Split into training and test sets
XTrain, XTest, yTrain, yTest = train_test_split(X_scaled, y, random_state=1)

from sklearn import metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Search for good hyperparameter values
# Specify values to grid search over
n_estimators = np.arange(1, 30, 5)
max_features  = np.arange(1, X.shape[1], 10)
max_depth    = np.arange(1, 100, 10)

hyperparameters   = {'n_estimators': n_estimators, 
                     'max_features': max_features, 
                     'max_depth': max_depth}

# Grid search using cross-validation
gridCV = GridSearchCV(RandomForestClassifier(), param_grid=hyperparameters, cv=10, n_jobs=4)
gridCV.fit(XTrain, yTrain)

best_n_estim      = gridCV.best_params_['n_estimators']
best_max_features = gridCV.best_params_['max_features']               
best_max_depth    = gridCV.best_params_['max_depth']

# Train classifier using optimal hyperparameter values
# We could have also gotten this model out from gridCV.best_estimator_
clfRDF = RandomForestClassifier(n_estimators=best_n_estim, max_features=best_max_features, max_depth=best_max_depth)
clfRDF.fit(XTrain, yTrain)
RF_predictions = clfRDF.predict(XTest)

print (metrics.classification_report(yTest, RF_predictions))
print ("Overall Accuracy:", round(metrics.accuracy_score(yTest, RF_predictions),2))

from sklearn.svm import SVC

# Search for good hyperparameter values
# Specify values to grid search over
g_range = 2. ** np.arange(-15, 5, step=2)
C_range = 2. ** np.arange(-5, 15, step=2)

hyperparameters = [{'gamma': g_range, 
                    'C': C_range}] 

# Grid search using cross-validation
grid = GridSearchCV(SVC(), param_grid=hyperparameters, cv= 10)  
grid.fit(XTrain, yTrain)

bestG = grid.best_params_['gamma']
bestC = grid.best_params_['C']

# Train SVM and output predictions
rbfSVM = SVC(kernel='rbf', C=bestC, gamma=bestG)
rbfSVM.fit(XTrain, yTrain)
SVM_predictions = rbfSVM.predict(XTest)

print metrics.classification_report(yTest, SVM_predictions)
print "Overall Accuracy:", round(metrics.accuracy_score(yTest, SVM_predictions),2)

from multilayer_perceptron import multilayer_perceptron

# Search for good hyperparameter values
# Specify values to grid search over
layer_size_range = [(3,2),(10,10),(2,2,2),10,5] # different networks shapes
learning_rate_range = np.linspace(.1,1,3)
hyperparameters = [{'hidden_layer_sizes': layer_size_range, 'learning_rate_init': learning_rate_range}]

# Grid search using cross-validation
grid = GridSearchCV(multilayer_perceptron.MultilayerPerceptronClassifier(), param_grid=hyperparameters, cv=10)
grid.fit(XTrain, yTrain)

# Output best hyperparameter values
best_size    = grid.best_params_['hidden_layer_sizes']
best_best_lr = grid.best_params_['learning_rate_init']

# Train neural network and output predictions
nnet = multilayer_perceptron.MultilayerPerceptronClassifier(hidden_layer_sizes=best_size, learning_rate_init=best_best_lr)
nnet.fit(XTrain, yTrain)
NN_predictions = nnet.predict(XTest)

print metrics.classification_report(yTest, NN_predictions)
print "Overall Accuracy:", round(metrics.accuracy_score(yTest, NN_predictions),2)

# here's a rough solution

import collections

# stick all predictions into a dataframe
predictions = pd.DataFrame(np.array([RF_predictions, SVM_predictions, NN_predictions])).T
predictions.columns = ['RF', 'SVM', 'NN']
predictions = pd.DataFrame(np.where(predictions=='yes', 1, 0), 
                           columns=predictions.columns, 
                           index=predictions.index)

# initialise empty array for holding predictions
ensembled_predictions = np.zeros(shape=yTest.shape)

# majority vote and output final predictions
for test_point in range(predictions.shape[0]):
    predictions.iloc[test_point,:]
    counts = collections.Counter(predictions.iloc[test_point,:])
    majority_vote = counts.most_common(1)[0][0]
    
    # output votes
    ensembled_predictions[test_point] = majority_vote.astype(int)
    print "The majority vote for test point", test_point, "is: ", majority_vote

# Get final accuracy of ensembled model
yTest[yTest == "yes"] = 1
yTest[yTest == "no"] = 0

print metrics.classification_report(yTest.astype(int), ensembled_predictions.astype(int))
print "Ensemble Accuracy:", round(metrics.accuracy_score(yTest.astype(int), ensembled_predictions.astype(int)),2)



