# Load CSV from URL using Pandas
from pandas import read_csv, DataFrame
import numpy
#url = 'https://goo.gl/vhm1eU'
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age', 'class']
#data = read_csv(url, names=names)
data = read_csv(filename, names=names)
print(data.shape)
df = DataFrame(data)
print(df.describe())

# View first 10 rows
peek = data.head(10)
#print(peek)
# Examine dimensions of the dataset
shape = data.shape
#print(shape)
# Examine data type
types = data.dtypes
#print(types)

# Class Distribution
class_counts = data.groupby('class').size()
print(class_counts)

# Compute ratio of insulin to glucose 
data['ratio'] = data['insul']/data['gluc']
print(data.shape)
print(data.describe())

# Statistical Summary
from pandas import set_option #, DataFrame
set_option('display.width', 100)
set_option('precision', 3)

# Use pandas DataFrame for descriptive statistics
#names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age', 'class', 'ratio']
#df = DataFrame(data)
#df.columns = names
#print(df.describe())

# Examine skew of the attribute distributions
skew = data.skew()
print(skew)

#Save new file to disk
#data.to_csv('ratio_data.csv')

# Load CSV from URL using Pandas
from pandas import read_csv
import numpy
#url = 'https://goo.gl/vhm1eU'
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age', 'class']
#data = read_csv(url, names=names)
data = read_csv(filename, names=names)
print(data.shape)

# Pairwise Pearson correlations
correlations = data.corr(method='pearson')
print(correlations)

# Univariate Histograms
from matplotlib import pyplot
data.hist(figsize=(10,8))
pyplot.show()
# Density plots
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(10,8))
pyplot.show()
# Box and Whiskers plots
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))
pyplot.show()

# Matrix Plots
from matplotlib import pyplot
from pandas import read_csv
import numpy
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age', 'class']
#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data, figsize=(10,8))
pyplot.show()

# Correlation matrix
correlations = data.corr()
fig = pyplot.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.show()


# 1. BASELINE MODEL USING MEAN IMPUTATION

# MEAN IMPUTATION OF ALL ATTRITUBES USING SKLEARN IMPUTER
from pandas import read_csv, DataFrame
from sklearn.preprocessing import Imputer
import numpy as np
#url = 'https://goo.gl/vhm1eU'
filename = 'pima-indians-diabetes.data.csv'
dataset = read_csv(filename, header=None)
# mark zero values as missing or NaN
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, np.NaN)

# Distributions with missing values == NaN
#print(dataset.isnull().sum())
missing = (dataset.isnull().sum()/768)*100
#print(missing)

# fill missing values with mean column values
values = dataset.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
#print(transformed_values.shape)
# count the number of NaN values in each column
#print(numpy.isnan(transformed_values).sum())
dataset_imp = DataFrame(transformed_values)
#print("Mean Imputation Descriptives")
#print(dataset_imp.describe())

#from pandas import read_csv
#import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# split dataset into inputs and outputs
values = dataset_imp.values
X = values[:,0:8]
y = values[:,8]

# evaluate an LDA model on the mean imputation dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=5, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
#print("Validation Accuracy = %.3f" % result.mean())
accuracy = result.mean()
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# SAVE DATASET
numpy.savetxt("mean_imp_data.csv", dataset_imp, delimiter=",")

# XGBoost model for Mean Imputation dataset
from pandas import read_csv, DataFrame
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# load data
#url = 'https://goo.gl/vhm1eU'
filename = 'mean_imp_data.csv'
dataset = read_csv(filename, header=None)

# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
#print(X.shape)
y = values[:,8]
#print(y.shape)

# split data into train and test sets
seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# SKIN FOLD THICKNESS ML MODEL
from pandas import read_csv, DataFrame
from sklearn.preprocessing import Imputer
import numpy as np

# Mean imputation for all attributes except skin(3) for training and validation samples

# 0. LOAD DATA AND MARK MISSING VALUES
filename = 'pima-indians-diabetes.data.csv'
dataset = read_csv(filename, header=None)
# mark zero values as missing or NaN - EXCEPT SKIN(3) 
dataset[[1,2,4,5]] = dataset[[1,2,4,5]].replace(0, np.NaN)
# Distributions with missing values == NaN
#print(dataset.isnull().sum())
missing3 = (dataset.isnull().sum()/768)*100
#print(missing3)

# 1. CREATE IMPUTATION DATASET FOR SKIN FOLD THICKNESS PREDICTIONS

# Fill missing values with mean column values
values = dataset.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
#print(transformed_values.shape)
# count the number of NaN values in each column
#print(numpy.isnan(transformed_values).sum())
dataset3_imp = DataFrame(transformed_values)

# Reorder cols in imputer dataset
dataset3_imp = dataset3_imp[[0,1,2,4,5,6,7,8,3]]
#print("Imputer dataset - new column order")
cols_list = dataset3_imp.columns.tolist()
#print(cols_list)
#print("Imputer dataset - Descriptives 3")
#print(dataset3_imp.describe())

# 2. CREATE TRAINING DATASET WITH MISSING REMOVED
# Remove cases with any missing values for model training
dataset.dropna(inplace=True)
# Reorder training dataset
dataset3 = dataset[[0,1,2,4,5,6,7,8,3]]
#print("Training dataset - new column order")
cols_list = dataset3.columns.tolist()
#print(cols_list)
# CHECK COL REORDER ON TRAINING DATASET
print("N size - MISSING REMOVED", dataset3.shape)
print("Descriptive Statistics - MISSING REMOVED")
print(dataset3.describe())


# 3. MODEL EVALUATION + JOBLIB TO SAVE FINAL MODEL

# 3a. Load libraries
import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
#from pickle import dump
#from pickle import load

# 3b. Prepare Reordered Data with Missing Removed

# Split-out validation dataset
array = dataset3.values
X = array[:,0:8]
Y = array[:,8]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 3c. Prepare Reordered Imputer Data

# Imputer-3 dataset
array3 = dataset3_imp.values
X3 = array3[:,0:8]
Y3 = array3[:,8]

# 3d. Choose Algorithm Evaluation Criteria
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# 3e. Algorithm check with default tuning parameters
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('LASSOCV', LassoCV()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# CV sequential evaluation of each model
results = []
names = []
print("Evaluate Model Performance")
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# 3f. Evaluate models on standardized dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledLASSOCV', Pipeline([('Scaler', StandardScaler()),('LASSOCV', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
print("Evaluate Model Performance on Standardized Data")
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#  Algorithm tuning for KNN to improve performance
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
print("Evaluate Performance of Ensembles on Standardized Data")
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Algorithm tuning for scaled GBM to improve performance 
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
print("Evaluate Performance with Hyper Parameter Tuning")
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    

# 4. FIT FINAL MODEL USING OUTPUT FROM ENSEMBLE AND MAKE PREDICTIONS

# 4a. Prepare the model - calculate mean and std on training data
scaler = StandardScaler().fit(X_train)
# standardize training data
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=50)
model.fit(rescaledX, Y_train)

# 4b. Transform the validation dataset and calculate prediction error
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print("Mean Squared Error on Validation Sample")
print(mean_squared_error(Y_validation, predictions))


# 5. SAVE MODEL USING JOBLIB DUMP FOR PREDICTION ON A NEW DATASET

# 5a. Dump final model to disk
filename = 'final_skin_model.sav'
dump(model, filename)

# 5b. Calculate mean and std on imputer dataset
scaler3 = StandardScaler().fit(X3)
rescaledX3 = scaler.transform(X3)


# 5c. Import model from disk
loaded_model = load(filename)

# 5d. Predict skin on imputer dataset and review MSE
predictions3 = loaded_model.predict(rescaledX3)
print("Mean Squared Error on Imputer3 Dataset")
print(mean_squared_error(Y3, predictions3))

# 5e. Review distribution of prediction using model imputation
Y3hat = DataFrame(predictions3)
#print(Y3hat.shape)
print("Descriptives Y3hat")
print(Y3hat.describe())


# 6. MERGE PREDICTIONS WITH DATASET AND PERFORM IMPUTATION

# 6a. Merge predicted skin back to original dataset and check
#print(data.shape)
#print(Y3hat.shape)
dataset3Yhat = np.concatenate((data,Y3hat),axis=1)
#print(dataset3Yhat.shape)
df3Yhat = DataFrame(dataset3Yhat)
print("Descriptives dataset3Yhat - NO MODEL IMPUTATION ON SKIN FOLD THICKNESS (3)")
print(df3Yhat.describe())
#print(df3Yhat.head(20))


# 6b. REPLACE MISSING VALUES WITH MODEL PREDICTIONS 
df3Yhat[3] = np.where(df3Yhat[3] == 0, df3Yhat[9], df3Yhat[3])
df3Yhat = df3Yhat[[0,1,2,3,4,5,6,7,8]]
print("Descriptives df3Yhat - WITH IMPUTATION ON SKIN FOLD THICKNESS (3)")
#print(df3Yhat.shape)
print(df3Yhat.describe())
#print(df3Yhat.head(20))

# 7. SAVE DATASET
numpy.savetxt("skin_imp_data.csv", df3Yhat, delimiter=",")

# 4. INSULIN FOLD THICKNESS ML MODEL

from pandas import read_csv, DataFrame
from sklearn.preprocessing import Imputer
import numpy as np

# 0. LOAD DATA AND MARK MISSING VALUES
filename = 'skin_imp_data.csv'
dataset = read_csv(filename, header=None)
#print(dataset.shape)

# mark zero values as missing or NaN - EXCEPT SKIN(3) 
dataset[[1,2,4,5]] = dataset[[1,2,4,5]].replace(0, np.NaN)
# Distributions with missing values == NaN
#print(dataset.isnull().sum())
missing4 = (dataset.isnull().sum()/768)*100
#print(missing4)

# 1. CREATE IMPUTATION DATASET FOR SKIN FOLD THICKNESS PREDICTIONS

# Fill missing values with mean column values
values = dataset.values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
#print(transformed_values.shape)
# count the number of NaN values in each column
#print(numpy.isnan(transformed_values).sum())
dataset4_imp = DataFrame(transformed_values)

# Reorder cols in imputer dataset
dataset4r_imp = dataset4_imp[[0,1,2,3,5,6,7,8,4]]
#print("Imputer dataset - new column order")
cols_list = dataset4r_imp.columns.tolist()
#print(cols_list)
print("Imputer dataset - Descriptives 4")
print(dataset4r_imp.describe())


# 2. CREATE TRAINING DATASET WITH MISSING REMOVED
# Remove cases with any missing values for model training
dataset.dropna(inplace=True)
# Reorder training dataset
dataset4 = dataset[[0,1,2,3,5,6,7,8,4]]
#print("Training dataset - new column order")
cols_list = dataset4.columns.tolist()
#print(cols_list)
# CHECK COL REORDER ON TRAINING DATASET
#print("N size - MISSING REMOVED", dataset4.shape)
print("Descriptive Statistics - MISSING REMOVED")
print(dataset4.describe())


# 3. MODEL EVALUATION + JOBLIB TO SAVE FINAL MODEL

# 3a. Load libraries
import numpy as np
from numpy import arange
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
#from pickle import dump
#from pickle import load

# 3b. Prepare Reordered Data with Missing Removed

# Split-out validation dataset
array = dataset4.values
X = array[:,0:8]
Y = array[:,8]
#Y = np.log(array[:,8])
#ln_Y = np.log(Y)

validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# 3c. Prepare Reordered Imputer Data

# Imputer dataset
array4 = dataset4r_imp.values
X4 = array4[:,0:8]
Y4 = array4[:,8]

# 3d. Choose Algorithm Evaluation Criteria
# Test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# 3e. Algorithm check with default tuning parameters
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

# evaluate each model in turn
results = []
names = []
print("Evaluate Model Performance")
for name, model in models:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# 3f. Evaluate models on standardized dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
print("Evaluate Model Performance on Standardized Data")
for name, model in pipelines:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# KNN Algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = numpy.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
print("Evaluate Performance of Ensembles on Standardized Data")
for name, model in ensembles:
	kfold = KFold(n_splits=num_folds, random_state=seed)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Tune scaled GBM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=numpy.array([50,100,150,200,250,300,350,400]))
model = GradientBoostingRegressor(random_state=seed)
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
print("Evaluate Performance with Hyper Parameter Tuning")
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    

# 4. FIT FINAL MODEL USING OUTPUT FROM ENSEMBLE AND MAKE PREDICTIONS

# 4a. Prepare the model - calculate mean and std on training data
scaler = StandardScaler().fit(X_train)
# standardize training data
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=100)
model.fit(rescaledX, Y_train)

# 4b. Transform the validation dataset and calculate prediction error
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print("Mean Squared Error on Validation Sample")
print(mean_squared_error(Y_validation, predictions))


# 5. SAVE MODEL USING JOBLIB DUMP FOR PREDICTION ON A NEW DATASET

# 5a. Dump final model to disk
filename = 'final_insulin_model.sav'
dump(model, filename)

# 5b. Calculate mean and std on imputer dataset
scaler4 = StandardScaler().fit(X4)
rescaledX4 = scaler.transform(X4)
#model = GradientBoostingRegressor(random_state=seed, n_estimators=400)

# 5c. Import model from disk
loaded_model = load(filename)

# 5d. Predict skin on imputer dataset and review MSE
predictions4 = loaded_model.predict(rescaledX4)
print("Mean Squared Error on Imputer4 Dataset")
print(mean_squared_error(Y4, predictions4))

# 5e. Review distribution of prediction using model imputation
Y4hat = DataFrame(predictions4)
print(Y4hat.shape)
print("Descriptives Y4hat")
print(Y4hat.describe())

# UNLOG PREDICTIONS
#Y4hat[0] = np.exp(Y4hat[0] - 1)

# 6. MERGE PREDICTIONS WITH DATASET AND PERFORM IMPUTATION

# 6a. Merge predicted skin back to original dataset and check
print(dataset4_imp.shape)
df4imp=DataFrame(dataset4_imp)
print(df4imp.describe())
#print(Y4hat.shape)

dataset4Yhat = np.concatenate((df4imp,Y4hat),axis=1)
print(dataset4Yhat.shape)

df4Yhat = DataFrame(dataset4Yhat)
print("Descriptives df4Yhat - NO IMPUTATION")
print(df4Yhat.describe())
print(df4Yhat.head(20))

# 6b. REPLACE MISSING VALUES WITH MODEL PREDICTIONS 
df4Yhat[4] = np.where(df4Yhat[4] == 0, df4Yhat[9], df4Yhat[4])
print("Descriptives df4Yhat - WITH IMPUTATION")
print(df4Yhat.describe())
#print(df4Yhat.head(20))

df4Yhat2 = df4Yhat[[0,1,2,3,4,5,6,7,8]]

#  7. SAVE DATA
numpy.savetxt("skin_insul_imp_data.csv", df4Yhat2, delimiter=",")

# MEAN IMPUTATION OF ALL ATTRITUBES USING SKLEARN IMPUTER
from pandas import read_csv, DataFrame
from sklearn.preprocessing import Imputer
import numpy as np
#url = 'https://goo.gl/vhm1eU'
filename = 'skin_insul_imp_data.csv'
dataset = read_csv(filename, header=None)
df = DataFrame(dataset)
#print(df.describe())

# 1. BASELINE LDA MODEL USING MEAN AND MODEL IMPUTATION

# MEAN IMPUTATION OF ALL ATTRITUBES USING SKLEARN IMPUTER
from pandas import read_csv, DataFrame
from sklearn.preprocessing import Imputer
import numpy as np
#url = 'https://goo.gl/vhm1eU'
filename = 'skin_insul_imp_data.csv'
dataset = read_csv(filename, header=None)
df = DataFrame(dataset)
#print(df.describe())

#from pandas import read_csv
#import numpy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
y = values[:,8]

# evaluate an LDA model on the mean imputation dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=5, random_state=7)
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
accuracy = result.mean()
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# First XGBoost model for Pima Indians dataset
from pandas import read_csv, DataFrame
import numpy as np
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#url = 'https://goo.gl/vhm1eU'
filename = 'skin_insul_imp_data.csv'
dataset = read_csv(filename, header=None)
df = DataFrame(dataset)
#print(df.describe())

# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
#print(X.shape)
y = values[:,8]
#print(y.shape)

# split data into train and test sets
seed = 7
test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

### REVIEW MULTIVARIATE CORRELATIONS ###
from pandas import read_csv, DataFrame
import numpy as np
#url = 'https://goo.gl/vhm1eU'
filename = 'skin_insul_imp_data.csv'
names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age', 'class']
dataset = read_csv(filename, names=names)
# Compute ratio of insulin to glucose 
dataset['ratio'] = dataset['insul']/dataset['gluc']

# Pairwise Pearson correlations
correlations = dataset.corr(method='pearson')
print(correlations)

# Scatterplot Matrix
from pandas.plotting import scatter_matrix
scatter_matrix(dataset, figsize=(10,8))
pyplot.show()

from pandas import read_csv, DataFrame
import numpy as np
filename = 'skin_insul_imp_data.csv'
names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age', 'class']
dataset = read_csv(filename, names=names)
# Compute ratio of insulin to glucose 
dataset['ratio'] = dataset['insul']/dataset['gluc']
#df = DataFrame(dataset)
#ratio = df[[0,1,2,3,4,5,6,7,9,8]]

#dataset = read_csv(filename, header=None)
#df = DataFrame(dataset)
#print(df.describe())

from __future__ import print_function
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from time import time

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence

# split dataset into inputs and outputs
values = dataset.values
X = values[:,0:8]
print(X.shape)
y = values[:,8]
#print(y.shape)

def main():

    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=1)
    names = ['preg', 'gluc', 'dbp', 'skin', 'insul', 'bmi', 'pedi', 'age']

    print("Training GBRT...")
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='deviance',
                                     random_state=1)
    t0 = time()
    clf.fit(X_train, y_train)
    print(" done.")
    
    print("done in %0.3fs" % (time() - t0))
    importances = clf.feature_importances_
    
    print(importances)

    #print('Convenience plot with ``partial_dependence_plots``')

    features = [0, 1, 2, 3, 4, 5, 6, 7, (6,4)]
    fig, axs = plot_partial_dependence(clf, X_train, features,
                                       feature_names=names,
                                       n_jobs=3, grid_resolution=50)
    #fig.suptitle('Partial dependence plots of Pre Diabetes on risk factors')
            
    plt.subplots_adjust(bottom=0.1, right=2, top=2)  # tight_layout causes overlap with suptitle

    
    print('Custom 3d plot via ``partial_dependence``')
    fig = plt.figure()

    target_feature = (4, 6)
    pdp, axes = partial_dependence(clf, target_feature,
                                   X=X_train, grid_resolution=50)
    XX, YY = np.meshgrid(axes[0], axes[1])
    Z = pdp[0].reshape(list(map(np.size, axes))).T
    ax = Axes3D(fig)
    surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                           cmap=plt.cm.BuPu, edgecolor='k')
    ax.set_xlabel(names[target_feature[0]])
    ax.set_ylabel(names[target_feature[1]])
    ax.set_zlabel('Partial dependence')
    #  pretty init view
    ax.view_init(elev=22, azim=122)
    plt.colorbar(surf)
    plt.suptitle('Partial dependence of Pre Diabetes risk factors')
                 
    plt.subplots_adjust(right=1,top=.9)

    plt.show()
    
    # Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()

