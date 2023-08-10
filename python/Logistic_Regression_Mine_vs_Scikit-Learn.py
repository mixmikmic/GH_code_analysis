# Standard data analysis
import pandas as pd
from pandas import Series,DataFrame
import numpy as np

# Standard visualization analysis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# Scikit-learn analysis
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.grid_search import GridSearchCV

# Import division from future
from __future__ import division

# Import my own logistic regression class (see accompanying .py file)
from LogReg_Class import LogReg

# Ensure reproducibility (see accompanying rePYduce.py script)
import os
work_PATH = os.getcwd()
script_PATH = '/Users/pwinslow/DataScienceProjects/Useful_Scripts/'
os.chdir(script_PATH)
get_ipython().magic('run rePYduce.py numpy pandas matplotlib seaborn scikit-learn')
os.chdir(work_PATH)

exam_df = pd.read_csv('ex2data1.txt', header = None)
exam_df.columns = ['score1', 'score2', 'admitted']
exam_df.info()

# Plot the two data sets (admitted and not admitted) to see how they compare in terms of exam scores
def plotdata(df):
    
    input1, input2 = df.columns[0:-1]
    class_colname = df.columns[-1]
    
    ax = plt.subplot(111)
    
    # Note: this assumes that the df has been formatted so that the two classes are defined as '0' or '1'
    
    plt.scatter(df[df[class_colname] == 1][input1], 
            df[df[class_colname] == 1][input2], 
            color = 'black')
    
    plt.scatter(df[df[class_colname] == 0][input1], 
            df[df[class_colname] == 0][input2], 
            color = 'y')
    
    return ax

ax = plotdata(exam_df)
plt.legend(['admitted', 'rejected'])
plt.xlabel('First exam score')
plt.ylabel('Second exam score')

# Extract design matrix and target vector
X = exam_df[['score1', 'score2']].values
Y = exam_df['admitted'].values
m, n = X.shape
# Add bias column to design matrix
X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)

# Perform an initial analysis using the entire dataset for training
initial_theta = np.zeros( X.shape[1] )
logreg_classifier = LogReg()
theta_best = logreg_classifier.fit(initial_theta, X, Y)
score = logreg_classifier.score(theta_best, X, Y)
print 'Best fit regression parameters: {0}'.format(theta_best)
print 'Accuracy score: {0}'.format(score)

# Implement a stratified split of the data into train and test sets
skf = StratifiedKFold(Y, n_folds=10)

# Loop through folds, collecting accuracies as you go
scores = []
thetas = []
for train_index, test_index in skf:
    
    # Split inputs and targets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    initial_theta = np.zeros( X_train.shape[1] )

    # Create an instance of the logistic regression classifier
    logreg_classifier = LogReg()

    # Determine best fit parameters
    theta_best = logreg_classifier.fit(initial_theta, X_train, Y_train)
    thetas.append(theta_best)

    # Determine accuracy based on best fit parameters
    scores.append(logreg_classifier.score(theta_best, X_test, Y_test))
    
# Calculate the average score for the model based on all 10 scores
avg_score = np.mean(scores)
# Calculate the average regression parameters
avg_thetas = np.mean(thetas, axis = 0)

# Plot a histogram of the scores
sns.distplot(scores, bins = 20, kde = False)
plt.title('Average score: {}'.format(avg_score))

# Create an instance of the logistic regression classifier
logreg_classifier = LogReg()

# Define a mesh for a contour plot of the decision boundary
x_min, x_max = X[:,1].min()-5, X[:,1].max()+5
y_min, y_max = X[:,2].min()-5, X[:,2].max()+5
contour_x = np.linspace(x_min, x_max)
contour_y = np.linspace(y_min, y_max)

# Define a function which returns the model class prediction for a given set of inputs and the average regression
# parameters obtained from the 10-fold cross validation
def calc_z(x,y):
    return logreg_classifier.predict(avg_thetas, np.array([1, x, y]))

# Calculate the class prediction for each point in the mesh
z = np.zeros((len(contour_x), len(contour_y)))
for i, c_x in enumerate(contour_x):
    for j, c_y in enumerate(contour_y):
        z[i,j] = calc_z(c_x, c_y)

# Plot original data and decision boundary contour on top 
ax = plotdata(exam_df)
ax.contour(contour_x, contour_y, z, levels = [0], colors='green', linestyles = ['dashed'])
plt.legend(['accepted', 'rejected'])
plt.xlabel('First test score')
plt.ylabel('Second test score')

# Implement a stratified split of the data into train and test sets
skf = StratifiedKFold(Y, n_folds=10)

# Loop through folds, collecting accuracies as you go
scores = []
thetas = []
for train_index, test_index in skf:
    
    # Split inputs and targets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Create an instance of the logistic regression classifier
    logreg_classifier = LogisticRegression()

    # Fit the model
    logreg_classifier.fit(X_train, Y_train)

    # Determine accuracy based on best fit parameters
    # Predict the classes of the test set
    predict_classes = logreg_classifier.predict(X_test)
    thetas.append(logreg_classifier.coef_.ravel())
    scores.append(metrics.accuracy_score(Y_test, predict_classes))
    
# Calculate the average score for the model based on all 10 scores
avg_score = np.mean(scores)
# Calculate the average regression parameters
avg_thetas = np.mean(thetas, axis = 0)

# Plot a histogram of the scores
sns.distplot(scores, bins = 20, kde = False)
plt.title('Average score: {}'.format(avg_score))

# Implement a stratified split of the data into train and test sets
skf = StratifiedKFold(Y, n_folds=10)
for train_index, test_index in skf:
    
    # Split inputs and targets
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    initial_theta = np.zeros( X_train.shape[1] )

    # Create an instance of the logistic regression classifier
    logreg_classifier = LogisticRegression()

    # Fit the model
    logreg_classifier.fit(X_train, Y_train)

    # Determine accuracy based on best fit parameters
    # Predict the classes of the test set
    predict_classes = logreg_classifier.predict(X_test)
    score = metrics.accuracy_score(Y_test, predict_classes)
    
    # Break out of the loop if the average score is obtained
    if .85 <= score <= .95:
        break
        
# Plot decision boundary

x_min, x_max = X[:,1].min()-5, X[:,1].max()+5
y_min, y_max = X[:,2].min()-5, X[:,2].max()+5
contour_x = np.linspace(x_min, x_max)
contour_y = np.linspace(y_min, y_max)

def calc_z(x,y):
    return logreg_classifier.predict(np.array([[1, x, y]]))

z = np.zeros((len(contour_x), len(contour_y)))
for i, c_x in enumerate(contour_x):
    for j, c_y in enumerate(contour_y):
        z[i,j] = calc_z(c_x, c_y)

ax = plotdata(exam_df)
ax.contour(contour_x, contour_y, z, levels = [0], colors='green', linestyles = ['dashed'])
plt.legend(['accepted', 'rejected'])
plt.xlabel('First test score')
plt.ylabel('Second test score')

# Upload the data
exam_df = pd.read_csv('ex2data2.txt', header = None)
exam_df.columns = ['score1', 'score2', 'accepted']

# Translate data into numpy arrays
X = exam_df[['score1', 'score2']].values
Y = exam_df['accepted'].values

# Plot the data
ax = plotdata(exam_df)
plt.legend(['accepted', 'rejected'])
plt.xlabel('First test score')
plt.ylabel('Second test score')

# Use the poly_map method of the LogReg class to map to polynomial values and insert the bias feature. Try a 6 degree
# polynomial mapping.
logreg_classifier = LogReg()
X_poly = logreg_classifier.poly_map(X[:,0], X[:,1], 6)

# Implement a stratified split of the data into train and test sets
skf = StratifiedKFold(Y, n_folds=10)

# Loop through folds, collecting accuracies as you go
scores = []
thetas = []
for train_index, test_index in skf:
    
    # Split inputs and targets
    X_train, X_test = X_poly[train_index], X_poly[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    initial_theta = np.zeros( X_train.shape[1] )

    # Create an instance of the logistic regression classifier. From trial-and-error, we learn that we require 
    # a non-zero value of the ridge regression parameter to avoid overfitting the data with each instance and 
    # ending up with completely unrealistic fit parameters. 
    logreg_classifier = LogReg(lmbda=1, threshold=.5)

    # Determine best fit parameters
    theta_best = logreg_classifier.fit(initial_theta, X_train, Y_train)
    thetas.append(theta_best)

    # Determine accuracy based on best fit parameters
    scores.append(logreg_classifier.score(theta_best, X_test, Y_test))
    
    
# Calculate the average score for the model based on all 10 scores
avg_score = np.mean(scores)
# Calculate the average regression parameters
avg_thetas = np.mean(thetas, axis = 0)

# Plot a histogram of the scores
sns.distplot(scores, bins = 20, kde = False)
plt.title('Average score: {}'.format(avg_score))

# What does the resulting decision boundary look like?
contour_x = np.linspace(-1, 1.5)
contour_y = np.linspace(-1, 1.5)

def calc_z(x,y):
    return logreg_classifier.poly_map(x, y, 6).dot(avg_thetas)

z = np.zeros((len(contour_x), len(contour_y)))
for i, c_x in enumerate(contour_x):
    for j, c_y in enumerate(contour_y):
        z[i,j] = calc_z(c_x, c_y)

ax = plotdata(exam_df)
ax.contour(contour_x, contour_y, z, levels = [0], colors='salmon', linestyles = ['dashed'])
plt.legend(['accepted', 'rejected'])
plt.xlabel('First test score')
plt.ylabel('Second test score')

# Define a parameter grid to scan over
import itertools as it
param_grid = {'lambda':[.75, .8, .9, 1, 3, 5, 10], 'threshold':[.35,.4,.5,.6,.7,.8,.9], 
              'poly_degree':[2,3,4,5,6,7,8,9,10]}
varNames = sorted(param_grid)
val_combos = [dict(zip(varNames, prod)) for prod in it.product(*(param_grid[varName] for varName in varNames))]

# Implement a stratified split of the data into train and test sets before entering grid search
skf = StratifiedKFold(Y, n_folds=10)
    
score_results = []
for combo in val_combos:
    
    logreg_classifier = LogReg(lmbda = combo['lambda'], threshold = combo['threshold'])
    X_poly = logreg_classifier.poly_map(X[:,0], X[:,1], combo['poly_degree'])
    
    scores = []
    for train_index, test_index in skf:
    
        # Split inputs and targets
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        initial_theta = np.zeros( X_train.shape[1] )

        # Determine best fit parameters
        theta_best = logreg_classifier.fit(initial_theta, X_train, Y_train)

        # Determine accuracy based on best fit parameters
        scores.append(logreg_classifier.score(theta_best, X_test, Y_test))
    
    avg_score = np.mean(scores)
    
    score_results.append([combo['lambda'], combo['threshold'], combo['poly_degree'], avg_score])


# Collect overall results
score_results = np.array(score_results)
max_index = score_results[:,3].argmax()
best_lambda, best_threshold, best_poly_degree, best_score = score_results[max_index]

# Print results of scan
print 'lambda_best: {0}, threshold_best: {1}, poly_degree_best: {2}, best score: {3}'.format(best_lambda, 
                                                                                            best_threshold, 
                                                                                            best_poly_degree, 
                                                                                            best_score)

# Implement a stratified split of the data into train and test sets
skf = StratifiedKFold(Y, n_folds=10)

# Loop through folds, collecting accuracies as you go
scores = []
thetas = []
for train_index, test_index in skf:
    
    # Initialize an instance of the optimized classifier and polynomial expansion of the dataset
    logreg_classifier = LogReg(lmbda=best_lambda, threshold=best_threshold)
    X_poly = logreg_classifier.poly_map(X[:,0], X[:,1], best_poly_degree)

    # Split inputs and targets
    X_train, X_test = X_poly[train_index], X_poly[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    initial_theta = np.zeros( X_train.shape[1] )

    # Determine best fit parameters
    theta_best = logreg_classifier.fit(initial_theta, X_train, Y_train)
    thetas.append(theta_best)

    # Determine accuracy based on best fit parameters
    scores.append(logreg_classifier.score(theta_best, X_test, Y_test))
    
    
# Calculate the average regression parameters
avg_thetas = np.mean(thetas, axis = 0)

# Plot decision boundary
contour_x = np.linspace(-1, 1.5)
contour_y = np.linspace(-1, 1.5)

def calc_z(x,y):
    return logreg_classifier.poly_map(x, y, best_poly_degree).dot(avg_thetas)

z = np.zeros((len(contour_x), len(contour_y)))
for i, c_x in enumerate(contour_x):
    for j, c_y in enumerate(contour_y):
        z[i,j] = calc_z(c_x, c_y)

ax = plotdata(exam_df)
ax.contour(contour_x, contour_y, z, levels = [0], colors='salmon', linestyles = ['dashed'])
plt.legend(['accepted', 'rejected'])
plt.xlabel('First test score')
plt.ylabel('Second test score')

# We'll still use the poly_map method of the LogReg class to map to polynomial values and insert the bias feature.
# Since GridSearchCV doesn't take this parameter as an internal one, we'll need to modify it by hand, looking for the 
# best score.
logreg_classifier = LogReg()
X_poly = logreg_classifier.poly_map(X[:,0], X[:,1], 2)

logreg_classifier = LogisticRegression()

parameter_grid = {'C':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}

cross_validation = StratifiedKFold(Y, n_folds = 10)

grid_search = GridSearchCV(logreg_classifier, param_grid = parameter_grid, cv = cross_validation)

grid_search.fit(X_poly, Y)

print 'Best score: {}'.format(grid_search.best_score_)
print 'Best ridge regression parameter: {}'.format(grid_search.best_params_['C'])

grid_search.best_estimator_

my_classifier = LogReg()
X_poly = my_classifier.poly_map(X[:,0], X[:,1], 2)

# Plot decision boundary
x_min, x_max = X_poly[:,1].min()-.5, X_poly[:,1].max()+.5
y_min, y_max = X_poly[:,2].min()-.5, X_poly[:,2].max()+.5
contour_x = np.linspace(x_min, x_max)
contour_y = np.linspace(y_min, y_max)

def calc_z(x,y):
    return grid_search.best_estimator_.predict([my_classifier.poly_map(x, y, 2)])

z = np.zeros((len(contour_x), len(contour_y)))
for i, c_x in enumerate(contour_x):
    for j, c_y in enumerate(contour_y):
        z[i,j] = calc_z(c_x, c_y)

ax = plotdata(exam_df)
ax.contour(contour_x, contour_y, z, levels = [0], colors='salmon', linestyles = ['dashed'])
plt.legend(['accepted', 'rejected'])
plt.xlim(-1.25, 1.5)
plt.ylim(-1.25, 1.5)
plt.xlabel('First test score')
plt.ylabel('Second test score')

