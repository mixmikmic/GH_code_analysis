# This tells matplotlib not to try opening a new window for each plot.
get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import seaborn as sns  # 'pip install seaborn' if not already installed
import numpy as np
import pandas as pd
import time
from numpy.linalg import inv
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

np.set_printoptions(precision=4, suppress=True)

boston = load_boston()
X, Y = boston.data, boston.target
plt.hist(Y, 50)
plt.xlabel('Median value (in $1000)')
print(boston.DESCR)

# Shuffle the data, but make sure that the features and accompanying labels stay in sync.
np.random.seed(0)
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]

# Split into train and test. JTODO upgrade to sklearn splitter
train_data, train_labels = X[:350], Y[:350]
test_data, test_labels = X[350:], Y[350:]

# Load Housing Data with Pandas Dataframe, include target
col_names = boston['feature_names'].tolist()
col_names.append('target')
boston_df = pd.DataFrame(np.c_[boston['data'], boston['target']], columns=col_names)

correlation_matrix = boston_df.corr()

# Use seaborn to create a pretty correlation heatmap.
sns.set(style="white")
fig, ax = plt.subplots(figsize=(9, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.zeros_like(correlation_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(correlation_matrix, cmap=cmap, mask=mask, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
fig.tight_layout()

# eta is the learning rate; smaller values will tend to give slower but more precise convergence.
# num_iters is the number of iterations to run.

import random

def gradient_descent(train_data, target_data, eta, num_iters):
    # Add a 1 to each feature vector so we learn an intercept.
    X = np.c_[np.ones(train_data.shape[0]), train_data]
    
    # m = number of samples, k = number of features
    m, k = X.shape
    
    # Initially, set all the parameters to 1.
    theta = np.array([random.random() for i in range(k)]) #np.ones(k)
    
    # Keep track of costs after each step.
    costs = []
    
    for iter in range(0, num_iters):
        # Get the current predictions for the training examples given the current estimate of theta.
        hypothesis = np.dot(X, theta)
        
        # The loss is the difference between the predictions and the actual target values.
        loss = hypothesis - target_data
        
        # In standard linear regression, we want to minimize the sum of squared losses.
        cost = np.sum(loss ** 2) / (2 * m)
        costs.append(cost)
        
        # Compute the gradient.
        gradient = np.dot(X.T, loss) / m

        # Update theta, scaling the gradient by the learning rate.
        theta = theta - eta * gradient
        
    return theta, costs

# Run gradient descent and plot the cost vs iterations.
theta, costs = gradient_descent(train_data[:,0:1], train_labels, .01, 500)
plt.plot(costs)
plt.xlabel('Iteration'), plt.ylabel('Cost')
plt.show()

def OLS(X, Y):
    # Add the intercept.
    X = np.c_[np.ones(X.shape[0]), X]
    
    # We use np.linalg.inv() to compute a matrix inverse.
    return np.dot(inv(np.dot(X.T, X)), np.dot(X.T, Y))

ols_solution = OLS(train_data[:,0:1], train_labels)

lr = LinearRegression(fit_intercept=True)
lr.fit(train_data[:,0:1], train_labels)

print('Our estimated theta:     %.4f + %.4f*CRIM' %(theta[0], theta[1]))
print('OLS estimated theta:     %.4f + %.4f*CRIM' %(ols_solution[0], ols_solution[1]))
print('sklearn estimated theta: %.4f + %.4f*CRIM' %(lr.intercept_, lr.coef_[0]))

num_feats = 5
theta, costs = gradient_descent(train_data[:,0:num_feats], train_labels, .01, 10)
plt.plot([k for k in map(np.log, costs)])
plt.xlabel('Iteration'), plt.ylabel('Log Cost')
plt.show()

start_time = time.time()
theta, costs = gradient_descent(train_data[:,0:num_feats], train_labels, .001, 100000)
train_time = time.time() - start_time
plt.plot([k for k in map(np.log, costs)])
plt.xlabel('Iteration'), plt.ylabel('Log Cost')
plt.show()

print('Training time: %.2f secs' %train_time)
print('Our estimated theta:', theta)
print('OLS estimated theta:', OLS(train_data[:,0:num_feats], train_labels))

plt.figure(figsize=(15, 3))
for feature in range(num_feats):
    plt.subplot(1, num_feats, feature+1)
    plt.hist(train_data[:,feature])
    plt.title(boston.feature_names[feature])

scaler = preprocessing.StandardScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

plt.figure(figsize=(15, 3))

for feature in range(5):
    plt.subplot(1, 5, feature+1)
    plt.hist(scaled_train_data[:,feature])
    plt.title(boston.feature_names[feature])

start_time = time.time()
theta, costs = gradient_descent(scaled_train_data[:,0:5], train_labels, .01, 5000)
train_time = time.time() - start_time
plt.plot([k for k in map(np.log, costs)])
plt.xlabel('Iteration'), plt.ylabel('Log Cost')
plt.show()

print('Training time: %.2f secs' %train_time)
print('Our estimated theta:', theta)
print('OLS estimated theta:', OLS(scaled_train_data[:,0:5], train_labels))

# Create an augmented training set that has 2 copies of the crime variable.
augmented_train_data = np.c_[scaled_train_data[:,0], scaled_train_data]

# Run gradient descent and OLS and compare the results.
theta, costs = gradient_descent(augmented_train_data[:,0:6], train_labels, .01, 5000)
print('Our estimated theta:', theta)

print('OLS estimated theta:')
try:
    print(OLS(augmented_train_data[:,0:6], train_labels))
except:
    print('ERROR, singular matrix not invertible')

from JSAnimation import IPython_display # pip install JSAnimation

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats 

from sklearn.datasets.samples_generator import make_regression 

#code adapted from http://tillbergmann.com/blog/python-gradient-descent.html
x, y = make_regression(n_samples = 100, 
                       n_features=1, 
                       n_informative=1, 
                       noise=20,
                       random_state=2017)
x = x.flatten()
slope, intercept, _,_,_ = stats.linregress(x,y)
best_fit = np.vectorize(lambda x: x * slope + intercept)
plt.plot(x,y, 'o', alpha=0.5)
grid = np.arange(-3,3,0.1)
plt.plot(grid,best_fit(grid), '.')

def gradient_descent(x, y, theta_init, step=0.001, maxsteps=0, precision=0.001, ):
    costs = []
    m = y.size # number of data points
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    counter = 0
    oldcost = 0
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        gradient = x.T.dot(error)/m 
        theta = theta - step * gradient  # update
        history.append(theta)
        
        pred = np.dot(x, theta)
        error = pred - y 
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 1 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
    return history, costs, preds, counter

xaug = np.c_[np.ones(x.shape[0]), x]
theta_i = [-1, 1] + np.random.rand(2)
history, cost, preds, iters = gradient_descent(xaug, y, theta_i, step=0.01)
theta = history[-1]
print("Gradient Descent: {:.2f}, {:.2f} {:d}".format(theta[0], theta[1], iters))
print("Least Squares: {:.2f}, {:.2f}".format(intercept, slope))

plt.plot(range(len(cost)), cost);

def init():
    line.set_data([], [])
    return line,

def animate(i):
    ys = preds[i]
    line.set_data(xaug[:, 1], ys)
    return line,

fig = plt.figure(figsize=(10,6))
ax = plt.axes(xlim=(-3, 2.5), ylim=(-170, 170))
ax.plot(xaug[:,1],y, 'o')
line, = ax.plot([], [], lw=2)
plt.plot(xaug[:,1], best_fit(xaug[:,1]), 'k-', color = "r")

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(preds), interval=75)
#anim.save('grad_images/gdline.mp4') Errors with html when run in notebook
anim

"""
Playground!

Parameters to change:
 * init_theta = [-100, -100] + np.random.rand(2)
 * step=0.01
"""

# TODO change the initial conditions and see how it changes the rate of convergence
init_theta = [-100, -100] + np.random.rand(2)
_, __, preds, ___ = gradient_descent(xaug, y, init_theta, step=1.0)

fig = plt.figure(figsize=(10,6))
ax = plt.axes(xlim=(-3, 2.5), ylim=(-170, 170))
ax.plot(xaug[:,1],y, 'o')
line, = ax.plot([], [], lw=2)

print('Converged in %i steps.' % len(preds))

# Plot the best solution
plt.plot(xaug[:,1], best_fit(xaug[:,1]), 'k-', color = "r")
ax.set_title('My Playground for different initial Conditions', size='xx-large')

# Animate the convergence
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(preds), interval=10)
anim

