import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

from SGD import Regression, Classification

def plot(l, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(range(len(l)), l)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.set_size_inches((5.5,3))
    fig.patch.set_alpha(0.0)

X = np.array([
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0.5, 0.5],
    [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0.5, 0, 0.5],
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0.5, 0.5, 0]
    ])
y = np.array([5, 3, 1, 4, 5, 1, 5])

n_sample = y.size

# learn incrementally
regressor = Regression(X, y)

RMSEs = []
for i in range(600):
    # update with randomly sampled one
    j = np.random.randint(n_sample)
    RMSE = regressor.update(X[j], y[j])
    RMSEs.append(RMSE)
    
plot(RMSEs, 'Iteration', 'RMSE')

# learn at once
# 1 iteration = descent for all samples
regressor = Regression(X, y)
RMSEs = regressor.fit()
plot(RMSEs, 'Iteration', 'RMSE')

# give labels for the mock dataset
y_labels = np.ones_like(y)
y_labels[y < np.mean(y)] = -1
y_labels

# learn incrementally
classifier = Classification(X, y_labels)

AUCs = []
for i in range(100):
    # update with random sample
    j = np.random.randint(n_sample)
    current = classifier.update(X[j], y_labels[j])
    AUCs.append(current)

plot(AUCs, 'Iteration', 'AUC')



