# Import useful libraries used in the notebook
import time
import numpy as np
import matplotlib.pyplot as plt

# Show plots inline 
get_ipython().magic('matplotlib inline')

from sklearn.datasets import make_classification

from pylab import rcParams

# Functions are defined in this module
from acm_imbalanced_library import *

# Auto-reload external modules
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
rcParams['figure.figsize'] = (10.0, 5.0)

def comp_scatter(x_left, y_left, title_left, x_right=None, y_right=None, title_right=None):
    '''
    Utility function to create comparison scatter plots. 
    '''
    # Plot left-hand scatterplot
    plt1 = plt.subplot(121)
    plt.title(title_left, fontsize='large')
    y_left = np.squeeze(y_left)
    plt1.scatter(x_left[y_left == 0][:, 0], x_left[y_left == 0][:, 1], s=100, marker = "x", color="black")
    plt1.scatter(x_left[y_left == 1][:, 0], x_left[y_left == 1][:, 1], s=100, marker = "s", color="red")
    if (title_right == None):
        return 
    
    # Plot right hand scatterplot
    plt.subplot(122, sharex=plt1, sharey=plt1)
    plt.title(title_right, fontsize='large')
    y_right = np.squeeze(y_right)
    plt.scatter(x_right[y_right == 0][:, 0], x_right[y_right == 0][:, 1], s=100, marker = "x", color="black")
    plt.scatter(x_right[y_right == 1][:, 0], x_right[y_right == 1][:, 1], s=100, marker = "s", color="red")

    return 

# plt1 = plt.subplot(121)
# plt.title("Original dataset", fontsize='large')
X, y = make_classification(n_samples=100, n_features=2, 
                           n_informative=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=2, weights=[0.9, 0.1],
                           random_state=23)
comp_scatter(X, y, 'Original dataset')

# Randomly oversample the data
X_oversamp, y_oversamp = randomOversample(X, y)

comp_scatter(X, y, 'Original dataset', 
             X_oversamp, y_oversamp, 'Random oversampled dataset')


print 'Y value counts (original data) = {}'.format(np.unique(y, return_counts = True))
print 'Y value counts (oversampled data) = {}'.format(np.unique(y_oversamp, return_counts=True))


# Randomly undersample the data
X_undersamp, y_undersamp = randomUndersample(X, y)

comp_scatter(X, y, 'Original dataset', 
             X_undersamp, y_undersamp, 'Random undersampled dataset')

X_smote, y_smote = SMOTEoversample(X, y)

comp_scatter(X, y, 'Original dataset', 
             X_smote, y_smote, 'SMOTE dataset')

# Apply tomek undersampling to the dataset 10 times

X_tomek = X
y_tomek = y

for i in range(10):
    X_tomek, y_tomek = TomekUndersample(X_tomek, y_tomek)

comp_scatter(X, y, 'Original dataset', 
             X_tomek, y_tomek, 'Tomek link dataset')

X_smote, y_smote = SMOTEoversample(X, y)

for i in range(10):
    X_smote_tomek, y_smote_tomek = TomekUndersample(X_smote, y_smote)

comp_scatter(X, y, 'Original dataset', 
             X_smote_tomek, y_smote_tomek, 'Tomek link dataset')



