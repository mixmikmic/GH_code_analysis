get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from learn_interactive import DecisionTree, make_random_tree, canvas_dim
from sklearn.cross_validation import train_test_split

real_tree = make_random_tree()
print real_tree

n_points = 100
noise = 0.2

# generate some noisy labels
points = np.random.randint(0, canvas_dim, (n_points, 2))
labels = np.zeros((points.shape[0],),dtype=np.bool_)

for i in range(len(labels)):
    label, _ = real_tree.evaluate(points[i,0], points[i,1])
    if np.random.rand() < noise:
        label = not label
    labels[i] = label

# plot the points (blue for positive, red for negative)
plt.plot(points[labels==1,0], points[labels==1,1],'b.')
plt.plot(points[labels==0,0], points[labels==0,1],'r.')

def best_split(data, targets):
    """ Find the best split for a given dataset.
        data is an nx2 numpy array where n is the number of data points.  The first column
        of data contains the x coordinates of the points, and the second contains the y
        coordinates.
        targets is an n dimensional numpy array containing the binary target values (0, 1)
        
        returns: a tuple consisting of the variable to split on and the threshold to split at.
                 all splits are evaluated as variable_name > threshold.
        """
    # this starter code is designed to help, but if you want to start from scratch, please
    # go ahead.
    best_impurity = np.inf
    split_variable = None
    split_threshold = None
    variable_names = ['x', 'y']
    for i in range(data.shape[1]):
        for threshold in np.arange(-1.0, canvas_dim+1, 1):
            # loop over all possible splitting variables and thresholds
            pass

split_variable, split_threshold = best_split(points, labels)
print split_variable, split_threshold

# visualize the best split computed above

plt.plot(points[labels==1,0], points[labels==1,1],'b.')
plt.plot(points[labels==0,0], points[labels==0,1],'r.')
if split_variable == "x":
    plt.plot([split_threshold, split_threshold], [0, canvas_dim], 'k')
else:
    plt.plot([0, canvas_dim],[split_threshold, split_threshold], 'k')



