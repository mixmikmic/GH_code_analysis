# Import all required libraries
from __future__ import division # For python 2.*

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

np.random.seed(0)
get_ipython().run_line_magic('matplotlib', 'inline')

X = np.genfromtxt("data/X_train.txt",delimiter=None)
Y = np.genfromtxt("data/Y_train.txt",delimiter=None)
[Xtr,Xva,Ytr,Yva] = ml.splitData(X,Y,0.80)

Xte = np.genfromtxt('data/X_test.txt',delimiter=None)
Xt, Yt = Xtr[:4000], Ytr[:4000]

tree_one = ml.dtree.treeClassify(Xt, Yt, minParent=2**6, maxDepth=25, nFeatures=6)  # The nFeatures makes it random
probs = tree_one.predictSoft(Xte)

print("{0:>15}: {1:.4f}".format('Train AUC', tree_one.auc(Xt, Yt)))
print("{0:>15}: {1:.4f}".format('Validation AUC', tree_one.auc(Xva, Yva)))

np.random.seed(0)  # Resetting the seed in case you ran other stuff.
n_bags = 10
bags = []   # self.learners
for l in range(n_bags):
    # Each boosted data is the size of the original data. 
    Xi, Yi = ml.bootstrapData(Xt, Yt, Xt.shape[0])

    # Train the model on that draw
    tree = ml.dtree.treeClassify(Xi, Yi, minParent=2**6,maxDepth=25, nFeatures=6)
    bags.append(tree)

for l in range(n_bags):
    print(l)
    print("{0:>15}: {1:.4f}".format('Train AUC', bags[l].auc(Xt, Yt)))
    print("{0:>15}: {1:.4f}".format('Validation AUC', bags[l].auc(Xva, Yva)))

class BaggedTree(ml.base.classifier):
    def __init__(self, learners):
        """Constructs a BaggedTree class with a set of learners. """
        self.learners = learners
    
    def predictSoft(self, X):
        """Predicts the probabilities with each bagged learner and average over the results. """
        n_bags = len(self.learners)
        preds = [self.learners[l].predictSoft(X) for l in range(n_bags)]
        return np.mean(preds, axis=0)

bt = BaggedTree(bags)
bt.classes = np.unique(Y)

print("{0:>15}: {1:.4f}".format('Train AUC', bt.auc(Xt, Yt)))
print("{0:>15}: {1:.4f}".format('Validation AUC', bt.auc(Xva, Yva)))

path_to_file = './data/poly_data.txt' 
data = np.genfromtxt(path_to_file, delimiter='\t') # Read data from file

X, Y = np.atleast_2d(data[:, 0]).T, data[:, 1]
X, Y = ml.shuffleData(X, Y)
Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)

# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-13, 18)
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=30, loc=0)

plt.show()

boosts = []
n_boosts = 20

Ytr_ = np.copy(Ytr)  # We're going to copy the data becuase each booster iteration we're going to mess with it.

for i in range(n_boosts):
    tree = ml.dtree.treeRegress(Xtr, Ytr_, maxDepth=1)
    boosts.append(tree)
    
    # Now "learning" from out mistakes.
    Ytr_ -= tree.predict(Xtr) 

xs = np.linspace(0, 4.2, 200)
xs = np.atleast_2d(xs).T

ys = boosts[0].predict(xs)

# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')

ax.plot(xs, ys, lw=3, color='black', alpha=0.75, label='Prediction')

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-13, 18)
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=30, loc=0)

plt.show()

def predict(X, boosts):
    """Predicts regression values using boosting. """
    preds = [boosts[i].predict(X) for i in range(len(boosts))]
    
    # Notice that in the bagging we returning the mean, here we return the sum
    return np.sum(preds, axis=0)

xs = np.linspace(0, 4.2, 200)
xs = np.atleast_2d(xs).T

ys = predict(xs, boosts)

# Plotting the data
f, ax = plt.subplots(1, 1, figsize=(10, 8))
    
ax.scatter(Xtr, Ytr, s=80, color='blue', alpha=0.75, label='Train')
ax.scatter(Xte, Yte, s=240, marker='*', color='red', alpha=0.75, label='Test')

ax.plot(xs, ys, lw=3, color='black', alpha=0.75, label='Prediction')

ax.set_xlim(-0.2, 4.3)
ax.set_ylim(-13, 18)
ax.set_xticklabels(ax.get_xticks(), fontsize=25)
ax.set_yticklabels(ax.get_yticks(), fontsize=25)   

# Controlling the size of the legend and the location.
ax.legend(fontsize=30, loc=0)

plt.show()



