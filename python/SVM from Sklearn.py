import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

## Load iris dataset
iris = datasets.load_iris()
x = iris.data[:,0:2]    ## we are taking only 2 features inorder to simplify plotting in 2d only.
y = iris.target

## Train test split of data
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 0)

clf = svm.SVC()
clf.fit(x_train, y_train)

clf.score(x_test, y_test)

def makegrid(x1, x2, h = 0.02):
    ##Calculating region where we would plot.
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    
    ## Making an array of all points between with separatin h
    a = np.arange(x1_min, x1_max, h)
    b = np.arange(x2_min, x2_max, h)
    
    ## Replicates each point multiple times to make a mesh grid
    xx, yy = np.meshgrid(a, b)
    
    return xx, yy

xx, yy = makegrid(x[:,0], x[:,1])

predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])
plt.scatter(xx.ravel(), yy.ravel(), c= predictions)
plt.show()

a = np.arange(1,3,0.2)
b = np.arange(4,6,0.2)
xx, yy = np.meshgrid(a,b)
(xx*yy*xx).shape

