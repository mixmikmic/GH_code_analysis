get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as datasets
import sklearn.metrics as metrics
import sklearn.utils as utils

import scipy.sparse.linalg as linalg
import scipy.cluster.hierarchy as hr
import sklearn.cluster as cluster
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import slideUtilities as sl
import laUtilities as ut
from importlib import reload
from datetime import datetime
from IPython.display import Image
from IPython.display import display_html
from IPython.display import display
from IPython.display import Math
from IPython.display import Latex
from IPython.display import HTML
print('')

get_ipython().run_cell_magic('html', '', '<style>\n .container.slides .celltoolbar, .container.slides .hide-in-slideshow {\n    display: None ! important;\n}\n</style>')

X, y = datasets.make_circles(noise=.1, factor=.5, random_state=1)
print('Shape of data: {}'.format(X.shape))
print('Unique labels: {}'.format(np.unique(y)))

plt.prism()  # this sets a nice color map
plt.scatter(X[:, 0], X[:, 1], c=y)
_ = plt.axis('equal')

# Normally we would randomly permute the data rows first, 
# but this is random synthetic data so it's not necessary.
# Make sure you permute X and y with the same permutation :)

X_train = X[:50]
y_train = y[:50]
X_test = X[50:]
y_test = y[50:]


plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, )
plt.axis('equal')
plt.title('Training Data')
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.title('Test Data')
_=plt.axis('equal')

k = 5
knn = KNeighborsClassifier(n_neighbors=k)    

knn.fit(X_train,y_train)
y_pred_test = knn.predict(X_test)
print('Accuracy on test data: {}'.format(knn.score(X_test, y_test)))

y_pred_train = knn.predict(X_train)
print('Accuracy on training data: {}'.format(knn.score(X_train, y_train)))

plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.axis('equal')
plt.title('Training')
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test)
plt.title('Testing $k$={}\nAccuracy: {}'.format(k,knn.score(X_test, y_test)))
_=plt.axis('equal')

test_point = np.argmax(X_test[:,1])
X_test[test_point]
neighbors = knn.kneighbors([X_test[test_point]])[1]

# This code generates the information about the decision region
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,)
plt.scatter(X_train[neighbors,0], X_train[neighbors,1], c = y_train[neighbors], marker='s', s=30)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired,
                  alpha=0.3)

plt.axis('equal')
plt.title(r'Training')
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test)
plt.scatter(X_test[test_point,0], X_test[test_point,1], c='b')
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired,
                  alpha=0.3)

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
plot_step = 0.02
plt.title('Testing $k$={}\nAccuracy: {}'.format(k,knn.score(X_test, y_test)))
_=plt.axis('equal')

k = 3
knn = KNeighborsClassifier(n_neighbors=k)    
knn.fit(X_train,y_train)
y_pred_test = knn.predict(X_test)
plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, )
plt.axis('equal')
plt.title(r'Training')
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test)
plt.title('Testing $k$={}\nAccuracy: {}'.format(k,knn.score(X_test, y_test)))
_=plt.axis('equal')

test_point = np.argmax(X_test[:,1])
X_test[test_point]
neighbors = knn.kneighbors([X_test[test_point]])[1]

# This code generates the information about the decision region
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(121)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,)
plt.scatter(X_train[neighbors,0], X_train[neighbors,1], c = y_train[neighbors], marker='s', s=30)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired,
                  alpha=0.3)
plt.axis('equal')
plt.title(r'Training')
plt.subplot(122)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test)
plt.scatter(X_test[test_point,0], X_test[test_point,1], c='b')
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired,
                  alpha=0.3)
plt.title('Testing $k$={}\nAccuracy: {}'.format(k,knn.score(X_test, y_test)))
_=plt.axis('equal')

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)
y_pred_test = dtc.predict(X_test)
print('DT accuracy on test data: ', dtc.score(X_test, y_test))
y_pred_train = dtc.predict(X_train)
print('DT accuracy on training data: ', dtc.score(X_train, y_train))

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, marker='^',s=30)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30)
plt.axis('equal')
_=plt.title('Decision Tree\n Triangles: Test Data, Circles: Training Data\nAccuracy: {}'.format(dtc.score(X_test, y_test)))

x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
plot_step = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
Z = dtc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired,
                  alpha=0.3)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, marker='^',s=30)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30)
plt.axis('equal')
plt.xlim((x_min, x_max))
plt.ylim((y_min, y_max))
_=plt.title('Decision Tree\n Triangles: Test Data, Circles: Training Data\nAccuracy: {}'.format(dtc.score(X_test, y_test)))

dot_data = tree.export_graphviz(dtc, out_file=None,
                         feature_names=['X','Y'],
                         class_names=['Red','Green'],
                         filled=True, rounded=True,  
                         special_characters=True) 
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_pdf("dt.pdf") 
Image(graph.create_png())  

sl.hide_code_in_slideshow()
display(Image("figs/R._A._Fisher.png", width=125))

iris = datasets.load_iris()

X = iris.data
y = iris.target
ynames = iris.target_names
print(X.shape, y.shape)
print(X[1,:])
print(iris.target_names)
print(y)

#Randomly suffle the data
X, y = utils.shuffle(X, y, random_state=1)
y

train_set_size = 100
X_train = X[:train_set_size]  # selects first 100 rows (examples) for train set
y_train = y[:train_set_size]
X_test = X[train_set_size:]   # selects from row 100 until the last one for test set
y_test = y[train_set_size:]
print(X_train.shape), y_train.shape
print(X_test.shape), y_test.shape

k = 5
knn = KNeighborsClassifier(n_neighbors=k)

knn.fit(X_train, y_train)
y_pred_test = knn.predict(X_test)
print("Accuracy of KNN test set:", knn.score(X_test, y_test))

# Create color maps
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# we will use only the first two (of four) features, so we can visualize
X = X_train[:, :2] 
h = .02  # step size in the mesh
k = 25
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y_train)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y_train, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
_ = plt.title("3-Class classification (k = {})".format(k))

dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred_test = dtc.predict(X_test)
print("Accuracy of DTC test set:", dtc.score(X_test, y_test))

# we will use only the first two (of four) features, so we can visualize
X = X_train[:, :2] 
h = .02  # step size in the mesh
dtc.fit(X, y_train)
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
Z = dtc.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y_train, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
_ = plt.title("3-Class classification (k = {})".format(k))

digits = datasets.load_digits()
X_digits, y_digits = digits.data, digits.target

print ('Data shape: {}'.format(X_digits.shape))
print ('Data labels: {}'.format(y_digits))
print ('Unique labels: {}'.format(digits.target_names))
X_digits, y_digits = utils.shuffle(X_digits, y_digits, random_state=1)

digits.images[3]

plt.gray() 
plt.rc('axes', grid=False)
_=plt.matshow(digits.images[3],cmap=plt.cm.gray_r) 

plt.rc('image', cmap='binary', interpolation='bilinear')
plt.rc('axes', grid=False)
plt.figure(figsize=(4,4))
_=plt.imshow(digits.images[3])

for t in range(4):
    plt.figure(figsize=(8,2))
    for j in range(4):
        plt.subplot(1, 4, 1 + j)
        plt.imshow(X_digits[4*t + j].reshape(8, 8))

step = 1000;
X_digits_train = X_digits[:step]
y_digits_train = y_digits[:step]
X_digits_test = X_digits[step:len(y_digits)]
y_digits_test = y_digits[step:len(y_digits)]

knn_digits = KNeighborsClassifier(n_neighbors=20)
_=knn_digits.fit(X_digits_train, y_digits_train)

acc = []
for k in range(1,60):
    knn_digits = KNeighborsClassifier(n_neighbors=k)
    knn_digits.fit(X_digits_train, y_digits_train)
    y_digits_test_pred = knn_digits.predict(X_digits_test)
    # print("KNN test accuracy on MNIST digits, k = {}, acc = {}: ".format(
            #k,knn_digits.score(X_digits_test, y_digits_test)))
    acc.append(knn_digits.score(X_digits_test, y_digits_test))

plt.plot(acc,'.')
plt.xlabel('k')
_=plt.ylabel('Accuracy')

neighbors = knn_digits.kneighbors(X_digits_test, n_neighbors=3, return_distance=False)
print(type(neighbors))
print(neighbors.shape)

plt.rc("image", cmap="binary")  # this sets a black on white colormap
# plot X_digits_valid[0]
for t in range(3):
    plt.figure(figsize=(8,2))
    plt.subplot(1, 4, 1)
    plt.imshow(X_digits_test[t].reshape(8, 8))
    plt.title("Query")
    # plot three nearest neighbors from the training set
    for i in [0, 1, 2]:
        plt.subplot(1, 4, 2 + i)
        plt.title("neighbor {}".format(i))
        plt.imshow(X_digits_train[neighbors[t, i]].reshape(8, 8)) 

