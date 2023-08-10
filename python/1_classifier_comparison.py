get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

cols_cmap = ['#FF0000', '#0000FF']
cmap = ListedColormap(cols_cmap)

from sklearn.datasets import load_iris
iris = load_iris()
iris.keys()

# These datasets come with a quick description
print(iris['DESCR'])

# We can see that "data" consists of an array
# It is shape (n_samples, n_features)
print(iris['data'][[0, 1, 2, -3, -2, -1], :])
print('---')
print(iris['target'][[0, 1, 2, -3, -2, -1]])

# Return a boolean mask for the targets we care about
targets = [0, 2]
mask_samples = np.array([ii in targets
                         for ii in iris['target']], dtype=bool)

# These are the features we'll use
features = [0, 1]
name1, name2 = [iris['feature_names'][ii] for ii in targets]

# Now pull the datapoints
X = iris['data'][:, features][mask_samples]
y = iris['target'][mask_samples]

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=100)
ax.set(xlabel=name1, ylabel=name2)

from sklearn.svm import SVC
svc = SVC(C=1., kernel='linear')

# Fit the model with the given features / classes
svc.fit(X, y)

# Now we can use this model to predict the class of new data
svc.predict(X[:10])

from sklearn.model_selection import train_test_split, KFold, ShuffleSplit

X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=1337)

print(X_train.shape)
print(X_test.shape)

# Training the model requires both input and output data
svc.fit(X_train, y_train)

predictions = svc.predict(X_test)
print(predictions)

# We'll set the training data to be white on the inside
# Then we can compare it with the model predictions
edgecolors = np.where(y_train == targets[0], cols_cmap[0], cols_cmap[1])

# White will be the training data
# Filled-in will be the testing data
fig, ax = plt.subplots()
ax.scatter(X_train[:, 0], X_train[:, 1], c='w', s=200,
           alpha=.2, edgecolors=edgecolors)
ax.scatter(X_test[:, 0], X_test[:, 1], c=predictions, s=100, cmap=cmap)
ax.set(xlabel=name1, ylabel=name2)

# There is one coefficient for each feature
svc.coef_

# Remake the scatterplot
fig, ax = plt.subplots()
ax.scatter(X_train[:, 0], X_train[:, 1], c='w', s=200,
           alpha=.2, edgecolors=edgecolors)
ax.scatter(X_test[:, 0], X_test[:, 1], c=predictions, s=100, cmap=cmap)

# Now plot the shape of the dividing hyperplane
plt_x = np.linspace(*ax.get_xlim(), num=100)
w = svc.coef_[0]
a = -w[0] / w[1]
line_y = a * plt_x - svc.intercept_[0] / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = svc.support_vectors_[0]
yy_down = a * plt_x + (b[1] - a * b[0])

b = svc.support_vectors_[-1]
yy_up = a * plt_x + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
ax.plot(plt_x, line_y, 'k')
ax.plot(plt_x, yy_down, 'k--')
ax.plot(plt_x, yy_up, 'k--')
plt.autoscale(tight=True)

# Classes like `ShuffleSplit` randomly shuffle all the data on each split
cv1 = ShuffleSplit(n_splits=10, test_size=.2)
label1 = 'Shuffle Split'

# Classes like `KFold` let you split the data in structured subsets
cv2 = KFold(n_splits=10)
label2 = 'KFold'

for cv, label in [(cv1, label1), (cv2, label2)]:
    fig, ax = plt.subplots()
    for ii, (tr, tt) in enumerate(cv.split(X)):
        ax.plot(np.tile(ii, [tr.shape[0]]), tr, 'or', ms=4)
        ax.plot(np.tile(ii, [tt.shape[0]]), tt, 'ob', ms=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Index')
    ax.set_title(label)

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# Collect some extra parameters for this viz
h = .02  # step size in the mesh
X_plt, y_plt = X.copy(), y.copy()  # So we don't overwrite these
linearly_separable = (X_plt, y_plt)

# Define the datasets we'll use (we'll loop through these)
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

# Create the figure and then populate it with data
fig, axs = plt.subplots(len(classifiers), len(datasets), figsize=(9, 27))
# iterate over datasets
for ii, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test =         train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    for ax in axs[:, ii]:
        if ii == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

    # iterate over classifiers
    for jj, (name, clf) in enumerate(zip(names, classifiers)):
        ax = axs[jj, ii]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ii == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')

plt.tight_layout()
plt.show()

