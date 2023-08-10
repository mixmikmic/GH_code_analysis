import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# make a random regression example
X, y = make_regression(n_features=1, n_samples=100, noise=10)

plt.scatter(X, y)

linear_regressor = LinearRegression(fit_intercept=True, normalize=False)  # define an instance
linear_regressor.fit(X, y)  # fit on the data

print("Slope is %.2f"%linear_regressor.coef_)
print("Intercept is %.2f"%linear_regressor.intercept_)

linear_regressor.predict(3)

x_new = np.linspace(-2, 2).reshape((-1, 1))
pred_new = linear_regressor.predict(x_new)

fig, ax = plt.subplots()

ax.scatter(X,y)
ax.plot(x_new, pred_new, "r-")

from sklearn.datasets import make_classification

X, y = make_classification(n_classes=3, n_clusters_per_class=1, n_samples=500, shuffle=True, n_features=2, class_sep=0.8, n_redundant=0)

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(X[:,0], X[:,1], c=y, s=40)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

# we will fit on the first 400 instances and test on the last 100 instances

classifier.fit(X[:400], y[:400])

from sklearn.metrics import classification_report

y_predicted = classifier.predict(X[400:])

print(classification_report(y_predicted, y[400:]))

xx1, xx2 = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 100), np.linspace(X[:,1].min(), X[:,1].max(), 100))
Z = classifier.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

fig, ax = plt.subplots(figsize=(10, 10))
ax.contourf(xx1, xx2, Z, alpha=0.4)
ax.scatter(X[:,0], X[:,1], c=y, s=40)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()
classifier = DecisionTreeClassifier()

scaling_classifier = Pipeline([("normalize", normalizer), ("classify", classifier)])

scaling_classifier.fit(X, y)

scaling_classifier.predict(X)

from sklearn.datasets import fetch_20newsgroups
newsgroup_train = fetch_20newsgroups(subset="train")
newsgroup_test = fetch_20newsgroups(subset="test")

for label in newsgroup_train.target_names:
    print(label)

print(newsgroup_train.data[0])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

count_vectorizer = CountVectorizer()
tfid = TfidfTransformer()
classifier = SGDClassifier(alpha=0.1, loss="hinge")

text_classifier = Pipeline([("vect", count_vectorizer), ("tfid", tfid), ("clf", classifier)])

text_classifier.fit(newsgroup_train.data, newsgroup_train.target)

prediced_labels = text_classifier.predict(newsgroup_test.data)
print(classification_report(newsgroup_test.target, prediced_labels))

from sklearn.grid_search import GridSearchCV

# define grid for parameters to search
parameters = {"clf__alpha": [1e-6, 1e-4, 0.001, 0.01], "clf__loss" : ["hinge", "log"]}

grid_search = GridSearchCV(text_classifier, parameters, n_jobs=-1, cv=4, verbose=True)  # use fourfold CV to obtain best model

grid_search.fit(newsgroup_train.data, newsgroup_train.target)

grid_search.best_params_

prediced_labels = grid_search.predict(newsgroup_test.data)
print(classification_report(newsgroup_test.target, prediced_labels))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds in datasets:
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ("%.2f" % score).lstrip("0"),
                size=15, horizontalalignment="right")
        i += 1

figure.subplots_adjust(left=.02, right=.98)

# for randomly choosing a subset
from random import shuffle
indices = range(150)
shuffle(indices)
train_indices = indices[:100]
test_indices = indices[100:]

# for dealing with the labels
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# classifiers
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

