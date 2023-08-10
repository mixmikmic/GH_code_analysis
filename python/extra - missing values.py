import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(precision=3)
get_ipython().magic('matplotlib inline')
plt.rcParams["figure.dpi"] = 300

from sklearn.datasets import load_iris
from sklearn.utils import shuffle
iris = load_iris()

X, y = iris.data, iris.target
X, y = shuffle(X, y)

print(X[:30])

# a column is mostly missing
rng = np.random.RandomState(0)
X_missing_column = X.copy()
mask = X.sum(axis=1) < rng.normal(loc=19, scale=3, size=X.shape[0])
X_missing_column[mask, 0] = np.NaN
X_missing_column[120:]

# only a few rows have missing data. but a lot of it
rng = np.random.RandomState(4)
X_missing_rows = X.copy()
for i in rng.randint(0, 30, 5):
    X_missing_rows[i, rng.uniform(size=4)> .2] = np.NaN
X_missing_rows[:30]

X[y==2].mean(axis=0)

# some values missing only
rng = np.random.RandomState(0)
X_some_missing = X.copy()
mask = np.abs(X[:, 2] - rng.normal(loc=5.5, scale=.7, size=X.shape[0])) < .6
X_some_missing[mask, 3] = np.NaN
# different random numbers
mask2 = np.abs(X[:, 2] - rng.normal(loc=5.5, scale=.7, size=X.shape[0])) < .6
X_some_missing[mask2, 2] = np.NaN
X_some_missing[:30]

# from now on use X_ = X_some_missing
X_ = X_some_missing

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X_, y, stratify=y, random_state=0)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

nan_columns = np.any(np.isnan(X_train), axis=0)
X_drop_columns = X_train[:, ~nan_columns]
logreg = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(logreg, X_drop_columns, y_train, cv=10)
np.mean(scores)

print(X_train[-30:])



from sklearn.preprocessing import Imputer
imp = Imputer(strategy="mean").fit(X_train)
X_mean_imp = imp.transform(X_train)
X_mean_imp[-30:]

X_mean_imp.shape

import matplotlib.patches as patches
imputed_mask = np.any(np.isnan(X_train), axis=1)

def plot_imputation(X_imp, title=None, ax=None):
    # helper function to plot imputed data points
    if ax is None:
        ax = plt.gca()
    if title is not None:
        ax.set_title(title)
    ax.scatter(X_imp[imputed_mask, 2], X_imp[imputed_mask, 3], c=plt.cm.Vega10(y_train[imputed_mask]), alpha=.6, marker="s")
    ax.scatter(X_imp[~imputed_mask, 2], X_imp[~imputed_mask, 3], c=plt.cm.Vega10(y_train[~imputed_mask]), alpha=.6)
    # this is for creating the legend...
    square = plt.Line2D((0,), (0,), linestyle='', marker="s", markerfacecolor="w", markeredgecolor="k", label='Imputed data')
    circle = plt.Line2D((0,), (0,), linestyle='', marker="o", markerfacecolor="w", markeredgecolor="k", label='Real data')
    plt.legend(handles=[square, circle], numpoints=1, loc="best")

plot_imputation(X_mean_imp, "Mean imputation")

# I designed the problem so that mean imputation wouldn't work

mean_pipe = make_pipeline(Imputer(), StandardScaler(), LogisticRegression())
scores = cross_val_score(mean_pipe, X_train, y_train, cv=10)
np.mean(scores)

from sklearn.neighbors import KNeighborsRegressor

# imput feature 2 with KNN
feature2_missing = np.isnan(X_train[:, 2])
knn_feature2 = KNeighborsRegressor().fit(X_train[~feature2_missing, :2],
                                         X_train[~feature2_missing, 2])

X_train_knn2 = X_train.copy()
X_train_knn2[feature2_missing, 2] = knn_feature2.predict(X_train[feature2_missing, :2])

# impute feature 3 with KNN
feature3_missing = np.isnan(X_train[:, 3])
knn_feature3 = KNeighborsRegressor().fit(X_train[~feature3_missing, :2],
                                         X_train[~feature3_missing, 3])

X_train_knn3 = X_train_knn2.copy()
X_train_knn3[feature3_missing, 3] = knn_feature3.predict(X_train[feature3_missing, :2])

plot_imputation(X_train_knn3, "Simple KNN imputation")

# this is cheating because I'm not using a pipeline
# we would need to write a transformer that does the imputation
scores = cross_val_score(logreg, X_train_knn3, y_train, cv=10)
np.mean(scores)

from sklearn.ensemble import RandomForestRegressor


# this is just because I'm lazy and don't want to special-case the first iteration
X_imputed = Imputer().fit_transform(X_train)
feature2_missing = np.isnan(X_train[:, 2])
feature3_missing = np.isnan(X_train[:, 3])

inds_not_2 = np.array([0, 1, 3])
inds_not_3 = np.array([0, 1, 2])

rf = RandomForestRegressor(n_estimators=100)

for i in range(10):
    last = X_imputed.copy()
    # imput feature 2 with rf
    
    rf.fit(X_imputed[~feature2_missing][:, inds_not_2], X_train[~feature2_missing, 2])

    X_imputed[feature2_missing, 2] = rf.predict(X_imputed[feature2_missing][:, inds_not_2])

    # impute feature 3 with rf
    
    rf.fit(X_imputed[~feature3_missing][:, inds_not_3], X_train[~feature3_missing, 3])
    X_imputed[feature3_missing, 3] = rf.predict(X_imputed[feature3_missing][:, inds_not_3])
    
    # this would make more sense if we scaled the data beforehand
    if (np.linalg.norm(last - X_imputed)) < .5:
        break

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plot_imputation(X_mean_imp, "Mean", ax=axes[0])
plot_imputation(X_train_knn3, "KNN", ax=axes[1])
plot_imputation(X_imputed, "Random Forest imputation", ax=axes[2])

scores = cross_val_score(logreg, X_imputed, y_train, cv=10)
np.mean(scores)

# you need to pip install fancyimpute for the rest! - and tensorflow
import fancyimpute
X_train_fancy_knn = fancyimpute.KNN().complete(X_train)

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
plot_imputation(X_train_knn3, "Naive KNN", ax=ax[0])
plot_imputation(X_train_fancy_knn, "Fancy KNN", ax=ax[1])

X_train_fancy_simple = fancyimpute.SimpleFill().complete(X_train)
X_train_fancy_mice = fancyimpute.MICE(verbose=0).complete(X_train)
X_train_fancy_si = fancyimpute.SoftImpute(verbose=0).complete(X_train)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=100)
for ax, name, X_imp in zip(axes.ravel(), ["simple", "KNN", "MICE", "Soft impute"],
                           [X_train_fancy_simple, X_train_fancy_knn, X_train_fancy_mice, X_train_fancy_si]):
    plot_imputation(X_imp, name, ax=ax)

mice = fancyimpute.MICE(verbose=0)
X_train_fancy_mice = mice.complete(X_train)
scores = cross_val_score(logreg, X_train_fancy_mice, y_train, cv=10)
scores.mean()

