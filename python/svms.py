get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import randint as sp_randint  # talk about this one
from scipy.stats import uniform

from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

ofaces = fetch_olivetti_faces()
print(ofaces.DESCR)

fig, axes = plt.subplots(10, 10, figsize=(16, 16), subplot_kw={'xticks':[], 'yticks':[]})
fig.subplots_adjust(hspace=0, wspace=0)
for i, ax in enumerate(axes.flat):
    ax.imshow(ofaces.images[i*2], cmap='binary_r')

model = make_pipeline(PCA(n_components=200, svd_solver='randomized'), SVC())
# we need to force a shuffle for this dataset
strategy = KFold(n_splits=5, shuffle=True)
# can be dist or list of dicts
param_grid = [
    {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100, 1000, 10000]},
    {'svc__kernel': ['rbf'], 'svc__C': [1, 10, 1000, 10000], 'svc__gamma': [0.001, 0.1, 1.0, 10.0]}
]
grid = GridSearchCV(model, param_grid, cv=strategy)
# this will take a while
grid.fit(ofaces.data, ofaces.target)
grid.best_estimator_

grid.best_score_

model = make_pipeline(PCA(n_components=200, svd_solver='randomized'), SVC())
strategy = KFold(n_splits=5, shuffle=True)
# cannot provide a list
param_dist = {
    'svc__kernel': ['linear', 'rbf'],
    'svc__C': sp_randint(1, 100000),
    'svc__gamma': uniform(0.001, 10.0)
}
search = RandomizedSearchCV(model, param_dist, cv=strategy, n_iter=20)
# this will take a while
search.fit(ofaces.data, ofaces.target)
search.best_estimator_

search.best_score_

faces = fetch_lfw_people(min_faces_per_person=50)
len(faces.data), faces.target_names

fig, axes = plt.subplots(1, 11, figsize=(16, 3), subplot_kw={'xticks':[], 'yticks':[]})
fig.subplots_adjust(hspace=0.1, wspace=0.1)
first_img = [np.argmax(faces.target == x) for x in list(range(len(faces.target_names)))]
for i, ax in enumerate(axes.flat):
    idx = first_img[i]
    ax.imshow(faces.data[idx].reshape(62, 47), cmap='binary_r')
    if i % 2:
        ax.set_title(faces.target_names[i], fontsize=10)
    else:
        ax.set_xlabel(faces.target_names[i], fontsize=10)

xtrain, xtest, ytrain, ytest = train_test_split(faces.data, faces.target, test_size=0.2)

model = make_pipeline(PCA(n_components=200, svd_solver='randomized'), SVC())
param_dist = {
    'svc__kernel': ['linear', 'rbf'],
    'svc__C': sp_randint(1, 100000),
    'svc__gamma': uniform(0.001, 10.0)
}
search = RandomizedSearchCV(model, param_dist, cv=5, n_iter=20)
# this will take a while
search.fit(xtrain, ytrain)
search.best_estimator_

search.best_score_

yfit = search.best_estimator_.predict(xtest)
print(classification_report(ytest, yfit, target_names=faces.target_names))

m = confusion_matrix(ytest, yfit)
fig = plt.figure(figsize=(12, 12))
ax = sns.heatmap(m.T, square=True, annot=True, fmt='d', cmap='Greens',
                 xticklabels=faces.target_names, yticklabels=faces.target_names)
ax.set(xlabel='true label', ylabel='predicted label');

fig, axes = plt.subplots(2, 8, figsize=(16, 5), subplot_kw={'xticks':[], 'yticks':[]})
fig.subplots_adjust(hspace=0.1, wspace=0.2)
correct = yfit == ytest
for i, ax in enumerate(axes.flat):
    ax.imshow(xtest[i].reshape(62, 47), cmap='binary_r')
    if correct[i]:
        ax.set_title(faces.target_names[ytest[i]], fontsize=10)
    else:
        ax.set_title(faces.target_names[yfit[i]], fontsize=10, color='red')

