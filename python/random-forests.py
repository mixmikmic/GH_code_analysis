get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV

wine = load_wine()
print(wine['DESCR'])

fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(wine.data[:, 0], wine.data[:, 1], s=30*wine.data[:, 2], c=wine.target, cmap='viridis');

pca = PCA()
pca.fit(wine.data)
fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(np.cumsum(pca.explained_variance_ratio_))
ax.set(xlabel='components', ylabel='explained variance');

pca = PCA(n_components=2)
wine_pca = pca.fit_transform(wine.data)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(wine_pca[:, 0], wine_pca[:, 1], s=60, c=wine.target, cmap='viridis');

df = pd.DataFrame(wine.data, columns=[wine.feature_names])
df.describe()

preprocess = make_pipeline(StandardScaler(), PCA(n_components=2))
wine_pca = preprocess.fit_transform(wine.data)
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(wine_pca[:, 0], wine_pca[:, 1], s=60, c=wine.target, cmap='viridis');

model = DecisionTreeClassifier()
model.fit(wine_pca, wine.target)

fig, ax = plt.subplots(figsize=(16, 7))
ax.scatter(wine_pca[:, 0], wine_pca[:, 1], s=60, c=wine.target, cmap='viridis', zorder=2)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x = np.linspace(*xlim, num=200)
y = np.linspace(*ylim, num=200)
xx, yy = np.meshgrid(x, y)
z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, z, alpha=0.2, cmap='viridis', zorder=1);

xtrain, xtest, ytrain, ytest = train_test_split(wine_pca, wine.target, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print('cross validation', cross_val_score(model, xtrain, ytrain))
print('test score', model.score(xtest, ytest))

xtrain, xtest, ytrain, ytest = train_test_split(wine_pca, wine.target, test_size=0.2)
models = ['Bagging', 'Random Forest', 'Ada Boost']
model_dict = {
    'Bagging': BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.7),
    'Random Forest': RandomForestClassifier(n_estimators=30),
    'Ada Boost': AdaBoostClassifier(n_estimators=30)
}
fig, ax = plt.subplots(len(models), figsize=(16, 24))
for i, model_name in enumerate(models):
    model = model_dict[model_name]
    model.fit(xtrain, ytrain)
    ax[i].scatter(wine_pca[:, 0], wine_pca[:, 1], s=60, c=wine.target, cmap='viridis', zorder=2)
    xlim = ax[i].get_xlim()
    ylim = ax[i].get_ylim()
    x = np.linspace(*xlim, num=200)
    y = np.linspace(*ylim, num=200)
    xx, yy = np.meshgrid(x, y)
    z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax[i].contourf(xx, yy, z, alpha=0.2, cmap='viridis', zorder=1)
    ax[i].set_title(model_name, fontsize=20)

model = RandomForestClassifier(n_estimators=30)
model.fit(xtrain, ytrain)
print('cross validation', cross_val_score(model, xtrain, ytrain))
print('test score', model.score(xtest, ytest))

iris = load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=0.2)

model = RandomForestClassifier()
param_dist = {
    'n_estimators': [10, 50, 100],
    'min_samples_leaf': [1, 2, 5]
}
grid = GridSearchCV(model, param_dist, cv=5)
# this will take a few seconds
grid.fit(xtrain, ytrain)
grid.best_estimator_

grid.best_score_

yfit = grid.best_estimator_.predict(xtest)
print(classification_report(ytest, yfit, target_names=iris.target_names))

# this is a tiny dataset, let's look at the errors for the full data
yfit_train = grid.best_estimator_.predict(xtrain)
m = confusion_matrix(np.hstack([ytrain, ytest]), np.hstack([yfit_train, yfit]))
fig = plt.figure(figsize=(5, 5))
ax = sns.heatmap(m.T, square=True, annot=True, fmt='d', cmap='ocean_r',
                 xticklabels=iris.target_names, yticklabels=iris.target_names)
ax.set(xlabel='true label', ylabel='predicted label');

