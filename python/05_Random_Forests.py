import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
from matplotlib.colors import ListedColormap
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

from IPython.core.display import HTML
#HTML("<style>.container { width:100% !important; }</style>")

from sklearn import datasets
iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_names'] = df.species
df.replace({'species_names':{
            0:iris['target_names'][0],
            1:iris['target_names'][1],
            2:iris['target_names'][2]            
        }}, inplace=True)
df.columns = [item.replace(' (cm)', '') for item in df.columns]
df.head()

plt.rcParams.update({'axes.labelsize': 'large'})
plt.rcParams.update({'axes.titlesize': 'large'})

g = sns.FacetGrid(df, hue='species_names', size=10)
g.map(plt.scatter, 'sepal length', 'petal width', s=70)
g.add_legend()
g.fig.gca().set_title('Iris species')

from sklearn import tree
X = df[['sepal length', 'petal width']].values
y = df.species

step = 0.05
    
def mesh_plot(x, y, species, colors, ax, clf):
    values = species.unique()
    xx, yy = np.meshgrid(
        np.arange(x.min() - 0.1, x.max() + 0.1, step),
        np.arange(y.min() - 0.1, y.max() + 0.1, step))
    mesh_predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    mesh_predict = mesh_predict.reshape(xx.shape)
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(y.min() - 0.2, y.max() + 0.2)
    ax.scatter(x, y, c=colors)
    ax.pcolormesh(xx, yy, mesh_predict,
        cmap=ListedColormap(sns.color_palette()[:3]), alpha=0.2)

uniqueSpecies = df.species.unique()
colorsMap = dict(zip(uniqueSpecies, sns.color_palette()[:len(uniqueSpecies)]))

sns.set(font_scale=1.25)
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(11, 4), squeeze=True)
fig.tight_layout()

for idx in range(0, 3):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.3, random_state=idx)    
    clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    colors = [colorsMap[item] for item in y_train]
    mesh_plot(X_train[:, 0], X_train[:, 1], y_train, colors, ax[idx], clf)
fig.suptitle('Decision trees using three subsets of the data', y=1.00)

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(11, 4), squeeze=True)
fig.tight_layout()

for idx in range(0, 3):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.3, random_state=idx)    
    clf = RandomForestClassifier(n_estimators=500).fit(X_train, y_train)
    colors = [colorsMap[item] for item in y_train]
    mesh_plot(X_train[:, 0], X_train[:, 1], y_train, colors, ax[idx], clf)
    
fig.suptitle('Random forests using three subsets of the data', y=1.00)

