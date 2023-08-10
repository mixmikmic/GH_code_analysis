import pandas as pd

ps = pd.read_csv('data/ps_merged_edited.csv', sep =';')
ps.head()

ps.shape

y = ps['Target']

X = ps
X = X.drop('Target', axis=1)
X.head() # set features as everything except the target column

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(X['Organism']) # converts the labels 'Conopeptide' and 'Other toxin' into binary labels: 1 and 0
le.classes_

le.transform(X['Organism'])

X['Organism'] = le.transform(X['Organism'])

X.head()

le.fit(X['ID'])
len(le.classes_)

X = X.drop('ID', axis=1) # remove the ID feature from the dataframe
X.head()

X.head()

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)
p = pca.transform(X) # transform features into their principal components

print(pca.explained_variance_ratio_)

from sklearn.preprocessing import StandardScaler

# use standardization
rescaledX = StandardScaler().fit_transform(X)
X = pd.DataFrame(data = rescaledX, columns = X.columns)
X.head()

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)
p = pca.transform(X) # transform features into their principal components

print(pca.explained_variance_ratio_)

print(pca.components_)

p.shape

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.scatter(p[:, 0], p[:, 1]) # plot the principal components
plt.ylabel("PC2")
plt.xlabel("PC1")
plt.show()

import numpy as np

pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()

clf = PCA(0.95)
X_trans = clf.fit_transform(X)
print(X.shape)
print(X_trans.shape)

