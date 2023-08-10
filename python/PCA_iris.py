from IPython.display import HTML, display

iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features.
y = iris.target

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from IPython.display import Image
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
scaler.fit(X)

print(scaler.transform(X))

df = pd.DataFrame(X, columns=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])

df

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

kmeans.labels_

y

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

X_reduced = PCA(n_components=3).fit_transform(X)

X_reduced

kmeansPCA = KMeans(n_clusters=3, random_state=0).fit(X_reduced)

kmeansPCA.labels_

fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=110)


ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],  c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])


plt.show()

kmeans.labels_

kmeansPCA.labels_



