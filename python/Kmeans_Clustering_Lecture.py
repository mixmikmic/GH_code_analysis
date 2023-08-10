#just a bit of the usual housekeeping

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

get_ipython().magic('matplotlib inline')

#reading in the data and doing some cleanup from the original format.
df = pd.read_csv("../data/whiskies.txt")
df.drop(['RowID'], inplace=True, axis=1)
remove_tab = lambda x: x.replace('\t','')
df.Postcode = df.Postcode.map(remove_tab)

df.head()

X = df.drop(['Distillery','Postcode',' Latitude',' Longitude'], axis=1)

X.head()

#feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled

def find_k (X, k_range, sample_percent=1):
    """
    k_range: a list of possible k values
    X, the data we're clustering on
    """
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist
    from sklearn.metrics import pairwise_distances

    N = X.shape[0]
    sampleSize = X.shape[0] * sample_percent

    if sampleSize > 0:
        index = np.arange(np.shape(X)[0])
        np.random.shuffle(index)
        X =  X[index, :]


    mean_distortions=[]
    for k in k_range:
        #cluster using k, then calculate the mean distortion (average distance to closest centroid)
        kmeans_model = KMeans(n_clusters=k, init='k-means++', n_jobs=-1).fit(X)
        mean_distortions.append(sum(np.min(pairwise_distances(X, kmeans_model.cluster_centers_,
                                                              metric='euclidean'),axis=1)) / X.shape[0])


    #visualize results
    plt.plot(k_range, mean_distortions)
    plt.xlabel("K Value")
    plt.ylabel("Mean Distortion")
    plt.title("Elbow Graph for Mean Distortion per K")
    plt.show()

find_k(np.matrix(X), range(1,10),1)

kmeans_model = KMeans(n_clusters=4, random_state=42)
kmeans_model.fit(X_scaled)

df['labels'] = kmeans_model.labels_

df.head(10)

df[df['labels'] == 2]

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

model = TSNE(n_components=3, learning_rate=12, random_state=10)
X_tsne = model.fit_transform(X) 
plt.figure(figsize=(10,10))
ax = plt.subplot(111, projection='3d')
ax.scatter(X_tsne[:, 0], X_tsne[:, 1],X_tsne[:,2], c=kmeans_model.labels_)







