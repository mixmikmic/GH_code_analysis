get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
from sklearn import cluster
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') 

import seaborn as sns

seeds = pd.read_csv("./datasets/seeds.csv")

# A:
seeds.info()

# Plot the Data to see the distributions/relationships
sns.pairplot(seeds, hue='species')

# Check for nulls
seeds.isnull().sum()

# Look at the real species labels.
feature_space = pd.DataFrame(seeds, columns=['perimeter', 'groove_length'])
feature_space.head()

# A:
df = seeds.drop(labels=['species'], axis=1)

df.head()

from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(df)
Xs.shape

from sklearn.cluster import KMeans

model = KMeans(n_clusters=8, random_state=0).fit(Xs)

predicted = model.labels_
centroids = model.cluster_centers_

print "Predicted clusters to points: ", predicted
print "Location of centroids: "
print centroids

df['predicted'] = predicted
df.head()

from matplotlib import pyplot as plt

plt.figure(figsize=(7,7))

df.plot(x="x", y="y", kind="scatter", color=df['predicted'], )#colormap='gist_rainbow', alpha=.7)
plt.scatter(centroids[:,:1], centroids[:,1:], marker='o', s=150, alpha=.7, c=range(0,3), cmap='gist_rainbow')

from sklearn.metrics import silhouette_score

# A:

import random

random.randint(1,25), random.randint(1,25)

# A:

# A:



