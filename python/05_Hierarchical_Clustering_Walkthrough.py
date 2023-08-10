import warnings
warnings.filterwarnings('ignore', module='scipy')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('white')
sns.set_palette('dark')

get_ipython().magic('matplotlib inline')

n_points_cluster = 10

data = np.append(np.random.normal(1, 1, (n_points_cluster, 2)), 
                 np.random.normal(3, 1, (n_points_cluster, 2)), 
                 axis = 0)

real_id = np.append([1]*n_points_cluster, [2]*n_points_cluster)

print(data.shape)

# custom red/blue palette
palette = {1:sns.color_palette()[0], 2:sns.color_palette()[2]}
colors = [palette[x] for x in real_id]

fig = plt.figure(figsize=(5,5))
ax = plt.axes()

ax.scatter(data[:,0], data[:,1], 
            c=colors);

from sklearn.metrics.pairwise import cosine_similarity

sim_cosine = cosine_similarity(data)

print(sim_cosine.shape)

dist_cosine = 1. - sim_cosine

from sklearn.metrics.pairwise import euclidean_distances

dist_euc = euclidean_distances(data)

from scipy.cluster.hierarchy import average

linkage_matrix_average = average(dist_euc)

print(linkage_matrix_average.shape)

from scipy.cluster.hierarchy import ward

linkage_matrix_ward = ward(dist_euc)

print(linkage_matrix_ward.shape)

from scipy.cluster.hierarchy import dendrogram

plt.show(dendrogram(linkage_matrix_average, orientation= "left"))

plt.show(dendrogram(linkage_matrix_ward, orientation= "left"))

from scipy.cluster.hierarchy import fcluster

k = 2
clusters1 = fcluster(linkage_matrix_ward, k, criterion='maxclust')
print(clusters1)

fig = plt.figure(figsize=(5,5))
ax = plt.axes()

marker_dict = {1:'o', 2:'s'}
markers = [marker_dict[x] for x in clusters1]

for x,y,c,m in zip(data[:,0], data[:,1], colors, markers):
    ax.scatter(x, y, c=c, marker=m)

