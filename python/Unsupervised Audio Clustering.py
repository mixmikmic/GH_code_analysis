get_ipython().magic('matplotlib inline')
import IPython
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster
from sklearn.decomposition import PCA

raw_data=pd.read_csv('data.csv')
train_data=raw_data.drop('audio', 1)
size=train_data.shape
print('No of features: '+str(size[1]))
print('No of Data Points: '+str(size[0]))

sns.pairplot(train_data.iloc[0:50][train_data.columns[0:10]])

train_data=pd.DataFrame(sklearn.preprocessing.scale(train_data))

kmean=sklearn.cluster.KMeans(n_clusters=5,n_jobs=4,max_iter=1000000)
kmean.fit(train_data)

def return_file_name(path):
    return os.path.split(path)[1]


clusters_kmean=[]
for i in range(0,kmean.n_clusters):
    new_cluster=raw_data.iloc[np.argwhere(kmean.predict(train_data)==i).T[0]]['audio'].str.strip()
    new_cluster=new_cluster.map(return_file_name)
    clusters_kmeanAgg.append(new_cluster)


for i in range(0,len(clusters_kmean)):
    print('\nCluster'+str(i)+'\n')
    print(clusters_kmean[i].sample(20,replace=True))
    

af=sklearn.cluster.AffinityPropagation()
af.fit(train_data)
print('Number of clusters: '+str(len(af.cluster_centers_indices_)))

aggl=sklearn.cluster.AgglomerativeClustering(n_clusters=5)
aggl.fit(train_data)

clusters_aggl=[]
for i in range(0,5):
    new_cluster=raw_data.iloc[np.argwhere(aggl.fit_predict(train_data)==i).T[0]]['audio'].str.strip()
    new_cluster=new_cluster.map(return_file_name)
    clusters_aggl.append(new_cluster)

for i in range(0,len(clusters)):
    print('\nCluster'+str(i)+'\n')
    print(clusters_aggl[i].sample(10,replace=True))

from scipy.spatial.distance import squareform, pdist

ecd_mat=pd.DataFrame(squareform(pdist(train_data.ix[:, :])))
ecd_mat.head()

mean_ecd=np.mean(ecd_mat.mean())
print('Mean Euclidean Distance: '+str(mean_ecd))


all_eps=np.arange(5,15,0.1)
for i in all_eps:
    dbscan=sklearn.cluster.DBSCAN(eps=i)
    dbscan.fit(train_data)
    labels=dbscan.labels_
    n_outlier=np.sum(labels==-1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print('EPS: '+str(i)+'\tNo of outlier: '+str(n_outlier)+'\tNo of cluster: '+str(n_clusters))

