import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.datasets import load_iris

from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap as lcm


inertia = []
for x in range(2,6):
    km = KMeans(n_clusters = x)
    labels = km.fit_predict(X)
    plt.figure(figsize = (10,6))
    plt.scatter(X.iloc[:,1], X.iloc[:,3], c=labels, cmap = lcm(['red','green','blue','yellow','purple']), label = 'data')
    plt.scatter(km.cluster_centers_[:,1], km.cluster_centers_[:,3], marker='*', c = 'black',s = 200, label = 'centroids')
    plt.title( 'Number of Clusters: ' +str(x))
    plt.legend()
    plt.show()
    inertia.append(km.inertia_)
def inerta_plot( X):
    plt.figure(figsize = (10,6))
    plt.plot(range(2,6), inertia, marker = 'o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('inertia')
    plt.title('Inertia Decrease with K')
    plt.xticks([2,3,4,5])
    plt.show()
    
inertia_plot(X)

from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score

def silh_samp_cluster( X, start=2, stop=5, metric = 'euclidean'):
    # Taken from sebastian Raschkas book Python Machine Learning second edition
    for x in range(start, stop):
        km = KMeans(n_clusters = x)
        y_km = km.fit_predict(X)
        cluster_labels = np.unique(y_km)
        n_clusters = cluster_labels.shape[0]
        silhouette_vals = silhouette_samples(X, y_km, metric = metric)
        y_ax_lower, y_ax_upper =0,0
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[y_km == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i)/n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper),
                    c_silhouette_vals,
                    height=1.0,
                    edgecolor='none',
                    color = color)
            yticks.append((y_ax_lower + y_ax_upper)/2.)
            y_ax_lower+= len(c_silhouette_vals)
            
        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg,
                   color = 'red',
                   linestyle = "--")
        plt.yticks(yticks, cluster_labels+1)
        plt.ylabel("cluster")
        plt.xlabel('Silhouette Coefficient')
        plt.title('Silhouette for ' + str(x) + " Clusters")
        plt.show()
    
silh_samp_cluster(X)

