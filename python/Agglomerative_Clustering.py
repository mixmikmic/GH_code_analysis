import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')
import seaborn as sns
from mpl_toolkits.basemap import Basemap


imgroot = 'C:/Users/lezhi/Dropbox/thesis/img/'
dataroot = 'C:/Users/lezhi/Dropbox/thesis/data/'

df = pd.read_csv(dataroot+'deep_features_all.csv')
df.head()

df_label = pd.read_csv(dataroot+'test_stats_all.csv')
df_label.head()

# 40, 56, 87, 119
df_label['city'] = df_label['label'].apply(lambda x: x.split('_')[0])

from sklearn import preprocessing 
X_normalized = preprocessing.normalize(df, norm='l2') #* 1000
X_normalized[0,:15]

from sklearn.cluster import AgglomerativeClustering

# http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering_metrics.html
X = X_normalized
n_clusters = 2
ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", affinity="euclidean").fit(X)

# trial, not important
a = np.zeros((4,2))
b = [[1,2],[3,4]]
a[[2,0]] = b
a

# check the sizes of the 2 main clusters
len(ac.labels_[ac.labels_ == 0]), len(ac.labels_[ac.labels_ == 1])

y = ac.labels_
children = ac.children_
len(children), children

nodes = np.unique(children.ravel())
len(nodes), nodes

pd.DataFrame(y_fit).to_csv(dataroot+'y_fit.csv')
pd.DataFrame(children).to_csv(dataroot+'children.csv')

def children_of(i):
    return children[i-len(y)]
children_of(17010)

from sklearn import cluster
from sklearn.cluster import SpectralClustering

y_preds = pd.DataFrame()
X = X_normalized
for num_cat in range(2,8):
    print num_cat
    spectral = cluster.SpectralClustering(n_clusters=num_cat, eigen_solver='arpack', affinity="rbf", assign_labels='discretize')
    y_pred = spectral.fit_predict(X)
    y_preds['cat_from_'+str(num_cat)] = y_pred
    
y_preds.head()

def most_frequent_label(arr):
    unique = np.unique(arr)
    count = [np.sum([arr == ele]) for ele in unique]
    return unique[np.argsort(count)[::-1]][0]

[np.sum([ y_preds['cat_from_4'] == ele]) for ele in np.unique( y_preds['cat_from_4'] )]

cityfilter = df_label['city'] == 'boston'
C = [el[0] for el in zip(y, cityfilter) if el[1]]
sum(cityfilter), len(cityfilter), len(C), len(df[cityfilter])

# https://basemaptutorial.readthedocs.org/en/latest/plotting_data.html

fig, axes = plt.subplots(2, 2, figsize=(18,18))
axes = axes.ravel()
cmap = sns.cubehelix_palette(8, as_cmap=True)

for i,city in enumerate(['boston','chicago','newyork','sanfrancisco']):
    
    axes[i].set_title(city)
    
    cityfilter = df_label['city'] == city
    C = [el[0] for el in zip(y, cityfilter) if el[1]]
    df0 = df_label[cityfilter]
    lat = df0['lat']
    lng = df0['lng']

    bounds = [np.min(lng), np.max(lng), np.min(lat), np.max(lat)]

    map = Basemap(resolution = 'l', 
                  epsg=4326, ax=axes[i],
                  llcrnrlon = bounds[0], llcrnrlat = bounds[2], urcrnrlon = bounds[1], urcrnrlat = bounds[3])

    map.hexbin(lng.values, lat.values, C=C, reduce_C_function = most_frequent_label, gridsize = 100, cmap=cmap)

    #map.colorbar(location='bottom') 
plt.tight_layout()
plt.show()

n_nodes = 34017
n_leaves = 17009

# find all leaf nodes that are decedants of a particular node
def find_leaves(node):
    siblings = children[node-n_leaves]
    
    if siblings[0] < n_leaves:
        children0 = [siblings[0]]
    else:
        children0 = find_leaves(siblings[0])        
    if siblings[1] < n_leaves:
        children1 = [siblings[1]]
    else:
        children1 = find_leaves(siblings[1])
    
    return children0 + children1

len(find_leaves(34015))

# calculte the category each image fall into given 2-200 categories
parent_list = [[]] * n_leaves

for i in range(1, 200):
    top = n_nodes - i
    leaves = find_leaves(top)
    for l in leaves:
#         if not l in parent_list:
#             parent_list[l] = [top]
#         else:
        parent_list[l] = parent_list[l] + [top]
parent_list

import json
with open(dataroot+'parents.json', 'w') as fp:
    json.dump(parent_list, fp)



# not important
def answer(x,y,z):
    m = []
    d_2 = []
    d_46911 = []
    d_other = []
    
    def add_num(d_list, el, i):
        if not len(d_list) == 0:
            if d_list[-1][0] == el:
                return d_list
        return d_list + [(el, i)]
    
    for i,el in enumerate([x,y,z]):  
                
        if el <= 12:            
            #m = add_num(m, el, i)
            m = m + [(el,i)]
        if el <= 28 and el > 12:
            d_2 = add_num(d_2, el, i)
        if el <= 30 and el > 12:
            d_46911 = add_num(d_46911, el, i)
        if el <= 31 and el > 12:
            d_other = add_num(d_other, el, i)
            
    # print m,d_2,d_46911,d_other[0]
    
    def to_date(nums, d_list):
        y = [el for el in range(3) if not el in [d_list[0][1], m[0][1]]][0]
        return "%02d/%02d/%02d" % (m[0][0], d_list[0][0], nums[y])    
    
    if len(m)==3: #1 and len(d_2)==0 and len(d_46911)==0 and len(d_other)==0 :
        return "%02d/%02d/%02d" % (m[0][0], m[0][0], m[0][0])
            
    elif len(m) == 2:
        return "Ambiguous"
    
    elif m[0] == 2:
        if len(d_2) > 1:
            return "Ambiguous"
        else:
            return to_date([x,y,z], d_2)
        
    elif m[0][0] in [4,6,9,11]:
        
        if len(d_46911) > 1:
            return "Ambiguous"
        else:
            return to_date([x,y,z], d_46911)
    else:
        if len(d_other) > 1:
            return "Ambiguous"
        else:
            return to_date([x,y,z], d_other)
        
        
answer(6,30,31)



