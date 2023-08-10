get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
plt.rcParams['image.cmap'] = 'RdBu'
import numpy as np
# Import custom code
import csv
import pandas as pd
import scipy.cluster.hierarchy as hier
from sklearn import manifold

def draw_dendrogram(co_count, terms, color_thresh, fig_size=(10,50)):
    #Y = hier.linkage(norm_co_count(co_count),method='complete')
    Y = hier.linkage(co_count,method='complete')
    plt.figure(figsize=fig_size)
    Z = hier.dendrogram(Y,orientation='left',labels=terms, color_threshold=color_thresh, leaf_font_size=20)
    return Z

def sorted_counts(co_count):
    sort_ind = np.argsort(np.diag(np.array(co_count)))[::-1]    
    terms = [list(co_count)[idx] for idx in sort_ind]
    return np.array(co_count)[sort_ind][:,sort_ind], terms

def norm_co_count(co_count):
    #normalize co-occurrence matrix row-wise, i.e. divide each row
    # by that row's diagonal element
    normed = np.zeros(np.shape(co_count))
    for idx in range(np.shape(normed)[0]):
        if co_count[idx,idx] is not 0:
            normed[idx,:] = co_count[idx,:]/co_count[idx,idx]
        
    return normed

def jacard_dist(co_counts):
    # get jacardi index
    D = np.shape(co_counts)[0]
    X = co_counts
    jacardD = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            if X[i,j] != 0.:            
                jacardD[i,j] = X[i,j]/(X[i,i]+X[j,j]-X[i,j])

    return jacardD

co_count = pd.read_csv('../data/paircount_neu.csv')
counts, terms = sorted_counts(co_count)
top_n = 50
Z = draw_dendrogram(jacard_dist(counts[:top_n, :top_n]), terms[:top_n], 1.4, (3,20));
plt.title('neu', fontsize=20)
plt.xticks(fontsize=20)

# use this to decide what threshold to use for coloring the clusters
cluster_vals = np.sort([Z['dcoord'][idx][1] for idx in range(len(Z['dcoord']))])
print cluster_vals[-10]
plt.plot(cluster_vals)
plt.plot([0., top_n],2*[cluster_vals[-10]])

query_term = 'fear'
datafiles = ('../data/paircount_cogsci.csv',
             '../data/paircount_cog.csv',
             '../data/paircount_neu.csv', 
             '../data/paircount_neumet.csv')

related_list = []
top_sim = 10
for f in datafiles:
    df = pd.read_csv(f)
    
    # check if query term is in database
    if query_term in list(df):
        # get term-term similarity using Jacard Index    
        similarity = pd.DataFrame(jacard_dist(np.array(df)),columns=list(df))

        # sort term distance based on queried term
        sim_inds = np.argsort(similarity[query_term])[::-1]
        sim = np.sort(similarity[query_term])[::-1]

        # get top-10 words
        related_list.append([list(df)[idx] for idx in sim_inds[:top_sim]])
        related_list.append([round(s,3) for s in sim[:top_sim]])
    else:
        # add blanks for database if term is not in it
        related_list.append(top_sim*[''])
        related_list.append(top_sim*[''])

pd.DataFrame(np.transpose(related_list), columns=('CogSci','', 'PM CS','', 'PM Neu','', 'PM NeuMet',''))

