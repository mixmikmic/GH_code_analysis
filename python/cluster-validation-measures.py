import json, ast, re
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')

def load_items(filepath):
    print("Loading %s ..." % filepath)
    lines = open(filepath,"r").readlines()
    items = []
    category_map = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if i == 0:
            task_name = line
        elif len(line) > 0:
            parts = line.split(":",1)
            category = parts[0].strip()
            for item in parts[1].strip().split(";"):
                if len(item) > 0:
                    items.append(item)
                    category_map[item] = category
    return task_name,items,category_map

def get_dict_data(filepath):
    print("Loading %s ..." % filepath)
    dict_data = json.load(open(filepath))
    dict_data = {ast.literal_eval(k): v for k, v in dict_data.items()}
    #dict_data = {frozenset(k): v for k, v in dict_data.items()}
    return dict_data

def create_similarity_matrix( items, dict_data ):
    n = len(items)
    S = np.zeros([n,n])
    index_map = {}
    for ind, item in enumerate(items):
        index_map[item] = ind
    for pair in dict_data:
        ind1 = index_map[pair[0]]
        ind2 = index_map[pair[1]]
        sim = dict_data[pair]
        S[ind1,ind2] = sim
        S[ind2,ind1] = sim
    return S

def create_distance_matrix( items, filepath ):
    sim_dict = get_dict_data( filepath )
    S = create_similarity_matrix( items, sim_dict )
    D = 1-(S/S.max())
    return D

# From sklearn calinski_harabaz_score, but need dispersion separately:
def between_within_dispersion(X, labels, cluster_names):
    """
    Based on
    .. [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
       analysis". Communications in Statistics
       <http://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_
    """    
    n_samples, _ = X.shape
    n_labels = max(labels)+1
    mean = np.mean(X, axis=0)
    extra_disp = 0.0
    intra_disp = 0.0
    results = pd.DataFrame(columns=["Categoey","Size","Within-SS", "Between-SS"])
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        bgss = len(cluster_k) * np.sum((mean_k - mean) ** 2)
        wgss = np.sum((cluster_k - mean_k) ** 2)
        extra_disp += bgss
        intra_disp += wgss        
        results.loc[k] = (cluster_names[k], len(cluster_k), np.round(wgss,2), np.round(bgss,2)) 
    mean_within = wgss/k
    mean_between = bgss/k
    ch = extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.))    
    return (results,ch,mean_within,mean_between)

methods = ["eve","word2vec_cbow","word2vec_sg","fasttext_cbow","fasttext_sg","glove"]
dataset_ids = ["european_cities", "movie_genres", "animal_classes", "cuisine", "music_genres", "nobel_laureates",
               "country_continent"]

result_cols = ["Dataset","Method","Mean Within","Mean Between","CH-Index"]
result_rows = []
for dataset_id in dataset_ids:
    # Load the data
    print()
    task_name,items,category_map = load_items("../dataset/tasks/%s.txt" % dataset_id )
    categories = list(set(category_map.values()))
    categories.sort()
    print("Task: %s" % task_name)
    print("%d categories: %s" % (len(categories),categories) )
    # Convert to labels
    labels = []
    for item in items:
        labels.append( categories.index( category_map[item] ) )
    labels = np.array(labels)    
    # Create distance matrices
    print("Creating distance matrices ...")
    D = {}
    for method_id in methods:
        filepath = "../output/pairwise_similarity/%s_%s.json" % (dataset_id,method_id) 
        D[method_id] = create_distance_matrix(items, filepath )
    # Calculate the scores
    print("Validating by measures ...")
    for method_id in methods:
        results,ch,mean_within,mean_between = between_within_dispersion( D[method_id], labels, categories )
        result_rows.append( [dataset_id, method_id, mean_within, mean_between, ch ] )

df_results = pd.DataFrame( result_rows, columns = result_cols )
df_results = df_results.round(3)
df_results

df_results.to_csv("cluster-validation.csv")



