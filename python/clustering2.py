import sys
sys.path.append("../classes")

import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.manifold import TSNE, Isomap, MDS, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import numpy as np, pandas as pd
from scipy.cluster import hierarchy
from sklearn.metrics import homogeneity_completeness_v_measure

from geno_classifier import *

from itertools import starmap, product

import GEOparse

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

results = pickle.load(open('../results/breast_cancer3.results','rb'))
labels = pickle.load(open('../datasets/breast_cancer3_y','rb'))

X_diff, y = preprocess_results(results, labels, True, True, False, False)
X_no_diff, _ = preprocess_results(results, labels, use_diff_score=False, use_pathways=False)

tsne = TSNE(n_components=2, random_state=42)
diseases_reduced_tsne = tsne.fit_transform(X_diff)

Z = hierarchy.linkage(diseases_reduced_tsne, method='single')

classes = {'unhealthy': 0, 'healthy': 1}
colors = {0 : 'darkmagenta', 1 : 'lightpink'}

labeled_colors = {k:colors[v] for k,v in classes.items()}

labeled_colors

def get_color(k, df):
    return labeled_colors[y[k-len(df)]]

plt.figure(figsize=(12, 7))
dn = hierarchy.dendrogram(Z, p=15, truncate_mode='lastp',
                          labels=y, link_color_func=lambda k: get_color(k, diseases_reduced_tsne))



