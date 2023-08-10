import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import cPickle as pickle
import sompy

# topic matrix, created with lemmatization, bigrams and 10 topics
topic_matrix = pickle.load(open("data_processed/document_topics_mtx.p", "rb"))

mapsize = [30,30]
som = sompy.SOMFactory.build(
    topic_matrix,
    mapsize,
    mask=None,
    mapshape='planar',
    lattice='rect',
    normalization='var',
    initialization='pca',
    neighborhood='gaussian',
    training='batch',
    name='sompy')
som.train(n_job=8, verbose='info')

mapsize = [50,50]
som50 = sompy.SOMFactory.build(
    topic_matrix,
    mapsize,
    mask=None,
    mapshape='planar',
    lattice='rect',
    normalization='var',
    initialization='pca',
    neighborhood='gaussian',
    training='batch',
    name='sompy')
som50.train(n_job=8, verbose='info')

# The method find_k_nodes will find the nearest neightbors given an input. 
# The nearest neightbor will then be the location of that observation in 
# the map.
node_hits = []
for i in range(len(topic_matrix)):
    data = topic_matrix[i].reshape(1,-1)
    loc = som50.find_k_nodes(data)[1][0][0]
    node_hits.append(loc)

from collections import Counter
node_counts = Counter(node_hits)

print(node_counts)

len(node_counts)

som.component_names = ["_".join([t,str(i)]) for i,t in enumerate(['topic']*10)]
v = sompy.mapview.View2DPacked(30, 30, '',text_size=10)
v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=5)

import cPickle as pickle
topic_words = pickle.load(open("data_processed/topic_words_final_model.p", "rb"))
topic_words[2]

topic_words[9]

v = sompy.mapview.View2DPacked(10, 10, 'Topic Clustering',text_size=10)
cl = som.cluster(n_clusters=10, random_state=0)
# I am using the default plotting methods. I am sure this can be made prettier...
v.show(som, what='cluster')

h = sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
h.show(som)

# Get the coorfinates in the map
import random
coords = som.bmu_ind_to_xy(np.arange(som.codebook.nnodes))

# let's make use of the fact that we know the document classes and say 
# that I have read 3 space science related documents at random 
document_class = pd.read_csv("data_processed/document_class.txt")
space_sci = document_class.loc[document_class.document_class == 'sci.space'].index.tolist()
rand_doc_idx = random.sample(space_sci, 3)

# Extract the corresponding documents 
space_sci_docs = topic_matrix[rand_doc_idx, :]

# Build the user profile by averaging. Note that here one can get as clever
# as you want. You could calculate a weighted average with weights that account
# for the the number of words in the document and the time spent reading, or any
# other proxy of "interest"
user_profile = np.mean(space_sci_docs, axis=0).reshape(1, -1)
weights = som.codebook.matrix.T
activations = user_profile.dot(weights)[0]

act_matrix = np.zeros([30,30])
for i,j,idx in coords:
    act_matrix[i,j] = activations[idx]

import matplotlib.pyplot as plt
plt.imshow(act_matrix, cmap='jet', interpolation='nearest')
plt.show()

