#sudo apt-get install -y libigraph0-dev
#then run
#!pip install python-igraph

get_ipython().run_line_magic('matplotlib', 'inline')
import os
import random
import numpy as np
import cPickle as pickle
import matplotlib.pyplot
from matplotlib.pyplot import imshow
from PIL import Image
from scipy.spatial import distance
from igraph import *
from tqdm import tqdm

images, pca_features = pickle.load(open('./results/features_caltech101.p', 'r'))

for i, f in zip(images, pca_features)[0:5]:
    print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... "%(i, f[0], f[1], f[2], f[3]))

num_images = 10000

if len(images) > num_images:
    sort_order = sorted(random.sample(xrange(len(images)), num_images))
    images = [images[i] for i in sort_order]
    pca_features = [pca_features[i] for i in sort_order]

#this might take about 90 minutes...
kNN = 30

graph = Graph(directed=True)
graph.add_vertices(len(images))

for i in tqdm(range(len(images))):
    distances = [ distance.cosine(pca_features[i], feat) for feat in pca_features ]
    idx_kNN = sorted(range(len(distances)), key=lambda k: distances[k])[1:kNN+1]
    for j in idx_kNN:
        graph.add_edge(i, j, weight=distances[j])
    
summary(graph)

pickle.dump([images, graph], open('./results/graph_caltech101_30knn.p', 'wb'))

#images, graph = pickle.load(open('./results/graph_caltech101_30knn.p', 'r'))

def get_concatenated_images(indexes, thumb_height):
    thumbs = []
    for idx in indexes:
        img = Image.open(images[idx])
        img = img.convert('RGB')
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

# pick two random indices
idx1 = int(len(images) * random.random())
idx2 = int(len(images) * random.random())

# run get_shortest_paths
path = graph.get_shortest_paths(idx1, to=idx2, mode=OUT, output='vpath', weights='weight')[0]

# retrieve the images, concatenate into one, and display them
results_image = get_concatenated_images(path, 200)
matplotlib.pyplot.figure(figsize = (16,12))
imshow(results_image)

# pick two random indices
idx1 = int(len(images) * random.random())
idx2 = int(len(images) * random.random())

# run get_shortest_paths
path = graph.get_shortest_paths(idx1, to=idx2, mode=OUT, output='vpath', weights='weight')[0]

# retrieve the images, concatenate into one, and display them
results_image = get_concatenated_images(path, 200)
matplotlib.pyplot.figure(figsize = (16,12))
imshow(results_image) 



