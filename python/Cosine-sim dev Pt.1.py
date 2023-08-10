# imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import datasets, model_selection
mnist = datasets.fetch_mldata('MNIST original')

data, target = mnist.data, mnist.target

resh = np.array([[4, 4, 1, 4],
                [4, 12, 4, 1]])
resh.reshape(1, 8)

# cosine similarity function
v1 = np.array([3, 5, 1, 0, 9, 9])
v2 = np.array([45, 1, 3, 2, 9, 9])

V = []
V.append(v1)
V.append(v2)
V = np.array(V)
V

cosine_similarity(V)[0][1]

data.shape

np.max(target)

np.min(target)

target.shape

target[18623]

target[0]

target[1]

target[50000]

# see number of unique labels
unique, counts = np.unique(target, return_counts=True)
dict(zip(unique, counts))

target[55000]

# some rnadom indices
indices = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 10000, 18000, 22000, 25000, 35000, 38000, 43000, 49000, 55000, 10, 15, 16, 2, 3, 4, 5, 1000]
for i in indices:
    print(target[i])

# store distance values
distance = []
for i in indices:
    ls = []
    ls.append(data[0])
    ls.append(data[i])
    ls = np.array(ls)
    
    # cosine similarity
    cosim = cosine_similarity(ls)[0][1]
    distance.append(cosim)
    
distance

data.shape

data.T.shape

import heapq
distance = np.array(distance)
heapq.nlargest(5, range(len(distance)), distance.take)[1]

vsdf = np.arange(0, 20000)
heapq.nlargest(5, range(len(vsdf)), vsdf.take)

# take top n
def vote(arr, n):
    """arr: array whose values will be checked
    n: number of top values you want to return
    return the determined number"""

target[54051]

get_ipython().run_cell_magic('time', '', 'distance = []\nfor i in range(0, len(target)):\n    ls = []\n    ls.append(data[12665])\n    ls.append(data[i])\n    ls = np.array(ls)\n    \n    # cosine similarity\n    cosim = cosine_similarity(ls)[0][1]\n    distance.append(cosim)\n    \ndistance = np.array(distance)\ndistance.shape')

top = heapq.nlargest(10, range(len(distance)), distance.take)
top[1]

for i in top:
    print(target[i])

indices = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051, 10000, 18000, 22000, 25000, 35000, 38000, 43000, 49000, 55000]
for i in indices:
    print(target[i])





















