get_ipython().magic('pylab inline')
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

kartta = pd.read_csv('../boreal_data/StandMap.txt', sep='\t', header=None)

np.shape(kartta)

kartta

all(kartta[0]== 0)

kartta[0].values[np.where(kartta[0] != 0)]

ind = 8
kartta[ind].values[np.where(kartta[ind] != 0)]

pylab.rcParams['figure.figsize'] = (10,8)
plt.imshow(1/kartta.values, interpolation='nearest',cmap='gist_ncar')
plt.show()

get_ipython().run_cell_magic('time', '', 'sizes = [len(np.where(kartta == i)[0]) for i in range(np.max(np.max(kartta)))]')

np.min(sizes), np.max(sizes[1:]), np.mean(sizes[1:])

np.where(sizes==np.max(sizes[1:]))

len(np.where(sizes==np.min(sizes))[0])

np.min(kartta.values), np.max(kartta.values)

import simplejson as json
with open('clusterings/new_600.json') as file:
    clust600 = json.load(file)

xtoc = np.array(clust600['6']['xtoc'])

clust_sizes = np.array([sum(xtoc==i) for i in range(max(xtoc))])

np.argmax(clust_sizes)

clust_kartta = np.zeros(np.shape(kartta))
max_id = len(xtoc)
for rowind, row in enumerate(kartta.values):
    for colind, value in enumerate(row):
        if value != 0 and value < max_id:
            clust_kartta[rowind,colind] = xtoc[value-1]

pylab.rcParams['figure.figsize'] = (40,25)
plt.imshow(clust_kartta, interpolation='nearest', cmap='Reds')
plt.show()

