get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
get_ipython().magic('matplotlib inline')
import sys
import numpy as np
import scipy.cluster.hierarchy as sch
import pylab
import scipy
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append('../src/')
import data_config  as dc
katodata = dc.kato.data()
series = katodata[0]['deltaFOverF']
series = series[:107,:107]

G = nx.from_numpy_matrix(series)
nx.draw_graphviz(G)

for i,data in katodata.iteritems():
    delta = data['deltaFOverF_deriv']
    calcium = data['deltaFOverF_bc']
    plt.figure()
    plt.pcolormesh(delta)
    plt.figure()
    plt.pcolormesh(calcium)





