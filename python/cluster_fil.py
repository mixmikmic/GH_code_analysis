#from __future__ import print_function
from os.path import split as pathsplit, join as pathjoin, splitext, isfile, basename
import sys
sys.path.append('../../GBT/filterbank_tools/')
get_ipython().magic('matplotlib inline')
#sys.path.append('/Users/urebbapr/research/ptf/trunk/util')

import numpy as np
import pylab as pl
from file_utils import *

from filterbank import Filterbank as FB, db
from skimage.feature import hog, daisy
from sklearn.cluster import SpectralClustering, KMeans

data = np.load('blc03.hog216.npy')
images = np.load('blc03.images.reduced.npy')

spectral = SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity="rbf")

labels = spectral.fit_predict(data)
print labels

print images.shape

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# # plot with dimensionality reduction

# pca = PCA(n_components=2)
# pca.fit(data)

# colours = ['b', 'g','r','c','m','y']

# pl.figure(figsize=(8,8))
# print data.shape
# print labels

# Y = np.array(pca.transform(data))
# print Y
# print Y.shape
# pl.figure()
# pl.scatter(Y[:,0],Y[:,1])
    

    
    

# CLUSTER 3

f_fillist = 'blc03_guppi_files.txt'
fillist = read_list_from_file(f_fillist)

for i, url_fil in enumerate(fillist):
    if labels[i] != 3:
        continue
    f_fil = basename(url_fil)
    f_filnpy = f_fil + "_rebin.npy"
    reduced_obs = np.load(f_filnpy)
    pl.figure()
    pl.imshow(reduced_obs, aspect='auto')
    #print("CLUSTER %d %s" % (labels[i], f_fil) )
    pl.title("CLUSTER %d %s" % (labels[i], f_fil))

for i, url_fil in enumerate(fillist):
    if labels[i] != 2:
        continue
    f_fil = basename(url_fil)
    f_filnpy = f_fil + "_rebin.npy"
    reduced_obs = np.load(f_filnpy)
    pl.figure()
    pl.imshow(reduced_obs, aspect='auto')
    #print("CLUSTER %d %s" % (labels[i], f_fil) )
    pl.title("CLUSTER %d %s" % (labels[i], f_fil))

for i, url_fil in enumerate(fillist):
    if labels[i] != 1:
        continue
    f_fil = basename(url_fil)
    f_filnpy = f_fil + "_rebin.npy"
    reduced_obs = np.load(f_filnpy)
    pl.figure()
    pl.imshow(reduced_obs, aspect='auto')
    #print("CLUSTER %d %s" % (labels[i], f_fil) )
    pl.title("CLUSTER %d %s" % (labels[i], f_fil))

for i, url_fil in enumerate(fillist):
    if labels[i] != 0:
        continue
    f_fil = basename(url_fil)
    f_filnpy = f_fil + "_rebin.npy"
    reduced_obs = np.load(f_filnpy)
    pl.figure()
    pl.imshow(reduced_obs, aspect='auto')
    #print("CLUSTER %d %s" % (labels[i], f_fil) )
    pl.title("CLUSTER %d %s" % (labels[i], f_fil))



