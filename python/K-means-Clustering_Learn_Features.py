from numpy import *
import numpy as np
import cPickle
import scipy.io as io
from random import randrange 
from matplotlib import pyplot as plt
from os.path import join
from sklearn.cluster import KMeans
from sklearn import metrics
import cPickle as pickle
import matplotlib
from sklearn.feature_extraction import image
from ipywidgets import FloatProgress
from IPython.display import display

def normalization(patches):
    means_patches = mean(patches, axis=0)
    std_patches = std(patches, axis=0)
    patches = (patches - means_patches[np.newaxis,:])/(std_patches[np.newaxis,:])
    return patches

def whitening(patches):
    eig_values, eig_vec = np.linalg.eig(np.cov(patches.T))
    zca = eig_vec.dot(np.diag((eig_values+0.01)**-0.5).dot(eig_vec.T))
    patches = np.dot(patches, zca)
    return patches

# READ THE DATA
with open(join('cifar-10-batches-py','data_batch_1'),'rb') as f:
    data = pickle.load(f)
    
images = data['data'].reshape((-1,3,32,32)).astype('float64')
images = np.rollaxis(images, 1, 4)

# EXTRACT RANDOM PATCHES
rng = np.random.RandomState(0)
NBPATCH = 16
patches = np.zeros((NBPATCH*10000,6,6,3))
indice =0
for i in range(10000):
    patches[indice:indice+NBPATCH] = image.extract_patches_2d(images[i], (6,6), NBPATCH, random_state=rng)
    indice+=NBPATCH

patches = patches.reshape(NBPATCH*10000,108)

patches = normalization(patches)
patches = whitening(patches)

# RUN K-MEANS
NUM_CLUSTERS= 50
km = KMeans(n_clusters=NUM_CLUSTERS, n_jobs=1, random_state=0, n_init=1, verbose=True)
km.fit_predict(patches)
centroids = km.cluster_centers_.reshape((NUM_CLUSTERS,6,6,3))

# READ THE DATA / YOU CAN READ EITHER THE SAME BATCH OR AN OTHER
np.set_printoptions(threshold=np.nan)

with open(join('cifar-10-batches-py','test_batch'),'rb') as f:
    data_2 = pickle.load(f)

data2 = data['data'].reshape((-1,3,32,32)).astype('float32')
data2 = np.rollaxis(images, 1, 4)
labels2 = data_2['labels']

#EXTRACT DETERMINIST PATCHES With STRIDE
patch_size = 6
s =1
loss = 32-(patch_size+1)*(32/(patch_size+s))
nb_patches = (32/(patch_size+s))
patches = np.zeros((0,patch_size,patch_size,3))
for x in range(0,32-loss,patch_size+s):
    for y in range(0,32-loss,patch_size+s):
        patches = np.concatenate((patches, images[:,x:x+patch_size,y:y+patch_size,:]), axis=0)

patches = patches.reshape((patches.shape[0],-1))

# REAPPLY THE SAME NORMALIZATION AND WHITENING
patches = normalization(patches)
patches = whitening(patches)

# GET THE CLUSTER ASSIEGNMENT FOR EACH PATCH
newCls = km.predict(patches)

# TRANSFORM THE PATCH TO BINARY VECTOR
Kpatches=np.zeros((160000,NUM_CLUSTERS))
for x in range(160000):
    Kpatches[x][newCls[x]]=1

# CONSTRUCT THE REPRESENTATION OF THE IMAGES USING THE BINARY VECTORS
cls_images =np.zeros((10000,nb_patches, nb_patches,NUM_CLUSTERS))
indices =0
a,b =nb_patches,nb_patches
for img in range(10000):
    for i in range(nb_patches):
        for j in range(nb_patches):
            cls_images[img][i][j] = Kpatches[indices]
            indices += 1

# CREATE THE FEATURES VECTORS THAT WILL BE USED IN NAIVE BAYES
# WE WILL CLASSIFY THE FEATURES(REPRESENTATION OF THE IMAFE) NOT THE IMAGES 

nb_features = 4*NUM_CLUSTERS
features = np.zeros((10000,nb_features))
half = nb_patches/2

for i in range(10000):
    im = cls_images[i]
    indice =0
    for k in range(NUM_CLUSTERS):
        features[i][indice]= sum(im[0:half,0:half,k])
        features[i][indice+1]= sum(im[0:half,half:,k])
        features[i][indice+2]= sum(im[half:,0:half,k])
        features[i][indice+3]= sum(im[half:,half:,k])
        indice+=4
#Save the features to be used in Naive Bayes        
pickle.dump(features, open("features/hard-k-150/raw-data/projecteatures-hard-300-16.obj", "wb"))

