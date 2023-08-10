import pickle
import gzip
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import time

start = time.time()
#Inselspital
get_ipython().magic('ls -lh /home/dueo/data/Inselspital_2015_07_08/')
with open('/home/dueo/data/Inselspital_2015_07_08/META.pickle') as f:
    Names,X,Y,l = pickle.load(f)
print ("Loaded data in " + str(time.time() - start))
print ("   " + str(X.shape) + " y " + str(Y.shape) + " " + str(np.max(Y)))
np.histogram(Y, bins=[0, 1, 2, 3,4,5,6])

start = time.time()
with open('/home/dueo/data/Inselspital_2015_07_08/META.pickle') as f:
    Names_M,X_M,Y_M,l_M = pickle.load(f)
print ("Loaded data in " + str(time.time() - start))
print ("   " + str(X_M.shape) + " y " + str(Y_M.shape) + " " + str(np.max(Y_M)))
np.histogram(Y_M, bins=[0, 1, 2, 3,4,5,6])

idx = 21
Names[idx]

get_ipython().magic('matplotlib inline')
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplot(131)
plt.imshow(Y[idx,0,:,:])
plt.subplot(132)
plt.imshow(X[idx,0,:,:])

Y[idx,0,100,100] #2 The big thing is a tumor
Y[idx,0,100,160] #1 The small thing is normal tissue
np.max(Y)

import pylab
idx = 46
pylab.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplot(131)
plt.imshow(Y_M[idx,0,:,:])
plt.subplot(132)
plt.imshow(X_M[idx,0,:,:])
Y_M[idx,0,130,75]  #3 The big one
Y_M[idx,0,130,170] #1 The small one (is also 1 just differnent color dueo to scaling teh color scale)

idx_Y = []
for i in range(0,len(Y)):
    if (np.sum(Y[1,0,:,:] == 2) > 1000):
        idx_Y.append(i)        

idx_Y_M = []
for i in range(0,len(Y_M)):
    if (np.sum(Y_M[1,0,:,:] == 3) > 1000):
        idx_Y_M.append(i)        

idx_Y = idx_Y[0:10]
idx_Y_M = idx_Y_M[0:10]
idx_Y_M

with open('META_sub.pickle', 'wb') as f:
    pickle.dump((Names_M[0:30],X_M[0:30,:,:,:],Y_M[0:30,:,:,:],l_M[0:30]), f, -1)
get_ipython().magic('ls -rtlh')

with open('GBM_sub.pickle', 'wb') as f:
    pickle.dump((Names[0:9],X[0:9,:,:,:],Y[0:9,:,:,:],l[0:9]), f, -1)

get_ipython().magic('ls -lh /home/dueo/data/Inselspital_2015_07_08/')



