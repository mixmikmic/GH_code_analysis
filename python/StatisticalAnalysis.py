get_ipython().magic('time')
import os
import numpy as np

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from sklearn import datasets
from sklearn import cross_validation
from sklearn import preprocessing

m_font_size = 20

get_ipython().magic('time')
# Create Data
n_events = 1000

m_means = [0]
m_vars = [0.5]

data = m_vars*np.random.randn(n_events,1)+m_means

get_ipython().magic('time')
# plot data
get_ipython().magic('matplotlib inline')

fig, ax = plt.subplots(figsize=(10,10),nrows=1, ncols=1)

n_bins = 50
m_bins = np.linspace(data.min(), data.max(), n_bins)

n, bins, patches = (ax.hist(data,bins=m_bins,fc=[0,0,1],alpha=0.8, normed=1))

from StatisticalAnalysis import *
[pdf,pts]= EstPDF(data,bins=m_bins)
fig, ax = plt.subplots(figsize=(10,10),nrows=1, ncols=1)
ax.plot(pts,pdf)
print sum(pdf)

get_ipython().magic('time')

from StatisticalAnalysis import *

# Create Data
n_events = 1000

m_means = [0]
m_vars = [0.5]

data1 = m_vars*np.random.randn(n_events,1)+m_means

m_means = [2.5]
m_vars = [0.5]

data2 = m_vars*np.random.randn(n_events,1)+m_means

n_bins = 100
m_bins = np.linspace(np.array([data1.min(),data2.min()]).min(),
                     np.array([data1.max(),data2.max()]).max(),
                     n_bins)

[kl,kl_vector] = KLDiv(data1, data1, mode='kernel',bins=m_bins,kernel_bw=0.5)
#print kl

data_1_2 = np.append(data1,data2,axis=1)
#data_1_2 = np.append(data_1_2,data2,axis=1)

mi1 = mutual_information(data_1_2)
mi2 = mutual_information_2d(data_1_2[:,1],data_1_2[:,1],normalized=True)

print mi, mi2
#fig, ax = plt.subplots(figsize=(10,10),nrows=1, ncols=1)
#ax.plot(m_bins[0:-1],kl_vector)


data_1_2.shape



