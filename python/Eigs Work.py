get_ipython().magic('pylab inline')

import pickle
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
e = 20

data1 = pd.read_csv('d11_2010_01_m1_pivot1.csv', header=None)

data1.head()

pca1 = PCA()
pca1.fit(data1.iloc[:,3:])

plt.plot(np.cumsum(pca1.explained_variance_ratio_[:e]))
print np.cumsum(pca1.explained_variance_ratio_[:e])

eigs1 = {}
with open('d11_2010_01_m1_eigs'+str(e)+'_1.pkl', 'rb') as pfile:
    eigs1['eigvalues'] = pickle.load(pfile)
    eigs1['eigvectors'] = pickle.load(pfile)
    eigs1['mean'] = pickle.load(pfile)

spk1_explained_variance_ratio_ = []
for i in xrange(shape(eigs1['eigvalues'])[0]):
    spk1_explained_variance_ratio_.append(eigs1['eigvalues'][i] / sum(eigs1['eigvalues']))

plt.plot(np.cumsum(spk1_explained_variance_ratio_[:20]))
print np.cumsum(spk1_explained_variance_ratio_[:20])

shape(eigs1['eigvectors'])

plt.plot(eigs1['mean'])

shape(data1.iloc[:,3:]), shape(eigs1['eigvectors'][:2].T)

vec2proj1 =  np.dot(data1.iloc[:,3:], eigs1['eigvectors'][:2].T)

plt.scatter([x[0] for x in vec2proj1], [y[1] for y in vec2proj1])

data2 = pd.read_csv('d11_2010_01_m1_pivot2.csv', header=None)

data2.head()

pca2 = PCA()
pca2.fit(data2.iloc[:,3:])

plt.plot(np.cumsum(pca2.explained_variance_ratio_[:e]))
print np.cumsum(pca2.explained_variance_ratio_[:e])

eigs2 = {}
with open('d11_2010_01_m1_eigs'+str(e)+'_2.pkl', 'rb') as pfile:
    eigs2['eigvalues'] = pickle.load(pfile)
    eigs2['eigvectors'] = pickle.load(pfile)
    eigs2['mean'] = pickle.load(pfile)

spk2_explained_variance_ratio_ = []
for i in xrange(shape(eigs2['eigvalues'])[0]):
    spk2_explained_variance_ratio_.append(eigs2['eigvalues'][i] / sum(eigs2['eigvalues']))

plt.plot(np.cumsum(spk2_explained_variance_ratio_[:20]))
print np.cumsum(spk2_explained_variance_ratio_[:20])

shape(eigs2['eigvectors'])

plt.plot(eigs2['mean'])

shape(data2.iloc[:,3:]), shape(eigs2['eigvectors'][:2].T)

vec2proj2 =  np.dot(data2.iloc[:,3:], eigs2['eigvectors'][:2].T)

plt.scatter([x[0] for x in vec2proj2], [y[1] for y in vec2proj2])



