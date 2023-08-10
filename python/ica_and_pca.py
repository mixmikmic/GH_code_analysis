import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA, PCA

import warnings
warnings.filterwarnings("ignore") # to avoid complex number warning

from create_datasets import SymptomDiseaseData
data = SymptomDiseaseData()
train, valid, test, train_labels, valid_labels, test_labels = data.generate_data(split=True)
data.get_data_insight()

# Restack valid and test cause we don't need them
test = np.vstack((valid, test))
test_labels = np.vstack((valid_labels, test_labels))

ica = FastICA(n_components=train_labels.shape[1])#, whiten=False)
S_ = ica.fit_transform(train)  # Reconstruct signals

pca = PCA(n_components=train_labels.shape[1], whiten=False)
pca_done = pca.fit_transform(train)

cov = pca.get_covariance()
eig = np.linalg.eig(cov)

bottom = 250
top = 350
zoom = False

f, axarr = plt.subplots(2, 2, figsize=(20, 10))
f.size=((20,20))

axarr[0, 0].plot(train_labels.T[0], label='Influenza')
axarr[0, 0].plot(train_labels.T[1], label='Windpocken')
axarr[0, 0].plot(train_labels.T[2], label='Noro')
if zoom: axarr[0, 0].set_xlim([bottom,top])
axarr[0, 0].legend()
axarr[0, 0].set_title('True Distribution of Disease')

axarr[0, 1].plot(S_)
if zoom: axarr[0, 1].set_xlim([bottom,top])
axarr[0, 1].set_title('ICA Reconstruction')

axarr[1, 0].plot(pca_done)
if zoom: axarr[1, 0].set_xlim([bottom,top])
axarr[1, 0].set_title('PCA Reconstruction')

axarr[1, 1].hist(eig[0][:15], bins=15)
axarr[1, 1].set_title('Eigenvalues')

plt.show()

pca_test = pca.transform(test)
error = test_labels - pca_test

fig = plt.figure()
fig.size=((20,20))

plt.plot(test_labels.T[0], label='Influenza')
plt.plot(test_labels.T[1], label='Windpocken')
plt.plot(test_labels.T[2], label='Noro')
# if zoom: axarr[0, 0].set_xlim([bottom,top])
plt.legend()
plt.title('True Test Distribution of Disease')
plt.show()

plt.plot(pca_test)
# if zoom: axarr[0, 1].set_xlim([bottom,top])
plt.title('PCA Test Reconstruction')
plt.show()

plt.plot(error)
# if zoom: axarr[1, 0].set_xlim([bottom,top])
plt.title('Error Rate')
plt.show()

plt.show()

