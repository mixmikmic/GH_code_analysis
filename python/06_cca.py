import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Add the optrans package to the Python path
# (change the path below to the location of the optimaltransport directory on your computer)
# Note: this step is not necessary if you have installed optimaltransport through pip.
import sys
sys.path.append('../../optimaltransport')

from optrans.datasets import oasis

# Load the image data, class labels, and metadata
fpath = '../optrans/datasets/oasis_data.npz'
img, label, metadata = oasis.load_data(fpath=fpath)

# Find incomplete metadata and remove those subjects
ind = ~np.isnan(metadata).any(axis=1)
img = img[ind]
label = label[ind]
metadata = metadata[ind]

# Select some metadata
# Demographics/clinical data: X = (age, mmse)
# Derived anatomical measure: Y = (etiv, nwbv, asf)
X = metadata[:,[1,4]]
Y = metadata[:,6:9]

_, ax = plt.subplots(2, 3, figsize=(16,6))

# Plot histograms of X data (demographics/clinical)
for x,a,lab in zip(X.T,ax[0],['Age','MMSE']):
    hist, bins = np.histogram(x)
    width = (bins[1]-bins[0]) * 0.4
    a.bar(bins[:-1], hist, width=width, color='b')
    a.set_xlabel(lab)

# Plot histograms of Y data (derived anatomical measures)
for y,a,lab in zip(Y.T,ax[1],['eTIV','nWBV','ASF']):
    hist, bins = np.histogram(y)
    width = (bins[1]-bins[0]) * 0.4
    a.bar(bins[:-1], hist, width=width, color='r')
    a.set_xlabel(lab)

ax[0,2].axis('off')
plt.tight_layout()
plt.show()

from optrans.decomposition import CCA

# Split into training and testing data
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=.5, random_state=42)

# Initialize CCA with 1 component
cca = CCA(n_components=1)

# Fit CCA and transform the training data
Xtr_cca, Ytr_cca = cca.fit_transform(Xtr, Ytr)

# Transform the testing data
Xte_cca, Yte_cca = cca.transform(Xte, Yte)

# Compute the correlation coefficient for the training and test data
corr_tr = cca.score(Xtr, Ytr)
corr_te = cca.score(Xte, Yte)

print('Corr. coefficient (train): {:.2f}'.format(corr_tr))
print('Corr. coefficient (test):  {:.2f}'.format(corr_te))

from optrans.decomposition import fit_line

# Get x-y coordinates of correlation line
xl, yl = fit_line(Xte_cca.squeeze(), Yte_cca.squeeze())

# Plot data projected on to first canonical direction
plt.scatter(Xtr_cca, Ytr_cca, c='b', label='Train')
plt.scatter(Xte_cca, Yte_cca, c='r', label='Test')
plt.plot(xl, yl, 'k--', lw=3, label='Corr. test')
plt.grid(alpha=.3)
plt.xlabel('X scores')
plt.ylabel('Y scores')
plt.legend()
plt.show()

# X data: images (reshaped into 1D arrays)
n_samples, h, w = img.shape
X = img.reshape((n_samples,h*w))

# Y data: age
Y = metadata[:,1]

# Split data into train and test sets
Xtr, Xte, Ytr, Yte = train_test_split(X, metadata[:,1], test_size=.5, random_state=42)

# Reshape the Y vectors into n-by-1 arrays
Ytr = Ytr.reshape(-1,1)
Yte = Yte.reshape(-1,1)

pca = PCA(n_components=10)
Xtr_pca = pca.fit_transform(Xtr)
Xte_pca = pca.transform(Xte)

print('Shape of original training data:  ', Xtr.shape)
print('Shape of training data after PCA: ', Xtr_pca.shape)

# Initialize CCA to return a single component
cca = CCA(n_components=1)

# Fit CCA to training data
cca.fit(Xtr_pca, Ytr)

# Transform the train and test data (images only)
Xtr_cca = cca.transform(Xtr_pca)
Xte_cca = cca.transform(Xte_pca)

# Compute the correlation coefficient for the training and test data
corr_tr = cca.score(Xtr_pca, Ytr)
corr_te = cca.score(Xte_pca, Yte)

print('Corr. coefficient (train): {:.2f}'.format(corr_tr))
print('Corr. coefficient (test):  {:.2f}'.format(corr_te))

# Get x-y coordinates of correlation line
xl, yl = fit_line(Xte_cca.squeeze(), Yte.squeeze())

# Plot X data projected on to first canonical direction
plt.scatter(Xtr_cca, Ytr, c='b', label='Train')
plt.scatter(Xte_cca, Yte, c='r', label='Test')
plt.plot(xl, yl, 'k--', lw=3, label='Corr. test')
plt.grid(alpha=.3)
plt.xlabel('X scores')
plt.ylabel('Age')
plt.legend()
plt.show()

from optrans.visualization import plot_mode_image

plot_mode_image([pca,cca], shape=(h,w), n_std=3., n_steps=5, cmap='gray', figsize=(16,4))
plt.show()



