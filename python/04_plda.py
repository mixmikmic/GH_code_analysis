import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Add the optrans package to the Python path
# (change the path below to the location of the optimaltransport directory on your computer)
# Note: this step is not necessary if you have installed optimaltransport through pip.
import sys
sys.path.append('../../code/python/optimaltransport')

from optrans.decomposition import PLDA

# Load iris dataset and class labels
X, y = load_iris(return_X_y=True)

# Initialize PLDA object
plda = PLDA(alpha=1., n_components=2)

# Fit PLDA and transform the data into two dimensions
X_plda = plda.fit_transform(X, y)

# Get the classification accuracy on the training data
acc = plda.score(X, y)

# Plot the 2D space
for lab in np.unique(y):
    plt.scatter(X_plda[y==lab,0], X_plda[y==lab,1], label='Class {}'.format(lab))
plt.grid(alpha=0.3)
plt.xlabel('1st PLDA component')
plt.ylabel('2nd PLDA component')
plt.title('Classification accuracy = {:.2f}'.format(acc))
plt.legend()
plt.show()

# Create a function to draw the scatter plot
def plot_scatter(X, y, ax):
    ax.grid(True, alpha=0.3)
    for lab in np.unique(y):
        ax.scatter(X[y==lab,0], X[y==lab,1])
    return

# Initialize the figure
fig, ax = plt.subplots(2, 3, figsize=(12,6))

# Compute PLDA using different alpha values
for i,alpha in enumerate([0.,10.,1000.]):
    # Fit and transform data using PLDA
    plda = PLDA(alpha=alpha, n_components=2)
    X_plda = plda.fit_transform(X, y)

    # Plot transformed data
    plot_scatter(X_plda, y, ax[0,i])
    ax[0,i].set_title('PLDA $\\alpha$={:.1f}'.format(alpha))
    
# For comparison, perform LDA
# Note: This should be the same as PLDA with alpha=0
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)
plot_scatter(X_lda, y, ax[1,0])
ax[1,0].set_title('LDA')

# For comparison, perform PCA
# Note: This should be the same as PLDA with very large alpha
pca = PCA()
X_pca = pca.fit_transform(X)
plot_scatter(X_pca, y, ax[1,-1])
ax[1,2].set_title('PCA')

# Ignore the bottom-center plot
ax[1,1].axis('off')

plt.tight_layout()
plt.show()

from optrans.datasets import adni

# Load the image data X and class labels y
fpath = '../../code/python/optimaltransport/optrans/datasets/adni_data.npz'
X, y = adni.load_data(fpath=fpath)

# Find the indices of the first healthy volunteer and first Alzheimer's patient
ind_hv = y.argmin()
ind_ad = y.argmax()

# Plot the images
_, ax = plt.subplots(1, 2)
ax[0].imshow(X[ind_hv], cmap='gray')
ax[0].set_title('Healthy')
ax[1].imshow(X[ind_ad], cmap='gray')
ax[1].set_title('Alzheimer\'s')
plt.show()

# Reshape data into a n-by-p 2d array
n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

# Split data into training and test sets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)

# Initialize PCA
pca = PCA()

# Learn PCA on the training data
Xtr_pca = pca.fit_transform(Xtr)

# Transform the test data accordingly
Xte_pca = pca.transform(Xte)

print("Dimensions before PCA: ", Xtr.shape[1])
print("Dimensions after PCA: ", Xtr_pca.shape[1])

# Train PLDA classifier using dimension-reduced data
plda = PLDA(n_components=2, alpha=1.)
plda.fit(Xtr_pca, ytr)

# Test PLDA classifier
y_pred = plda.predict(Xte_pca)
acc = plda.score(Xte_pca, yte)

print("Classification accuracy on test data: {:.3f}".format(acc))

# Transform the training and testing data using our learned PLDA directions
Xtr_plda = plda.transform(Xtr_pca)
Xte_plda = plda.transform(Xte_pca)

# Plot the transformed data on a scatter plot
for lab,col,name in zip([0,1],['b','r'],['Healthy','AD']):
    plt.scatter(Xtr_plda[ytr==lab,0], Xtr_plda[ytr==lab,1], c=col, marker='x', alpha=0.3, label=name+' - train')
    plt.scatter(Xte_plda[yte==lab,0], Xte_plda[yte==lab,1], c=col, label=name+' - test')
plt.grid(alpha=0.3)
plt.xlabel('1st PLDA component')
plt.ylabel('2nd PLDA component')
plt.title('Classification accuracy = {:.2f}'.format(acc))
plt.legend()
plt.show()

# Plot the transformed data on a scatter plot
for lab,col,name in zip([0,1],['b','r'],['Healthy','AD']):
    plt.scatter(Xtr_plda[ytr==lab,0], Xtr_plda[ytr==lab,1], c=col, marker='x', alpha=0.1, label=name+' - train')
    plt.scatter(Xte_plda[yte==lab,0], Xte_plda[yte==lab,1], c=col, alpha=0.3, label=name+' - test')
plt.grid(alpha=0.3)
plt.xlabel('1st PLDA component')
plt.ylabel('2nd PLDA component')
plt.title('Classification accuracy = {:.2f}'.format(acc))
plt.legend()

# Plot evenly-spaced black dots
std0 = np.sqrt(plda.explained_variance_[0])
std_range = np.linspace(-1.5,1.5,5)
std_steps = std0 * std_range
plt.plot(std0*std_range, np.zeros(5), marker='o', ls='-', c='k')
for s in std_range:
    plt.text(std0*s, -0.003, '{:.1f}$\sigma$'.format(s), ha='center', color='k', fontsize=10)
    
plt.show()

from optrans.visualization import plot_mode_image

ax = plot_mode_image([pca, plda], component=0, shape=(h,w), n_std=3, n_steps=5, cmap='gray', figsize=(14,4))
plt.show()

from optrans.visualization import plot_mode_histogram

# Plot histogram of the test data projected on to the 1st PLDA component
ax = plot_mode_histogram(Xte_plda, y=yte, component=0, decomp=plda, n_bins=11)
plt.show()

# Load the transport maps X and class labels y
fpath = '../../code/python/optimaltransport/optrans/datasets/'
X, y = adni.load_rcdt_maps(fpath=fpath+'adni_rcdt_maps.npz')
img0 = adni.load_img0(fpath=fpath+'adni_img0.npy')

_, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].imshow(X[0])
ax[0].set_title('Sample transport map')
ax[1].imshow(img0, cmap='gray')
ax[1].set_title('Reference image')
plt.show()

# Reshape data into a n-by-p 2d array
n_imgs, h, w = X.shape
X = X.reshape((n_imgs,h*w))

# Split data into training and test sets
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=11, stratify=y)

# Initialize PCA
pca = PCA()

# Learn PCA on the training data
Xtr_pca = pca.fit_transform(Xtr)

# Transform the test data accordingly
Xte_pca = pca.transform(Xte)

print("Dimensions before PCA: ", Xtr.shape[1])
print("Dimensions after PCA: ", Xtr_pca.shape[1])

# Train PLDA classifier using dimension-reduced data
plda = PLDA(n_components=2, alpha=10.)
plda.fit(Xtr_pca, ytr)

# Test PLDA classifier
y_pred = plda.predict(Xte_pca)
acc = plda.score(Xte_pca, yte)

print("Classification accuracy on test data: {:.3f}".format(acc))

# Transform the training and testing data using our learned PLDA directions
Xtr_plda = plda.transform(Xtr_pca)
Xte_plda = plda.transform(Xte_pca)

# Plot the transformed data on a scatter plot
for lab,col,name in zip([0,1],['b','r'],['Healthy','AD']):
    plt.scatter(Xtr_plda[ytr==lab,0], Xtr_plda[ytr==lab,1], c=col, marker='x', alpha=0.3, label=name+' - train')
    plt.scatter(Xte_plda[yte==lab,0], Xte_plda[yte==lab,1], c=col, label=name+' - test')
plt.grid(alpha=0.3)
plt.xlabel('1st PLDA component')
plt.ylabel('2nd PLDA component')
plt.title('Classification accuracy = {:.2f}'.format(acc))
plt.legend()
plt.show()

from optrans.continuous import RadonCDT

ax = plot_mode_image([pca, plda], component=0, shape=(h,w), transform=RadonCDT(), img0=img0, 
                     n_std=3, n_steps=5, cmap='gray', figsize=(14,4))
plt.show()

# Plot histogram of the test data projected on to the 1st PLDA component
ax = plot_mode_histogram(Xte_plda, y=yte, component=0, decomp=plda, n_bins=11)
plt.show()



