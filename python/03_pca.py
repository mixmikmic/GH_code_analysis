import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.decomposition import PCA

# Add the optrans package to the Python path
# (change the path below to the location of the optimaltransport directory on your computer)
# Note: this step is not necessary if you have installed optimaltransport through pip.
import sys
sys.path.append('../../optimaltransport')

from optrans.datasets import adni

# Load the image data
fpath = '../optrans/datasets/adni_data.npz'
img, label = adni.load_data(fpath=fpath)

# Find the indices of the first healthy volunteer and first Alzheimer's patient
ind_hv = label.argmin()
ind_ad = label.argmax()

# Plot the images
_, ax = plt.subplots(1, 2)
ax[0].imshow(img[ind_hv], cmap='gray')
ax[0].set_title('Healthy')
ax[1].imshow(img[ind_ad], cmap='gray')
ax[1].set_title('Alzheimer\'s')
plt.show()

# Get the dimensions of our dataset
n_imgs, h, w = img.shape

# Reshape the 2D images into 1D arrays
img = img.reshape((n_imgs,h*w))

# Initialize PCA in image space
pcai = PCA(n_components=5)

# Transform the data and reduce the it to just 5 features
img_pca = pcai.fit_transform(img)

# Get the proportion of variance explained by each component
var_ratio = pca.explained_variance_ratio_

# Plot explained variance
plt.bar(range(5), var_ratio, align='center', color='b')
plt.xlabel('Component')
plt.ylabel('Proportion of variance explained')
plt.show()

# Plot the first 5 PCA components
_, ax = plt.subplots(1, 5, figsize=(16,4))
for i,a in enumerate(ax):
    # Reshape the component back to 2D
    component = pcai.components_[i].reshape((h,w))
    a.imshow(component, cmap='gray')
    a.set_title("Component {}".format(i))
plt.show()

# Plot data projected on to first two components
plt.scatter(img_pca[:,0], img_pca[:,1], marker='+')
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.grid(True, alpha=0.3)

# Set the axes limits
lim = 1.2 * np.max(np.abs(img_pca[:,:2]), axis=0)
plt.xlim([-lim[0],lim[0]])
plt.ylim([-lim[1],lim[1]])
plt.show()

# Get standard deviation of data along 1st component
std0 = np.sqrt(pcai.explained_variance_[0])

# Plot data projected on to first two components
plt.scatter(img_pca[:,0], img_pca[:,1], marker='+', alpha=0.3)

# Plot evenly-spaced red dots
std_steps = np.linspace(-1.5,1.5,5)
plt.plot(std0*std_steps, np.zeros(std_steps.size), marker='o', ls='-', c='r')
for s, in zip(std_steps):
    plt.text(std0*s, -0.001, '{:.1f}$\sigma$'.format(s), ha='center', color='r', fontsize=14)

# Axis formatting, as before
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.grid(True, alpha=0.3)
plt.xlim([-lim[0],lim[0]])
plt.ylim([-lim[1],lim[1]])
plt.show()

from optrans.decomposition import get_mode_variation

# Set some parameters
component = 0  # Index of the component
n_std = 3.     # Range (in standard deviations) of data along component
n_steps = 5    # Number of steps along the component (i.e. number of images to visualize)

# Get the data variation along the PCA component 
mode = get_mode_variation(pcai, component=component, n_std=n_std, n_steps=n_steps)

# The range (in standard deviations) along this component
std_range = np.linspace(-n_std/2, n_std/2, n_steps)

# Plot the results
fig, ax = plt.subplots(1, n_steps, figsize=(16,4))
for m,s,a in zip(mode,std_range,ax):
    img_recon = m.reshape((h,w))
    a.imshow(img_recon, cmap='gray')
    a.set_title("{:.1f}$\sigma$".format(s))
plt.show()

# Load the transport maps corresponding to the images above
fpath = '../optrans/datasets/adni_rcdt_maps.npz'
f, _ = adni.load_rcdt_maps(fpath=fpath)

# Load the reference image that was used to compute the transport maps
fpath = '../optrans/datasets/adni_img0.npy'
img0 = adni.load_img0(fpath=fpath)

# Plot the images and corresponding transport maps
_, ax = plt.subplots(2, 2, figsize=(10,5))
ax[0,0].imshow(img[ind_hv].reshape((h,w)), cmap='gray')
ax[0,0].set_title('Image')
ax[0,0].set_ylabel('Healthy')
ax[0,1].imshow(f[ind_hv])
ax[0,1].set_ylabel('Healthy')
ax[0,1].set_title('Transport map')
ax[1,0].imshow(img[ind_ad].reshape((h,w)), cmap='gray')
ax[1,0].set_ylabel('Alzheimer\'s')
ax[1,1].imshow(f[ind_ad])
ax[1,1].set_ylabel('Alzheimer\'s')
plt.show()

# Get the dimensions of our dataset
n_imgs, h, w = f.shape

# Reshape the 2D transport maps into 1D arrays
f = f.reshape((n_imgs,h*w))

# Initialize PCA in transport space
pcat = PCA(n_components=5)

# Transform the data and reduce the it to just 5 features
f_pca = pcat.fit_transform(f)

# Get the proportion of variance explained by each component
var_ratio = pcat.explained_variance_ratio_

# Plot explained variance for transport space and image space
plt.bar(range(5), var_ratio, width=-.4, align='edge', color='r', label='Transport space')
plt.bar(range(5), pcai.explained_variance_ratio_, width=.4, align='edge', color='b', label='Image space')
plt.xlabel('Component')
plt.ylabel('Proportion of variance explained')
plt.legend()
plt.show()

# Plot the first 5 PCA components
_, ax = plt.subplots(1, 5, figsize=(16,4))
for i,a in enumerate(ax):
    # Reshape the component back to 2D
    component = pcat.components_[i].reshape((h,w))
    a.imshow(component)
    a.set_title("Component {}".format(i))
plt.show()

# Plot data projected on to first two components
plt.scatter(f_pca[:,0], f_pca[:,1], marker='+')
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.grid(True, alpha=0.3)

# Set the axes limits
lim = 1.2 * np.max(np.abs(f_pca[:,:2]), axis=0)
plt.xlim([-lim[0],lim[0]])
plt.ylim([-lim[1],lim[1]])
plt.show()

from optrans.continuous import RadonCDT

# Get the data variation along the PCA component 
mode = get_mode_variation(pcat, component=0, n_std=n_std, n_steps=n_steps)

# Initialize the RadonCDT so we can apply the inverse transport maps
radoncdt = RadonCDT()

# Plot the results
fig, ax = plt.subplots(2, n_steps, figsize=(16,6))
for i,s in enumerate(std_range):
    # Reshape reconstructed transport map
    f_recon = mode[i].reshape((h,w))
    ax[0,i].imshow(f_recon)
    ax[0,i].set_title("{:.1f}$\sigma$".format(s))
    
    # Reconstruct corresponding image
    img_recon = radoncdt.apply_inverse_map(f_recon, img0)
    ax[1,i].imshow(img_recon, cmap='gray')
    ax[1,i].set_title("{:.1f}$\sigma$".format(s))
    
plt.show()



