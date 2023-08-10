#Download the data (1MB)
from urllib.request import urlretrieve, urlopen
from zipfile import ZipFile
files = urlretrieve("https://www.dropbox.com/s/ecdlgwxjq04m5mx/HyperSpy_demos_EDS_TEM_files.zip?raw=1", "./HyperSpy_demos_EDX_TEM_files.zip")
with ZipFile("HyperSpy_demos_EDX_TEM_files.zip") as z:
    z.extractall()

# Set up HyperSpy
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

# Let's go with inline plots
get_ipython().magic('matplotlib inline')

# Load data
cs = hs.load('core_shell.hdf5')
cs.change_dtype('float')
cs.plot()

# Apply PCA
cs.decomposition()
cs.plot_explained_variance_ratio()

# Apply ICA
cs.blind_source_separation(3)
axes = hs.plot.plot_images(cs.get_bss_loadings(), axes_decor=None, cmap='RdBu', colorbar=None)

# Corrupt the data with sparse errors
sparse = 0.05 # Fraction of corrupted data

cserror = cs.copy()
E = 1000 * np.random.binomial(1, sparse, cs.data.shape)
cserror.data = cserror.data + E

cserror.plot()

# Apply PCA
cserror.decomposition()
cserror.plot_explained_variance_ratio()

# Apply ICA
cserror.blind_source_separation(3)
axes = hs.plot.plot_images(cserror.get_bss_loadings(), axes_decor=None, cmap='RdBu', colorbar=None)

# Try online robust PCA
cserror.decomposition(normalize_poissonian_noise=False,
                      algorithm='ORPCA',
                      output_dimension=10,                  
                      init='rand',                      
                      lambda1=0.005,
                      lambda2=0.005)
cserror.plot_explained_variance_ratio()

# Apply ICA
cserror.blind_source_separation(3)
axes = hs.plot.plot_images(cserror.get_bss_loadings(), axes_decor=None, cmap='RdBu', colorbar=None)

