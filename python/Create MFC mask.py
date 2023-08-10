get_ipython().magic('matplotlib inline')
import nibabel as nib
from nilearn import plotting as niplt
import numpy as np

cortex = nib.load('cerbcort.nii.gz')

# Binarize
cortex = nib.Nifti1Image((cortex.get_data() > 0).astype('int'), cortex.get_header().get_best_affine())
niplt.plot_roi(cortex)

i, j, k = np.meshgrid(*map(np.arange, cortex.get_data().shape), indexing='ij')

# Maximum left and right X coordinates
X_l = nib.affines.apply_affine(np.linalg.inv(cortex.get_affine()), [-10, 0, 0])[0]
X_r = nib.affines.apply_affine(np.linalg.inv(cortex.get_affine()), [10, 0, 0])[0]

# Maximum Y and Z coordinates
Y = nib.affines.apply_affine(np.linalg.inv(cortex.get_affine()), [0, -22, 0])[1]
Z = nib.affines.apply_affine(np.linalg.inv(cortex.get_affine()), [0, 0, -32])[2]

## Exclude lateral 
cortex.get_data()[
    np.where((i < X_r) | 
             (i > X_l))] = 0

# Exclude posterior
cortex.get_data()[
    np.where(j < Y)] = 0

## Exclude ventral 
cortex.get_data()[
    np.where(k < Z)] = 0

# Binarize
cortex.get_data()[cortex.get_data() < 1] = 0
cortex.get_data()[cortex.get_data() >= 1] = 1

niplt.plot_roi(cortex)



