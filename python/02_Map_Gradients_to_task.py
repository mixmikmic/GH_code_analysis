import glob
import h5py as h5
import time
import os

import nibabel as nib
import numpy as np
from fgrad.regress import regress_subject, trim_data

gradients = nib.load('../data/rsFC_eigenvectors.dscalar.nii').get_data().T

subjects = sorted(glob.glob('/Users/marcel/projects/HCP/data/*'))
n_sub = len(subjects)
n_runs = 2 # maximum number of runs per subject
n_grad = 10 # number of gradients to fit
n_tp = 405 # maximum number of time points in time series

ts_gradients = np.zeros([n_sub, n_runs, n_tp, n_grad])

for j, s in enumerate(subjects):
    print "Subject %s, %d of %d" % (s, j+1, n_sub)
    files = sorted(glob.glob('%s/MNINonLinear/Results/tfMRI_WM_*/tfMRI_*_Atlas_MSMAll.dtseries.nii' % s))
    print "Found %d runs" % len(files)
    if len(files) > 0:
        for i, f in enumerate(files):
            t = time.time()
            ts_gradients[j, i, :, :] = regress_subject(files[0], gradients[:,0:n_grad])
            print "Run %d/%d took %.2f seconds" % (i+1, len(files), time.time() - t)
    else:
        print "Found no runs for subject %s!" % s

ts_gradients, r, subjects = trim_data(ts_gradients, subjects)

if os.path.isfile('../data/reconstructed_WM.hdf5'):
    os.remove('../data/reconstructed_WM.hdf5')

f = h5.File('../data/reconstructed_WM.hdf5')
g = f.create_group('Working_memory')
g.create_dataset('LR', data = ts_gradients[:,0,:,:], compression = "gzip", chunks = (1,n_tp,n_grad))
g.create_dataset('RL', data = ts_gradients[:,1,:,:], compression = "gzip", chunks = (1,n_tp,n_grad))
g.create_dataset('Subjects', data = s)

f.close()

