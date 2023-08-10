import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')

import nibabel as nib

import dipy.core.gradients as grad
import dipy.sims.voxel as sims
import dipy.reconst.dti as dti
import dipy.reconst.sfm as sfm
import dipy.reconst.peaks as peaks

gtab = grad.gradient_table(op.join('data', 'SUB1_b2000_1.bvals'), op.join('data', 'SUB1_b2000_1.bvecs'))

SNR = 100
S0 = 100
mevals = np.array(([0.0015, 0.0005, 0.0005],
                   [0.0015, 0.0005, 0.0005]))
angles = [(0, 0), (90, 0)]
sig, sticks = sims.multi_tensor(gtab, mevals, S0, angles=angles,
                                fractions=[50, 50], snr=SNR)

sticks

dti_model = dti.TensorModel(gtab)
dti_fit = dti_model.fit(sig)

dti_fit.evecs[0]

plt.plot(sig, 'o-')
plt.plot(dti_fit.predict(gtab, S0=100))

sf_model = sfm.SparseFascicleModel(gtab)
sf_fit = sf_model.fit(sig)

peak_dirs, _, _ = peaks.peak_directions(sf_fit.odf(sf_model.sphere), sf_model.sphere)

peak_dirs

vox_idx = (53, 43, 47)
data1 = nib.load(op.join('data', 'SUB1_b2000_1.nii.gz')).get_data()[vox_idx]
data2 = nib.load(op.join('data', 'SUB1_b2000_2.nii.gz')).get_data()[vox_idx]

dti_fit = dti_model.fit(data1)
dti_predict = dti_fit.predict(gtab, S0=np.mean(data1[gtab.b0s_mask]))

fig, ax = plt.subplots(1)
ax.plot(data2,'o-')
ax.plot(dti_predict,'o-')
fig.set_size_inches([10,6])

dti_fit.evecs[0]

sf_fit = sf_model.fit(data1)
sf_predict = sf_fit.predict(gtab, S0=np.mean(data1[gtab.b0s_mask]))
peak_dirs, _, _ = peaks.peak_directions(sf_fit.odf(sf_model.sphere), sf_model.sphere)
print(peak_dirs)

fig, ax = plt.subplots(1)
ax.plot(data2,'o-')
ax.plot(dti_predict,'o-')
ax.plot(sf_predict,'o-')
fig.set_size_inches([10,6])

plt.hist(np.abs(sf_predict - data2), histtype='step')
plt.hist(np.abs(dti_predict - data2), histtype='step')
plt.hist(np.abs(data1 - data2), histtype='step')

rmse_retest = np.sqrt(np.mean((data1 - data2)**2))
rmse_dti = np.sqrt(np.mean((dti_predict - data2)**2))
rmse_sfm = np.sqrt(np.mean((sf_predict - data2)**2))
print("Test-retest RMSE: %2.2f"%rmse_retest)
print("DTI RMSE: %2.2f"%rmse_dti)
print("SFM RMSE: %2.2f"%rmse_sfm)

