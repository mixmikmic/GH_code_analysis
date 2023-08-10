import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic('matplotlib inline')
from mpl_toolkits.mplot3d import Axes3D

def dti_signal(theta, b, Q, S0=100):
    """ 
    A function to compute the diffusion signal from the DTI model
    
    Parameters
    ----------
    theta : n by 3 array
        The directions to compute the signal.
    
    b : float or array of length n
        The b-value(s) used for the measurement.
    
    Q : 3 by 3 array
        The diffusion tensor, a symmetrical rank 2 tensor.
    
    S0 : float
        The baseline signal, relative to which the signal is computed
    """
    # We take the diagonal of this, because we are only interested in the multiplication of each vector with the 
    # matrix, not in various cross-products:
    adc = np.diag(np.dot(np.dot(theta, Q), theta.T))
    # ADC stands for 'apparent diffusion coefficient', which is an estimate of the diffusivity (in mm^2/s) 
    # in each direction of measurement. We will estimate that later with data
    return S0 * np.exp(-b * adc) 

Q = np.eye(3)
theta = np.array([[1, 0, 0],  [0, 1, 0],  [0, 0, 1], [1/np.sqrt(2), 1/np.sqrt(2), 0]])
b = 2.0 # We need to scale b to the right units
dti_signal(theta, b, Q)

Q = np.array([[1.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
dti_signal(theta, b, Q)

import dipy.reconst.dti as dti 
import dipy.core.gradients as grad
import dipy.core.sphere as sph
import nibabel as nib

gtab = grad.gradient_table(op.join('data', 'SUB1_b2000_1.bvals'), op.join('data', 'SUB1_b2000_1.bvecs'))

sig1 = dti_signal(gtab.bvecs, b, Q)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for ii in range(sig1.shape[0]):
    x,y,z = gtab.bvecs[ii]
    this_sig = np.array([sig1[ii]])
    ax.plot3D(this_sig * x , this_sig * y, this_sig * z, 'b.')

plt.plot(sig1, 'o-')

model = dti.TensorModel(gtab)
fit = model.fit(sig1)

fit.evals

plt.scatter(fit.predict(gtab, S0=100), sig1)

vox_idx = (40, 74, 34)
data1 = nib.load(op.join('data', 'SUB1_b2000_1.nii.gz')).get_data()[vox_idx]

plt.plot(data1, 'o-')

data2 = nib.load(op.join('data', 'SUB1_b2000_2.nii.gz')).get_data()[vox_idx]

plt.plot(data1, 'o-')
plt.plot(data2, 'o-')

fit = model.fit(data1)

fit.evals

fit.md

fit.fa

fit.evecs[0]

predict1 = fit.predict(gtab, S0=np.mean(data1[:10]))

plt.scatter(predict1, data1)

plt.scatter(predict1, data2)

plt.scatter(data1, data2)

fig, ax = plt.subplots(1)
plt.plot(data1, '-o')
plt.plot(data2, '-o')
plt.plot(predict1, '-o')
fig.set_size_inches([10, 6])

plt.hist(np.abs(data1 - data2), histtype='step')
plt.hist(np.abs(predict1 - data2), histtype='step')

rmse_retest = np.sqrt(np.mean((data1 - data2)**2))
rmse_model = np.sqrt(np.mean((predict1 - data2)**2))
print("Test-retest RMSE: %2.2f"%rmse_retest)
print("Model prediction RMSE: %2.2f"%rmse_model)

