import sys
print(sys.version)

import numpy as np
#import pylab
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import dicom
from algorithms import gamma_evaluation 

dcm_ref = dicom.read_file("Case7_dose_AAA.dcm")
dcm_evl = dicom.read_file("Case7_dose_Dm.dcm")

def load_dose_from_dicom(dcm):
    """Imports the dose in matplotlib format, with the following index mapping:
        i = y
        j = x
        k = z
    
    Therefore when using this function to have the coords match the same order,
    ie. coords_reference = (y, x, z)
    """
    pixels = np.transpose(
        dcm.pixel_array, (1, 2, 0))
    dose = pixels * dcm.DoseGridScaling

    return dose

dose_reference = load_dose_from_dicom(dcm_ref)
print dose_reference.shape

dose_evaluation = load_dose_from_dicom(dcm_evl)
print dose_evaluation.shape

get_ipython().magic('pinfo gamma_evaluation')

# Perform gamma evaluation at 4mm, 2%, resoution x=2, y=1
gamma_map = gamma_evaluation(dose_evaluation, dose_reference, 4., 2., (2, 1), signed=True)

plt.imshow(gamma_map, cmap='RdBu_r') # aspect=2, vmin=-2, vmax=2
plt.colorbar()
plt.show()

# Two subplots, unpack the axes array immediately
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

ax1.imshow(reference)
ax1.set_title('reference')

ax2.imshow(sample)
ax2.set_title('sample')

ax3.imshow(gamma_map, cmap='RdBu_r', aspect=2, vmin=-2, vmax=2)
ax3.set_title('gamma_map')
plt.show()

