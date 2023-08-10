import copy
import dicom2nifti
import dicom2nifti.compressed_dicom as compressed_dicom
import importlib
import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.registration as regs
import numpy as np
import os
import pandas as pd
import pylab
import shutil
get_ipython().run_line_magic('matplotlib', 'inline')

importlib.reload(hf)

img_dir = r"D:\CBCT\AARON"
dcm_path = img_dir+"\\15814.dcm"

dicom_input = compressed_dicom.read_file(dcm_path,
                                       defer_size=100,
                                       stop_before_pixels=False,
                                       force=False)

cine_rate = float(dicom_input[('0018', '0040')].value)
kvp = float(dicom_input[('0018', '0060')].value)
d_detector = int(dicom_input[('0018', '1110')].value)
d_patient = int(dicom_input[('0018', '1111')].value)
t_exposure = int(dicom_input[('0018', '1150')].value)
current = int(dicom_input[('0018', '1151')].value)
exposure = int(dicom_input[('0018', '1152')].value)
pixel_spacing = dicom_input[('0018', '1164')].value
angle1 = float(dicom_input[('0018', '1510')].value)
angle2 = float(dicom_input[('0018', '1511')].value)
angle1_increment = dicom_input[('0018', '1520')].value
angle2_increment = dicom_input[('0018', '1521')].value
shutter_edges = [int(dicom_input[('0018', str(x))].value) for x in [1602, 1604, 1606, 1608]]

proj_imgs, _ = hf.dcm_load(dcm_path)
proj_imgs = np.transpose(proj_imgs, (1,2,0))
proj_imgs_unscaled = copy.deepcopy(proj_imgs)
proj_imgs = (proj_imgs - np.amin(proj_imgs)) / (np.amax(proj_imgs) - np.amin(proj_imgs))

proj_imgs = proj_imgs[:, :, 50:-50]
rows, frames, cols = proj_imgs.shape

pylab.imshow(proj_imgs[:,100,:], cmap='gray')

proj_imgs.shape

import astra





rec_cropped = rec[:,10:120,20:170]
rec_cropped[rec_cropped < 0] = 0
rec_cropped = np.transpose(rec_cropped, (2,1,0))

rec_cropped = rec_cropped * -1

hf.draw_slices(rec_cropped)

print(np.amin(rec[80,10:120,20:170]), np.amax(rec[80,10:120,20:170]))

pylab.imshow(rec[100,10:120,20:170])#, vmin=-50, vmax=150) #vmin=.0010, vmax=0.0035)



recon_img = hf.dcm_load(img_dir+"\\15810.dcm")
recon_img = np.transpose(recon_img, (1,2,0))[::-1,:,:]

