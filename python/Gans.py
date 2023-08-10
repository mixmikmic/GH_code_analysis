import niftiutils.helper_fxns as hf
import niftiutils.transforms as tr
import niftiutils.registration as regs
import os
import shutil
import pandas as pd
import numpy as np
import pylab
import importlib
import shutil
import copy

import dicom2nifti
import dicom2nifti.compressed_dicom as compressed_dicom
get_ipython().run_line_magic('matplotlib', 'inline')

import wgan
import dcgan
import argparse
import data

import neural_style_transfer as nst

importlib.reload(nst)

args = nst.init_args()

cbct_dcm_path = r"Z:\CBCT\imgs\E105393873\xper ct abdomen dual phase high quality open_5006"

cbct_img, _ = hf.dcm_load(cbct_dcm_path)
cbct_img = tr.apply_window(cbct_img, wl=-60, ww=800)

importlib.reload(hf)

hf.draw_slices(cbct_img)

cbct_png_path = "D:\\cbct.png"
hf.save_slice_as_img(cbct_img, cbct_png_path)

args['base_image_path'] = cbct_png_path

mdct_dcm_path = r"Z:\Isa\spect\12973122\y90 sirs   5.0  b31s_3"

mdct_img, _ = hf.dcm_load(mdct_dcm_path)
mdct_img = tr.apply_window(mdct_img, wl=-60, ww=800)

mdct_img = mdct_img[mdct_img.shape[0]//4:mdct_img.shape[0]*3//4,mdct_img.shape[1]//4:mdct_img.shape[1]*2//3]

mdct_png_path = "D:\\mdct.png"
hf.save_slice_as_img(mdct_img, mdct_png_path)

args['style_reference_image_path'] = mdct_png_path

importlib.reload(nst)

nst.run(args)



importlib.reload(wgan)

x_sampler = data.DataSampler()
z_sampler = data.NoiseSampler()

discriminator = dcgan.Discriminator()
generator = dcgan.Generator()

model = wgan.WassersteinGAN(generator, discriminator, x_sampler, z_sampler, dcgan)

#model.train(nb_epoch=500, batch_size=64)

model = wgan.WassersteinGAN(generator, discriminator, x_sampler, z_sampler, dcgan)
model.train(nb_epoch=5000, batch_size=64)

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

proj_imgs = hf.dcm_load(img_dir+"\\15814.dcm")
proj_imgs = np.transpose(proj_imgs, (1,2,0))
proj_imgs_unscaled = copy.deepcopy(proj_imgs)
proj_imgs = (proj_imgs - np.amin(proj_imgs)) / (np.amax(proj_imgs) - np.amin(proj_imgs))

proj_imgs = proj_imgs[:, :, 50:-50]
rows, frames, cols = proj_imgs.shape

recon_img = hf.dcm_load(img_dir+"\\15810.dcm")
recon_img = np.transpose(recon_img, (1,2,0))[::-1,:,:]

pylab.imshow(proj_data[:,100,:], cmap='gray')

proj_imgs.shape

hf.draw_slices(np.transpose(proj_imgs[-100:300:-1,:,50:-50],(2,1,0)), width=10)

hf.save_nii(rec_cropped, "D:\\recon.nii")

hf.draw_slices(rec_cropped, width=10)

rec_cropped = rec[:,10:120,20:170]
rec_cropped[rec_cropped < 0] = 0
rec_cropped = np.transpose(rec_cropped, (2,1,0))

rec_cropped = rec_cropped * -1

hf.draw_slices(rec_cropped)

hf.draw_slices(np.transpose(mdct1_iso,(2,1,0)))

print(np.amin(rec[80,10:120,20:170]), np.amax(rec[80,10:120,20:170]))

pylab.imshow(rec[100,10:120,20:170])#, vmin=-50, vmax=150) #vmin=.0010, vmax=0.0035)



