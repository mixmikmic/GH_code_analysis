from occiput.Reconstruction.SPECT import SPECT_Static_Scan 
get_ipython().run_line_magic('pylab', 'inline')

spect = SPECT_Static_Scan()

spect.set_n_pixels(128,128)
spect.set_gantry_angular_positions(0.0, 360.0, 59) 
spect.load_measurement_from_file('../data/spect/projection.nii')
spect.load_attenuation_from_file('../data/spect/attenuation.nii')
spect.set_pixel_size(4.8,4.8)
spect.set_radius(200.0)
spect.set_psf(fwhm0_mm=5.0, depth_dependence=0.0001)

spect.set_use_gpu(True)

figure(figsize=[18,6])
subplot(1,2,1); imshow(spect._measurement[:,:,10],cmap='gray');
subplot(1,2,2); imshow(spect._attenuation[64,:,:],cmap='gray');

activity = spect.estimate_activity(iterations=10, subset_size=16, subset_mode='random', method='EM') 

figure(figsize=[18,6])
subplot(131); imshow(activity.data[:,:,55],cmap='gray',vmax=180.0); 
subplot(132); imshow(activity.data[:,55,:],cmap='gray',vmax=180.0); 
subplot(133); imshow(activity.data[64,:,:],cmap='gray',vmax=180.0)

activity.save_to_file('../data/spect/spect_01_reconstruction.nii')













