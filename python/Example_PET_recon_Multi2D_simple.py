import occiput_suite.occiput as occiput
from numpy import *
get_ipython().run_line_magic('pylab', 'inline')

N_x = 64                 # Number of image pixels in x direction 
N_y = 64                 # Number of image pixels in y direction 
N_projections = 120      # Number of directions of the Lines of Response 
N_bins = 150             # Number of lines of response per direction 
N_reconstructions = 32   # Number of 2D reconstructions to perform in parallel 

pet = occiput.Reconstruction.PET.PET_Multi2D_Scan()#PET_Static_Scan()
shape = [N_x, N_y, N_reconstructions]
pet.n_slices = N_reconstructions
pet.set_activity_shape(shape)
pet.set_activity_size(shape)
pet.set_attenuation_shape(shape)
pet.set_attenuation_size(shape)

pet.binning.N_axial = N_projections
pet.binning.N_azimuthal = 1 
pet.binning.N_u = N_bins
pet.binning.N_v = N_reconstructions + 1
pet.binning.size_u = N_bins
pet.binning.size_v = N_reconstructions + 1
pet.binning.angles_axial = linspace(0,pi,pet.binning.N_axial) 
pet.binning.angles_azimuthal = linspace(0,pi,pet.binning.N_azimuthal) 
pet.set_binning(pet.binning)

pet.binning.display_geometry()

center = [0.5,0.5,0.5]
activity = occiput.DataSources.Synthetic.Shapes.uniform_cylinder(shape,center=center, axis=2, radius=0.3, length=1.5)

figure(figsize=(14,4))
subplot(1,3,1); imshow(activity.data[int(N_x/2),:,:].reshape([N_x,N_reconstructions]), cmap='hot')
subplot(1,3,2); imshow(activity.data[:,int(N_y/2),:].reshape([N_y,N_reconstructions]), cmap='hot')
subplot(1,3,3); imshow(activity.data[:,:,int(N_reconstructions/2)].reshape([N_x,N_y]), cmap='hot')

projection = pet.project_activity(activity)

pet.set_prompts(projection)

a = pet.osem_reconstruction(iterations=15, subset_size=120)

figure(figsize=(14,4))
subplot(1,3,1); imshow(a.data[int(N_x/2),:,:].reshape([N_x,N_reconstructions]), cmap='hot')
subplot(1,3,2); imshow(a.data[:,int(N_y/2),:].reshape([N_y,N_reconstructions]), cmap='hot')
subplot(1,3,3); imshow(a.data[:,:,int(N_reconstructions/2)].reshape([N_x,N_y]), cmap='hot')



