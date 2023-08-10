get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# IO
import cPickle as pickle
import yaml

# Custom modules
import sys
sys.path.append('..')
from common import voxel_data
from common import mesh
from common import images

# setting the paths to the data we want to load
base_path = ('/media/michael/Seagate/phd_projects/'
             'volume_completion_data/data/for_release/')

scene_folder = base_path + 'fold_0/saved_00169/'
image_path = scene_folder + 'poses/000035.yaml'
voxel_data_path = scene_folder + 'tsdf.dat'
voxel_meta_path = scene_folder + 'tsdf_meta.yaml'

# Before calling load_voxels, we must have 'common' on the path
# This is done above, in the line:
#  sys.path.append('..')
vox = voxel_data.WorldVoxels.load_from_dat(
    voxel_data_path, voxel_meta_path)

# Plotting a slice through the TSDF
vox_slice = vox.V[:, :, vox.V.shape[2] / 2]
vox_slice[np.isnan(vox_slice)] = np.nanmin(vox_slice)

plt.figure(figsize=(7, 7))
plt.imshow(vox_slice, cmap = 'bwr')
plt.colorbar(label='TSDF value')

# converting voxel data to a mesh, using marching cubes
ms = mesh.Mesh()
ms.from_volume(vox)

# extract vertices from mesh above a certain height
ms_x, ms_y, ms_z = ms.vertices.T
threshold = (np.nanmin(ms_z) + np.nanmax(ms_z)) / 2.0
idxs_to_plot = ms_z > threshold

# plotting these vertices
plt.figure(figsize=(12, 12))
plt.plot(ms_x[idxs_to_plot], ms_y[idxs_to_plot], '.')
plt.axis('equal');

# loading in image data
image_dict = yaml.load(open(image_path))
im = images.RGBDImage.load_from_dict(scene_folder, image_dict)

# displaying image
plt.figure(figsize=(10, 5))
im.print_info()
im.disp_channels()

# projecting the depth image into 3D and plotting
im_xyz = im.get_world_xyz()
plt.figure(figsize=(10, 5))
plt.plot(im_xyz[:, 0], im_xyz[:, 1], '.', ms=0.5)
plt.axis('equal')

# Plotting the mesh in the same coordinates as the depth image
plt.figure(figsize=(10, 5))
plt.plot(im_xyz[:, 0], im_xyz[:, 1], '.', ms=0.5)
plt.plot(ms_x[idxs_to_plot], ms_y[idxs_to_plot], '.r')
plt.axis('equal')





