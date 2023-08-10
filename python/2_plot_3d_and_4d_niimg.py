get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings("ignore")

from nilearn import datasets
import nibabel as nb
import os.path as op

path_data = op.expanduser('~/data_ucsf/machine_learning')
print('Datasets are stored in: %r' % datasets.get_data_dirs(path_data))

data = datasets.fetch_localizer_button_task(data_dir=path_data)
print(data['description'])

tmap_filenames = data['tmaps']
print(tmap_filenames)

# tmap_filenames is a list of filenames. We need to take the first one.
tmap_filename = tmap_filenames[0]

# Now load the data
brain = nb.load(tmap_filename)

brain.affine

brain.get_data()[:2]

# This is the dimensionality of the volume
# it is 3D, AKA a snapshot of activity
brain.get_data().shape

from nilearn import plotting

# Here we'll load the nifty object according to the statistical map
tmap = nb.load(tmap_filename)

# Then plot it
plotting.plot_stat_map(tmap)

# Alternatively, we could simply point it to the filename
# plotting.plot_stat_map(tmap_filename)

plotting.plot_stat_map(tmap_filename, threshold=3)

# For the y-axis
plotting.plot_stat_map(tmap_filename, threshold=3, display_mode='y')

plotting.plot_stat_map(tmap_filename, threshold=3,
                       display_mode='z', cut_coords=10)

rsn = datasets.fetch_atlas_smith_2009(data_dir=path_data)['rsn10']
print(rsn)

from nilearn import image

print(nb.load(rsn).shape)

first_rsn = image.index_img(rsn, 0)
print(first_rsn.shape)

plotting.plot_stat_map(first_rsn)

for img in image.iter_img(rsn):
    # img is now an in-memory 3D img
    plotting.plot_stat_map(img, threshold=3, colorbar=False,
                           cut_coords=(0, 0, 0))

