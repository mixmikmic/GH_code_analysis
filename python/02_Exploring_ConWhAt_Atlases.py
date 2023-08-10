# ConWhAt stuff
from conwhat import VolConnAtlas,StreamConnAtlas,VolTractAtlas,StreamTractAtlas
from conwhat.viz.volume import plot_vol_scatter

# Neuroimaging stuff
import nibabel as nib
from nilearn.plotting import plot_stat_map,plot_surf_roi

# Viz stuff
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import seaborn as sns

# Generic stuff
import glob, numpy as np, pandas as pd, networkx as nx

atlas_dir = '/scratch/hpc3230/Data/conwhat_atlases'
atlas_name = 'CWL2k8Sc33Vol3d100s_v01'

vca = VolConnAtlas(atlas_dir=atlas_dir + '/' + atlas_name,
                   atlas_name=atlas_name)

vca.atlas_name

vca.atlas_dir

vca.vfms.head()

vca.Gnx.edges[(10,35)]

img = vca.get_vol_from_vfm(1637)

plot_stat_map(img)

vca.bbox.ix[1637]

ax = plot_vol_scatter(vca.get_vol_from_vfm(1),c='r',bg_img='nilearn_destrieux',
                      bg_params={'s': 0.1, 'c':'k'},figsize=(20, 15))

ax.set_xlim([0,200]); ax.set_ylim([0,200]); ax.set_zlim([0,200]);

fig, ax = plt.subplots(figsize=(16,12))

sns.heatmap(np.log1p(vca.weights),xticklabels=vca.region_labels,
            yticklabels=vca.region_labels,ax=ax);
plt.tight_layout()

vca.cortex

vca.hemispheres

vca.region_mapping_fsav_lh

vca.region_mapping_fsav_rh

f = '/opt/freesurfer/freesurfer/subjects/fsaverage/surf/lh.inflated'
vtx,tri = nib.freesurfer.read_geometry(f)
plot_surf_roi([vtx,tri],vca.region_mapping_fsav_lh);

