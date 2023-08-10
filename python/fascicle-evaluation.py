import os.path as op

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import dipy.core.gradients as grad

from IPython.display import display, Image

dwi_ni = nib.load(op.join('data', 'SUB1_b2000_1.nii.gz'))
data = dwi_ni.get_data()
affine = dwi_ni.get_affine()
gtab = grad.gradient_table(op.join('data','SUB1_b2000_1.bvals'), op.join('data', 'SUB1_b2000_1.bvecs'))

candidate_prob = [s[0] for s in nib.trackvis.read('prob-track.trk', points_space='voxel')[0]]
candidate_det = [s[0] for s in nib.trackvis.read('det-track.trk', points_space='voxel')[0]]

import dipy.tracking.life as life
fiber_model = life.FiberModel(gtab)

fit_prob = fiber_model.fit(data, candidate_prob, affine=np.eye(4))
fit_det = fiber_model.fit(data, candidate_det, affine=np.eye(4))

fig, ax = plt.subplots(1)
ax.hist(fit_prob.beta, bins=100, histtype='step')
ax.hist(fit_det.beta, bins=100, histtype='step')
ax.set_xlim([-0.01, 0.05])
ax.set_xlabel('Streamline weights')
ax.set_label('# streamlines')

optimizied_prob = list(np.array(candidate_prob)[np.where(fit_prob.beta>0)[0]])

len(optimizied_prob)/float(len(candidate_prob))

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from dipy.data import read_stanford_t1
from dipy.tracking.utils import move_streamlines
from numpy.linalg import inv
t1 = nib.load(op.join('data', 'SUB1_t1_resamp.nii.gz'))
t1_data = t1.get_data()
t1_aff = t1.get_affine()

streamlines_actor = fvtk.streamtube(optimizied_prob, line_colors(optimizied_prob))

vol_actor = fvtk.slicer(t1_data)
vol_actor.display_extent(0, t1_data.shape[0]-1, 0, t1_data.shape[1]-1, 25, 25)
ren = fvtk.ren()
fvtk.add(ren, streamlines_actor)
fvtk.add(ren, vol_actor)
fvtk.camera(ren, viewup=(1,0,1), verbose=False)
fvtk.record(ren, out_path='life-prob-track.png', size=(600,600))

display(Image(filename='life-prob-track.png'))

