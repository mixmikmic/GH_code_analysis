import os.path as op
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import urllib as url
import nibabel as nib
import os.path as op
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as wdg
import IPython.display as display
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import dipy.core.geometry as geo

bvecs = np.loadtxt(op.join('data', 'SUB1_b2000_1.bvecs'))

bvecs.shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot3D expects something with length, so we convert into 1-item arrays:
for x,y,z in bvecs.T:
    ax.plot3D(np.ones(1) * x, np.ones(1) * y, np.ones(1) * z, 'o')

ni = nib.load(op.join('data', 'SUB1_b2000_1.nii.gz'))
data = ni.get_data()

data.shape

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_bvecs(elev=45, azim=45, x=40, y=50, z=35):
    this_vec = geo.sphere2cart(1, np.deg2rad(elev), np.deg2rad(azim))
    angles = np.arccos(np.dot(this_vec, bvecs[:, 10:]))
    closest = np.argmin(angles)
    n = closest + 10
    fig = plt.figure()
    #ax = fig.add_subplot(221, projection='3d')
    #a = Arrow3D([0, bvecs[0][n]],[0,bvecs[1][n]],[0,bvecs[2][n]], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax = fig.add_subplot(221)
    a = ax.arrow(0,0, bvecs[1][n], bvecs[0][n], linewidth=4, width=0.005, color=[np.abs(bvecs[0][n]), np.abs(bvecs[1][n]), 
                                                                    np.abs(bvecs[2][n])])
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    #ax.add_artist(a)
    ax.axis('off')
    #for set_lim in [ax.set_xlim, ax.set_ylim, ax.set_zlim]:
    #    set_lim([-1,1])
    ax = fig.add_subplot(222)
    ax.arrow(2,2,0,10, linewidth=4, width=0.1, color='r')
    ax.arrow(2,2,10,0, linewidth=4, width=0.1, color='g')
    ax.matshow(data[:, :, z, n], cmap=matplotlib.cm.bone)
    ax.axis('off')
    ax = fig.add_subplot(223)
    ax.axis('off')
    ax.arrow(2,2,0,10, linewidth=4, width=0.1, color='b')
    ax.arrow(2,2,10,0, linewidth=4, width=0.1, color='r')
    ax.matshow(np.rot90(data[:, y, :, n]), cmap=matplotlib.cm.bone)
    ax = fig.add_subplot(224)
    ax.axis('off')
    ax.arrow(2,2,0,10, linewidth=4, width=0.1, color='b')
    ax.arrow(2,2,10,0, linewidth=4, width=0.1, color='g')
    ax.matshow(np.rot90(data[x, :, :, n]), cmap=matplotlib.cm.bone)
    fig.set_size_inches(10, 10)

pb_widget = wdg.interactive(plot_bvecs, 
                            elev=wdg.FloatSlider(min=-180, max=180, step=1.0),
                            azim=wdg.FloatSlider(min=-180, max=180, step=1.0, value=90),
                            x=wdg.IntSlider(min=1, max=data.shape[0], value=data.shape[0]//2),
                            y=wdg.IntSlider(min=1, max=data.shape[1], value=data.shape[1]//2),
                            z=wdg.IntSlider(min=1, max=data.shape[2], value=data.shape[2]//2)
                            )
d = display.display(pb_widget)



