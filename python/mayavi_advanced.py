get_ipython().magic('matplotlib qt')
import numpy as np
import matplotlib.image as mpimg

from mayavi import mlab
from tvtk.api import tvtk

mlab.clf()
im = mpimg.imread( 'ucbseal_139_540.png')

sigma = 50
xg, yg = np.mgrid[:im.shape[0], :im.shape[1]]
z = 100*np.exp( -(xg-im.shape[0]*.5)**2/sigma**2)*        np.exp( -(yg-im.shape[1]*.5)**2/sigma**2)

pts = mlab.points3d(xg, yg, z, mode='2dvertex')

im = (im*255).astype(np.int8)

sc=tvtk.UnsignedCharArray()
#the number of components can be 1 (scalar), 2 (scalar, alpha), 3 (RGB), or
#4 (RGBA)
sc.from_array(im.reshape(-1,4))
pts.mlab_source.dataset.point_data.scalars=sc
pts.mlab_source.dataset.modified()



