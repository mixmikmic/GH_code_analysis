get_ipython().magic('matplotlib inline')


import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib import gridspec

import numpy as np

def add_relative_to_current_source_file_path_to_sys_path(relpath):
    import os, sys, inspect
    path = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0],relpath)))
    if path not in sys.path:
        sys.path.insert(0,path)

add_relative_to_current_source_file_path_to_sys_path("../trajectories")   

import optimized_spherical_paths as osp
import numpy as np

PA = np.array([-98.200, -40.600,      0.])
PB = np.array([-98.200, -38.600,      0.])
C0 = np.array([-95.8613047,  -35.354918,    -1.58666675])
C1 = np.array([-95.8613047,  -43.84508,     -1.58666675])
#C0 = np.array([-95.8613047,  -35.354918,    0])
#C1 = np.array([-95.8613047,  -43.84508,     0])

params = {u'minDist': 4.0}

sigma, wA, sigmaAvg, sigmaA, sigmaB, t = osp.calculate_position_trajectory_as_optimized_blend_of_spherical_trajectories(
    PA, PB, C0, C1, osp.real_optimizer_unconstrained_at_endpoints, params)

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

ax.scatter(sigma[:,0],sigma[:,1],sigma[:,2])



