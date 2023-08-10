from utilities import *
import numpy as np
from MDPlus.core import Fasu, Cofasu
from MDPlus.analysis import pca
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#=====================
# You may play with the variables in this section later on in the tutorial:
n_reps = 8
#n_reps = 6
atom_selection = 'all'
#atom_selection = 'name CA'
#atom_selection = 'name CA and resid 7 to 89'
first_frame = 1
#first_frame = 100
last_frame = 2500
#======================

trajectory_files = ['rep{}/1rhw.md1.nc'.format(i + 1) for i in range(n_reps)]
replicate_ids = ['rep{}'.format(i + 1) for i in range(n_reps)]
topology_file = '1rhw_prot.pdb'
frames = slice(first_frame-1, last_frame)
f_list = [Fasu(trajectory_file, top=topology_file, selection=atom_selection, frames=frames) 
          for trajectory_file in trajectory_files]
dynein_data = Cofasu(f_list)
dynein_data.align()

plt.figure(figsize=(15,5))
plt.subplot(121)
plot_rmsd(dynein_data, replicate_ids)

plt.subplot(122)
plot_rmsf(dynein_data)

p = pca.fromtrajectory(dynein_data)

plt.figure(figsize=(15,15))
for dataset in range(8):
    plt.subplot(3, 3, dataset + 1)
    plot_pca(p, replicate_ids, highlight=dataset)



