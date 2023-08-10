get_ipython().magic('run EUVIP_1_defaults.ipynb')

get_ipython().magic('cd -q ../test/')

import numpy as np
import MotionClouds as mc
import matplotlib.pyplot as plt

# PARAMETERS
seed = 2042
np.random.seed(seed=seed)
N_sparse = 5
sparse_base = 2.e5
sparseness =  np.logspace(-1, 0, N_sparse, base=sparse_base, endpoint=True)
print(sparseness)

# TEXTON
N_X, N_Y, N_frame = 256, 256, 1
fx, fy, ft = mc.get_grids(N_X, N_Y, 1)
mc_i = mc.envelope_gabor(fx, fy, ft, sf_0=0.05, B_sf=0.025, B_theta=np.inf)#, V_X=)

values = np.random.randn(N_X, N_Y, N_frame)
chance = np.argsort(-np.abs(values.ravel()))
chance = np.array(chance, dtype=np.float)
chance /= chance.max()
chance = chance.reshape((N_X, N_Y, N_frame))

fig, axs = plt.subplots(1, N_sparse, figsize=(fig_width, fig_width/N_sparse))
for i_ax, l0_norm in enumerate(sparseness):

    threshold = 1 - l0_norm
    mask = np.zeros_like(chance)
    mask[chance > threshold] = 1.
    
    im = 2*mc.rectif(mc.random_cloud(mc_i, events=mask*values))-1
                
    axs[i_ax].imshow(im[:, :, 0], vmin=-1, vmax=1, cmap=plt.gray())
    axs[i_ax].text(4, 40, r'$\epsilon=%.0e$' % l0_norm, color='white', fontsize=8)
    axs[i_ax].set_xticks([])
    axs[i_ax].set_yticks([])
plt.tight_layout()
fig.subplots_adjust(hspace = .0, wspace = .0, left=0.0, bottom=0., right=1., top=1.)
mp.savefig(fig,  experiment + '_droplets', figpath = '../docs/');

get_ipython().magic('cd -q ../notebooks/')

