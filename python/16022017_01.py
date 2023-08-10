import numpy as np
import scipy as sp
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.pyplot as plt
from matplotlib import animation
import networkx as nx
from time import strftime, gmtime
#import graph_tool.all as gt
get_ipython().magic('matplotlib inline')
import sys


cwd = getcwd()
dir_root = dirname(cwd)
filepath = os.path.join(dir_root, 'src')
sys.path.append(filepath) #('/home/tianpei/Dropbox/Codes/Python/LatNet/src/')
print(filepath)
get_ipython().magic('load_ext Cython')

from latent_random_network import latent_random_field as lat_rf 

savefigure = False
seed = 1000
choice = 'grid'
d = 10
if choice == 'newman':
    size = 20
    prob = 0
    option= {'seed': seed, 'node_dim': d, 'model': 'newman'}
    option['k-NN'] = 2  
    RF = lat_rf(size, prob, option)
    G0 = RF.graph_build(size=size, prob=prob, option=option, save_fig=savefigure)
elif choice == 'partition':
    size = [10, 10]
    prob = [0.9, 0.1]
    option= {'seed': seed, 'node_dim': d, 'model': 'partition'}
    option['cutoff_freq'] = 12
    RF = lat_rf(size, prob, option)
    G0 = RF.graph_build(size=size, prob=prob, option=option, save_fig=savefigure)
elif choice == 'grid':
    size = [20, 20]
    prob = 0.6
    d = 1
    option= {'seed': seed, 'node_dim': d, 'model': 'grid'}
    RF = lat_rf(size, prob, option)
    G0 = RF.graph_build(size=size, prob=prob, option=option)
elif choice == 'binomial':
    size = 20
    prob = 0.18
    option= {'seed': seed, 'node_dim': d, 'model': 'binomial'}
    RF = lat_rf(size, prob, option)
    G0 = RF.graph_build(size=size, prob=prob, option=option)
elif choice == 'tree':
    size = 20
    gamma = 3
    prob = 0
    option= {'seed': seed, 'node_dim': d, 'gamma': gamma ,'model': 'tree'}
    option['cutoff_freq'] = 11
    RF = lat_rf(size, prob, option)
    G0 = RF.graph_build(size=size, prob=prob, option=option, save_fig=savefigure)

from PGMcommon import ising_2dmodel

state, hist = ising_2dmodel(G0, size[0], size[1], 20)

ss = [v for v in G0.nodes_iter()]
list(zip(*[(n, i) for i, n in enumerate(G0.nodes_iter())]))
dict([(n, i) for i, n in enumerate(G0.nodes_iter())])

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure(4)
ax = fig.add_subplot(111)
cax = ax.matshow(hist[0], vmin=-0.2, vmax=0.2)

def animate(i):
    #print(i)
    cax.set_data(hist[i])  # update the data
    return cax,


# Init only required for blitting to give a clean slate.
def init():
    cax.set_data(hist[0]) 
    return cax,





# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(hist), interval=10, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('ising_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

