get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import numpy as np
from catalog import Pink
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

pink = Pink.loader('Script_Experiments_Fractions_Trials/FIRST_WISE_W1_Norm_Log_3_12x12_Trial0/trained.pink')

pink.show_som()
first = pink.retrieve_som_data(channel=0)

pink.show_som(channel=1)
wise = pink.retrieve_som_data(channel=1)

def extract_neuron(dataset, x, y):
    '''Helper to extract a specific neuron fromt eh data
    '''
    binary, data, som_height, som_width, channel, n_width, n_height = dataset
    
    return data[y*n_width:(y+1)*n_width, 
                x*n_width:(x+1)*n_width]

def plot_neurons(f_n, w_n, save=None):
    fig, ax = plt.subplots(1,2)
    cmap=plt.get_cmap('bwr')

    if isinstance(f_n, int):
        f_nd = extract_neuron(first, f_n, w_n)
        w_nd = extract_neuron(wise, f_n, w_n)    
        f_n, w_n = f_nd, w_nd
    
    PIX=1.8
    OFF_X = f_n.shape[0]//2
    OFF_Y = f_n.shape[1]//2
    extent = [-OFF_X*PIX, OFF_X*PIX, -OFF_Y*PIX,OFF_Y*PIX]


    ax[0].imshow(f_n, cmap=cmap, norm=mpl.colors.SymLogNorm(0.03), extent=extent)
    ax[0].contour(w_n,origin='image', extent=extent, levels=np.arange(5*w_n.std(), w_n.max(), 8*w_n.std()),
                 linewidths=1, colors='black', cmap=None)
#     ax[0].contour(f_n,origin='image', extent=extent)
    ax[0].set(xlabel="$\Delta''$", ylabel="$\Delta''$", title='FIRST')
# 
#     ax[1].imshow(w_n, cmap=cmap, norm=mpl.colors.SymLogNorm(0.15), extent=extent)
    ax[1].imshow(w_n, cmap=cmap, extent=extent)
    ax[1].contour(f_n,origin='image', extent=extent, levels=np.arange(2*f_n.std(), f_n.max(), 8*f_n.std()),
                 linewidths=1, colors='black', cmap=None)
    ax[1].set(xlabel="$\Delta''$", ylabel="$\Delta''$", title='WISE W1')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position('right')

    fig.tight_layout()
    if save is None:
        fig.show()
    else:
        fig.savefig(save)
#     plt.close(fig)

fig, ax = plt.subplots(1,1, figsize=(10,10))
cmap=plt.get_cmap('bwr')

ax.imshow(first[1], cmap=cmap, norm=mpl.colors.SymLogNorm(0.03))

fig.show()

fig, ax = plt.subplots(1,1, figsize=(10,10))

ax.imshow(wise[1], cmap=cmap)
# ax.imshow(wise[1], cmap=cmap, norm=mpl.colors.SymLogNorm(0.03))

fig.show()

plot_neurons(4,9)

plot_neurons(8,6)

plot_neurons(0,0)

plot_neurons(6,7)

for x in range(12):
    for y in range(12):
        plot_neurons(x,y, save=f'Images/Neurons/{x}-{y}.pdf')

plt.close('all')













