import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy
import seaborn
import numpy

get_ipython().magic('matplotlib inline')



# Setup the plot

cols = ['Southern Hemisphere', 'Northern Hemisphere']
rows = ['TOA', 'Surface radiation', 'Surface heat', 'Ocean']

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 12))
#plt.setp(axes.flat, ylabel='Y-label')

pad = 5 # in points

for ax, col in zip(axes[0], cols):
    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

for ax, row in zip(axes[:,0], rows):
    ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation='vertical')

# Put in the data

rsdt_trend = 0.5
rsut_trend = 0.4
rlut_trend = -0.1
rndt_trend = 0.8

toa_values = (rndt_trend, rsdt_trend, rsut_trend, rlut_trend)

ind = numpy.arange(len(toa_values))  # the x locations for the groups
width = 0.7  
    
axes[0,0].bar(ind, toa_values, width,
              color=['r', 'None', 'None', 'None'],
              edgecolor=['r', 'r', 'b', 'b'],
              tick_label=['rndt', 'rsdt', 'rsut', 'rlut'],
              linewidth=1.0)
    
fig.tight_layout()
# tight_layout doesn't take these labels into account. We'll need 
# to make some room. These numbers are are manually tweaked. 
# You could automatically calculate them, but it's a pain.
fig.subplots_adjust(left=0.15, top=0.95)

plt.show()

axes.shape



