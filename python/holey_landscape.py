get_ipython().magic('matplotlib inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pandas as pd
import seaborn as sns
sns.set(
    style='ticks',
    context='paper',
    font_scale=2
)

red = sns.color_palette('Reds',255)[-1]
blue = sns.color_palette('Blues',255)[-1]

def show(m, cmap='viridis'):
    plt.matshow(m, cmap=cmap)
    plt.xticks([])
    plt.yticks([])

n = 10
d = 2
p = 0.2

shape = (n,) * d
r = np.random.rand(*shape)
show(r)

m = r < 0.2
show(m)

labels, k = ndimage.label(m)
show(labels, cmap='Set1')

biggest = max((labels==lbl).sum() for lbl in range(1, k))
biggest

def biggest_component(n, d, p):
    shape = (n,) * d
    m = np.random.rand(*shape)
    m = m < p
    m, k = ndimage.label(m)
    if k < 2:
        return 0
    return max((m == lbl).sum() for lbl in range(1, k)) / m.size

biggest_component(n, d, p)

n = 30
d = 3
ps  = np.linspace(0.1, 0.5, 50)
nsamples = 10
biggest = [
    [
        biggest_component(n, d, p) 
        for _ in range(nsamples)
    ]
    for p in ps
]

df = pd.DataFrame(biggest)
df['p'] = ps
df = pd.melt(df, id_vars='p', var_name='sample', value_name='max size')
df.head()

fig = plt.figure(figsize=(12, 12)) 
gs = mpl.gridspec.GridSpec(nrows=2, ncols=2)

reds = [(1,1,1), (1, 0, 0), (0.25, 0, 0)]
red_cm = mpl.colors.LinearSegmentedColormap.from_list(
        'red_cm', colors=reds)
blues = [(1,1,1), (0, 0, 1), (0, 0, 0.25)]
blue_cm = mpl.colors.LinearSegmentedColormap.from_list(
        'blue_cm', colors=blues)

ax0 = plt.subplot(gs[0, :])
ax1 = plt.subplot(gs[1, 0])
ax2 = plt.subplot(gs[1, 1])
p1 = 0.2
p2 = 0.4

sns.tsplot(df, time='p', unit='sample', value='max size', ci=99,
           color='k', lw=3, ax=ax0)
ax0.plot(p1, 0.001, 'o', color=blues[2], markersize=11)
ax0.plot(p2, 0.345, 'o', color=reds[2], markersize=11)

ax0.axvline(0.31, color='k', ls='--')
ax0.text(0.308, 0.52, '$P_c$')
ax0.set(
    xlabel=('P'),
    ylabel=('Relative size of largest component')
)

shape = (n,) * d
m = np.random.rand(*shape)
m = m < p1
labels, k = ndimage.label(m)
largest_comp = max(range(1,k), key=lambda i: (labels==i).sum())
labels[(labels>0) & (labels!=largest_comp)] = 1
labels[labels==largest_comp] = 2
ax1.matshow(labels[0], cmap=blue_cm, vmin=0, vmax=2)
patches = [ 
    mpl.patches.Patch(color=blues[2], label="largest"),
    mpl.patches.Patch(color=blues[1], label="all other")
]
ax1.legend(handles=patches, frameon=True, fancybox=True, loc='lower right', 
           borderpad=0.5, edgecolor='k', framealpha=0.9, 
           handlelength=0.5, handletextpad=0.2)
ax1.set(
    xticks=[], yticks=[]
)
m = np.random.rand(*shape)
m = m < p2
labels, k = ndimage.label(m)
largest_comp = max(range(1,k), key=lambda i: (labels==i).sum())
labels[(labels>0) & (labels!=largest_comp)] = 1
labels[labels==largest_comp] = 2
ax2.matshow(labels[0], cmap=red_cm , vmin=0, vmax=2)
patches = [ 
    mpl.patches.Patch(color=reds[2], label="largest"),
    mpl.patches.Patch(color=reds[1], label="all other")
]
ax2.legend(handles=patches, frameon=True, fancybox=True, loc='lower right', 
           borderpad=0.5, edgecolor='k', framealpha=0.9, handlelength=0.5, handletextpad=0.2)
ax2.set(
    xticks=[], yticks=[]
)

for ax, letter in zip([ax0, ax1, ax2], 'ABCDEFG'):
    ax.annotate(letter, xy=(0.02, 0.91), xycoords='axes fraction', fontsize=36, bbox=dict(fc="w", lw=0, alpha=0.75))

fig.tight_layout() 
sns.despine(ax=ax0)
fig.savefig('holey_landscape.tif', dpi=300, papertype='a4')

