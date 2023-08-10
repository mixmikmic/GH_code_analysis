get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import matplotlib as mpl
import numpy as np
from catalog import Pink
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

pink = Pink.loader('Script_Experiments_Fractions_Trials/FIRST_Norm_NoLog_3_12x12_Trial0/trained.pink')

cl_csv = '/Users/gal16b/Documents/Postdoc_Work/rgz_rcnn/data/RGZdevkit2017/RGZ2017/ImageSets/Main/full_catalogue.csv'

cl_df = pd.read_csv(cl_csv)
cl_df

def reduc_func(s):
    '''Function to acquire the census levels of the objects
    '''
    filename = s.filename.split('_')[0]
    rows = cl_df[cl_df['first_id']==filename]
    res = []
    for k, r in rows.iterrows():
        label = f"{r['num_cpnts']}_{r['num_peaks']}"
        res.append((label, r['cl']))

    return res

book, counts = pink.attribute_heatmap(func=reduc_func, plot=False, realisations=100)

book

ex = book[(0,0)]
labels = list(set([i[0][0] for i in ex if len(i) > 0]))
labels.sort()

ex = book[(0,0)]
fig, ax = plt.subplots(1,1)
x = [labels.index(i[0][0]) for i in ex if len(i) > 0]
y = [i[0][1] for i in ex if len(i)>0]
ax.plot(x,y,'ro')

ax.set(xlim=[-0.5,5.5], ylim=[0.6,1.])

ax.xaxis.set_ticklabels(['a']+labels)

fig.show()

labels

def get_shape(book):
    max_shape = 0

    for i in book.keys():
        curr_shape = i[0]*i[1]
        if curr_shape > max_shape:
            max_shape = curr_shape
            shape = i

    return shape

def label_plot(book, shape, save=None, xtick_rotation=None, 
                color_map='gnuplot2', title=None, weights=None, figsize=(6,6),
                literal_path=False, count_text=False):
    '''Isolated function to plot the attribute histogram if the data is labelled in 
    nature

    book - dict
        A dictionary whose keys are the location on the heatmap, and values
        are the list of values of sources who most belonged to that grid
    shape - tuple
        The shape of the grid. Should attempt to get this from the keys or
        possible recreate it like in self.attribute_heatmap() 
    save - None or Str
        If None, show the figure on screen. Otherwise save to the path in save
    xtick_rotation - None or float
        Will rotate the xlabel by rotation
    color_map - str
        The name of the matplotlib.colormap that will be passed directly to matplotlib.pyplot.get_map()
    title - None of str
        A simple title strng passed to fig.suptitle()
    weights - None or dict
        If not None, the dict will have keys corresponding to the labels, and contain the total
        set of counts from the Binary file/book object. This will be used to `weigh` the contribution
        per neuron, to instead be a fraction of dataset type of statistic. 
    figsize - tuple of int
        Size of the figure to produce. Passed directly to plt.subplots
    literal_path - bool
        If true, take the path and do not modify it. If False, prepend the project_dir path
    count_label - bool
        If true, put as an anotation the counts of items in that neuron plot
    '''
    # Need access to the Normalise and ColorbarBase objects
    import matplotlib as mpl
    from collections import Counter
    from collections import defaultdict
    unique_labels = []

    for k, v in book.items():
        v = [i[0][0] for i in v if len(i) > 0]
        c = Counter(v)
        unique_labels.append(c.keys())

    unique_labels = list(set([u for labels in unique_labels for u in labels]))
    unique_labels.sort()

    fig, ax = plt.subplots(nrows=shape[0]+1, ncols=shape[1]+1, figsize=figsize)

    # Set empty axis labels for everything
    for a in ax.flatten():
#         a.set(xticklabels=[], yticklabels=[])
        a.set(xticklabels=[])

    for k, v in book.items():
        
        # Guard agaisnt most similar empty neuron
        if len(v) > 0:
            vals = defaultdict(list)
            for item in v:
                if len(item) > 0:
                    vals[item[0][0]].append(item[0][1])

            ax[k].boxplot([vals[i] for i in unique_labels], labels=unique_labels,sym='k.',
                         flierprops={'markersize':0.25})
            ax[k].set(ylim=[0.45,1.05])
            
        if k[1] != 0: 
            ax[k].set(yticklabels=[])
        if k[0] != shape[1]:
            ax[k].set(xticklabels=[])
        else:
            if xtick_rotation is not None:
                ax[k].tick_params(axis='x', rotation=xtick_rotation)
#                 for item in ax[k].get_xticklabels():
#                     item.set_fontsize(8.5)

    if title is not None:
        fig.suptitle(title, y=0.9)

    fig.subplots_adjust(hspace=0.05, wspace=0.05)
#     fig.tight_layout()
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

label_plot(book, get_shape(book), figsize=(12,9), xtick_rotation=90, save='Images/Consensus_Level_Example.png')



