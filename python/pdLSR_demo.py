from __future__ import print_function
import os
import inspect

import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.2f' % x)
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_palette('dark')
sns.set_style('ticks')

get_ipython().magic('matplotlib inline')

# Import pdLSR
try:
    # If pdLSR is in PYTHONPATH (or is installed), then use a direct import
    import pdLSR
    
except ImportError:
    # Attempt to demo pdLSR without installing by importing from directory
    pdLSR_path = '../../pdLSR'
    
    print("Module pdLSR was not found in PYTHONPATH. Looking for module in directory '{:s}'".format(pdLSR_path))
    
    if os.path.exists(pdLSR_path):
        import imp
        pdLSR = imp.load_package('pdLSR', pdLSR_path)
        print("Module pdLSR was found in the directory '{:s}' and imported.".format(pdLSR_path))
    else:
        raise ImportError("Module pdLSR could not be found in the directory '{:s}'.".format(pdLSR_path) +         "This demonstration will not run until the module is located.")

data = pd.read_csv('GCN4_twofield.tsv', sep='\t')
get_ipython().system(' head GCN4_twofield.tsv')

groupby = ['resi', 'field']
xname = 'time'
yname = 'intensity'
yerr = None

exponential_decay = pdLSR.functions.exponential_decay
print(inspect.getsource(exponential_decay))

params = [{'name':'inten', 
           'value':np.asarray(data.groupby(groupby)[yname].max()), 
           'vary':True},
          {'name':'rate', 
           'value':20.0, 
           'vary':True}]

minimizer_kwargs = {'params':params,
                    'method':'leastsq',
                    'sigma':0.95,
                    'threads':None}

fit_data = pdLSR.pdLSR(data, exponential_decay, groupby, 
                       xname, yname,
                       minimizer='lmfit',
                       minimizer_kwargs=minimizer_kwargs)
fit_data.fit()
fit_data.predict()

resi = 51
field = 14.1

fit_data.data.loc[(resi, field)]

fit_data.results.head(n=4)

fit_data.stats.head(n=4)

fit_data.covar.loc[(resi, field)]

fit_data.pivot_covar().loc[(resi, field)].values

fit_itensities = fit_data.results.loc[:,('inten','value')]

fit_data.data['intensity'] = fit_data.data.intensity.div(fit_itensities)
fit_data.model['ycalc'] = fit_data.model.ycalc.div(fit_itensities)

plot_data = pd.concat([fit_data.data, 
                       fit_data.model], axis=0).reset_index()

colors = sns.color_palette()

palette = {14.1:colors[0], 18.8:colors[2]}

grid = sns.FacetGrid(plot_data, col='resi', hue='field', palette=palette,
                     col_wrap=3, size=2.0, aspect=0.75, 
                     sharey=True, despine=True)


grid.map(plt.plot, 'xcalc', 'ycalc', marker='', ls='-', lw=1.0)
grid.map(plt.plot, 'time', 'intensity', marker='o', ms=5, ls='')

grid.set(xticks=np.linspace(0.05, 0.25, 5),
         ylim=(-0.1, 1.05))

ax = grid.axes[0]
legend = ax.get_legend_handles_labels()
ax.legend(legend[0][2:], legend[1][2:], loc=0, frameon=True)

f = plt.gcf()
f.set_size_inches(12,8)
f.subplots_adjust(wspace=0.2, hspace=0.25)

plot_data = (fit_data.results
             .sort_index(axis=1)
             .loc[:,('rate',['value','stderr'])]
             )
plot_data.columns = plot_data.columns.droplevel(0)
plot_data.reset_index(inplace=True)

fig = plt.figure()
fig.set_size_inches(7,5)
ax = plt.axes()

palette = [colors[0], colors[2]]

for pos, (field, dat) in enumerate(plot_data.groupby('field')):
    _ = dat.plot('resi', 'value', yerr='stderr',
                 kind='bar', label=field, color=palette[pos],
                 position=(-pos)+1, ax=ax, width=0.4)
    
ax.set_ylabel('decay rate (s$^{-1}$)')
ax.set_xlabel('residue')
ax.set_xlim(ax.get_xlim()[0]-0.5, ax.get_xlim()[1])
plt.xticks(rotation=0)

sns.despine()
plt.tight_layout()

