__depends__ = ["/Volumes/Seagate Expansion Drive/seq1_jcole_gene-editing_gene-editing_multiple/multiple"]
__dest__ = []

get_ipython().magic('matplotlib inline')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.options.display.mpl_style = 'default'
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
import seaborn as sns

print pd.__version__

plt.rcdefaults()
# Typeface sizes
from matplotlib import rcParams
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Computer Modern Roman']
#rcParams['text.usetex'] = True

# Optimal figure size
WIDTH = 350.0  # the number latex spits out
FACTOR = 0.90  # the fraction of the width you'd like the figure to occupy
fig_width_pt  = WIDTH * FACTOR

inches_per_pt = 1.0 / 72.27
golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good

fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
fig_height_in = fig_width_in * golden_ratio   # figure height in inches
fig_dims      = [fig_width_in, fig_height_in] # fig dims as a list

rcParams['figure.figsize'] = fig_dims

methods = ['noedits', 'zfn', 'talen', 'crispr', 'perfect']
for method in methods:
    # We have 10 relicates for each simulation
    for sim in xrange(1,11):
        # Load the individual allele frequency history files
        af = pd.read_csv('multiple/01_00/%s/%s/minor_allele_frequencies_%s.txt'%(method,sim, method), sep='\t')
        af.columns = ['generation', 'Brachyspina', 'HH1', 'HH2', 'HH3', 'HH4', 'HH5', 'BLAD', 'CVM', 'DUMPS',
                     'Mulefoot', 'Horned', 'Red']
        af['replicate'] = sim
        af['method'] = method
        if sim == 1 and method == methods[0]:
            all_replicates = af
        else:
            all_replicates = pd.concat([all_replicates, af])

all_replicates.head()

all_replicates = pd.melt(all_replicates,
                         id_vars = ['generation', 'replicate', 'method'],
                         value_vars = ['Brachyspina', 'HH1', 'HH2', 'HH3', 'HH4', 'HH5', 'BLAD', 'CVM',
                                       'DUMPS', 'Mulefoot', 'Horned', 'Red'],
                        var_name = 'recessive',
                        value_name = 'frequency')

all_replicates.head()

grouped = all_replicates.groupby(['generation', 'method', 'recessive']).mean().reset_index()
grouped.head()

expected = {}
actual = {}

for r in ['Brachyspina', 'HH1', 'HH2', 'HH3', 'HH4', 'HH5', 'BLAD', 'CVM', 'DUMPS', 'Mulefoot', 'Horned', 'Red']:
    expected[r] = {}
    actual[r] = {}
    for method in methods:
        expected[r][method] = []
        actual[r][method] = []
        # Red and horned are NOT lethals
        if r in ['Horned', 'Red']:
            for g in xrange(1,21):
                if g == 1:
                    expected[r][method].append(float(grouped[(grouped['generation']==g) & 
                                                             (grouped['method']==method) &
                                                             (grouped['recessive']==r)]
                                                     ['frequency']))
                else:
                    q0 = expected[r][method][g-2]
                    p0 = 1. - q0
                    q1 = (p0*q0) + q0**2
                    #print 'g: ', g, ', q0: ', q0, ', p0: ', p0, ', q1: ', q1
                    expected[r][method].append(q1)
                actual[r][method].append(float(grouped[(grouped['generation']==g) &
                                                 (grouped['method']==method) &
                                                 (grouped['recessive']==r)]
                                               ['frequency']))

        # The other recessives are lethals        
        else:
            for g in xrange(1,21):
                if g == 1:
                    expected[r][method].append(float(grouped[(grouped['generation']==g) & 
                                                             (grouped['method']==method) &
                                                             (grouped['recessive']==r)]
                                                     ['frequency']))
                else:
                    q0 = expected[r][method][g-2]
                    p0 = 1. - q0
                    q1 = (p0*q0) / (p0**2 + (2*p0*q0))
                    #print 'g: ', g, ', q0: ', q0, ', p0: ', p0, ', q1: ', q1
                    expected[r][method].append(q1)
                actual[r][method].append(float(grouped[(grouped['generation']==g) &
                                                 (grouped['method']==method) &
                                                 (grouped['recessive']==r)]
                                               ['frequency']))
#for k in expected[r].keys():
#print k, ':\t', expected[r][k], '\n'
    
print expected['HH1']['talen']
print
print actual['HH1']['talen']

sns.set(style="darkgrid")
for r in all_replicates['recessive'].unique():
    print r
    #sns.tsplot(data=grouped, time="generation", unit="replicate", condition="method", value="frequency")
    sns_plot = sns.tsplot(data=all_replicates[all_replicates['recessive']==r], time="generation", unit="replicate",
               condition="method", value="frequency")
    sns.plt.title('Change in recessive allele frequencies using %s (1%% of bulls and 0%% of cows edited)'%(r))
    plt.show()
    sns_plot.get_figure().savefig('multiple/01_00/rate_of_allele_frequency_change_multiple_01_00_%s.png'%(r), dpi=300)

sns.set(style="darkgrid")
for m in all_replicates['method'].unique():
    print m
    sns_plot = sns.tsplot(data=all_replicates[(all_replicates['method']==m) & (all_replicates['recessive'] != 'Horned')],
               time="generation", unit="replicate", condition="recessive", value="frequency")
    sns.plt.title('Change in recessive allele frequencies using %s (1%% of bulls and 0%% of cows edited)'%(m))
    plt.show()
    sns_plot.get_figure().savefig('multiple/01_00/rate_of_allele_frequency_change_multiple_01_00_%s.png'%(m), dpi=300)

xlabels = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
colors = itertools.cycle(['r', 'g', 'b'])
for r in ['Brachyspina', 'HH1', 'HH2', 'HH3', 'HH4', 'HH5', 'BLAD', 'CVM', 'DUMPS', 'Mulefoot', 'Horned', 'Red']:
    fig = plt.figure(figsize=(16, 12), dpi=300, facecolor='white')
    for i, m in enumerate(methods):
        ax = fig.add_subplot(3, 2, i+1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)  
        ax.set_title(m, weight='bold')
        ax.set_xlabel('Birth year')
        ax.set_ylabel('Allele frequency')
        ax.plot(actual[r][m], label='Observed', marker='o', c='b', mec='b')
        ax.plot(expected[r][m], label='Expected', c='gray')
        if r == 'Horned':
            ax.set_ylim(0.0, 1.1)
        else:
            ax.set_ylim(0.0, 0.10)
        ax.set_xlim(0, 20)
        # Manually set the number of ticks on the plot
        ax.set_xticks(np.arange(20))
        # Apply the tick labels
        ax.set_xticklabels(xlabels)
        # Turn off the top and right tick marks
        plt.tick_params(
            axis='both',       # changes apply to both axes
            which='both',      # both major and minor ticks are affected
            right='off',
            labelright='off',
            top='off',         # ticks along the top edge are off
            labeltop='off')    # labels along the bottom edge are off
        # Place the legend
        ax.legend(loc='best')
    # Use the recessive name as the title for each set of subplots
    plt.suptitle('%s (1%% bulls, 0%% cows edited)'%r, fontsize=20, weight='bold')
    plt.tight_layout(pad=1., w_pad=0.5, h_pad=0.95)
    # Tweak the layout so that the subplot titles don't overlap because tight_layout()
    # ignores suptitle().
    plt.subplots_adjust(top=0.915)
    plt.show()
    filename = 'holstein-act-vs-exp-%s-by-method_01_00.png' % ( r )
    fig.savefig(filename, dpi=300)

from statsmodels.sandbox.regression.predstd import wls_prediction_std
def fit_line(x, y):
    """Return RegressionResults instance of best-fit line."""
    #X = sm.add_constant(x)
    data = {'x':np.array(x), 'y':np.array(y)}
    fit = smf.ols(formula = 'y ~ x + I(x**2)', data=data).fit()
    print fit.summary()
    
    prstd, iv_l, iv_u = wls_prediction_std(fit)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, y, 'o', label="data")
    ax.plot(x, fit.fittedvalues, 'r', label="OLS")
    ax.plot(x, iv_u, 'r--')
    ax.plot(x, iv_l, 'r--')
    ax.legend(loc='best');
    ax.set_ylim(0.0, 0.10)
    ax.set_xlim(0., 20.)
    plt.show()
    
    return fit

fits = {}
years = [float(i) for i in xrange(1,21)]
for r in ['HH3']:
    fits[r] = {}
    for m in methods:
        print '\n\n==> ', m
        fit = fit_line(y=actual[r][m], x=years)
        fits[r][m] = fit

import statsmodels
print statsmodels.__version__
import matplotlib
print matplotlib.__version__
import seaborn
print seaborn.__version__

#import dask.dataframe as dd
methods = ['crispr', 'noedits', 'perfect', 'talen', 'zfn']
for method in methods:
    print method
    # We have 10 replicates for each simulation
    for sim in xrange(1,11):
        if sim == 1: print '\tReplicate: ', sim,
        elif sim < 10: print ', ', sim,
        else: print ', ', sim, ''
        # Load the individual history files
        #print '\t\tReading live cows.'
        lc = pd.read_csv('multiple/01_00/%s/%s/cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        #print '\t\tReading dead cows.'
        dc = pd.read_csv('multiple/01_00/%s/%s/dead_cows_history_%s_20.txt'%(method,sim,method), sep='\t')
        #print '\t\tReading live bulls.'
        lb = pd.read_csv('multiple/01_00/%s/%s/bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
        #print '\t\tReading dead bulls.'
        db = pd.read_csv('multiple/01_00/%s/%s/dead_bulls_history_%s_20.txt'%(method,sim,method), sep='\t')
        # Stack the individual animal datasets
        #print '\t\tConcatenating animal datasets'
        all_animals = pd.concat([lc, dc, lb, db], axis=0)
        all_animals['replicate'] = sim
        all_animals['method'] = method
        if method == methods[0] and sim == 1:
            #print '\t\tCreating initial dataframe for replicates.'
            #all_replicates = all_animals
            grouped = all_animals.groupby(['born', 'method']).mean().reset_index()
        else:
            #print '\t\tCreating successive dataframe for replicates.'
            grouped = pd.concat([grouped, all_animals.groupby(['born', 'method']).mean().reset_index()])

all_replicates['method'].value_counts()

#grouped = all_replicates.groupby(['born', 'method']).mean().reset_index()
grouped.head()

grouped.sort_values(by=['method','born'], inplace=True)

# Plot inbreeding by method
sns_plot = sns.tsplot(data=grouped, time="born", unit="replicate", condition="method", value="inbreeding")
sns.plt.title('Inbreeding rate by editing method (1% of bulls and 0% of cows edited)')
plt.show()
sns_plot.get_figure().savefig('multiple/01_00/rate_of_inbreeding_change_multiple_01_00.png', dpi=300)

sns.tsplot(data=grouped, time="born", unit="replicate", condition="method", value="TBV")
plt.show()

# http://stackoverflow.com/questions/22650833/pandas-groupby-cumulative-sum
grouped['TBV_cumulative'] = grouped.groupby(['method'])['TBV'].apply(lambda x: x.cumsum())

grouped.describe()

grouped['method'].value_counts()

for method in grouped['method'].value_counts().keys():
    print method, grouped.loc[(grouped['method'] == method) & (grouped['born'] == 20)]['TBV_cumulative']

for method in grouped['method'].value_counts().keys():
    print method, 'mean: ', grouped.loc[(grouped['method'] == method) & (grouped['born'] == 20)]['TBV_cumulative'].mean(),
    print 'stderr: ', grouped.loc[(grouped['method'] == method) & (grouped['born'] == 20)]['TBV_cumulative'].std()

import scipy.stats as stats
for pop1 in grouped['method'].value_counts().keys():
    for pop2 in grouped['method'].value_counts().keys():
        if pop1 != pop2:
            print pop1, ' ', pop2
            print stats.ttest_ind(a=grouped.loc[(grouped['method'] == pop1) & (grouped['born'] == 20)]['TBV_cumulative'],
                            b=grouped.loc[(grouped['method'] == pop2) & (grouped['born'] == 20)]['TBV_cumulative'],
                            equal_var=False)

sns.set(style="darkgrid")
for method in grouped['method'].value_counts().keys():
    sns_plot = sns.barplot(grouped[grouped['method'] == method]['born'],
                grouped[grouped['method'] == method]['TBV_cumulative'],
                palette="RdBu_r")
    plt.ylim(0,400000)
    plt.title(method)
    sns.plt.title('Cumulative genetic gain for %s (1%% of bulls and 0%% of cows edited)'%(method))
    sns_plot.get_figure().savefig('multiple/01_00/genetic_gain_multiple_%s_01_00.png'%(method), dpi=300)
    plt.show()



