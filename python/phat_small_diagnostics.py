get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'svg'")

from beast.plotting.plot_stats_check import make_diagnostic_plots
from beast.plotting.beastplotlib import set_params

set_params()

results_file = 'beast_example_phat/beast_example_phat_stats.fits'
fig = make_diagnostic_plots(results_file)

from beast.plotting.beastplotlib import plot_generic

from astropy.table import Table
from matplotlib.colors import LogNorm

# read in FITS table; this happens under the hood in make_diagnostic_plots
t = Table.read(results_file)
# make scatterplot of all values with a log-scale colormap of chi^2 values
fig, ax, cbar = plot_generic(t, 'logT_Exp', 'logL_Exp', plottype='scatter', 
                             plot_kwargs={'c'    : t['chi2min'],
                                          'norm' : LogNorm() })
fig.set_size_inches(6,5)

# add transparent circles for points with chi^2 values above 50
plot_generic(t, 'logT_Exp', 'logL_Exp', plottype='scatter', fig=fig, ax=ax, 
             thresh_col='chi2min', thresh=50, thresh_op='greater', 
             plot_kwargs={'facecolor' : 'none',
                          'edgecolor' : 'k' })

from pandas.tools.plotting import scatter_matrix
from beast.plotting.beastplotlib import fancify_colname

df = t.to_pandas() # convert table to Pandas dataframe

df_logL = df.filter(regex='logL') # select only columns with logL in name

sm = scatter_matrix(df_logL, diagonal='kde', figsize=(8,8))

# use fancify_colname function to convert column name axis labels to Texified format
for ax in sm.ravel():
    xlabel = fancify_colname(ax.get_xlabel())
    ax.set_xlabel(xlabel)
    ylabel = fancify_colname(ax.get_ylabel())
    ax.set_ylabel(ylabel)

sm = scatter_matrix(df[['logL_Exp','logT_Exp','M_ini_Exp','logg_Exp']], diagonal='kde', figsize=(8,8))
for ax in sm.ravel():
    xlabel = fancify_colname(ax.get_xlabel())
    ax.set_xlabel(xlabel)
    ylabel = fancify_colname(ax.get_ylabel())
    ax.set_ylabel(ylabel)



