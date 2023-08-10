import seaborn as sns
import psyutils as pu

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('matplotlib inline')

sns.set_style("white")
sns.set_style("ticks")

# load data:
dat = pu.psydata.load_psy_data()
dat.info()

pu.psydata.binomial_binning(dat, y='correct', 
                            grouping_variables=['contrast', 'sf'])

pu.psydata.binomial_binning(dat, y='correct', 
                            grouping_variables=['contrast', 'sf'],
                            rule_of_succession=True)

g = pu.psydata.plot_psy(dat, 'contrast', 'correct', 
                        function='weibull',
                        hue='sf', 
                        col='subject',
                        log_x=True,
                        col_wrap=3,
                        errors=False,                        
                        fixed={'gam': .5, 'lam':.02}, 
                        inits={'m': 0.01, 'w': 3})
g.add_legend()
g.set(xlabel='Log Contrast', ylabel='Prop correct')
g.fig.subplots_adjust(wspace=.8, hspace=.8);

g = pu.psydata.plot_psy_params(dat, 'contrast', 'correct', 
                               x="sf", y="m",
                               function='weibull',
                               hue='subject', 
                               fixed={'gam': .5, 'lam':.02})
g.set(xlabel='Spatial Frequency', ylabel='Contrast threshold');

