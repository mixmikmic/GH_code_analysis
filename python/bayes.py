import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_palette('colorblind'); sns.set_color_codes()
import holoviews as hv
hv.notebook_extension('bokeh')

import param
import paramnb

import numpy as np
import pandas as pd

import scipy.stats as stats
import pymc3 as pm

get_ipython().run_cell_magic('opts', "Curve[width=200 height=200 tools=['hover']]", '%%opts Overlay [legend_position=\'top\' ]\n# ----------------------------------------------------------------------\ndef update_binomial_distribution( prior, success=6, trials=9):\n    likelihood      = stats.binom.pmf( success, trials, prior[\'p\'] )\n    unstd_posterior = likelihood * prior[\'Pr\']                # update the belief\n    posterior       = unstd_posterior / unstd_posterior.sum() # standardize to sum to 1\n\n    return pd.DataFrame( { "p" : prior[\'p\'], \'Pr\' : posterior } )\n# ----------------------------------------------------------------------\nsuccesses = [0,1,1,2,3,3,3,4,5]\nn_points  = 30\n\nprior     = pd.DataFrame( { \'p\' : np.linspace(0, 1, n_points), \\\n                            \'Pr\': np.repeat(1./n_points, n_points) })\n# --------------------------------------------------------------------\np = hv.Curve( prior, kdims=[\'p\'], vdims=[\'Pr\'], label=\'Initial Belief\')\np = p.redim(p=dict(range=(0,1)),Pr=dict(range=(0,0.23)))\nfor i in range(1, len(successes)):\n    s     = successes[i]\n    lbl   = \'n: \'+str(i)+\', w: \'+str(s)\n    p_prior = hv.Curve( prior, kdims=[\'p\'], vdims=[\'Pr\'])\n    \n    prior = update_binomial_distribution( prior, success=s-successes[i-1], trials=1)\n    \n    p = p + hv.Curve( prior, kdims=[\'p\'], vdims=[\'Pr\'], label=lbl)   \\\n            *p_prior\n\np=p.redim.range(Pr=(0,.095))\nhv.Layout(p).cols(3)\n#p.cols(3)')



