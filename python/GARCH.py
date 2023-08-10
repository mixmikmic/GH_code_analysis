import arch as arch
import pandas as pd
import numpy as np

def get_inst_vol(y, 
                 annualize,
                 x = None, 
                 mean = 'Constant', 
                 vol = 'Garch', 
                 dist = 'normal', 
                 ):

    """Fn: to calculate conditional volatility of an array using Garch:


    params
    --------------
    y : {numpy array, series, None}
        endogenous array of returns
    x : {numpy array, series, None}
        exogneous
    mean : str, optional
           Name of the mean model.  Currently supported options are: 'Constant',
           'Zero', 'ARX' and  'HARX'
    vol : str, optional
          model, currently supported, 'GARCH' (default),  'EGARCH', 'ARCH' and 'HARCH'
    dist : str, optional
           'normal' (default), 't', 'ged'

    returns
    ----------

    series of conditioanl volatility. 
    """
    if isinstance(y, pd.core.series.Series):
        ## remove nan.
        y = y.dropna()
    elif isinstance(y, np.ndarray):
        y = y[~np.isnan(y)]

    # provide a model
    model = arch.arch_model(y * 100, mean = 'constant', vol = 'Garch')

    # fit the model
    res = model.fit(update_freq= 5)

    # get the parameters. Here [1] means number of lags. This is only Garch(1,1)
    omega = res.params['omega']
    alpha = res.params['alpha[1]']
    beta = res.params['beta[1]']

    
    # more interested in conditional vol
    if annualize.lower() == 'd':
        ann_cond_vol = res.conditional_volatility * np.sqrt(252)
    elif annualize.lower() == 'm':
        ann_cond_vol = res.conditional_volatility * np.sqrt(12)
    elif annualize.lower() == 'w':
        ann_cond_vol = res.conditional_volatility * np.sqrt(52)
    return ann_cond_vol * 0.01    

## This is for styling notebook on nbviewer
from IPython.display import HTML
def css():
    style = open('css/custom.css', 'r').read()
    return HTML(style)
css()

