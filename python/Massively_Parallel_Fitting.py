get_ipython().magic('pylab inline')
import pandas as pd
import pysd
import scipy.optimize
import multiprocessing
import numpy as np
import seaborn

data = pd.read_csv('../../data/Census/Males by decade and county.csv', header=[0,1], skiprows=[2])
data.head()

model = pysd.read_vensim('../../models/Aging_Chain/Aging_Chain.mdl')

param_names = ['dec_%i_loss_rate'%i for i in range(1,10)]

def error(param_vals, measurements):
    predictions = model.run(params=dict(zip(param_names, param_vals)),
                            initial_condition=(2000,measurements['2000']),
                            return_timestamps=2010,
                            rtol=1).loc[2010]

    errors = predictions - measurements['2010']
    return sum(errors.values[1:]**2) #ignore first decade: no birth info

def fit(row):
    res = scipy.optimize.minimize(error, args=row,
                                  x0=[.05]*9,
                                  method='L-BFGS-B');
    return pd.Series(index=['dec_%i_loss_rate'%i for i in range(1,10)], data=res['x'])

get_ipython().run_cell_magic('capture', '', 'county_params = data.apply(fit, axis=1)')

df2 = county_params.drop('dec_1_loss_rate',1)
df2.plot(kind='hist', bins=np.arange(-.15,.4,.01), alpha=.4, histtype='stepfilled')
plt.xlim(-.15,.4)
plt.title('Fit yearly loss rates from each US county\n by age bracket from 2000 to 2010', fontsize=16)
plt.ylabel('Number of Counties', fontsize=16)
plt.xlabel('Yearly Loss Rate in 1% Brackets', fontsize=16)
plt.legend(frameon=False, fontsize=14)
plt.savefig('Loss_Histogram.svg')

get_ipython().run_cell_magic('capture', '', '\ndef _apply_df(args):\n    df, func, kwargs = args\n    return df.apply(func, **kwargs)\n\ndef apply_by_multiprocessing(df, func, workers, **kwargs):\n    pool = multiprocessing.Pool(processes=workers)\n    result = pool.map(_apply_df, [(d, func, kwargs) for d in np.array_split(df, workers)])\n    pool.close()\n    return pd.concat(list(result))\n\ncounty_params = apply_by_multiprocessing(data[:10], fit, axis=1, workers=4)')

