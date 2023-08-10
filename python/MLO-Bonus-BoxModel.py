from IPython.display import Image
Image(url='http://www.atmosedu.com/physlets/GlobalPollution/waterbucket.gif')

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

import datetime

mlo=pd.read_csv('weekly_mlo.csv',skiprows=20,index_col='Date',parse_dates=True)
url='http://scrippsco2.ucsd.edu/sites/default/files/data/in_situ_co2/weekly_mlo.csv'
mlo=pd.read_csv(url,skiprows=21,names=['Date','CO2'],index_col='Date',parse_dates=True)
mlo.plot()

mlo.index

mlo.index[0]

mlo['elapsed'] = (mlo.index.to_series() - mlo.index.to_series()[0])

mlo.head()

mlo.elapsed.dtype

mlo['elapsed'] = (mlo.index.to_series() - mlo.index.to_series()[0])/datetime.timedelta(days=365)

mlo.head()

So=3.
R=.01
mlo['S']=So*np.exp(R*mlo.elapsed)
mlo.S.plot()







