import pandas as pd
import numpy as np
import matplotlib
from IPython.display import IFrame

import matplotlib
# display graphs inline
get_ipython().magic('matplotlib inline')

# Make graphs prettier
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
#pd.set_option('display.mpl_style', 'default')
matplotlib.style.use('ggplot')
# Make the fonts bigger
matplotlib.rc('figure', figsize=(14, 7))
matplotlib.rc('font', family='normal', weight='bold', size=22)
import matplotlib.pyplot as plt

IFrame('https://data.stackexchange.com/stackoverflow/query/new', width=700, height=700)

fixed_df = pd.read_csv('../data/so_pandas.csv', sep=',', encoding='latin1', parse_dates=['CreationDate'], dayfirst=True)



fixed_df.head()


fixed_df['year'] = fixed_df['CreationDate'].apply(lambda x : x.year)



new_df = fixed_df.groupby('year').aggregate(np.sum)

new_df['PostTypeId'].plot(kind = 'bar')





