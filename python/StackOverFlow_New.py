import pandas as pd
import numpy as np
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
import seaborn as sns

raw_pandas_data = pd.read_csv('../data/QueryResults.csv', sep=',', encoding='latin1', parse_dates=['CreationDate'], dayfirst=True)


raw_pandas_data['year'] = raw_pandas_data['CreationDate'].apply(lambda x : x.year)

raw_pandas_data = raw_pandas_data[['year','PostTypeId']]

panda_data  = raw_pandas_data.groupby('year').aggregate(np.sum)

panda_data['average_post_per_month']  = panda_data['PostTypeId'].apply(lambda x : x/5 if x== 5937 else x/12 )

panda_data = panda_data[['average_post_per_month']]

panda_data

sns.set_context(rc={"figure.figsize": (15, 10)})
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
panda_data['average_post_per_month'].plot(kind = 'bar',color=sns.color_palette(flatui,3))

raw_pandas_data



