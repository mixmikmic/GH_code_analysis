import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data/newyork/NYClatlong.csv').iloc[:,1:]
df.rename({'neighbourhood_cleansed': 'neighborhood', 
           'host_is_superhost': 'superhost',
           'is_business_travel_ready': 'business'
          }, axis='columns', inplace=True)
df.info()

def str2bool(x):
    return str(x).lower() == 't'

df['superhost'] = df.superhost.apply(str2bool)
df['business'] = df.business.apply(str2bool)


def clean_price(x):
    y = x.split('.')[0]          .replace('$', '')          .replace(',', '') 
    return y

df['price'] = df.price.apply(clean_price).astype('uint16')

df['price_p_person'] = round(df.price / df.accommodates, 2)

df.price[(df.price < 1000)].plot(kind='hist', bins=40)
df.price.describe(percentiles=list(np.arange(0,1,.1)))

df.price_p_person[(df.price_p_person < 250)].plot(kind='hist', bins=35)
df.price_p_person.describe(percentiles=list(np.arange(0,1,.1)))

columns = ['price','superhost','business','price_p_person','latitude','longitude']
aggs = df.groupby('neighborhood')[columns].agg({'average':'mean', 'count':'count'})
aggs.columns = aggs.columns.get_level_values(1)

colnames = {'price':'avg_price','superhost':'avg_superhost',
            'business':'avg_business','price_p_person':'avg_price_p_person',
            'latitude':'avg_lat','longitude':'avg_long'}
aggs = aggs.reset_index().iloc[:,:8].rename(colnames, axis='columns')
aggs['count'] = aggs.iloc[:,-1]

data = pd.merge(df, aggs, how='left', on='neighborhood')
cnt = data.iloc[:, -1]
data = data.iloc[:, :19]
data['count'] = cnt
data.head(2)

data.to_csv('data/clean_latlong.csv')







