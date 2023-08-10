import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style('whitegrid')

get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

chip_file = 'datasets/chipotle.tsv'

# A:
df = pd.read_csv(chip_file, delimiter='\t')
df.tail()

# A:
def add_sub_order(x):
    x['sub_order'] = np.arange(len(x)) + 1
    return x

grouped_pd = df.groupby('order_id')
grouped_pd = grouped_pd.apply(add_sub_order)
df_new = grouped_pd[['order_id', 'quantity', 'sub_order', 'item_name', 'choice_description', 'item_price']]
df_new.head()

def clean_column(x):
    x = x.replace('$','')
    return x

# A:
df_new['item_price'] = df_new['item_price'].map(clean_column)
df_new['item_price'] = df_new['item_price'].astype(dtype='float64')
df_new.info()

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

# A:

